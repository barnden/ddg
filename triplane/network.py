import torch
import numpy as np
from torch import nn
from score_sde.models import up_or_down_sampling, utils, layers, layerspp
from score_sde.models.layers import default_init
from score_sde.models.layerspp import conv3x3, Combine
from score_sde.models.discriminator import TimestepEmbedding
from score_sde.models.dense_layer import dense, conv2d
from score_sde.models.ncsnpp_generator_adagn import PixelNorm
from triplane.layers import BigGANResNetBlock
from triplane.volumetric_rendering.ray_sampler import RaySampler
import functools

class NeRFPosEnc(nn.Module):
    def __init__(self, max_freq):
        super().__init__()

        self._scales = 2 ** torch.arange(0, max_freq)

    def forward(self, x):
        xb = x[..., None, :] * self._scales[:, None]
        xb = torch.reshape(xb, (*x.shape[:-1], -1))
        emb = torch.sin(torch.concat([xb, xb + torch.pi / 2], dim=-1))
        emb = torch.concat([x, emb], dim=-1)

        return emb

    def _apply(self, fn):
        super(NeRFPosEnc, self)._apply(fn)
        self._scales = fn(self._scales)

class RayEncoder(nn.Module):
    def __init__(self, nf, max_pos_freq, max_dir_freq):
        super().__init__()

        self.posenc = NeRFPosEnc(max_pos_freq)
        self.direnc = NeRFPosEnc(max_dir_freq)

        emb_ch = (max_pos_freq + 1) * (max_dir_freq + 1)

        self.Dense_0 = nn.Linear(emb_ch, nf * 4)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
        nn.init.zeros_(self.Dense_0.bias)

        self.act = nn.SiLU()

        self.Dense_1 = nn.Linear(nf * 4, nf * 4)
        self.Dense_1.weight.data = default_init()(self.Dense_1.weight.shape)
        nn.init.zeros_(self.Dense_1.bias)

    def forward(self, rpos, rdir):
        emb = torch.concat([self.posenc(rpos), self.direnc(rdir)], dim=-1)
        emb = self.Dense_0(emb)
        emb = self.act(emb)
        emb = self.Dense_1(emb)

        return emb

class RayUpsampler(nn.Module):
    def __init__(self, nf, factor=2):
        super().__init__()

        self.conv = conv3x3(nf * 4, nf * 4)
        self.factor = factor

    def forward(self, x):
        x = up_or_down_sampling.upsample_2d(x, k=[1, 3, 3, 1], factor=self.factor)
        x = self.conv(x)

        return x

@utils.register_model(name="trincsnpp")
class TriNCSNpp(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        embed_rays = config.embed_rays

        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        if resblock_type != 'biggan':
            raise NotImplementedError("Unsupported resblock type")

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            #assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        ResnetBlock = functools.partial(
            BigGANResNetBlock,
            act=act,
            dropout=dropout,
            fir=fir,
            fir_kernel=fir_kernel,
            init_scale=init_scale,
            skip_rescale=skip_rescale,
            temb_dim=nf * 4,
            zemb_dim=z_emb_dim,
        )

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                        with_conv=resamp_with_conv,
                                        fir=fir,
                                        fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample,
                                        with_conv=resamp_with_conv,
                                        fir=fir,
                                        fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)
        
        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)

        # Downsampling block
        channels = config.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf

        # Conv layers to downsample

        if embed_rays:
            RayDownsampler = functools.partial(conv3x3, nf * 4, nf * 4)
            self.ray_downsample = nn.ModuleList([
                RayDownsampler(stride=2 ** i_level)
                for i_level in range(num_resolutions)
            ])

        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))

                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch


            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')
            
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            
            if False:
                # Generate x_0 prediction directly from UNet
                modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            elif False:
                # Generate feature planes for triplane rendering
                modules.append(conv3x3(in_ch, 32 * 3, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [
            PixelNorm(),
            dense(config.nz, z_emb_dim),
            self.act,
        ]

        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)

        self.z_transform = nn.Sequential(*mapping_layers)

        if embed_rays:
            self.ray_transform = RayEncoder(nf, 15, 8)

        # Triplane representation:
        #   Original NCSNpp outputs predicted mean with size M x M x 3
        #   Apply more upsampling blocks to achieve features N x N x 3f
        #   RenderDiffusion (arXiv:2112.07804) uses M=64, N=256 for ShapeNet, and M=N=32 for CLEVR1

        # FIXME: Detect if N > M, if so then add more ResNet layers to upsample
        if True:
            # FIXME: Don't hardcode.
            nf_mult = [3]
            self.num_feature_resolutions = num_feature_resolutions = len(nf_mult)

            if embed_rays:
                self.ray_upsample = nn.ModuleList([RayUpsampler(nf, 2 ** (i_level + 1)) for i_level in range(num_feature_resolutions)])

            self.feature_modules = nn.ModuleList([layerspp.conv3x3(in_ch, self.nf)])
            in_ch = self.nf

            for i_level in range(num_feature_resolutions):
                for i_block in range(num_res_blocks):
                    # FIXME: Don't hardcode
                    out_ch = nf_mult[i_level] * 32
                    self.feature_modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                    in_ch = out_ch

                # RenderDiffusion always has an attention layer in each block
                self.feature_modules.append(AttnBlock(channels=in_ch))

                if self.resblock_type == 'ddpm':
                    self.feature_modules.append(Upsample(in_ch=in_ch))
                else:
                    self.feature_modules.append(ResnetBlock(in_ch=in_ch, up=True))

            self.feature_modules.append(conv3x3(in_ch, 3 * 32))

    def forward(self, x, time_cond, z, rpos=None, rdir=None):
        # timestep/noise_level embedding; only for continuous training
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")

        if rpos is not None and rdir is not None:
            cemb = self.ray_transform(rpos.type(torch.float), rdir.type(torch.float))
            cemb = cemb.reshape((cemb.shape[0], cemb.shape[-1], *x.shape[-2:]))
        else:
            cemb = None

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != "none":
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1

        if cemb is not None:
            cembs = [self.ray_downsample[i_level](cemb) for i_level in range(self.num_resolutions)]
        else:
            cembs = None

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                if cembs is not None:
                    h = modules[m_idx](hs[-1], temb, zemb, cembs[i_level])
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)

                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    if cembs is not None:
                        h = modules[m_idx](hs[-1], temb, zemb, cembs[i_level + 1])
                    else:
                        h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1

                if self.progressive_input == "input_skip":
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        if cembs is not None:
            h = modules[m_idx](h, temb, zemb, cembs[-1])
        else:
            h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        if cembs is not None:
            h = modules[m_idx](h, temb, zemb, cembs[-1])
        else:
            h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if cembs is not None:
                    h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb, cembs[i_level])
                else:
                    h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != "none":
                if i_level == self.num_resolutions - 1:
                    if self.progressive == "output_skip":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == "residual":
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name.")
                else:
                    if self.progressive == "output_skip":
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{self.progressive} is not a valid name")

            if i_level != 0:
                if self.resblock_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    if cembs is not None:
                        h = modules[m_idx](h, temb, zemb, cembs[i_level - 1])
                    else:
                        h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if False:
            if self.progressive == "output_skip":
                h = pyramid
            else:
                h = self.act(modules[m_idx](h))
                m_idx += 1
                h = modules[m_idx](h)
                m_idx += 1
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1

        if True:
            f_idx = 0

            if cemb is not None and cembs is not None:
                cembs_up = [cemb] + [self.ray_upsample[i](cemb) for i in range(self.num_feature_resolutions)]
            else:
                cembs_up = None
            h = self.feature_modules[f_idx](h)
            f_idx += 1
            for i_level in range(self.num_feature_resolutions):
                for i_block in range(self.num_res_blocks):
                    # h = self.feature_modules[f_idx](h, temb, zemb, up_or_down_sampling.upsample_2d(cemb, k=[1, 3, 3, 1], factor=2 ** (i_level)))
                    if cembs_up is not None:
                        h = self.feature_modules[f_idx](h, temb, zemb, cembs_up[i_level])
                    else:
                        h = self.feature_modules[f_idx](h, temb, zemb)
                    f_idx += 1

                h = self.feature_modules[f_idx](h)
                f_idx += 1

                if self.resblock_type == "ddpm":
                    h = self.feature_modules[f_idx](h)
                    f_idx += 1
                else:
                    # h = self.feature_modules[f_idx](h, temb, zemb, up_or_down_sampling.upsample_2d(cemb, k=[1, 3, 3, 1], factor=2 ** (i_level + 1)))
                    if cembs_up is not None:
                        h = self.feature_modules[f_idx](h, temb, zemb, cembs_up[i_level + 1])
                    else:
                        h = self.feature_modules[f_idx](h, temb, zemb)
                    f_idx += 1

            h = self.act(self.feature_modules[f_idx](h))
            f_idx += 1

            assert f_idx == len(self.feature_modules)

        assert m_idx == len(modules)

        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h

class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding),
                    )

        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
                    )

        self.dense_t1= dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input, t_emb, c_emb=None):
        out = self.act(input)
        out = self.conv1(out)

        if c_emb is not None:
            emb = (t_emb[..., None, None] + c_emb)
            emb = emb.permute(0, 2, 3, 1)
            B, H, W, C = emb.shape
            emb = emb.reshape(B, H * W, C)
            emb = self.dense_t1(emb)
            emb = emb.reshape(B, H, W, -1)
            emb = emb.permute(0, -1, 1, 2)
        else:
            emb = t_emb[..., None, None]

        out += emb

        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out

class Epilogue(nn.Module):
    def __init__(self, in_ch, cmap_dim, resolution, img_channels, act=nn.LeakyReLU(0.2)):
        super().__init__()

class SG2PoseAwareDiscriminator(nn.Module):
    def __init__(self, img_resolution=64, nc=3, ngf=64, t_emb_dim=128, act=nn.LeakyReLU(0.2), mapping_kwargs={}):
        from stylegan2.networks_stylegan2 import MappingNetwork
        from math import log2
        super().__init__()

        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act
        )

        # Magic number from EG3D
        cmap_dim = min(19 - log2(img_resolution), 12)
        self.mapping = MappingNetwork(z_dim=0, c_dim=25, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

        self.start_conv = conv2d(nc, ngf*2, kernel_size=1, padding=0)
        self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim=t_emb_dim, act=act)

        self.conv2 = DownConvBlock(ngf*2, ngf*4, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv3 = DownConvBlock(ngf*4, ngf*8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf*8 + 1, ngf*8, kernel_size=3, padding=1, init_scale=0.)
        self.end_linear = dense(ngf*8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t, c):
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)
        h2 = self.conv2(h1, t_embed)
        h3 = self.conv3(h2, t_embed)
        out = self.conv4(h3, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        c_embed = self.mapping(c)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out


class PoseAwareDiscriminator(nn.Module):
  def __init__(self, nc = 3, ngf = 64, t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.act = act

    self.t_embed = TimestepEmbedding(
        embedding_dim=t_emb_dim,
        hidden_dim=t_emb_dim,
        output_dim=t_emb_dim,
        act=act,
        )

    self.c_embed = RayEncoder(nf=t_emb_dim // 4, max_pos_freq=15, max_dir_freq=8)
    self.ray_sampler = RaySampler()

    # Encoding layers where the resolution decreases
    self.start_conv = conv2d(nc,ngf*2,1, padding=0)
    self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim = t_emb_dim,act=act)
    self.emb1 = conv3x3(t_emb_dim, t_emb_dim, 1)

    self.conv2 = DownConvBlock(ngf*2, ngf*4,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.emb2 = conv3x3(t_emb_dim, t_emb_dim, 2)

    self.conv3 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act)
    self.emb3 = conv3x3(t_emb_dim, t_emb_dim, 4)

    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act)

    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1)

    self.stddev_group = 4
    self.stddev_feat = 1

  def forward(self, x, t, x_t, c):
    t_embed = self.act(self.t_embed(t))

    cam2world_matrix = c[:, :16].view(-1, 4, 4)
    intrinsics = c[:, 16:25].view(-1, 3, 3)
    ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, 64)
    c_embed = self.act(self.c_embed(ray_origins, ray_directions))
    c_embed = c_embed.reshape((c_embed.shape[0], c_embed.shape[-1], *x.shape[-2:]))

    input_x = torch.cat((x, x_t), dim=1)

    h0 = self.start_conv(input_x)
    h1 = self.conv1(h0, t_embed, c_embed)
    c1 = self.emb1(c_embed)
    h2 = self.conv2(h1, t_embed, c1)
    c2 = self.emb2(c_embed)
    h3 = self.conv3(h2, t_embed, c2)
    c3 = self.emb3(c_embed)
    out = self.conv4(h3,t_embed,c3)

    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)

    out = self.final_conv(out)
    out = self.act(out)

    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    out = self.end_linear(out)

    return out
