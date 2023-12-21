from functools import cache
import torch
from torch import nn
from triplane.network import TriNCSNpp as Generator
from triplane.volumetric_rendering.renderer import ImportanceRenderer
from triplane.volumetric_rendering.ray_sampler import RaySampler
import numpy as np
import torchvision

from score_sde.models import utils

@utils.register_model(name="triplane")
class Triplane(nn.Module):
    def __init__(self,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        rendering_kwargs    = {},
        sr_kwargs = {},
        ncsn_config = {},
    ):
        super().__init__()

        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.generator = Generator(ncsn_config)

        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # TODO: Implement superresolution
        self.superresolution = None

        self.decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32})

        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None
        self.embed_rays = ncsn_config.embed_rays

    def synthesis(self, x_tp1, t, z, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, embed_c=None, **_):
        extrinsics = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        neural_rendering_resolution = neural_rendering_resolution or self.neural_rendering_resolution

        # Create a batch of rays for volume rendering
        rendering_rays = self.ray_sampler(extrinsics, intrinsics, neural_rendering_resolution)

        # Create triplanes by running generator
        N, *_ = rendering_rays[0].shape

        if not cache_backbone and use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            if self.embed_rays:
                # Create a batch of rays for UNet/discriminator embedding based on original view
                if embed_c is None:
                    embed_extrinsics = extrinsics
                    embed_intrinsics = intrinsics
                else:
                    embed_extrinsics = embed_c[:, :16].view(-1, 4, 4)
                    embed_intrinsics = embed_c[:, 16:25].view(-1, 3, 3)

                embedding_rays = self.ray_sampler(embed_extrinsics, embed_intrinsics, neural_rendering_resolution)
                planes = self.generator(x_tp1, t, z, *embedding_rays)
            else:
                planes = self.generator(x_tp1, t, z)

        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three planes
        planes = planes.view(len(planes), 3, -1, *planes.shape[-2:])

        # Perform volume rendering on planes
        feature_samples, depth_samples, _ = self.renderer(planes, self.decoder, *rendering_rays, self.rendering_kwargs)

        # Reshape into raw neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Treat first 3 channels as 'RGB' raw-image
        rgb_image = feature_image[:, :3]

        # FIXME: Look at implementing super resolution on feature image to generate final image

        return {
            'image': rgb_image,
            'image_depth': depth_image
        }

    def forward(self, x_tp1, t, latent_z, c, embed_c=None):
        return self.synthesis(x_tp1, t, latent_z, c, embed_c=embed_c)

class OSGDecoder(nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, 1 + options['decoder_output_dim'])
        )

    def forward(self, sampled_features, *_):
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:]) * (1 + 2 * 0.001) - 0.001 # Sigmoid clamping (MipNeRF)
        sigma = x[..., 0:1]

        return {'rgb': rgb, 'sigma': sigma}
