import argparse
import torch
import numpy as np
from PIL import Image
from triplane.camera import create_lookat_extrinsics, random_lookat, get_relative

import os

import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths

from pathlib import Path
from triplane.triplane import Triplane

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_random_camera() -> torch.Tensor:
    c = torch.zeros(25)
    c[16:] = torch.Tensor([
            525.0000 / 512.0,   0.0000, 0.5,
            0.0000, 525.0000 / 512.0, 0.5,
            0.0000,   0.0000,   1.0000])

    eye = torch.randn(3)
    eye /= torch.linalg.norm(eye)
    c[:16] = create_lookat_extrinsics(torch.zeros(3)[None], eye=1.3 * eye[None], up=torch.Tensor([0, 0, -1])[None]).ravel()

    c = c[None]
    c.to('cuda:0')

    return c

def get_canonical_camera() -> torch.Tensor:
    c = torch.zeros(25)
    if 0:
        c = torch.Tensor([ 0.8540,      0.3085,    -0.4188,   0.5445,   0.5202,
                                -0.5066,      0.6876,    -0.8939,   0.0000,  -0.8051,
                                -0.5931,      0.7711,    -0.0000,   0.0000,  -0.0000,
                                1.0000,      525.0000,   0.0000, 256.0000,   0.0000,
                                525.0000,    256.0000,   0.0000,   0.0000,   1.0000])
        c[16] /= 128 * 4
        c[18] /= 128 * 4
        c[20] /= 128 * 4
        c[21] /= 128 * 4
    else:
        c[16:] = torch.Tensor([
                525.0000 / 512.0,   0.0000, 0.5,
                0.0000, 525.0000 / 512.0, 0.5,
                0.0000,   0.0000,   1.0000])

        eye = torch.Tensor([1, 0, 0])
        eye /= torch.linalg.norm(eye)
        c[:16] = create_lookat_extrinsics(torch.zeros(3)[None], eye=1.3 * eye[None], up=torch.Tensor([0, 0, -1])[None]).ravel()

    c = c[None]
    c.to('cuda:0')

    return c

def load_from_file(side='left'):
    from torchvision import transforms
    xf = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
        ])

    with Image.open(f"./saved_info/other/{side}.png").convert('RGB') as im:
        x = xf(im)

    x = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(x)
    x = x[None]

    x = x.to('cuda:0')

    c = torch.zeros(25)
    c[16:] = torch.Tensor([
            525.0000,   0.0000, 256.0000,
            0.0000, 525.0000, 256.0000,
            0.0000,   0.0000,   1.0000])
    c[16] /= 128 * 4
    c[18] /= 128 * 4
    c[20] /= 128 * 4
    c[21] /= 128 * 4

    if side == 'left':
        c[:16] = torch.Tensor([
            0.13318568468093872,-0.23039288818836212,0.9639401435852051,-1.2531222105026245,-0.9910910129547119,-0.03096095845103264,0.12953701615333557,-0.16839824616909027,6.332992796842518e-08,-0.9726049900054932,-0.23246392607688904,0.3022032380104065,-0.0,0.0,-0.0,1.0
        ])
    elif side == 'right':
        c[:16] = torch.Tensor([
            0.054104309529066086,-0.010577673092484474,-0.9984792470932007,1.298022985458374,0.9985352754592896,0.0005731878918595612,0.05410126596689224,-0.07033174484968185,5.931360647082329e-08,-0.9999438524246216,0.01059318333864212,-0.013770990073680878,-0.0,0.0,-0.0,1.0
        ])
    elif side == 'front':
        c[:16] = torch.Tensor([
            -0.9889143109321594,-0.015064507722854614,-0.14772097766399384,0.19203761219978333,0.1484871357679367,-0.10032907128334045,-0.9838118553161621,1.2789554595947266,-5.960463766996327e-08,-0.9948403239250183,0.10145364701747894,-0.13188958168029785,-0.0,0.0,-0.0,1.0
        ])
    elif side == 'back':
        c[:16] = torch.Tensor([
            0.992717981338501,-0.008494961075484753,0.12016164511442184,-0.15621012449264526,-0.12046155333518982,-0.07000653445720673,0.9902464151382446,-1.2873204946517944,-1.1175870007207322e-08,-0.9975103139877319,-0.0705200582742691,0.09167630225419998,-0.0,0.0,-0.0,1.0
        ])

    c = c[None]

    c = c.to('cuda:0')

    return x, c

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_circular_path(coefficients, generator, n_time, opt, x_init=None):
    # Create a triplane
    x_tp1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to('cuda:0')
    get_sample(coefficients, generator, n_time, x_tp1, args, c=get_random_camera(), cache=True)

    # Create circular camera path
    angles = torch.pi * torch.arange(0, 96, 1) / 48
    path = torch.stack([torch.cos(angles), torch.sin(angles), torch.ones_like(angles)], dim=-1)
    path = 1.3 * torch.nn.functional.normalize(path, dim=-1)

    # Set ShapeNet intrinsics
    c = torch.zeros(25)
    c[16:] = torch.Tensor([
            525.0000 / 512,   0.0000, 256.0000 / 512,
            0.0000, 525.0000 / 512, 256.0000 / 512,
            0.0000,   0.0000,   1.0000])

    c = c[None]

    rgb_images = []
    depth_images = []
    for i in range(0, path.shape[0]):
        c[:, :16] = create_lookat_extrinsics(lookat=torch.zeros(3), eye=path[i], up=torch.Tensor([0, 0, -1])).ravel()

        rgb, depth = get_sample(coefficients, generator, n_time, None, opt=opt, c=c, cache=False)

        rgb_images.append(rgb)
        depth_images.append(depth)

    return {'image': torch.cat(rgb_images), 'image_depth': torch.cat(depth_images)}


def get_sample(coefficients, generator, n_time, x_init, opt, c=None, cache=False):
    x = x_init

    batch_size = c.shape[0]
    device = x.device if x is not None else 'cuda:0'


    with torch.no_grad():
        if c is None:
            c = get_canonical_camera()

        if c.ndim == 1:
            c = c[None].repeat(x.shape[0], 1)

        c = c.to(device)

        render_pose = c.clone()
        embed_pose = c.clone()

        if opt.relative_render:
            # Instead of rendering at same input pose...
            # FIXME: Don't hardcode ShapeNet cars radius parameter (r=1.3)
            if opt.random_render:
                # FIXME: Implement
                pass
                # compute random in-distribution view v*, render relative pose (v* - v), pass v* to discriminator
                # v_star = random_lookat(r=1.3, B=batch_size)

                # discriminator_pose[:, :16] = v_star.flatten(start_dim=1)
                # render_pose[:, :16] = get_relative(c, discriminator_pose).flatten(start_dim=1)
            else:
                # render triplane at canonical, pass original view v to discriminator
                eye = 1.3 * torch.Tensor([1, 0, 0])
                eye = eye[None].repeat(batch_size, 1)
                up = torch.Tensor([0, 1, 0])[None].repeat(batch_size, 1)

                render_pose[:, :16] = create_lookat_extrinsics(lookat=torch.zeros(batch_size, 3), eye=eye, up=up).flatten(start_dim=1)


        embed_pose = embed_pose.to(device)
        render_pose = render_pose.to(device)

        if cache:
            for i in reversed(range(n_time)):
                t = torch.full((x.shape[0],), i, dtype=torch.int64).to(device)
                z = torch.randn(x.shape[0], opt.nz, device=device)

                pred = generator.synthesis(x, t, z, render_pose, use_cached_backbone=True, cache_backbone=True, embed_pose=embed_pose)
                x_new = sample_posterior(coefficients, pred['image'], x, t)

                x = x_new.detach()
                depth = pred['image_depth'].detach()
        else:
            pred = generator.synthesis(None, None, None, render_pose, use_cached_backbone=True, embed_pose=embed_pose)

            return (v.detach() for v in pred.values())

    return x, depth


def get_view_from(side):
    # Hardcoded extrinsics for ShapenetCars
    if side == 'left':
        return torch.Tensor([
            0.13318568468093872,-0.23039288818836212,0.9639401435852051,-1.2531222105026245,-0.9910910129547119,-0.03096095845103264,0.12953701615333557,-0.16839824616909027,6.332992796842518e-08,-0.9726049900054932,-0.23246392607688904,0.3022032380104065,-0.0,0.0,-0.0,1.0
        ])

    if side == 'right':
        return torch.Tensor([
            0.054104309529066086,-0.010577673092484474,-0.9984792470932007,1.298022985458374,0.9985352754592896,0.0005731878918595612,0.05410126596689224,-0.07033174484968185,5.931360647082329e-08,-0.9999438524246216,0.01059318333864212,-0.013770990073680878,-0.0,0.0,-0.0,1.0
        ])

    if side == 'front':
        return torch.Tensor([
            -0.9889143109321594,-0.015064507722854614,-0.14772097766399384,0.19203761219978333,0.1484871357679367,-0.10032907128334045,-0.9838118553161621,1.2789554595947266,-5.960463766996327e-08,-0.9948403239250183,0.10145364701747894,-0.13188958168029785,-0.0,0.0,-0.0,1.0
        ])

    if side == 'back':
        return torch.Tensor([
            0.992717981338501,-0.008494961075484753,0.12016164511442184,-0.15621012449264526,-0.12046155333518982,-0.07000653445720673,0.9902464151382446,-1.2873204946517944,-1.1175870007207322e-08,-0.9975103139877319,-0.0705200582742691,0.09167630225419998,-0.0,0.0,-0.0,1.0
        ])


#%%
def sample_and_test(args):
    device = 'cuda:0'

    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir

    to_range_0_1 = lambda x: (x + 1.) / 2.

    rendering_options = {
        'image_resolution': args.image_size,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',

        # FIXME: Implement super resolution
        'superresolution_module': 'no impl',
        'superresolution_noise_mode': 'no impl',
        'sr_antialias': True,

        # FIXME: Implement density regularization
        'density_reg': 0,
        'density_reg_p_dist': 0,
        'reg_type': 0,

        'decoder_lr_mul': 0,
        # These settings are taken from EG3D for ShapeNet
        'depth_resolution': 64,
        'depth_resolution_importance': 64,
        'ray_start': 0.1,
        'ray_end': 2.6,
        'box_warp': 1.6,
        'white_back': True,
        'avg_camera_radius': 1.7,
        'avg_camera_pivot': [0, 0, 0],
    }

    netG = Triplane(args.image_size, 3, rendering_kwargs=rendering_options, ncsn_config=args).to(device)
    ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)

    print('loading ckpt {}'.format('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id)))

    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)

    netG.load_state_dict(ckpt)
    netG.eval()

    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 50000 //args.batch_size

    save_dir = Path("./generated_samples/{}".format(args.dataset))
    save_dir.mkdir(exist_ok=True)

    if args.compute_fid:
        # FIXME: This does not work
        for i in range(iters_needed):
            with torch.no_grad():
                x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
                fake_sample = get_sample(pos_coeff, netG, args.num_timesteps, x_t_1, args)['image']

                fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    elif args.random:
        c = torch.zeros(25)
        c[16:] = torch.Tensor([
                525.0000 / 512,   0.0000, 256.0000 / 512,
                0.0000, 525.0000 / 512, 256.0000 / 512,
                0.0000,   0.0000,   1.0000])

        c = c[None]
        if (side := args.render_from_side) is not None:
            c[:, :16] = get_view_from(side)
        else:
            c[:, :16] = get_canonical_camera()

        images = []
        for _ in range(24):
            x_tp1 = torch.randn(args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)

            fake_sample, depth = get_sample(pos_coeff, netG, args.num_timesteps, x_tp1, args, c=c, cache=True)

            rgb = to_range_0_1(fake_sample.detach())
            images.append(rgb)

        torchvision.utils.save_image(torch.concat(images, dim=0), './samples_random_{}.jpg'.format(args.dataset))
    elif args.circular:
        from math import ceil
        x_t_1 = torch.randn(args.batch_size, args.num_channels, args.image_size, args.image_size).to(device)
        samples = sample_circular_path(pos_coeff, netG, args.num_timesteps, args, x_t_1)

        rgb = to_range_0_1(samples['image'])
        torchvision.utils.save_image(rgb, './samples_{}.jpg'.format(args.dataset))

        depth = samples['image_depth']
        # depth = depth.max() - depth
        # torchvision.utils.save_image(depth, './samples_depth_{}.jpg'.format(args.dataset))

        fig, axs = plt.subplots(ceil(depth.shape[0] / 8), 8, layout='constrained', figsize=(8, ceil(depth.shape[0] / 8)))
        fig.tight_layout()
        axs = axs.flatten()
        for img, ax in zip(depth, axs):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_axis_off()
            ax.imshow(img.permute(1, 2, 0).cpu())

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.savefig('samples_depth_{}.jpg'.format(args.dataset))
        print("saved samples")

        if args.animate:
            fig, ax = plt.subplots()

            rgb = rgb.permute(0, 2, 3, 1).cpu()
            frames = [[ax.imshow(rgb[i])] for i in range(rgb.shape[0])]

            ani = animation.ArtistAnimation(fig, frames)

            ani.save('./animation.gif', writer='imagemagick', fps=32)

            depth = depth.permute(0, 2, 3, 1).cpu()
            frames = [[ax.imshow(depth[i])] for i in range(depth.shape[0])]

            ani = animation.ArtistAnimation(fig, frames)

            ani.save('./animation-depth.gif', writer='imagemagick', fps=32)
    elif args.load_from_file:
        sides = ['left', 'right', 'front', 'back', 'all']
        images = []
        depths = []

        assert args.load_from_file in sides
        from train_ddgan import q_sample_pairs, Diffusion_Coefficients
        coeff = Diffusion_Coefficients(args, device)
        NN = 1

        if args.load_from_file == 'all':
            inputs = [load_from_file(side) for side in sides[:-1]]
            inputs = [(x.to(device), c.to(device)) for x, c in inputs]
            t = torch.full((1,), NN).to(device)

            for i in range(len(sides[:-1])):
                x, c = inputs[i]
                _, x_tp1 = q_sample_pairs(coeff, x, t)

                get_sample(pos_coeff, netG, NN, x_tp1, args, c=c, cache=True)

                images.append(to_range_0_1(x))

                for side in sides[:-1]:
                    c[:, :16] = get_view_from(side)
                    fake_sample, depth = get_sample(None, netG, None, None, args, c=c)
                    rgb = to_range_0_1(fake_sample.detach())
                    images.append(rgb)

            torchvision.utils.save_image(torch.concat(images, dim=0), './samples_{}.jpg'.format(args.dataset), nrow=5)
        else:
            x, c = load_from_file(args.load_from_file)
            x = x.to('cuda:0')
            t = torch.full((1,), NN).to('cuda:0')

            if (side := args.render_from_side) is not None:
                c[:, :16] = get_view_from(side)

            for _ in range(16):
                _, x_tp1 = q_sample_pairs(coeff, x, t)
                get_sample(pos_coeff, netG, NN, x_tp1, args, c=c, cache=True)

                fake_sample, depth = get_sample(None, netG, None, None, args, c=c)

                rgb = to_range_0_1(fake_sample.detach())
                images.append(rgb)
                depths.append(rgb)

            torchvision.utils.save_image(torch.concat(images, dim=0), './samples_{}.jpg'.format(args.dataset))
            # depth = depth.max() - depth
            # torchvision.utils.save_image(depth, './samples_depth_{}.jpg'.format(args.dataset))
    else:
        NN = int(args.num_timesteps)
        netG = netG.to('cuda:0')

        images = []
        for _ in range(16):
            x_tp1 = torch.randn(1, args.num_channels, args.image_size, args.image_size).to(device)
            #coefficients, generator, n_time, x_init, opt
            fake_sample, _ = get_sample(pos_coeff, netG, int(args.num_timesteps), x_tp1, args, cache=True)
            rgb = to_range_0_1(fake_sample.detach())
            images.append(rgb)

        torchvision.utils.save_image(torch.concat(images, dim=0), './samples_{}.jpg'.format(args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=2048,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--load_from_file', type=str, default=None)
    parser.add_argument('--render_from_side', type=str, default=None)
    parser.add_argument('--animate', action='store_true', default=False)
    parser.add_argument('--circular', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--embed_rays', action='store_true', default=False)
    parser.add_argument('--relative_render', action='store_true', default=False)
    parser.add_argument('--random_render', action='store_true', default=False)
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)


    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')

    args = parser.parse_args()
    sample_and_test(args)
