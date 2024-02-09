import torch as th

from improved_diffusion import dist_util
from improved_diffusion.script_util import create_model_and_diffusion
from improved_diffusion.resample import create_named_schedule_sampler

def load_teacher_model(path, options=None):
    if options is None:
        options = dict(
            image_size=64,
            num_channels=128,
            num_res_blocks=3,
            num_heads=4,
            num_heads_upsample=-1,
            attention_resolutions="16,8",
            dropout=0.0,
            learn_sigma=False,
            sigma_small=False,
            class_cond=False,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )

    model, diffusion = create_model_and_diffusion(**options)

    model.load_state_dict(dist_util.load_state_dict(path, map_location='cpu'))

    model.to(dist_util.dev())
    model.eval()

    return model, diffusion

def sample_from_teacher(model, diffusion, image_size=64, clip_denoised=True, use_ddim=False, batch_size=1):
    """
    Sample image from noise using model.
    """
    sample_fn = diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop

    sample = sample_fn(model, (batch_size, 3, image_size, image_size), clip_denoised=clip_denoised, model_kwargs={})
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

    return sample # [B, 3, H, W]

def inference(model, diffusion, x, t, normalize=False, convert_to_uint8=False):
    """
    Perform one step of diffusion

    model, diffusion from load_teacher_model()
    x - [B, C, H, W] Batch tensor of clean images
    t - [B,] Batch tensor of timesteps [0, 999]
    normalize - Normalize output to [0, 255]
    convert_to_uint8 - quantize to int
    """
    # Add noise to x to get x_t, then apply model to get x'
    x_t = diffusion.q_sample(x, t, noise=th.randn_like(x))
    prediction = diffusion.p_mean_variance(model, x_t, t)

    prediction = prediction["pred_xstart"] # [B, C, H, W]

    if normalize:
        prediction = ((prediction + 1) * 127.5).clamp(0, 255)

    if convert_to_uint8:
        prediction = prediction.to(th.uint8)

    return prediction
