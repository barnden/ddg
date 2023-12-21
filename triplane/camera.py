import torch as th
from torch import (
    Tensor,
    cross
)
from torch.linalg import norm

def normalize(x: Tensor) -> Tensor:
    return th.nn.functional.normalize(x, dim=-1)

def create_lookat_extrinsics(lookat: Tensor, eye: Tensor, up: Tensor=Tensor([0, 1, 0]), device=None) -> Tensor:
    device = device if device is not None else lookat.device
    lookat = lookat.to(device)
    eye = eye.to(device)
    up = up.to(device)

    w = normalize(lookat - eye)
    u = normalize(cross(up, w))
    v = cross(w, u)

    B = 1
    if lookat.ndim > 1:
        B = lookat.shape[0]

    mat = th.zeros(B, 4, 4, device=device)

    mat[:, :3, 0] = u
    mat[:, :3, 1] = v
    mat[:, :3, 2] = w

    mat[:, :3, 3] = eye

    mat[:, 3, 3] = 1

    return mat

def random_lookat(r: float=1, B: int=1, up: Tensor=Tensor([0, 1, 0])) -> Tensor:
    return create_lookat_extrinsics(lookat=th.zeros(B, 3), eye=r * normalize(th.randn(B, 3)), up=up[None].repeat(B, 1))

def ext2sph(x: Tensor) -> Tensor:
    eye = x[:, :3, 3]
    fwd = x[:, :3, 2]
    right = x[:, :3, 0]

    tgt = eye + fwd
    up = cross(fwd, right)

    rho = norm(eye, dim=-1)
    theta = th.atan2(norm(eye[:, :2]), eye[:, 2])
    phi = th.atan2(eye[:, 1], eye[:, 0])

    return rho, theta, phi, tgt, up

def sph2ext(rho: Tensor, theta: Tensor, phi: Tensor, lookat: Tensor=th.zeros(3)[None], up: Tensor=Tensor([0, 1, 0])) -> Tensor:
    x = phi.cos() * theta.sin()
    y = phi.sin() * theta.sin()
    z = theta.cos()

    eye = rho[:, None] * th.stack([x, y, z], dim=1)

    return create_lookat_extrinsics(lookat, eye=eye, up=up)

def get_relative(v1: Tensor, v2: Tensor) -> Tensor:
    v1 = v1[:, :16].view(-1, 4, 4)
    v2 = v2[:, :16].view(-1, 4, 4)

    R1_inv = th.linalg.inv(v1[:, :3, :3])

    mat = th.zeros(v1.shape[0], 4, 4)
    mat[:, :3, :3] = R1_inv.bmm(v2[:, :3, :3])
    mat[:, :3, 3] = R1_inv.bmm((v2[:, :3, 3] - v1[:, :3, 3])[..., None]).squeeze()
    mat[:, 3, 3] = 1

    return mat