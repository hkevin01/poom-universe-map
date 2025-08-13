import torch
from typing import Tuple


def make_ic(
    N: int,
    box_size: float,
    distribution: str = "uniform_jitter",
    jitter_sigma: float = 0.5,
    vel_sigma: float = 0.05,
    device=None,
    dtype=None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (positions [N,2], velocities [N,2], masses [N])
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    if distribution == "uniform_jitter":
        # Uniform positions with small Gaussian jitter
        pos = (torch.rand((N, 2), generator=g) - 0.5) * box_size
        pos += torch.randn((N, 2), generator=g) * jitter_sigma
    elif distribution == "gaussian":
        pos = torch.randn((N, 2), generator=g)
        pos = pos / pos.std() * (0.2 * box_size)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    vel = torch.randn((N, 2), generator=g) * vel_sigma
    m = torch.ones((N,))

    if dtype is not None:
        pos = pos.to(dtype)
        vel = vel.to(dtype)
        m = m.to(dtype)
    if device is not None:
        pos = pos.to(device)
        vel = vel.to(device)
        m = m.to(device)

    return pos, vel, m
