import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from .physics import apply_periodic


def positions_to_density(
    positions: torch.Tensor,
    box_size: float,
    grid_size: int = 256,
) -> np.ndarray:
    """
    Histogram (NGP) density field from positions on a grid.
    Returns density array [grid_size, grid_size].
    """
    with torch.no_grad():
        pos = apply_periodic(positions, box_size).detach().cpu().numpy()
    L = box_size
    # Map [-L/2, L/2) -> [0, L)
    shifted = pos + 0.5 * L
    H, xedges, yedges = np.histogram2d(
        shifted[:, 0], shifted[:, 1],
        bins=grid_size,
        range=[[0, L], [0, L]],
    )
    return H.T  # transpose so y is vertical axis


def save_density_png(
    density: np.ndarray,
    out_path: str,
    cmap: str = "magma",
    dpi: int = 160,
    vmin=None,
    vmax=None,
    title: str | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 6), dpi=dpi)
    plt.imshow(density, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="counts")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
