import torch

@torch.no_grad()
def apply_periodic(positions: torch.Tensor, box_size: float) -> torch.Tensor:
    """Wrap particle positions back into the periodic domain [-L/2, L/2).

    Args:
        positions: Tensor [N, 2]
        box_size: Box side length L
    Returns:
        Wrapped positions [N, 2]
    """
    L = box_size
    return ((positions + 0.5 * L) % L) - 0.5 * L


def pairwise_minimum_image(delta: torch.Tensor, box_size: float) -> torch.Tensor:
    """Apply the minimum image convention for a periodic box.

    Args:
        delta: pairwise displacements [..., 2]
        box_size: side length L
    """
    L = box_size
    return delta - L * torch.round(delta / L)


def gravitational_acceleration(
    positions: torch.Tensor,
    masses: torch.Tensor,
    G: float,
    softening: float,
    box_size: float,
) -> torch.Tensor:
    """
    Compute softened pairwise gravitational accelerations with periodic images.

    Args:
        positions: [N, 2]
        masses: [N] or [N, 1]
        G: gravitational constant (toy)
        softening: Plummer softening length
        box_size: domain side length L

    Returns:
        accelerations [N, 2]
    """
    pos = positions
    m = masses.view(-1)  # [N]

    # Compute pairwise displacements with periodic wrapping
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 2]
    diff = pairwise_minimum_image(diff, box_size)  # periodicity

    dist2 = (diff ** 2).sum(dim=-1) + softening**2  # [N, N]
    inv_r3 = dist2.pow(-1.5)
    # Zero self-interaction
    inv_r3.fill_diagonal_(0.0)

    # a_i = -G * sum_j m_j * (r_i - r_j) / |r_i - r_j|^3
    # Note: diff = r_i - r_j
    weighted = inv_r3 * m.unsqueeze(0)  # [N, N]
    acc = -G * (diff * weighted.unsqueeze(-1)).sum(dim=1)
    return acc
