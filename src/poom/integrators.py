import torch
from .physics import gravitational_acceleration, apply_periodic


def leapfrog_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    dt: float,
    G: float,
    softening: float,
    box_size: float,
):
    """
    Velocity-Verlet (leapfrog) step with periodic wrapping. Operates in-place.
    """
    # Half kick
    a0 = gravitational_acceleration(positions, masses, G, softening, box_size)
    velocities += 0.5 * dt * a0

    # Drift
    positions += dt * velocities
    positions.copy_(apply_periodic(positions, box_size))

    # Recompute acceleration
    a1 = gravitational_acceleration(positions, masses, G, softening, box_size)

    # Half kick
    velocities += 0.5 * dt * a1

    return positions, velocities
