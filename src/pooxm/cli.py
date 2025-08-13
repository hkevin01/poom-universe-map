import argparse
import os
from datetime import datetime

import torch
from tqdm import tqdm

from .devices import get_device, get_dtype
from .initial_conditions import make_ic
from .integrators import leapfrog_step
from .render import positions_to_density, save_density_png
from .utils import load_yaml, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="POOXM Universe Map demo (PyTorch + CUDA)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config).raw

    # Setup
    device = get_device(cfg.get("device", "auto"))
    dtype = get_dtype(cfg.get("dtype", "float32"))

    out_dir = cfg.get("out_dir", "outputs")
    ensure_dir(out_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f'{cfg.get("experiment_name","pooxm")}_{run_stamp}')
    ensure_dir(run_dir)

    torch.manual_seed(cfg.get("seed", 42))

    N = int(cfg["num_particles"]) if "num_particles" in cfg else int(cfg.get("N", 2048))
    L = float(cfg["box_size"]) if "box_size" in cfg else 100.0
    G = float(cfg.get("G", 1.0))
    soft = float(cfg.get("softening", 0.5))
    dt = float(cfg.get("dt", 0.02))
    steps = int(cfg.get("steps", 1000))
    save_every = int(cfg.get("save_every", 100))

    ic = cfg.get("ic", {})
    render_cfg = cfg.get("render", {})
    grid_size = int(render_cfg.get("grid_size", 256))

    # Initial conditions
    positions, velocities, masses = make_ic(
        N=N,
        box_size=L,
        distribution=ic.get("distribution", "uniform_jitter"),
        jitter_sigma=float(ic.get("jitter_sigma", 0.5)),
        vel_sigma=float(ic.get("vel_sigma", 0.05)),
        device=device,
        dtype=dtype,
        seed=int(cfg.get("seed", 42)),
    )

    # Warm-up density
    density0 = positions_to_density(positions, L, grid_size)
    save_density_png(
        density0,
        os.path.join(run_dir, "density_0000.png"),
        cmap=render_cfg.get("cmap", "magma"),
        dpi=int(render_cfg.get("dpi", 160)),
        vmin=render_cfg.get("vmin", None),
        vmax=render_cfg.get("vmax", None),
        title="t=0",
    )

    # Time integration
    pbar = tqdm(range(1, steps + 1), desc="Simulating", unit="step")
    for step in pbar:
        positions, velocities = leapfrog_step(
            positions, velocities, masses, dt, G, soft, L
        )

        if step % save_every == 0 or step == steps:
            d = positions_to_density(positions, L, grid_size)
            save_density_png(
                d,
                os.path.join(run_dir, f"density_{step:04d}.png"),
                cmap=render_cfg.get("cmap", "magma"),
                dpi=int(render_cfg.get("dpi", 160)),
                vmin=render_cfg.get("vmin", None),
                vmax=render_cfg.get("vmax", None),
                title=f"t={step*dt:.2f}",
            )

    print(f"Run complete. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
