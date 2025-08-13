# poom-universe-map

A CUDA-accelerated PyTorch demo that simulates a simple 2D self-gravitating particle system and renders a density "map of the universe." This is an illustrative sandbox inspired by the fictional "POOM Theory" to explore how local interaction rules can lead to large-scale structure patterns.ooxm-universe-map

A CUDA-accelerated PyTorch demo that simulates a simple 2D self-gravitating particle system and renders a density “map of the universe.” This is an illustrative sandbox inspired by the fictional “POOXM Theory” to explore how local interaction rules can lead to large-scale structure patterns.

Disclaimer: “POOXM Theory” is used here purely as a narrative device for a demo; this code does not claim physical accuracy or novelty.

## What this project does

- Evolves N particles in a 2D periodic box with a softened gravity-like force using a leapfrog integrator (O(N^2)).
- Renders particle positions to a density grid and saves PNG snapshots over time.
- Uses PyTorch with CUDA automatically if available.
- Provides a clean CLI and YAML config for reproducible runs.

## Project layout (src/ layout)

```text
poom-universe-map/
├─ src/
│  └─ poom/               # library code (CLI, physics, integrators, rendering)
├─ configs/                # YAML experiment configs
├─ tests/                  # unit tests (pytest)
├─ docs/                   # documentation and plans
├─ prompts/                # narrative prompt(s)
├─ experiments/            # runnable demo scripts
├─ scripts/                # helper scripts (lint, test, run)
├─ data/                   # data inputs (if any)
├─ assets/                 # images, diagrams
└─ outputs/                # generated images/results (gitignored)
```

## Installation

1) Create and activate a virtual environment.

2) Install dependencies (CPU):

```bash
pip install -r requirements.txt
```

3. Optional: install a CUDA build of PyTorch (example for CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Quickstart

- Run the demo:

```bash
python -m poom.cli --config configs/default.yaml
```

- Or after install in editable mode:

```bash
pip install -e .
poom --config configs/default.yaml
```

- Output density maps and snapshots appear in `outputs/`.

## Tuning

Edit `configs/default.yaml` for:

- number of particles
- timestep and steps
- softening, G, box size
- snapshot cadence and grid size for rendering

## Development

- Run tests: `pytest`
- Lint: `ruff check .` and `black --check .`
- Scripts are available in `scripts/`:
  - `scripts/test.sh`
  - `scripts/lint.sh`
  - `scripts/format.sh`
  - `scripts/run.sh`

## Contributing
See `.github/CONTRIBUTING.md`. Please run lint and tests before opening a PR. Follow the workflow in `docs/WORKFLOW.md`.

## License
MIT (see `LICENSE`).
