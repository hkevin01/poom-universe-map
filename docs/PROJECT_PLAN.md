# POOM Universe Map – Project Plan

This plan reflects the current codebase: a PyTorch (+CUDA if available) toy N-body simulator using an O(N^2) softened gravity-like force with a leapfrog integrator, rendering density maps to PNG over time from a YAML-driven CLI.

## Phase 1 – Repository Modernization

- [ ] Adopt src/ layout and adjust packaging (pyproject with setuptools find).
  - Actions: Move library to `src/poom/`, configure `pyproject.toml` with `package-dir` and entry point.
  - Options: Poetry/PDM for dependency management; this repo keeps setuptools + requirements.txt for simplicity.
- [ ] Add CI with GitHub Actions for lint + tests.
  - Actions: Cache pip, run ruff+black, run pytest.
  - Options: Add CUDA matrix or nightly tests; default CPU-only for speed.
- [ ] Standardize editor config and VS Code settings.
  - Actions: Add `.editorconfig`, `.vscode/settings.json` with Python/CPP/Java style hints.
  - Options: Enforce pre-commit hooks (pre-commit) later.
- [ ] Improve .gitignore and add common support folders (data, assets, scripts, docs).
  - Actions: Ensure outputs/ is ignored; add scripts for test/lint/run.
- [ ] Core docs: README, WORKFLOW, PROJECT_GOALS, CHANGELOG.
  - Actions: Document purpose, install/usage, contribution, and workflow.

## Phase 2 – Simulation Quality and UX

- [ ] Validate physics and stability bounds.
  - Actions: Add tests for energy drift bounds and periodicity; document recommended dt ranges.
  - Options: Double vs float precision switch in config.
- [ ] Improve CLI UX and logging.
  - Actions: Add `--out-dir`, `--seed`, and verbosity flags; structured logs.
  - Options: Rich/Typer for nicer CLI; keep argparse for stdlib minimalism.
- [ ] Rendering pipeline enhancements.
  - Actions: Add colormap presets and optional Gaussian smoothing.
  - Options: Export .npy density arrays and CSV of diagnostics.
- [ ] Reproducibility and configs.
  - Actions: Add experiment metadata and config snapshot in run folder.
  - Options: Hydra/OmegaConf for hierarchical configs.
- [ ] Performance pass.
  - Actions: Batch saves, torch.compile for PyTorch 2.1+, simple mixed precision.
  - Options: CUDA Graphs for repeated kernels if helpful.

## Phase 3 – Scaling Up

- [ ] Particle-Mesh (FFT) force solver.
  - Actions: Grid assignment, Poisson solver via FFT, gradient for force.
  - Options: torch.fft or CuPy; compare performance/accuracy.
- [ ] Barnes–Hut tree (2D quadtree).
  - Actions: Build tree, compute multipole approximations.
  - Options: PyTorch vs custom CUDA kernel path.
- [ ] 3D extension.
  - Actions: Generalize to [N,3]; camera-based render.
  - Options: Volume rendering via marching cubes or MIP.
- [ ] Diagnostics and analysis.
  - Actions: Track energy, momentum; compute two-point correlation.
  - Options: Power spectrum with windowing and normalization.
- [ ] Packaging and distribution.
  - Actions: Publish to PyPI; prebuilt wheels optional.
  - Options: Dockerfile and devcontainer setup.
