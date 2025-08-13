# Project Goals

## Purpose
Demonstrate a small, CUDA-ready PyTorch sandbox simulating a 2D self-gravitating particle system and rendering a density “map of the universe,” framed by the fictional POOXM narrative.

## Intended Audience
- Developers and students exploring GPU-accelerated numerical simulations.
- Researchers prototyping interaction rules and emergent patterns.

## Short-term Goals
- Reliable O(N^2) leapfrog integrator with periodic, softened gravity.
- CLI with YAML config and reproducible outputs.
- CI to lint and test on push/PR.

## Long-term Goals
- Swap O(N^2) force for Particle-Mesh (FFT) or Barnes–Hut.
- Extend to 3D and add diagnostics (energy, power spectrum).
- Interactive front-end for live visualization.
