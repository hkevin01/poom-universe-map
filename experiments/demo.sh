#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-}:src
python -m pooxm.cli --config configs/default.yaml
