#!/usr/bin/env bash
set -euo pipefail
ruff check .
black --check .
