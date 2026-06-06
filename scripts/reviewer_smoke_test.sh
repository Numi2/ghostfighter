#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
PYTHONPATH=. python -m ghostfighter.cli all --out runs/smoke --episodes-per-style 2 --epochs 1 --eval-episodes 8 --max-steps 40
