#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python -m ghostfighter.cli all --out runs/reference --episodes-per-style 60 --epochs 6 --eval-episodes 32 --max-steps 90 --demo-style pressure --stress
