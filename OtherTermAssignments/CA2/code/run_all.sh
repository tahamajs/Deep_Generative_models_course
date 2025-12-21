#!/usr/bin/env bash
# run_all.sh — convenience script to run all CA2 experiments
# Usage:
#   ./run_all.sh quick   # run short smoke-tests for MAF and CycleGAN
#   ./run_all.sh full    # run full experiments (longer)
#   ./run_all.sh help

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNPY="python3 ${SCRIPT_DIR}/run.py"

function usage() {
  cat <<EOF
Usage: $0 {quick|full|help}

Modes:
  quick    Run short smoke tests (single epoch / small batch sizes) for faster feedback.
  full     Run longer experiments (increase epochs; may take hours).
  help     Show this message.

Examples:
  $0 quick
  $0 full
EOF
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

echo "Running mode: ${MODE}"

if [ "${MODE}" = "help" ]; then
  usage
  exit 0
fi

if [ "${MODE}" = "quick" ]; then
  echo "== MAF: quick train (1 epoch, small batch)"
  ${RUNPY} maf --mode train --epochs 1 --batch_size 3 --quick

  echo "== MAF: quick generate (3 samples, uninitialized model if no checkpoint)"
  ${RUNPY} maf --mode generate --num_samples 3 || true

  echo "== MAF: quick eval (synthetic fallback if datasets missing)"
  ${RUNPY} maf --mode eval --model maf_final.pth --quick || ${RUNPY} maf --mode eval --quick || true

  echo "== CycleGAN: quick train (1 epoch, small batch)"
  ${RUNPY} cyclegan --mode train --epochs 1 --batch_size 2 --quick || true

  echo "== CycleGAN: quick test (small sample set)"
  ${RUNPY} cyclegan --mode test --num_samples 3 --quick || true

  echo "Quick mode finished. Inspect outputs in the project folders or rerun with 'full' for longer runs."
  exit 0
fi

if [ "${MODE}" = "full" ]; then
  echo "== MAF: full train (recommended to run on GPU)"
  ${RUNPY} maf --mode train --epochs 100 --batch_size 8

  echo "== MAF: generate (5 samples)"
  ${RUNPY} maf --mode generate --num_samples 5 --model maf_final.pth || true

  echo "== MAF: anomaly evaluation"
  ${RUNPY} maf --mode eval --model maf_final.pth || true

  echo "== CycleGAN: full training"
  ${RUNPY} cyclegan --mode train --epochs 20 --batch_size 16

  echo "== CycleGAN: test (use saved checkpoints if available)"
  ${RUNPY} cyclegan --mode test --num_samples 5 --model cyclegan_models/G_AB_final.pth || true

  echo "Full mode finished. Review checkpoints and logs in the output folders."
  exit 0
fi

usage
exit 1
