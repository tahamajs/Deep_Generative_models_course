#!/usr/bin/env bash
# run_all.sh — convenience script to run CA2 experiments and save artifacts
# Usage:
#   ./run_all.sh quick
#   ./run_all.sh full
#   ./run_all.sh quick --save_dir ../report/images --tag quick

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNPY="python3 ${SCRIPT_DIR}/run.py"

function usage() {
  cat <<EOF
Usage: $0 {quick|full|help} [--save_dir DIR] [--tag TAG] [--show_plots]

Modes:
  quick    Run short smoke tests for MAF and CycleGAN and save report artifacts.
  full     Run longer experiments (dataset/GPU recommended).
  help     Show this message.
EOF
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

MODE="$1"
shift || true

SAVE_DIR="${SCRIPT_DIR}/../report/images"
TAG="${MODE}"
SHOW_PLOTS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --show_plots)
      SHOW_PLOTS=1
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

COMMON_ARGS=(--save_dir "$SAVE_DIR" --tag "$TAG")
if [ "$SHOW_PLOTS" -eq 1 ]; then
  COMMON_ARGS+=(--show_plots)
fi

mkdir -p "$SAVE_DIR"

echo "Running mode: ${MODE}"
echo "Artifacts directory: ${SAVE_DIR}"
echo "Tag: ${TAG}"

if [ "${MODE}" = "help" ]; then
  usage
  exit 0
fi

if [ "${MODE}" = "quick" ]; then
  MAF_MODEL="${SAVE_DIR}/maf_final_${TAG}.pth"

  echo "== MAF: quick train"
  ${RUNPY} maf --mode train --epochs 1 --batch_size 3 --quick --out "$MAF_MODEL" "${COMMON_ARGS[@]}"

  echo "== MAF: quick generate"
  ${RUNPY} maf --mode generate --num_samples 5 --quick --model "$MAF_MODEL" "${COMMON_ARGS[@]}"

  echo "== MAF: quick eval"
  ${RUNPY} maf --mode eval --model "$MAF_MODEL" --quick "${COMMON_ARGS[@]}"

  echo "== CycleGAN: quick train"
  ${RUNPY} cyclegan --mode train --epochs 1 --batch_size 2 --quick "${COMMON_ARGS[@]}"

  echo "== CycleGAN: quick test"
  ${RUNPY} cyclegan --mode test --num_samples 3 --quick "${COMMON_ARGS[@]}"

  echo "Quick mode finished. Saved artifacts in: ${SAVE_DIR}"
  exit 0
fi

if [ "${MODE}" = "full" ]; then
  MAF_MODEL="${SAVE_DIR}/maf_final_${TAG}.pth"

  echo "== MAF: full train"
  ${RUNPY} maf --mode train --epochs 100 --batch_size 8 --out "$MAF_MODEL" "${COMMON_ARGS[@]}"

  echo "== MAF: generate"
  ${RUNPY} maf --mode generate --num_samples 5 --model "$MAF_MODEL" "${COMMON_ARGS[@]}"

  echo "== MAF: anomaly eval"
  ${RUNPY} maf --mode eval --model "$MAF_MODEL" "${COMMON_ARGS[@]}"

  echo "== CycleGAN: full train"
  ${RUNPY} cyclegan --mode train --epochs 20 --batch_size 16 "${COMMON_ARGS[@]}"

  echo "== CycleGAN: full test"
  ${RUNPY} cyclegan --mode test --num_samples 5 "${COMMON_ARGS[@]}"

  echo "Full mode finished. Saved artifacts in: ${SAVE_DIR}"
  exit 0
fi

usage
exit 1
