#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ACTIVATE="/Users/tahamajs/Documents/uni/venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Missing virtualenv activate script: $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

echo "[1/3] Exporting notebook-embedded report figures..."
python "$REPO_ROOT/tools/export_notebook_results.py" \
  --notebook "$REPO_ROOT/codes/CA3_Score_Based_Models.ipynb" \
  --images-dir "$REPO_ROOT/images" \
  --manifest "$REPO_ROOT/report/asset_manifest.json"

echo "[2/3] Verifying report asset references..."
python "$REPO_ROOT/tools/verify_report_assets.py" \
  --tex "$REPO_ROOT/report/DGM_CA3_Complete_Report.tex" \
  --images-dir "$REPO_ROOT/images" \
  --manifest "$REPO_ROOT/report/asset_manifest.json"

echo "[3/3] Building report PDF..."
(
  cd "$REPO_ROOT/report"
  latexmk -pdf -interaction=nonstopmode -halt-on-error DGM_CA3_Complete_Report.tex
)

echo "Done. PDF available at: $REPO_ROOT/report/DGM_CA3_Complete_Report.pdf"
