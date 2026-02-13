# CA1: Variational Autoencoder (VAE) for Face Images

This project implements a convolutional **beta-VAE** for 128x128 RGB face images (smile/non-smile classes), with a script-first workflow for training, evaluation, and report figure regeneration.

## Project Structure

- `code/CA1_VAE_training_and_evaluation.ipynb`: notebook runner/documentation hook.
- `code/vae_training.py`: canonical pipeline (training + checkpoints + report plots + summaries).
- `train/`: ImageFolder dataset root (`smile/`, `non_smile/`).
- `images/`: report figures (including required `output_cell_*` filenames).
- `report/`: LaTeX report sources and PDFs.

## Environment

Use the fixed environment for this project:

```bash
source /Users/tahamajs/Documents/uni/venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

## Canonical Run (120 Epochs + Report Plots)

From repository root:

```bash
source /Users/tahamajs/Documents/uni/venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib
python code/vae_training.py \
  --data-root train \
  --out-dir output \
  --images-dir images \
  --checkpoint-dir output/checkpoints \
  --epochs 120 \
  --batch-size 64 \
  --lr 5e-4 \
  --beta 1.0 \
  --device auto \
  --num-workers 0 \
  --analysis-samples 500 \
  --save-report-plots
```

## Smoke Test (1 Epoch)

```bash
source /Users/tahamajs/Documents/uni/venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib
python code/vae_training.py \
  --data-root train \
  --out-dir output_smoke \
  --images-dir images \
  --checkpoint-dir output_smoke/checkpoints \
  --epochs 1 \
  --batch-size 64 \
  --device auto \
  --num-workers 0 \
  --save-report-plots
```

## CLI Highlights

`code/vae_training.py` supports:

- `--device {auto,cuda,mps,cpu}`
- `--num-workers`
- `--resume <checkpoint>`
- `--save-report-plots / --no-save-report-plots`
- `--images-dir`
- `--analysis-samples`
- `--checkpoint-dir`

## Outputs

Main outputs under `output/`:

- `train_log.csv`: per-epoch train/val ELBO + recon + KL.
- `run_summary.json`: metric summary for report synchronization.
- `run_metrics.txt`: compact metrics snapshot.
- `checkpoints/best.pt`, `checkpoints/last.pt`.

Report figures in `images/`:

- `output_cell_11_img_0.png`
- `output_cell_19_img_0.png`, `output_cell_19_img_1.png`, `output_cell_19_img_2.png`
- `output_cell_21_img_0.png`
- `output_cell_23_img_0.png`
- `output_cell_25_img_0.png`
- `output_cell_27_img_0.png`
- `output_cell_29_img_0.png`

Descriptive duplicate names are also saved for traceability.

## Report Build

```bash
cd report
pdflatex -interaction=nonstopmode DGM_CA1_Exercise_Solutions.tex
pdflatex -interaction=nonstopmode DGM_CA1_Exercise_Solutions.tex
cp DGM_CA1_Exercise_Solutions.pdf DGM_CA1_final_EN.pdf
```
