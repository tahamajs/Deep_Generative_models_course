# CA2 (MAF + CycleGAN)

## Project layout
- `code/maf.py`: MADE/MAF model and training helpers.
- `code/cyclegan.py`: CycleGAN models and training loop.
- `code/datasets.py`: dataset loaders and transforms.
- `code/utils.py`: plotting/evaluation helpers with save support.
- `code/run.py`: main CLI for train/generate/eval/test.
- `code/run_all.sh`: convenience wrapper for quick/full runs.
- `report/report.tex`: report source.
- `report/images/`: generated report figures.

## Environment
Example (existing local venv):
```bash
source /Users/tahamajs/Documents/uni/venv/bin/activate
pip install -r requirements.txt
```

## Quick reproducible run (saves report plots)
From project root:
```bash
cd code
bash run_all.sh quick --save_dir ../report/images --tag quick
```

This produces files such as:
- `report/images/maf_loss_quick.png`
- `report/images/maf_generated_quick.png`
- `report/images/maf_roc_quick.png`
- `report/images/maf_score_dist_quick.png`
- `report/images/cyclegan_loss_quick.png`
- `report/images/cyclegan_a2b_1_quick.png` .. `report/images/cyclegan_a2b_3_quick.png`
- `report/images/cyclegan_b2a_1_quick.png` .. `report/images/cyclegan_b2a_3_quick.png`

If datasets are not available locally, quick mode falls back to synthetic data and still saves figures.

## Full run (real dataset expected)
```bash
cd code
bash run_all.sh full --save_dir ../report/images --tag full
```

## Report build
```bash
cd report
latexmk -pdf -interaction=nonstopmode report.tex
```
