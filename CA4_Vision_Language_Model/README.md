# CA4: Fine-Tuning PaliGemma on CLEVR-COGEN-A (LoRA + 8-bit)

This project fine-tunes `google/paligemma-3b-mix-224` for visual question answering on CLEVR-COGEN-A using parameter-efficient methods:

- **8-bit quantization** (BitsAndBytes)
- **LoRA adapters** (PEFT)

Both notebook and script pipelines are provided.

## Project Structure

- `code/CA4_VLM_training.ipynb`: notebook training pipeline.
- `code/training_pipeline.py`: scriptable training pipeline.
- `code/eval_pipeline.py`: side-by-side base vs fine-tuned evaluation.
- `code/eval_p1/eval_part1.py`: wrapper for 3% subset evaluation.
- `code/eval_p2/eval_part2.py`: wrapper for 20% subset evaluation.
- `code/run.sh`: helper commands for HF login/model download.
- `images/`: evaluation visualizations used in report.
- `description/DGM_HW4.pdf`: assignment statement.
- `report/`: final report files.

## Setup

```bash
cd CA4_Vision_Language_Model
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install transformers datasets peft evaluate bitsandbytes rouge_score huggingface_hub torch torchvision
```

Authenticate and (optionally) pre-download model files:

```bash
cd code
bash run.sh
```

## Report-Only Regeneration (No Training/Eval Run)

Use this path when you want report-ready plots/tables from existing notebook result artifacts only.

```bash
cd CA4_Vision_Language_Model
python3 -m pip install -r requirements-report.txt
python3 code/report_assets/generate_report_assets.py --config report/report_assets_config.json
python3 code/report_assets/validate_report_assets.py --config report/report_assets_config.json
```

Generated outputs:

- `images/generated/rouge_comparison_p1.png`
- `images/generated/rouge_comparison_p2.png`
- `images/generated/accuracy_comparison_p1_p2.png`
- `images/generated/prediction_pattern_breakdown_p1.png`
- `images/generated/prediction_pattern_breakdown_p2.png`
- `images/generated/qualitative_examples_grid.png`
- `report/generated/metrics_summary.json`
- `report/generated/metrics_table.tex`

Run tests for the report-asset pipeline:

```bash
python3 -m unittest discover -s code/report_assets/tests -p 'test_*.py'
```

## Training Methods Used

### 1) Reproducibility and Configuration

- `set_seed(seed)`:
  - Seeds Python, NumPy, and PyTorch.
  - Configures deterministic CuDNN when CUDA is available.
- `TrainConfig` dataclass (script) / `config` dict (notebook):
  - Centralized hyperparameters and paths.

### 2) Data Loading and Split Method

- `load_splits(cfg)`:
  - Loads `leonardPKU/clevr_cogen_a_train` via Hugging Face Datasets.
  - Applies `train_test_split(test_size=0.1, seed=42)`.

### 3) Preprocessing Method

- `preprocess_function(batch, processor, cfg)`:
  - Converts each image to RGB, resizes to `224x224`.
  - Builds prompt format: `"<image> {question}"`.
  - Tokenizes input text/image and target answer text.
  - Pads/truncates to `max_length`.
  - Sets padded label tokens to `-100` for loss masking.

### 4) Model Construction Method

- `build_model(cfg)`:
  - Loads `PaliGemmaForConditionalGeneration` with:
    - `BitsAndBytesConfig(load_in_8bit=True)`
    - `device_map="auto"`
  - Adds LoRA adapters using `LoraConfig`:
    - `r=64`, `lora_alpha=64`, `lora_dropout=0.05`
    - target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Freezes base weights and trains LoRA params only.

### 5) Trainer + Metric Method

- `build_trainer(model, processor, splits, cfg)`:
  - Maps preprocessing over train/val splits.
  - Uses Hugging Face `Trainer` and `TrainingArguments`.
  - Defines `compute_metrics(eval_pred)` using ROUGE:
    - `rouge1`, `rouge2`, `rougeL`.

### 6) Execution Methods

- `parse_args()`:
  - Controls run mode and smoke test step cap (`--run-training`, `--max-steps`).
- `main()`:
  - Builds config, datasets, model, trainer.
  - Runs training if enabled and saves best model.

## Evaluation Methods Used

### 1) Evaluation Config and Data Methods

- `EvalConfig` dataclass:
  - Paths to base and fine-tuned models, split settings, output locations.
- `load_splits(cfg)` + `preprocess_function(...)` (evaluation version):
  - Same dataset and prompt formatting logic as training.

### 2) Dataloader Method

- `to_dataloaders(splits, processor, cfg)`:
  - Builds:
    - processed loader (tokenized tensors)
    - original loader (for recovering raw image/question for reporting)
  - Includes custom collate function for tensor stacking.

### 3) Model Loading Method

- `load_models(cfg)`:
  - Loads base and fine-tuned models as `PaliGemmaForConditionalGeneration`.
  - Sets `eval()` mode for both.

### 4) Metric Methods

- `calculate_accuracy(results, prediction_key)`:
  - Extracts numeric answers via regex and computes exact-number accuracy.
- `evaluate(cfg)`:
  - Generates answers from both models.
  - Computes ROUGE + numeric accuracy.
  - Saves per-sample JSON and optional evaluation images.

### 5) Wrapper Methods

- `eval_p1/eval_part1.py`: runs evaluation on `train[:3%]` subset.
- `eval_p2/eval_part2.py`: runs evaluation on `train[:20%]` subset.

## Default Hyperparameters in Code

- Dataset split for training: `train[:20%]`
- Train/eval batch size: `4`
- Gradient accumulation: `4`
- Learning rate: `1e-4`
- Epochs: `1`
- Eval/save every `100` steps
- Max sequence length: `300` (training), `512` default in eval script

## How to Run

### Notebook training

```bash
cd CA4_Vision_Language_Model/code
jupyter lab CA4_VLM_training.ipynb
```

### Script training

```bash
cd CA4_Vision_Language_Model/code
python training_pipeline.py --run-training
```

Smoke test:

```bash
python training_pipeline.py --run-training --max-steps 20
```

### Evaluation

```bash
cd CA4_Vision_Language_Model/code
python eval_p1/eval_part1.py
python eval_p2/eval_part2.py
```

Or run configurable pipeline directly:

```bash
python eval_pipeline.py \
  --dataset-split "train[:3%]" \
  --base-model ./paligemma-3b-mix-224 \
  --finetuned-model ./final_merged_paligemma_model
```

## Output Artifacts

- Training outputs/checkpoints: `code/paligemma-clevr-finetuned/` (default).
- Evaluation images: `code/evaluation_images_comparison*/`.
- Evaluation JSON: `code/eval_p1/full_evaluation_results_comparison.json`, `code/eval_p2/full_evaluation_results_comparison.json`.
- Report figures: `images/`.

## Notes

- This project needs GPU resources for practical runtime.
- Keep base and fine-tuned model paths aligned with actual local directories before running evaluation.
