"""
Training pipeline for CA4: fine-tuning PaliGemma-3B on CLEVR-COGEN-A.

This mirrors the Jupyter notebook while enabling scriptable runs and smoke tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_id: str = os.getenv("MODEL_ID", "google/paligemma-3b-mix-224")
    dataset_split: str = "train[:20%]"
    test_size: float = 0.1
    max_length: int = 300
    train_batch_size: int = 4
    eval_batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    output_dir: str = "./paligemma-clevr-finetuned"
    max_steps: int = -1  # override for smoke tests
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_splits(cfg: TrainConfig) -> Tuple[DatasetDict, PaliGemmaProcessor]:
    """Load and split the CLEVR-COGEN-A dataset, and return splits and processor."""
    raw = load_dataset("leonardPKU/clevr_cogen_a_train", split=cfg.dataset_split)
    splits = raw.train_test_split(test_size=cfg.test_size, seed=cfg.seed)
    processor = PaliGemmaProcessor.from_pretrained(cfg.model_id)
    return splits, processor


def preprocess_function(
    batch: Dict[str, Any], processor: PaliGemmaProcessor, cfg: TrainConfig
) -> Dict[str, Any]:
    """Preprocess a batch: resize images, format prompts, and tokenize."""
    questions = batch["problem"]
    images = batch["image"]
    answers = batch["solution"]

    processed_images = []
    prompts = []
    targets = []
    for q, img, ans in zip(questions, images, answers):
        try:
            img = img.convert("RGB").resize((224, 224))
            processed_images.append(img)
            prompts.append(f"<image> {q}")
            targets.append(str(ans))
        except Exception as exc:
            logger.warning("Skipping sample due to image issue: %s", exc)

    if not processed_images:
        return {}

    encoder_inputs = processor(
        images=processed_images,
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )

    decoder_inputs = processor.tokenizer(
        text_target=targets,
        padding="max_length",
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )

    labels = decoder_inputs["input_ids"].clone()
    labels[decoder_inputs["attention_mask"] == 0] = -100
    encoder_inputs["labels"] = labels
    return encoder_inputs


def build_model(cfg: TrainConfig) -> PaliGemmaForConditionalGeneration:
    """Build and return a quantized PaliGemma model with LoRA adapters."""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %.1fM (%.2f%%)", trainable / 1e6, 100 * trainable / total
    )
    return model


def build_trainer(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    splits: DatasetDict,
    cfg: TrainConfig,
) -> Trainer:
    train_ds = splits["train"].map(
        lambda batch: preprocess_function(batch, processor, cfg),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    val_ds = splits["test"].map(
        lambda batch: preprocess_function(batch, processor, cfg),
        batched=True,
        remove_columns=splits["test"].column_names,
    )
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        max_steps=cfg.max_steps,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune PaliGemma on CLEVR-COGEN-A with LoRA + 8-bit."
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Execute training instead of dry run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override max steps (use small number for smoke).",
    )
    return parser.parse_args()



def main() -> None:
    """Main entry point for training pipeline."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = TrainConfig(max_steps=args.max_steps if args.max_steps is not None else -1)
    set_seed(cfg.seed)
    logger.info("Config: %s", cfg)

    splits, processor = load_splits(cfg)
    model = build_model(cfg)
    trainer = build_trainer(model, processor, splits, cfg)

    if args.run_training:
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(cfg.output_dir)
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))
    else:
        logger.info("Dry run completed. Use --run-training to start training.")


if __name__ == "__main__":
    main()

