"""
Evaluation pipeline for PaliGemma fine-tuning on CLEVR-COGEN-A.

Supports side-by-side evaluation of a base model and a fine-tuned (merged) model,
computing ROUGE and exact-number accuracy, and saving per-sample outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    base_model_path: str = "./paligemma-3b-mix-224"
    finetuned_model_path: str = "./final_merged_paligemma_model"
    dataset_split: str = "train[:20%]"  # override to "train[:3%]" for quick eval
    test_size: float = 0.1
    max_length: int = 512
    batch_size: int = 8
    output_dir: str = "./evaluation_images_comparison"
    output_json: str = "full_evaluation_results_comparison.json"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_splits(cfg: EvalConfig) -> Tuple[DatasetDict, PaliGemmaProcessor]:
    raw = load_dataset("leonardPKU/clevr_cogen_a_train", split=cfg.dataset_split)
    splits = raw.train_test_split(test_size=cfg.test_size, seed=cfg.seed)
    processor = PaliGemmaProcessor.from_pretrained(cfg.base_model_path)
    return splits, processor


def preprocess_function(
    batch: Dict[str, Any], processor: PaliGemmaProcessor, cfg: EvalConfig
) -> Dict[str, Any]:
    questions = batch["problem"]
    images = batch["image"]
    answers = batch["solution"]

    processed_images: List[Image.Image] = []
    prompts: List[str] = []
    targets: List[str] = []
    for q, img, ans in zip(questions, images, answers):
        try:
            pil_img = img.convert("RGB").resize((224, 224))
            processed_images.append(pil_img)
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


def to_dataloaders(
    splits: DatasetDict, processor: PaliGemmaProcessor, cfg: EvalConfig
) -> Tuple[DataLoader, DataLoader]:
    processed_test: Dataset = splits["test"].map(
        lambda batch: preprocess_function(batch, processor, cfg),
        batched=True,
        remove_columns=splits["test"].column_names,
    )
    processed_test.set_format(type="torch")

    def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        collated_batch = {key: [] for key in batch[0].keys()}
        for sample in batch:
            for key, value in sample.items():
                collated_batch[key].append(value)
        for key in ["pixel_values", "input_ids", "labels"]:
            if key in collated_batch and collated_batch[key]:
                collated_batch[key] = torch.stack(
                    [
                        item if torch.is_tensor(item) else torch.tensor(item)
                        for item in collated_batch[key]
                    ]
                )
        return collated_batch

    # original test for recovering images/questions
    splits["test"].set_format("torch")

    processed_loader = DataLoader(
        processed_test, batch_size=cfg.batch_size, collate_fn=custom_collate_fn
    )
    original_loader = DataLoader(splits["test"], batch_size=cfg.batch_size)
    return processed_loader, original_loader


def load_models(
    cfg: EvalConfig,
) -> Tuple[PaliGemmaForConditionalGeneration, PaliGemmaForConditionalGeneration]:
    logger.info("Loading base model: %s", cfg.base_model_path)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading fine-tuned (merged) model: %s", cfg.finetuned_model_path)
    finetuned_model = PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.finetuned_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    finetuned_model.eval()
    base_model.eval()
    return base_model, finetuned_model


def calculate_accuracy(results: List[Dict[str, str]], prediction_key: str) -> float:
    hits = 0
    total = 0
    for result in results:
        gt_match = re.search(r"\\d+", result["ground_truth"])
        pred_match = re.search(r"\\d+", result[prediction_key])
        if gt_match and pred_match:
            total += 1
            if int(gt_match.group(0)) == int(pred_match.group(0)):
                hits += 1
    return (hits / total) * 100 if total else 0.0


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    splits, processor = load_splits(cfg)
    processed_loader, original_loader = to_dataloaders(splits, processor, cfg)
    base_model, finetuned_model = load_models(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    rouge_metric = evaluate.load("rouge")
    device = next(finetuned_model.parameters()).device

    evaluation_results: List[Dict[str, str]] = []
    batch_counter = 0

    with torch.no_grad():
        for processed_batch, original_batch in zip(processed_loader, original_loader):
            if hasattr(finetuned_model, "device_map") and finetuned_model.device_map:
                pixel_values = processed_batch["pixel_values"]
                input_ids = processed_batch["input_ids"]
            else:
                pixel_values = processed_batch["pixel_values"].to(device)
                input_ids = processed_batch["input_ids"].to(device)

            labels_list_batch = processed_batch["labels"]
            finetuned_outputs = finetuned_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=50,
                early_stopping=True,
            )
            base_outputs = base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=50,
                early_stopping=True,
            )

            finetuned_texts = processor.tokenizer.batch_decode(
                finetuned_outputs, skip_special_tokens=True
            )
            base_texts = processor.tokenizer.batch_decode(
                base_outputs, skip_special_tokens=True
            )

            labels_for_decoding = labels_list_batch.clone()
            labels_for_decoding[labels_for_decoding == -100] = (
                processor.tokenizer.pad_token_id
            )
            label_texts = processor.tokenizer.batch_decode(
                labels_for_decoding, skip_special_tokens=True
            )

            for i, label in enumerate(label_texts):
                global_idx = batch_counter * cfg.batch_size + i
                image_tensor = original_batch["image"][i]
                image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                pil_image = Image.fromarray(image_np)
                image_path = os.path.join(
                    cfg.output_dir, f"eval_image_{global_idx}.png"
                )
                pil_image.save(image_path)

                evaluation_results.append(
                    {
                        "image_path": image_path,
                        "question": original_batch["problem"][i],
                        "ground_truth": label,
                        "finetuned_prediction": finetuned_texts[i],
                        "base_model_prediction": base_texts[i],
                    }
                )
            batch_counter += 1

    finetuned_preds = [res["finetuned_prediction"] for res in evaluation_results]
    base_preds = [res["base_model_prediction"] for res in evaluation_results]
    labels = [res["ground_truth"] for res in evaluation_results]
    finetuned_rouge = rouge_metric.compute(
        predictions=finetuned_preds, references=labels
    )
    base_rouge = rouge_metric.compute(predictions=base_preds, references=labels)
    finetuned_acc = calculate_accuracy(evaluation_results, "finetuned_prediction")
    base_acc = calculate_accuracy(evaluation_results, "base_model_prediction")

    metrics = {
        "finetuned": {**finetuned_rouge, "accuracy": finetuned_acc},
        "base": {**base_rouge, "accuracy": base_acc},
    }

    with open(cfg.output_json, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info("Saved evaluation results to %s", cfg.output_json)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PaliGemma fine-tuned model vs base."
    )
    parser.add_argument(
        "--dataset-split",
        default="train[:20%]",
        help='HF split string, e.g. "train[:3%]".',
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max sequence length."
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation_images_comparison",
        help="Where to save eval images.",
    )
    parser.add_argument(
        "--output-json",
        default="full_evaluation_results_comparison.json",
        help="Results JSON path.",
    )
    parser.add_argument(
        "--base-model",
        default="./paligemma-3b-mix-224",
        help="Base (pretrained) model path or ID.",
    )
    parser.add_argument(
        "--finetuned-model",
        default="./final_merged_paligemma_model",
        help="Fine-tuned merged model path.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = EvalConfig(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        dataset_split=args.dataset_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        output_json=args.output_json,
    )
    evaluate(cfg)


if __name__ == "__main__":
    main()
ص
