"""Shared utilities for report-only asset generation and validation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rouge_score import rouge_scorer

NUMERIC_PATTERN = re.compile(r"\d+")
REQUIRED_CONFIG_KEYS = [
    "p1_json",
    "p2_json",
    "qualitative_images",
    "plot_output_dir",
    "table_output_tex",
    "metrics_output_json",
    "style",
]
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
MODEL_KEYS = ["finetuned", "base"]
MODEL_TO_PREDICTION_KEY = {
    "finetuned": "finetuned_prediction",
    "base": "base_model_prediction",
}


def repo_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str, root: Path) -> Path:
    """Resolve an absolute or repository-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate report asset configuration."""
    path = Path(config_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    root = repo_root()
    style = {
        "dpi": 200,
        "font_size": 11,
        "title_size": 13,
        "color_finetuned": "#1f77b4",
        "color_base": "#ff7f0e",
        "color_correct": "#2ca02c",
        "color_incorrect": "#d62728",
        "color_non_numeric": "#7f7f7f",
    }
    style.update(payload.get("style", {}))

    return {
        "config_path": str(path),
        "p1_json": str(resolve_path(payload["p1_json"], root)),
        "p2_json": str(resolve_path(payload["p2_json"], root)),
        "qualitative_images": [
            str(resolve_path(item, root)) for item in payload["qualitative_images"]
        ],
        "plot_output_dir": str(resolve_path(payload["plot_output_dir"], root)),
        "table_output_tex": str(resolve_path(payload["table_output_tex"], root)),
        "metrics_output_json": str(resolve_path(payload["metrics_output_json"], root)),
        "style": style,
    }


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load one evaluation JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def extract_first_number(text: Any) -> Optional[int]:
    """Extract first integer token from text, returning None if absent."""
    if text is None:
        return None
    match = NUMERIC_PATTERN.search(str(text))
    if match is None:
        return None
    return int(match.group(0))


def compute_rouge(records: Iterable[Mapping[str, Any]], prediction_key: str) -> Dict[str, float]:
    """Compute mean ROUGE F1 metrics for records."""
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=False)
    totals = {key: 0.0 for key in ROUGE_KEYS}
    count = 0

    for record in records:
        prediction = str(record.get(prediction_key, ""))
        reference = str(record.get("ground_truth", ""))
        scores = scorer.score(reference, prediction)
        for key in ROUGE_KEYS:
            totals[key] += float(scores[key].fmeasure)
        count += 1

    if count == 0:
        return {key: 0.0 for key in ROUGE_KEYS}
    return {key: totals[key] / count for key in ROUGE_KEYS}


def compute_numeric_metrics(
    records: Iterable[Mapping[str, Any]], prediction_key: str
) -> Dict[str, float | int]:
    """Compute numeric accuracy/coverage with a shared regex parser."""
    total = 0
    comparable = 0
    hits = 0

    for record in records:
        total += 1
        gt_number = extract_first_number(record.get("ground_truth"))
        pred_number = extract_first_number(record.get(prediction_key))
        if gt_number is not None and pred_number is not None:
            comparable += 1
            if gt_number == pred_number:
                hits += 1

    accuracy_overall = (100.0 * hits / total) if total else 0.0
    accuracy_conditional = (100.0 * hits / comparable) if comparable else 0.0
    numeric_coverage = (100.0 * comparable / total) if total else 0.0

    return {
        "total_samples": total,
        "numeric_comparable_samples": comparable,
        "correct_numeric_predictions": hits,
        "accuracy_overall_pct": accuracy_overall,
        "accuracy_conditional_pct": accuracy_conditional,
        "numeric_coverage_pct": numeric_coverage,
    }


def compute_prediction_breakdown(
    records: Iterable[Mapping[str, Any]], prediction_key: str
) -> Dict[str, int]:
    """Compute correct/incorrect/non-numeric breakdown for one model."""
    correct_numeric = 0
    incorrect_numeric = 0
    non_numeric = 0

    for record in records:
        gt_number = extract_first_number(record.get("ground_truth"))
        pred_number = extract_first_number(record.get(prediction_key))

        if gt_number is None or pred_number is None:
            non_numeric += 1
            continue

        if gt_number == pred_number:
            correct_numeric += 1
        else:
            incorrect_numeric += 1

    return {
        "correct_numeric": correct_numeric,
        "incorrect_numeric": incorrect_numeric,
        "non_numeric": non_numeric,
    }


def compute_pairwise_outcomes(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Compute pairwise correctness outcomes (fine-tuned vs base)."""
    outcomes = {
        "both_correct": 0,
        "finetuned_only_correct": 0,
        "base_only_correct": 0,
        "both_wrong": 0,
    }

    for record in records:
        gt_number = extract_first_number(record.get("ground_truth"))
        finetuned = extract_first_number(record.get("finetuned_prediction"))
        base = extract_first_number(record.get("base_model_prediction"))

        finetuned_correct = gt_number is not None and finetuned == gt_number
        base_correct = gt_number is not None and base == gt_number

        if finetuned_correct and base_correct:
            outcomes["both_correct"] += 1
        elif finetuned_correct and not base_correct:
            outcomes["finetuned_only_correct"] += 1
        elif base_correct and not finetuned_correct:
            outcomes["base_only_correct"] += 1
        else:
            outcomes["both_wrong"] += 1

    return outcomes


def compute_dataset_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics and breakdowns for one dataset."""
    result: Dict[str, Any] = {
        "sample_count": len(records),
        "pairwise_outcomes": compute_pairwise_outcomes(records),
    }

    for model_key in MODEL_KEYS:
        prediction_key = MODEL_TO_PREDICTION_KEY[model_key]
        result[model_key] = {
            "rouge": compute_rouge(records, prediction_key),
            "numeric": compute_numeric_metrics(records, prediction_key),
            "prediction_breakdown": compute_prediction_breakdown(records, prediction_key),
        }

    return result


def build_metrics_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build deterministic metrics summary for p1 and p2 evaluation outputs."""
    p1_records = load_records(config["p1_json"])
    p2_records = load_records(config["p2_json"])

    return {
        "inputs": {
            "p1_json": config["p1_json"],
            "p2_json": config["p2_json"],
            "numeric_regex": NUMERIC_PATTERN.pattern,
        },
        "datasets": {
            "p1": compute_dataset_metrics(p1_records),
            "p2": compute_dataset_metrics(p2_records),
        },
    }


def _set_matplotlib_style(style: Dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.size": style["font_size"],
            "axes.titlesize": style["title_size"],
            "axes.labelsize": style["font_size"],
            "legend.fontsize": style["font_size"] - 1,
        }
    )


def plot_rouge_comparison(
    dataset_label: str,
    dataset_metrics: Dict[str, Any],
    output_path: str | Path,
    style: Dict[str, Any],
) -> None:
    """Plot ROUGE comparison chart for one dataset."""
    _set_matplotlib_style(style)

    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    metric_keys = ["rouge1", "rouge2", "rougeL"]
    x = np.arange(len(metric_labels))
    width = 0.35

    finetuned_values = [dataset_metrics["finetuned"]["rouge"][key] for key in metric_keys]
    base_values = [dataset_metrics["base"]["rouge"][key] for key in metric_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_ft = ax.bar(
        x - width / 2,
        finetuned_values,
        width,
        label="Fine-Tuned",
        color=style["color_finetuned"],
    )
    bars_base = ax.bar(
        x + width / 2,
        base_values,
        width,
        label="Base",
        color=style["color_base"],
    )

    ax.set_ylabel("Score")
    ax.set_title(f"ROUGE Comparison ({dataset_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, max(0.1, max(finetuned_values + base_values) * 1.25))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (bars_ft, bars_base):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(
    summary: Dict[str, Any], output_path: str | Path, style: Dict[str, Any]
) -> None:
    """Plot overall numeric accuracy comparison for p1 and p2."""
    _set_matplotlib_style(style)

    datasets = ["p1", "p2"]
    labels = ["Part 1 (3%)", "Part 2 (20%)"]
    x = np.arange(len(datasets))
    width = 0.35

    ft_acc = [
        summary["datasets"][dataset]["finetuned"]["numeric"]["accuracy_overall_pct"]
        for dataset in datasets
    ]
    base_acc = [
        summary["datasets"][dataset]["base"]["numeric"]["accuracy_overall_pct"]
        for dataset in datasets
    ]

    ft_cov = [
        summary["datasets"][dataset]["finetuned"]["numeric"]["numeric_coverage_pct"]
        for dataset in datasets
    ]
    base_cov = [
        summary["datasets"][dataset]["base"]["numeric"]["numeric_coverage_pct"]
        for dataset in datasets
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_ft = ax.bar(
        x - width / 2,
        ft_acc,
        width,
        label="Fine-Tuned",
        color=style["color_finetuned"],
    )
    bars_base = ax.bar(
        x + width / 2,
        base_acc,
        width,
        label="Base",
        color=style["color_base"],
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Numeric Exact-Match Accuracy (Overall)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx, bar in enumerate(bars_ft):
        ax.annotate(
            f"{bar.get_height():.2f}%\nCov {ft_cov[idx]:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for idx, bar in enumerate(bars_base):
        ax.annotate(
            f"{bar.get_height():.2f}%\nCov {base_cov[idx]:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_prediction_pattern_breakdown(
    dataset_label: str,
    dataset_metrics: Dict[str, Any],
    output_path: str | Path,
    style: Dict[str, Any],
) -> None:
    """Plot stacked breakdown of prediction patterns per model."""
    _set_matplotlib_style(style)

    models = ["finetuned", "base"]
    labels = ["Fine-Tuned", "Base"]

    correct = [
        dataset_metrics[model]["prediction_breakdown"]["correct_numeric"]
        for model in models
    ]
    incorrect = [
        dataset_metrics[model]["prediction_breakdown"]["incorrect_numeric"]
        for model in models
    ]
    non_numeric = [
        dataset_metrics[model]["prediction_breakdown"]["non_numeric"] for model in models
    ]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x, correct, color=style["color_correct"], label="Correct numeric")
    ax.bar(
        x,
        incorrect,
        bottom=correct,
        color=style["color_incorrect"],
        label="Incorrect numeric",
    )

    stacked = [c + i for c, i in zip(correct, incorrect)]
    ax.bar(
        x,
        non_numeric,
        bottom=stacked,
        color=style["color_non_numeric"],
        label="Non-numeric",
    )

    total = dataset_metrics["sample_count"]
    ax.set_title(f"Prediction Pattern Breakdown ({dataset_label})")
    ax.set_ylabel("Sample Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(total, 1) * 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx in range(len(models)):
        ax.annotate(
            str(correct[idx]),
            xy=(x[idx], correct[idx] / 2 if correct[idx] else 1),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        ax.annotate(
            str(incorrect[idx]),
            xy=(x[idx], correct[idx] + (incorrect[idx] / 2 if incorrect[idx] else 1)),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        ax.annotate(
            str(non_numeric[idx]),
            xy=(x[idx], stacked[idx] + (non_numeric[idx] / 2 if non_numeric[idx] else 1)),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
    plt.close(fig)


def make_qualitative_grid(
    image_paths: List[str], output_path: str | Path, style: Dict[str, Any]
) -> None:
    """Build deterministic qualitative image grid from selected samples."""
    _set_matplotlib_style(style)

    images = [Image.open(path).convert("RGB") for path in image_paths]
    cols = 3
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    if rows == 1:
        axes = np.array([axes])

    flat_axes = axes.flatten()
    for idx, ax in enumerate(flat_axes):
        if idx >= len(images):
            ax.axis("off")
            continue

        image = images[idx]
        filename = Path(image_paths[idx]).name
        ax.imshow(image)
        ax.set_title(filename.replace("_", r"\_"), fontsize=9)
        ax.axis("off")

    fig.suptitle("Selected Qualitative Examples (Deterministic Grid)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=style["dpi"], bbox_inches="tight")
    plt.close(fig)


def format_diff(value: float) -> str:
    """Format difference values for LaTeX table."""
    return f"{value:+.4f}"


def build_metrics_table_tex(summary: Dict[str, Any]) -> str:
    """Build LaTeX table sourced entirely from regenerated metrics."""
    lines: List[str] = [
        "% Auto-generated by code/report_assets/generate_report_assets.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Generated ROUGE and Numeric Metrics (Notebook Result Artifacts)}",
        "\\label{tab:generated_metrics}",
        "\\begin{tabular}{l|l|c|c|c}",
        "\\toprule",
        "Dataset & Metric & Fine-Tuned & Base Model & Difference \\",
        "\\midrule",
    ]

    dataset_rows = [
        ("Part 1 (3%)", "p1"),
        ("Part 2 (20%)", "p2"),
    ]

    for idx, (label, key) in enumerate(dataset_rows):
        dataset = summary["datasets"][key]
        if idx > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{c}}{{\\textit{{{label}, {dataset['sample_count']} samples}}}} \\")
        lines.append("\\midrule")

        for rouge_key, rouge_name in [
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougeL", "ROUGE-L"),
        ]:
            ft = dataset["finetuned"]["rouge"][rouge_key]
            base = dataset["base"]["rouge"][rouge_key]
            lines.append(
                f"{label} & {rouge_name} & {ft:.4f} & {base:.4f} & {format_diff(ft - base)} \\")

        ft_numeric = dataset["finetuned"]["numeric"]
        base_numeric = dataset["base"]["numeric"]

        lines.append(
            f"{label} & Accuracy (overall \\%) & {ft_numeric['accuracy_overall_pct']:.2f} & {base_numeric['accuracy_overall_pct']:.2f} & {ft_numeric['accuracy_overall_pct'] - base_numeric['accuracy_overall_pct']:+.2f} \\")
        lines.append(
            f"{label} & Accuracy (conditional \\%) & {ft_numeric['accuracy_conditional_pct']:.2f} & {base_numeric['accuracy_conditional_pct']:.2f} & {ft_numeric['accuracy_conditional_pct'] - base_numeric['accuracy_conditional_pct']:+.2f} \\")
        lines.append(
            f"{label} & Numeric coverage (\\%) & {ft_numeric['numeric_coverage_pct']:.2f} & {base_numeric['numeric_coverage_pct']:.2f} & {ft_numeric['numeric_coverage_pct'] - base_numeric['numeric_coverage_pct']:+.2f} \\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )

    return "\n".join(lines) + "\n"


def expected_output_files(config: Dict[str, Any]) -> List[str]:
    """Return deterministic expected generated output file list."""
    plot_dir = Path(config["plot_output_dir"])
    return [
        str(plot_dir / "rouge_comparison_p1.png"),
        str(plot_dir / "rouge_comparison_p2.png"),
        str(plot_dir / "accuracy_comparison_p1_p2.png"),
        str(plot_dir / "prediction_pattern_breakdown_p1.png"),
        str(plot_dir / "prediction_pattern_breakdown_p2.png"),
        str(plot_dir / "qualitative_examples_grid.png"),
        config["metrics_output_json"],
        config["table_output_tex"],
    ]


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Create output directories for generated artifacts."""
    Path(config["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["metrics_output_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["table_output_tex"]).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write deterministic JSON output."""
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_text(path: str | Path, text: str) -> None:
    """Write text file with UTF-8 encoding."""
    Path(path).write_text(text, encoding="utf-8")
