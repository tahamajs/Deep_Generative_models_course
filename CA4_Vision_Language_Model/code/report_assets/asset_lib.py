"""Shared utilities for report-only asset generation and validation."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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
        "width": 1400,
        "height": 900,
        "dpi": 220,
        "font": "Helvetica",
        "color_finetuned": "#2E6F95",
        "color_base": "#E67E22",
        "color_correct": "#2A9D8F",
        "color_incorrect": "#E76F51",
        "color_non_numeric": "#7F8C8D",
        "color_axis": "#222222",
        "color_grid": "#D9D9D9",
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


def _tokenize(text: Any) -> List[str]:
    return str(text).lower().split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1_from_overlap(pred_count: Counter[Any], ref_count: Counter[Any]) -> float:
    overlap = sum((pred_count & ref_count).values())
    pred_total = sum(pred_count.values())
    ref_total = sum(ref_count.values())
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return (2.0 * precision * recall) / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for idx, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[idx - 1], prev[idx]))
        prev = curr
    return prev[-1]


def compute_rouge(records: Iterable[Mapping[str, Any]], prediction_key: str) -> Dict[str, float]:
    """Compute deterministic mean ROUGE-F1 style metrics (rouge1/2/L)."""
    total = 0
    rouge1_sum = 0.0
    rouge2_sum = 0.0
    rouge_l_sum = 0.0

    for record in records:
        pred_tokens = _tokenize(record.get(prediction_key, ""))
        ref_tokens = _tokenize(record.get("ground_truth", ""))

        rouge1_sum += _f1_from_overlap(_ngram_counts(pred_tokens, 1), _ngram_counts(ref_tokens, 1))
        rouge2_sum += _f1_from_overlap(_ngram_counts(pred_tokens, 2), _ngram_counts(ref_tokens, 2))

        lcs = _lcs_length(pred_tokens, ref_tokens)
        if pred_tokens and ref_tokens and lcs > 0:
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            rouge_l = (2.0 * precision * recall) / (precision + recall)
        else:
            rouge_l = 0.0
        rouge_l_sum += rouge_l

        total += 1

    if total == 0:
        return {key: 0.0 for key in ROUGE_KEYS}

    return {
        "rouge1": rouge1_sum / total,
        "rouge2": rouge2_sum / total,
        "rougeL": rouge_l_sum / total,
    }


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
            "rouge_definition": "Deterministic ROUGE-style F1 over tokenized text (rouge1, rouge2, rougeL).",
        },
        "datasets": {
            "p1": compute_dataset_metrics(p1_records),
            "p2": compute_dataset_metrics(p2_records),
        },
    }


def _escape_draw_text(label: str) -> str:
    return label.replace("\\", "\\\\").replace("'", "\\'")


def _run_command(command: List[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def ensure_plot_tools() -> None:
    """Ensure required external plotting tools are available."""
    if shutil.which("magick") is None:
        raise RuntimeError(
            "ImageMagick 'magick' command is required for report plot generation."
        )


def _draw_grouped_bar_chart(
    title: str,
    categories: Sequence[str],
    series_names: Sequence[str],
    series_values: Sequence[Sequence[float]],
    series_colors: Sequence[str],
    output_path: str | Path,
    style: Dict[str, Any],
    y_max: Optional[float] = None,
    value_suffix: str = "",
) -> None:
    width = int(style["width"])
    height = int(style["height"])

    left = 120
    right = width - 80
    top = 120
    bottom = height - 150

    plot_w = right - left
    plot_h = bottom - top
    values = [value for row in series_values for value in row]
    max_value = y_max if y_max is not None else (max(values) if values else 1.0)
    max_value = max(max_value, 1e-6)

    draws: List[str] = []
    axis_color = style["color_axis"]
    grid_color = style["color_grid"]

    draws.append(
        f"stroke {axis_color} stroke-width 2 fill none line {left},{top} {left},{bottom}"
    )
    draws.append(
        f"stroke {axis_color} stroke-width 2 fill none line {left},{bottom} {right},{bottom}"
    )

    for idx in range(6):
        tick_value = max_value * idx / 5.0
        y = int(bottom - (plot_h * idx / 5.0))
        draws.append(f"stroke {grid_color} stroke-width 1 line {left},{y} {right},{y}")
        draws.append(
            f"fill {axis_color} font-size 20 text {left - 95},{y + 7} '{tick_value:.2f}{value_suffix}'"
        )

    group_count = len(categories)
    series_count = len(series_names)
    group_w = plot_w / max(group_count, 1)
    bar_gap = 8
    bar_w = max(20.0, (group_w - 40 - bar_gap * (series_count - 1)) / max(series_count, 1))

    for group_idx, category in enumerate(categories):
        group_start = left + group_w * group_idx + 20
        for series_idx, series_name in enumerate(series_names):
            value = series_values[series_idx][group_idx]
            x1 = int(group_start + series_idx * (bar_w + bar_gap))
            x2 = int(x1 + bar_w)
            bar_h = int((value / max_value) * plot_h) if max_value else 0
            y1 = bottom - bar_h
            y2 = bottom
            color = series_colors[series_idx]

            draws.append(f"fill {color} stroke none rectangle {x1},{y1} {x2},{y2}")
            draws.append(
                f"fill {axis_color} font-size 18 text {x1},{max(y1 - 8, top + 18)} '{value:.2f}{value_suffix}'"
            )

        label_x = int(group_start + (series_count * bar_w + (series_count - 1) * bar_gap) / 2 - 30)
        draws.append(
            f"fill {axis_color} font-size 22 text {label_x},{bottom + 40} '{_escape_draw_text(category)}'"
        )

    legend_x = right - 260
    legend_y = top - 40
    for idx, series_name in enumerate(series_names):
        y = legend_y + idx * 35
        color = series_colors[idx]
        draws.append(f"fill {color} rectangle {legend_x},{y - 15} {legend_x + 20},{y + 5}")
        draws.append(
            f"fill {axis_color} font-size 20 text {legend_x + 30},{y} '{_escape_draw_text(series_name)}'"
        )

    draws.append(f"fill {axis_color} font-size 36 text 60,70 '{_escape_draw_text(title)}'")

    command = [
        "magick",
        "-size",
        f"{width}x{height}",
        "xc:white",
        "-font",
        style["font"],
    ]
    for draw in draws:
        command.extend(["-draw", draw])
    command.append(str(output_path))
    _run_command(command)


def _draw_stacked_bar_chart(
    title: str,
    labels: Sequence[str],
    stacks: Sequence[Dict[str, float]],
    stack_order: Sequence[str],
    stack_colors: Dict[str, str],
    output_path: str | Path,
    style: Dict[str, Any],
) -> None:
    width = int(style["width"])
    height = int(style["height"])

    left = 120
    right = width - 80
    top = 120
    bottom = height - 150
    plot_w = right - left
    plot_h = bottom - top

    totals = [sum(stack[key] for key in stack_order) for stack in stacks]
    max_total = max(totals) if totals else 1
    max_total = max(max_total, 1)

    draws: List[str] = []
    axis_color = style["color_axis"]
    grid_color = style["color_grid"]

    draws.append(
        f"stroke {axis_color} stroke-width 2 fill none line {left},{top} {left},{bottom}"
    )
    draws.append(
        f"stroke {axis_color} stroke-width 2 fill none line {left},{bottom} {right},{bottom}"
    )

    for idx in range(6):
        tick = max_total * idx / 5.0
        y = int(bottom - (plot_h * idx / 5.0))
        draws.append(f"stroke {grid_color} stroke-width 1 line {left},{y} {right},{y}")
        draws.append(f"fill {axis_color} font-size 20 text {left - 70},{y + 7} '{tick:.0f}'")

    group_count = len(labels)
    group_w = plot_w / max(group_count, 1)
    bar_w = max(80, int(group_w * 0.45))

    for idx, label in enumerate(labels):
        x_center = left + group_w * idx + group_w / 2
        x1 = int(x_center - bar_w / 2)
        x2 = int(x_center + bar_w / 2)
        current_bottom = bottom

        for key in stack_order:
            value = stacks[idx][key]
            segment_h = int((value / max_total) * plot_h)
            y1 = current_bottom - segment_h
            y2 = current_bottom
            draws.append(
                f"fill {stack_colors[key]} stroke none rectangle {x1},{y1} {x2},{y2}"
            )
            if segment_h > 18:
                draws.append(
                    f"fill white font-size 18 text {x1 + 10},{y1 + segment_h // 2 + 7} '{int(value)}'"
                )
            current_bottom = y1

        draws.append(
            f"fill {axis_color} font-size 22 text {int(x_center - 55)},{bottom + 40} '{_escape_draw_text(label)}'"
        )

    legend_x = right - 350
    legend_y = top - 35
    for legend_idx, key in enumerate(stack_order):
        y = legend_y + legend_idx * 35
        draws.append(
            f"fill {stack_colors[key]} rectangle {legend_x},{y - 15} {legend_x + 20},{y + 5}"
        )
        draws.append(
            f"fill {axis_color} font-size 20 text {legend_x + 30},{y} '{_escape_draw_text(key.replace('_', ' '))}'"
        )

    draws.append(f"fill {axis_color} font-size 36 text 60,70 '{_escape_draw_text(title)}'")

    command = [
        "magick",
        "-size",
        f"{width}x{height}",
        "xc:white",
        "-font",
        style["font"],
    ]
    for draw in draws:
        command.extend(["-draw", draw])
    command.append(str(output_path))
    _run_command(command)


def plot_rouge_comparison(
    dataset_label: str,
    dataset_metrics: Dict[str, Any],
    output_path: str | Path,
    style: Dict[str, Any],
) -> None:
    """Render ROUGE comparison chart for one dataset."""
    categories = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    finetuned = [
        dataset_metrics["finetuned"]["rouge"]["rouge1"],
        dataset_metrics["finetuned"]["rouge"]["rouge2"],
        dataset_metrics["finetuned"]["rouge"]["rougeL"],
    ]
    base = [
        dataset_metrics["base"]["rouge"]["rouge1"],
        dataset_metrics["base"]["rouge"]["rouge2"],
        dataset_metrics["base"]["rouge"]["rougeL"],
    ]

    _draw_grouped_bar_chart(
        title=f"ROUGE Comparison ({dataset_label})",
        categories=categories,
        series_names=["Fine-Tuned", "Base"],
        series_values=[finetuned, base],
        series_colors=[style["color_finetuned"], style["color_base"]],
        output_path=output_path,
        style=style,
        y_max=max(max(finetuned + base) * 1.2, 0.1),
    )


def plot_accuracy_comparison(
    summary: Dict[str, Any], output_path: str | Path, style: Dict[str, Any]
) -> None:
    """Render overall numeric accuracy comparison for p1 and p2."""
    categories = ["Part 1 (3%)", "Part 2 (20%)"]
    finetuned = [
        summary["datasets"]["p1"]["finetuned"]["numeric"]["accuracy_overall_pct"],
        summary["datasets"]["p2"]["finetuned"]["numeric"]["accuracy_overall_pct"],
    ]
    base = [
        summary["datasets"]["p1"]["base"]["numeric"]["accuracy_overall_pct"],
        summary["datasets"]["p2"]["base"]["numeric"]["accuracy_overall_pct"],
    ]

    _draw_grouped_bar_chart(
        title="Numeric Exact-Match Accuracy (Overall)",
        categories=categories,
        series_names=["Fine-Tuned", "Base"],
        series_values=[finetuned, base],
        series_colors=[style["color_finetuned"], style["color_base"]],
        output_path=output_path,
        style=style,
        y_max=100.0,
        value_suffix="%",
    )


def plot_prediction_pattern_breakdown(
    dataset_label: str,
    dataset_metrics: Dict[str, Any],
    output_path: str | Path,
    style: Dict[str, Any],
) -> None:
    """Render stacked breakdown of prediction patterns per model."""
    stacks = [
        dataset_metrics["finetuned"]["prediction_breakdown"],
        dataset_metrics["base"]["prediction_breakdown"],
    ]

    _draw_stacked_bar_chart(
        title=f"Prediction Pattern Breakdown ({dataset_label})",
        labels=["Fine-Tuned", "Base"],
        stacks=stacks,
        stack_order=["correct_numeric", "incorrect_numeric", "non_numeric"],
        stack_colors={
            "correct_numeric": style["color_correct"],
            "incorrect_numeric": style["color_incorrect"],
            "non_numeric": style["color_non_numeric"],
        },
        output_path=output_path,
        style=style,
    )


def make_qualitative_grid(
    image_paths: List[str], output_path: str | Path, style: Dict[str, Any]
) -> None:
    """Build deterministic qualitative image grid from selected samples."""
    temp = Path(output_path).with_suffix(".tmp.png")

    montage_cmd = [
        "magick",
        "montage",
        *image_paths,
        "-tile",
        "3x",
        "-geometry",
        "320x240+10+10",
        "-background",
        "white",
        str(temp),
    ]
    _run_command(montage_cmd)

    annotate_cmd = [
        "magick",
        str(temp),
        "-background",
        "white",
        "-splice",
        "0x80",
        "-gravity",
        "north",
        "-font",
        style["font"],
        "-pointsize",
        "34",
        "-fill",
        "black",
        "-annotate",
        "+0+46",
        "Selected Qualitative Examples (Deterministic Grid)",
        str(output_path),
    ]
    _run_command(annotate_cmd)

    if temp.exists():
        temp.unlink()


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
        "Dataset & Metric & Fine-Tuned & Base Model & Difference \\\\",
        "\\midrule",
    ]

    dataset_rows = [
        ("Part 1 (3%)", "p1"),
        ("Part 2 (20%)", "p2"),
    ]

    for idx, (label, key) in enumerate(dataset_rows):
        dataset = summary["datasets"][key]
        label_tex = label.replace("%", r"\%")
        if idx > 0:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{5}}{{c}}{{\\textit{{{label_tex}, {dataset['sample_count']} samples}}}} \\\\"
        )
        lines.append("\\midrule")

        for rouge_key, rouge_name in [
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("rougeL", "ROUGE-L"),
        ]:
            ft = dataset["finetuned"]["rouge"][rouge_key]
            base = dataset["base"]["rouge"][rouge_key]
            lines.append(
                f"{label_tex} & {rouge_name} & {ft:.4f} & {base:.4f} & {format_diff(ft - base)} \\\\"
            )

        ft_numeric = dataset["finetuned"]["numeric"]
        base_numeric = dataset["base"]["numeric"]

        lines.append(
            f"{label_tex} & Accuracy (overall \\%) & {ft_numeric['accuracy_overall_pct']:.2f} & {base_numeric['accuracy_overall_pct']:.2f} & {ft_numeric['accuracy_overall_pct'] - base_numeric['accuracy_overall_pct']:+.2f} \\\\"
        )
        lines.append(
            f"{label_tex} & Accuracy (conditional \\%) & {ft_numeric['accuracy_conditional_pct']:.2f} & {base_numeric['accuracy_conditional_pct']:.2f} & {ft_numeric['accuracy_conditional_pct'] - base_numeric['accuracy_conditional_pct']:+.2f} \\\\"
        )
        lines.append(
            f"{label_tex} & Numeric coverage (\\%) & {ft_numeric['numeric_coverage_pct']:.2f} & {base_numeric['numeric_coverage_pct']:.2f} & {ft_numeric['numeric_coverage_pct'] - base_numeric['numeric_coverage_pct']:+.2f} \\\\"
        )

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
