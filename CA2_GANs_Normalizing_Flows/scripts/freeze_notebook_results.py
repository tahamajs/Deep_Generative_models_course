#!/usr/bin/env python3
"""Freeze existing notebook outputs into deterministic report assets.

This script does NOT execute notebooks. It only parses saved .ipynb JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


MAIN_PIXEL_CELL = 36
MAIN_LATENT_CELL = 50
Q2_FID_CELL = 57


WORD_NUMBERS = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
    13: "Thirteen",
    14: "Fourteen",
    15: "Fifteen",
    16: "Sixteen",
    17: "Seventeen",
    18: "Eighteen",
    19: "Nineteen",
    20: "Twenty",
}


@dataclass
class Config:
    repo_root: Path
    main_notebook: Path
    q2_notebook: Path
    report_tex: Path
    images_dir: Path
    artifacts_dir: Path
    metrics_json: Path
    figure_manifest_json: Path
    generated_metrics_tex: Path
    generated_fid_table_tex: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Freeze notebook outputs into report assets.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    return Config(
        repo_root=repo_root,
        main_notebook=repo_root / "code" / "CA2_GANs_and_NormalizingFlows_main.ipynb",
        q2_notebook=repo_root / "code" / "CA2_question2_results.ipynb",
        report_tex=repo_root / "report" / "CA2_Complete_Solutions.tex",
        images_dir=repo_root / "images",
        artifacts_dir=repo_root / "artifacts" / "frozen_results",
        metrics_json=repo_root / "artifacts" / "frozen_results" / "metrics.json",
        figure_manifest_json=repo_root / "artifacts" / "frozen_results" / "figure_manifest.json",
        generated_metrics_tex=repo_root / "report" / "generated_metrics.tex",
        generated_fid_table_tex=repo_root / "report" / "generated_fid_table.tex",
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def get_code_cell(notebook: dict, one_based_index: int) -> dict:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError("Notebook JSON has no 'cells' list")
    if one_based_index < 1 or one_based_index > len(cells):
        raise RuntimeError(f"Cell index {one_based_index} out of range (1..{len(cells)})")
    cell = cells[one_based_index - 1]
    if cell.get("cell_type") != "code":
        raise RuntimeError(f"Cell {one_based_index} is not a code cell")
    return cell


def output_text_from_cell(cell: dict) -> str:
    outputs = cell.get("outputs", [])
    parts: List[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        if "text" in output:
            text = output["text"]
            if isinstance(text, list):
                parts.append("".join(text))
            else:
                parts.append(str(text))

        data = output.get("data")
        if isinstance(data, dict) and "text/plain" in data:
            text_plain = data["text/plain"]
            if isinstance(text_plain, list):
                parts.append("".join(text_plain))
            else:
                parts.append(str(text_plain))

    return "\n".join(parts)


def parse_likelihood_block(text: str, block_name: str) -> Dict[str, float]:
    patterns = {
        "train_fashionmnist": r"(?:Fashion\s*MNIST|FashionMNIST)\s*Train\s*Log(?:\s*|-)Likelihood:\s*([-+]?\d+(?:\.\d+)?)",
        "test_fashionmnist": r"(?:Fashion\s*MNIST|FashionMNIST)\s*Test\s*Log(?:\s*|-)Likelihood:\s*([-+]?\d+(?:\.\d+)?)",
        "mnist": r"\bMNIST\s*Log(?:\s*|-)Likelihood:\s*([-+]?\d+(?:\.\d+)?)",
        "kmnist": r"\bKMNIST\s*Log(?:\s*|-)Likelihood:\s*([-+]?\d+(?:\.\d+)?)",
    }

    values: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError(f"Could not extract {key} from {block_name} output")
        values[key] = float(match.group(1))

    return values


def parse_fid_from_text(text: str) -> Dict[int, float]:
    matches = re.findall(r"FID\s*Score\s*at\s*Epoch\s*(\d+)\s*:\s*([-+]?\d+(?:\.\d+)?)", text)
    if not matches:
        raise RuntimeError("Could not find FID lines in Q2 notebook outputs")

    fid_per_epoch: Dict[int, float] = {}
    for epoch_str, value_str in matches:
        fid_per_epoch[int(epoch_str)] = float(value_str)

    return dict(sorted(fid_per_epoch.items(), key=lambda kv: kv[0]))


def first_nonempty_match(texts: List[str], parser_fn, desc: str):
    for text in texts:
        if text.strip():
            try:
                return parser_fn(text)
            except RuntimeError:
                continue
    raise RuntimeError(f"Failed to parse {desc} from notebook outputs")


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def to_macro_name_for_epoch(epoch: int) -> str:
    if epoch not in WORD_NUMBERS:
        raise RuntimeError(
            f"Epoch {epoch} is unsupported for macro generation (only 0..20 configured)"
        )
    return f"FIDEpoch{WORD_NUMBERS[epoch]}"


def resolve_figure_path(repo_root: Path, report_tex_dir: Path, include_path: str) -> Tuple[Path, bool]:
    include_path = include_path.strip()
    candidates = [
        repo_root / include_path,
        report_tex_dir / include_path,
        repo_root / "images" / Path(include_path).name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(), True

    return (repo_root / include_path).resolve(), False


def extract_figure_manifest(report_text: str, repo_root: Path, report_tex_dir: Path) -> dict:
    include_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    figure_entries: List[dict] = []

    for index, match in enumerate(include_pattern.finditer(report_text), start=1):
        include_path = match.group(1)

        # Caption extraction: search in the remainder of the current figure block.
        tail = report_text[match.end() :]
        end_figure = tail.find("\\end{figure}")
        scope = tail if end_figure == -1 else tail[:end_figure]

        caption_match = re.search(r"\\caption\{([^}]*)\}", scope, flags=re.DOTALL)
        label_match = re.search(r"\\label\{([^}]*)\}", scope)

        caption = ""
        if caption_match:
            caption = " ".join(caption_match.group(1).split())

        label = label_match.group(1) if label_match else ""

        resolved_path, exists = resolve_figure_path(repo_root, report_tex_dir, include_path)
        figure_entries.append(
            {
                "index": index,
                "include_path": include_path,
                "resolved_path": str(resolved_path),
                "exists": exists,
                "label": label,
                "caption": caption,
            }
        )

    missing_count = sum(1 for entry in figure_entries if not entry["exists"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_source": "report/CA2_Complete_Solutions.tex",
        "figure_count": len(figure_entries),
        "missing_count": missing_count,
        "figures": figure_entries,
    }


def write_generated_metrics_tex(path: Path, metrics: dict) -> None:
    pixel = metrics["realnvp_pixel"]
    latent = metrics["realnvp_latent"]
    fid = metrics["gan_fid"]

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/freeze_notebook_results.py")
    lines.append("% Do not edit manually.")
    lines.append(f"% Generated at: {metrics['generated_at_utc']}")

    macro_values = {
        "PixelTrainLL": fmt(pixel["train_fashionmnist"]),
        "PixelTestLL": fmt(pixel["test_fashionmnist"]),
        "PixelMNISTLL": fmt(pixel["mnist"]),
        "PixelKMNISTLL": fmt(pixel["kmnist"]),
        "LatentTrainLL": fmt(latent["train_fashionmnist"]),
        "LatentTestLL": fmt(latent["test_fashionmnist"]),
        "LatentMNISTLL": fmt(latent["mnist"]),
        "LatentKMNISTLL": fmt(latent["kmnist"]),
        "InitialFID": fmt(fid["initial_fid"]),
        "FinalFID": fmt(fid["final_fid"]),
        "BestFID": fmt(fid["best_fid"]),
        "BestFIDEpoch": str(fid["best_epoch"]),
        "BestFIDImproveAbs": fmt(fid["improvement_initial_to_best_abs"]),
        "BestFIDImprovePct": fmt(fid["improvement_initial_to_best_pct"], 1),
        "FinalFIDImproveAbs": fmt(fid["improvement_initial_to_final_abs"]),
        "FinalFIDImprovePct": fmt(fid["improvement_initial_to_final_pct"], 1),
    }

    for epoch_str, value in fid["per_epoch"].items():
        epoch = int(epoch_str)
        macro_values[to_macro_name_for_epoch(epoch)] = fmt(value)

    for macro_name in sorted(macro_values):
        lines.append(f"\\providecommand{{\\{macro_name}}}{{{macro_values[macro_name]}}}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generated_fid_table_tex(path: Path, metrics: dict) -> None:
    fid = metrics["gan_fid"]
    per_epoch = {int(k): v for k, v in fid["per_epoch"].items()}
    epochs = sorted(per_epoch)

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/freeze_notebook_results.py")
    lines.append("\\begin{table}[!t]")
    lines.append("\\renewcommand{\\arraystretch}{1.3}")
    lines.append("\\caption{FID Score Progression Throughout Training (Frozen Notebook Results)}")
    lines.append("\\label{tab:fid_scores}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|c|c|c|p{3.5cm}|}")
    lines.append("\\hline")
    lines.append("Epoch & FID Score & Change & Notes \\\\")
    lines.append("\\hline")

    prev = None
    for epoch in epochs:
        score = per_epoch[epoch]

        if prev is None:
            change_text = "Baseline"
            note = "Initial high score, model learning"
        else:
            change = ((score - prev) / prev) * 100.0
            change_text = f"{change:+.1f}\\%"
            if epoch == fid["best_epoch"]:
                note = "\\textbf{Best FID Score}"
            elif epoch == fid["final_epoch"]:
                note = "Final FID Score"
            elif change < 0:
                note = "Improvement"
            else:
                note = "Increase (training instability)"

        score_text = f"{score:.2f}"
        if epoch == fid["best_epoch"]:
            score_text = f"\\textbf{{{score_text}}}"
            if change_text != "Baseline":
                change_text = f"\\textbf{{{change_text}}}"

        lines.append(f"{epoch} & {score_text} & {change_text} & {note} \\\\")
        prev = score

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_metrics(main_nb: dict, q2_nb: dict) -> dict:
    main_pixel_text = output_text_from_cell(get_code_cell(main_nb, MAIN_PIXEL_CELL))
    main_latent_text = output_text_from_cell(get_code_cell(main_nb, MAIN_LATENT_CELL))
    q2_fid_text = output_text_from_cell(get_code_cell(q2_nb, Q2_FID_CELL))

    pixel = parse_likelihood_block(main_pixel_text, f"main notebook cell {MAIN_PIXEL_CELL}")
    latent = parse_likelihood_block(main_latent_text, f"main notebook cell {MAIN_LATENT_CELL}")
    fid_per_epoch = parse_fid_from_text(q2_fid_text)

    if not fid_per_epoch:
        raise RuntimeError("No FID epochs extracted")

    initial_epoch = min(fid_per_epoch)
    final_epoch = max(fid_per_epoch)
    best_epoch = min(fid_per_epoch, key=lambda epoch: fid_per_epoch[epoch])

    initial_fid = fid_per_epoch[initial_epoch]
    final_fid = fid_per_epoch[final_epoch]
    best_fid = fid_per_epoch[best_epoch]

    improvement_initial_to_best_abs = initial_fid - best_fid
    improvement_initial_to_best_pct = (
        (improvement_initial_to_best_abs / initial_fid) * 100.0 if initial_fid else 0.0
    )

    improvement_initial_to_final_abs = initial_fid - final_fid
    improvement_initial_to_final_pct = (
        (improvement_initial_to_final_abs / initial_fid) * 100.0 if initial_fid else 0.0
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "main_notebook": {
                "path": "code/CA2_GANs_and_NormalizingFlows_main.ipynb",
                "pixel_metrics_cell": MAIN_PIXEL_CELL,
                "latent_metrics_cell": MAIN_LATENT_CELL,
            },
            "q2_notebook": {
                "path": "code/CA2_question2_results.ipynb",
                "fid_cell": Q2_FID_CELL,
            },
            "execution_policy": "frozen_outputs_only_no_notebook_rerun",
        },
        "realnvp_pixel": pixel,
        "realnvp_latent": latent,
        "gan_fid": {
            "per_epoch": {str(epoch): value for epoch, value in fid_per_epoch.items()},
            "initial_epoch": initial_epoch,
            "initial_fid": initial_fid,
            "best_epoch": best_epoch,
            "best_fid": best_fid,
            "final_epoch": final_epoch,
            "final_fid": final_fid,
            "improvement_initial_to_best_abs": improvement_initial_to_best_abs,
            "improvement_initial_to_best_pct": improvement_initial_to_best_pct,
            "improvement_initial_to_final_abs": improvement_initial_to_final_abs,
            "improvement_initial_to_final_pct": improvement_initial_to_final_pct,
        },
    }


def main() -> int:
    config = parse_args()

    for required in [config.main_notebook, config.q2_notebook, config.report_tex]:
        if not required.exists():
            print(f"ERROR: Required input missing: {required}", file=sys.stderr)
            return 1

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    main_nb = read_json(config.main_notebook)
    q2_nb = read_json(config.q2_notebook)
    report_text = config.report_tex.read_text(encoding="utf-8")

    metrics = build_metrics(main_nb, q2_nb)
    figure_manifest = extract_figure_manifest(report_text, config.repo_root, config.report_tex.parent)

    config.metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config.figure_manifest_json.write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_generated_metrics_tex(config.generated_metrics_tex, metrics)
    write_generated_fid_table_tex(config.generated_fid_table_tex, metrics)

    print("Frozen results generated successfully:")
    print(f"- Metrics: {config.metrics_json}")
    print(f"- Figure manifest: {config.figure_manifest_json}")
    print(f"- TeX macros: {config.generated_metrics_tex}")
    print(f"- TeX FID table: {config.generated_fid_table_tex}")
    print(f"- Figures referenced: {figure_manifest['figure_count']}")
    print(f"- Missing figures: {figure_manifest['missing_count']}")

    if figure_manifest["missing_count"] > 0:
        print("ERROR: Missing figure references found in report source.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
