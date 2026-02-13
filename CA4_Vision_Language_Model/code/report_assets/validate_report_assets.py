"""Validate generated report assets and report figure references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from asset_lib import expected_output_files, load_config, repo_root

INCLUDEGRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHICSPATH_PATTERN = re.compile(r"\\graphicspath\{([^}]*)\}")
GRAPHICSPATH_ITEM_PATTERN = re.compile(r"\{([^}]*)\}")


def _resolve_include_path(
    report_path: Path, include_path: str, graphicspaths: List[str]
) -> List[Path]:
    include = Path(include_path)
    candidates: List[Path] = []

    # Candidate 1: plain report-relative include path.
    candidates.append((report_path.parent / include).resolve())

    # Candidate 2+: include with each graphicspath prefix.
    for prefix in graphicspaths:
        candidates.append((report_path.parent / Path(prefix) / include).resolve())

    return candidates


def _extract_graphicspaths(report_text: str) -> List[str]:
    match = GRAPHICSPATH_PATTERN.search(report_text)
    if not match:
        return []
    payload = match.group(1)
    return [item.strip() for item in GRAPHICSPATH_ITEM_PATTERN.findall(payload)]


def _validate_metrics_summary(path: Path) -> List[str]:
    errors: List[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "datasets" not in payload:
        errors.append("metrics_summary.json missing top-level key: datasets")
        return errors

    for key in ("p1", "p2"):
        if key not in payload["datasets"]:
            errors.append(f"metrics_summary.json missing dataset key: {key}")
            continue
        dataset = payload["datasets"][key]
        for model in ("finetuned", "base"):
            if model not in dataset:
                errors.append(f"metrics_summary.json missing model block: {key}.{model}")
                continue
            for metric_key in ("rouge", "numeric", "prediction_breakdown"):
                if metric_key not in dataset[model]:
                    errors.append(
                        f"metrics_summary.json missing block: {key}.{model}.{metric_key}"
                    )

    return errors


def validate_from_config(
    config_path: str | Path, report_path: str | Path | None = None
) -> List[str]:
    """Run full validation and return list of errors (empty when valid)."""
    config = load_config(config_path)
    errors: List[str] = []

    # Validate configured input files.
    for input_key in ("p1_json", "p2_json"):
        input_path = Path(config[input_key])
        if not input_path.exists():
            errors.append(f"Missing input file: {input_path}")

    for image_path in config["qualitative_images"]:
        image_file = Path(image_path)
        if not image_file.exists():
            errors.append(f"Missing qualitative image: {image_file}")

    # Validate generated output files.
    for output_file in expected_output_files(config):
        output_path = Path(output_file)
        if not output_path.exists():
            errors.append(f"Missing generated output: {output_path}")

    metrics_summary = Path(config["metrics_output_json"])
    if metrics_summary.exists():
        errors.extend(_validate_metrics_summary(metrics_summary))

    # Validate every includegraphics reference in report source.
    report_source = Path(report_path) if report_path else repo_root() / "report" / "CA4_Full_Report.tex"
    report_source = report_source.resolve()
    if not report_source.exists():
        errors.append(f"Missing report source: {report_source}")
        return errors

    report_text = report_source.read_text(encoding="utf-8")
    graphicspaths = _extract_graphicspaths(report_text)
    includes = INCLUDEGRAPHICS_PATTERN.findall(report_text)

    for include in includes:
        candidate_paths = _resolve_include_path(report_source, include, graphicspaths)
        if not any(path.exists() for path in candidate_paths):
            errors.append(
                f"Broken includegraphics reference: {include} (checked: {candidate_paths})"
            )

    if "\\input{generated/metrics_table.tex}" not in report_text:
        errors.append(
            "Report source does not include generated metrics table via \\input{generated/metrics_table.tex}"
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate report-asset generation outputs and report figure references."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to report_assets_config.json",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path to LaTeX report source (defaults to report/CA4_Full_Report.tex).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_from_config(args.config, args.report)

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed: all configured assets and report references are valid.")


if __name__ == "__main__":
    main()
