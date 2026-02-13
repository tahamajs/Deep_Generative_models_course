#!/usr/bin/env python3
"""Verify frozen-results assets and report consistency."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple


REQUIRED_METRIC_KEYS = {
    "realnvp_pixel",
    "realnvp_latent",
    "gan_fid",
    "source",
    "generated_at_utc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify frozen notebook result assets.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    return parser.parse_args()


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


def check_metrics_schema(metrics: dict) -> List[str]:
    errors: List[str] = []
    missing_top = REQUIRED_METRIC_KEYS - set(metrics.keys())
    if missing_top:
        errors.append(f"metrics.json missing top-level keys: {sorted(missing_top)}")

    for key in ["realnvp_pixel", "realnvp_latent"]:
        if key not in metrics:
            continue
        required = {"train_fashionmnist", "test_fashionmnist", "mnist", "kmnist"}
        got = set(metrics[key].keys()) if isinstance(metrics[key], dict) else set()
        if required - got:
            errors.append(f"{key} missing keys: {sorted(required - got)}")

    gan_fid = metrics.get("gan_fid", {})
    if not isinstance(gan_fid, dict):
        errors.append("gan_fid is not an object")
    else:
        required = {
            "per_epoch",
            "initial_epoch",
            "initial_fid",
            "best_epoch",
            "best_fid",
            "final_epoch",
            "final_fid",
            "improvement_initial_to_best_abs",
            "improvement_initial_to_best_pct",
            "improvement_initial_to_final_abs",
            "improvement_initial_to_final_pct",
        }
        missing = required - set(gan_fid.keys())
        if missing:
            errors.append(f"gan_fid missing keys: {sorted(missing)}")

    return errors


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    report_tex = repo_root / "report" / "CA2_Complete_Solutions.tex"
    metrics_json = repo_root / "artifacts" / "frozen_results" / "metrics.json"
    manifest_json = repo_root / "artifacts" / "frozen_results" / "figure_manifest.json"
    generated_metrics_tex = repo_root / "report" / "generated_metrics.tex"
    generated_fid_table_tex = repo_root / "report" / "generated_fid_table.tex"

    required_files = [
        report_tex,
        metrics_json,
        manifest_json,
        generated_metrics_tex,
        generated_fid_table_tex,
    ]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        for path in missing_files:
            print(f"ERROR: Missing required file: {path}", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    report_text = report_tex.read_text(encoding="utf-8")

    errors: List[str] = []
    errors.extend(check_metrics_schema(metrics))

    if "TODO" in report_text:
        errors.append("Report source still contains TODO marker(s)")

    include_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    include_paths = [m.group(1) for m in include_pattern.finditer(report_text)]

    missing_figures = []
    for include_path in include_paths:
        resolved, exists = resolve_figure_path(repo_root, report_tex.parent, include_path)
        if not exists:
            missing_figures.append((include_path, str(resolved)))

    for include_path, resolved in missing_figures:
        errors.append(f"Missing figure for includegraphics path '{include_path}' (resolved: {resolved})")

    if manifest.get("missing_count") not in (0, None):
        errors.append(f"figure_manifest.json reports missing_count={manifest.get('missing_count')}")

    if errors:
        print("Frozen-results verification FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Frozen-results verification passed.")
    print(f"- includegraphics references: {len(include_paths)}")
    print(f"- missing figure references: 0")
    print(f"- metrics schema: OK")
    print(f"- TODO markers in report: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
