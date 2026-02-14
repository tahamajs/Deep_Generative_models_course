#!/usr/bin/env python3
"""Backfill report figures from extracted notebook assets when available."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils import REPO_ROOT


EXTRACTED_FIGS = [
    "main_1_6_Step_3__DDPM_Training_Loop_cell026_out42.png",
    "main_2_5_Step_5__Inference_and_Testing_cell049_out06.png",
    "main_2_5_Step_5__Inference_and_Testing_cell049_out09.png",
    "main_3_4_Flow_Matching_Training_cell064_out102.png",
    "main_3_6_Steps_4-5__Visualizing_Generated_Time_Series_cell069_out02.png",
    "main_3_7_Step_6__Distribution_Comparison__Histogram_KDE_cell071_out02.png",
    "main_3_8_Step_7__Basic_Statistical_Evaluation_and_Volatility_cell073_out02.png",
    "main_3_9_Step_8__Advanced_Structural_and_Temporal_Evaluation_cell080_out02.png",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Copy available extracted notebook figures into report directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite destination files if present.")
    return parser.parse_args()


def copy_if_available(src: Path, dst: Path, force: bool):
    if not src.exists():
        return False
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
    return True


def main():
    args = parse_args()
    src_dir = REPO_ROOT / "code" / "notebook" / "extracted_notebook_images"
    dst_dir = REPO_ROOT / "report" / "En_report" / "figures"
    copied = 0
    for name in EXTRACTED_FIGS:
        copied += int(copy_if_available(src_dir / name, dst_dir / name, args.force))
    print(f"Backfill complete. Files copied: {copied}")


if __name__ == "__main__":
    main()
