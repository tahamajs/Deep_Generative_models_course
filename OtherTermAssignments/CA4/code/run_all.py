#!/usr/bin/env python3
"""Run CA4 pipelines and prepare report figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run all CA4 model stages and sync report plots.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--dreambooth", choices=["auto", "run", "skip"], default="auto")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run_stage(name, cmd):
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def list_expected_figures(fig_dir: Path):
    expected = [
        "forward_diffusion_viz.png",
        "ddpm_training_loss.png",
        "ddpm_samples_grid.png",
        "ddim_samples_grid.png",
        "ddpm_ddim_samples.png",
        "fm_loss_curve.png",
        "fm_generated_samples.png",
        "fm_real_vs_gen.png",
        "fm_dist_comparison.png",
        "fm_autocorr.png",
        "dreambooth_instance_sample.png",
        "dreambooth_generated_grid.png",
    ]
    print("\n=== Figure Summary ===")
    for name in expected:
        status = "OK" if (fig_dir / name).exists() else "MISSING"
        print(f"{status:8} {fig_dir / name}")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve() if args.output_dir else script_dir.parent / "report" / "En_report" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = [sys.executable]
    common = ["--preset", args.preset, "--output-dir", str(output_dir)]
    if args.show_plots:
        common.append("--show-plots")

    ddpm_cmd = base + [str(script_dir / "run_ddpm.py")] + common
    if args.allow_download:
        ddpm_cmd.append("--allow-download")
    run_stage("DDPM", ddpm_cmd)

    flow_cmd = base + [str(script_dir / "run_flow_matching.py")] + common
    run_stage("Flow Matching", flow_cmd)

    dream_cmd = base + [
        str(script_dir / "run_stable_diffusion.py"),
        "--mode",
        args.dreambooth,
        "--output-dir",
        str(output_dir),
    ]
    if args.show_plots:
        dream_cmd.append("--show-plots")
    run_stage("DreamBooth (optional)", dream_cmd)

    sync_cmd = base + [str(script_dir / "sync_report_figures.py")]
    run_stage("Backfill notebook figures", sync_cmd)

    list_expected_figures(output_dir)
    print("\nAll selected CA4 stages completed.")


if __name__ == "__main__":
    main()
