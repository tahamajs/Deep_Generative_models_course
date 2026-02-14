#!/usr/bin/env python3
"""Generate LaTeX table snippets for report metrics from JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _fmt_beta(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}.0"
    return f"{value:g}"


def _latex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _ordered_betas(summary: dict, mig: dict) -> list[float]:
    betas = [float(b) for b in summary.get("betas", [])]
    if not betas and "betas" in mig:
        betas = [float(b) for b in mig["betas"]]
    return sorted(betas)


def _get_mig_value(mig: dict, beta: float) -> float:
    return float(mig["mig"].get(str(beta), {}).get("mig", 0.0))


def _write_quantitative_table(output_dir: Path, summary: dict, mig: dict, betas: list[float]) -> None:
    rows = []
    for beta in betas:
        metrics = summary["final_metrics"][str(beta)]
        model_name = "VAE (executed)" if abs(beta - 1.0) < 1e-9 else "$\\beta$-VAE (executed)"
        rows.append(
            f"{model_name} & {_fmt_beta(beta)} & {_fmt_float(_get_mig_value(mig, beta))} & "
            f"{_fmt_float(float(metrics['val_recon']))} & {_fmt_float(float(metrics['val_kl']))} & "
            f"{_fmt_float(float(metrics['val_loss']))} \\\\"
        )

    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\begin{tabular}{@{}cccccc@{}}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{$\\beta$} & \\textbf{MIG} & \\textbf{Recon Loss} & \\textbf{KL Loss} & \\textbf{Total Loss} \\\\ \\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Measured results for executed VAE configurations in this report.}",
            "\\label{tab:comprehensive_results}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "quantitative_results_table.tex").write_text(content, encoding="utf-8")


def _write_per_factor_table(output_dir: Path, summary: dict, mig: dict, betas: list[float], mig_samples: int) -> None:
    factor_names = ["Color", "Shape", "Scale", "Orientation", "Position X", "Position Y"]
    col_spec = "l" + ("c" * len(betas))
    beta_headers = " & ".join([f"\\textbf{{$\\beta$={_fmt_beta(beta)}}}" for beta in betas])

    rows = []
    for i, factor_name in enumerate(factor_names):
        vals = []
        for beta in betas:
            per_factor = mig["mig"].get(str(beta), {}).get("per_factor", [])
            value = float(per_factor[i]) if i < len(per_factor) else 0.0
            vals.append(_fmt_float(value))
        rows.append(f"{factor_name} & " + " & ".join(vals) + " \\\\")

    avg_vals = " & ".join([_fmt_float(_get_mig_value(mig, beta)) for beta in betas])
    rows.append("\\midrule")
    rows.append("\\textbf{Average (MIG)} & " + avg_vals + " \\\\")

    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
            "\\toprule",
            f"\\textbf{{Factor}} & {beta_headers} \\\\ \\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{Per-factor MIG scores from the executed run "
                f"(epochs={int(summary['epochs'])}, subset={_latex_int(int(summary['subset']))}, "
                f"MIG sample size={_latex_int(int(mig_samples))})."
                "}"
            ),
            "\\label{tab:per_factor_mig}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "per_factor_mig_table.tex").write_text(content, encoding="utf-8")


def _write_executed_table(output_dir: Path, summary: dict, mig: dict, betas: list[float]) -> None:
    rows = []
    epochs = int(summary["epochs"])
    for beta in betas:
        metrics = summary["final_metrics"][str(beta)]
        rows.append(
            f"{_fmt_beta(beta)} & {epochs} & {_fmt_float(float(metrics['val_recon']))} & "
            f"{_fmt_float(float(metrics['val_kl']))} & {_fmt_float(float(metrics['val_loss']))} & "
            f"{_fmt_float(_get_mig_value(mig, beta))} \\\\"
        )

    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\begin{tabular}{@{}lccccc@{}}",
            "\\toprule",
            "\\textbf{$\\beta$} & \\textbf{Epochs} & \\textbf{Val Recon} & \\textbf{Val KL} & \\textbf{Val Total} & \\textbf{MIG} \\\\ \\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Directly measured metrics from the executed run (\\texttt{report\\_run\\_summary.json} and \\texttt{mig\\_summary.json}).}",
            "\\label{tab:latent_analysis}",
            "\\end{table}",
            "",
        ]
    )
    (output_dir / "executed_run_table.tex").write_text(content, encoding="utf-8")


def _write_run_config_snippet(
    output_dir: Path,
    summary: dict,
    betas: list[float],
    figures_rel: str,
    report_rel: str,
    assets_subset: int,
    mig_samples: int,
) -> None:
    betas_raw = ",".join([_fmt_beta(b).rstrip("0").rstrip(".") if not float(b).is_integer() else str(int(b)) for b in betas])
    epochs = int(summary["epochs"])
    subset = int(summary["subset"])

    lines = [
        "\\begin{verbatim}",
        "python run.py report \\",
        f"  --output-dir {figures_rel} \\",
        f"  --epochs {epochs} --subset {subset} --betas {betas_raw}",
        "\\end{verbatim}",
        "\\begin{verbatim}",
        "python generate_report_assets.py \\",
        f"  --figures-dir {figures_rel} \\",
        f"  --report-dir {report_rel} \\",
        f"  --subset {assets_subset} --betas {betas_raw} \\",
        f"  --mig-max-samples {mig_samples}",
        "\\end{verbatim}",
        "",
    ]
    (output_dir / "run_config_snippet.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("../report/DGM_Report_Template/figures/report_run_summary.json"),
    )
    parser.add_argument(
        "--mig-json",
        type=Path,
        default=Path("../report/DGM_Report_Template/figures/mig_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../report/auto"),
    )
    parser.add_argument("--figures-rel", type=str, default="../report/DGM_Report_Template/figures")
    parser.add_argument("--report-rel", type=str, default="../report")
    parser.add_argument("--assets-subset", type=int, default=5000)
    parser.add_argument("--mig-max-samples", type=int, default=1000)
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(args.mig_json, "r", encoding="utf-8") as f:
        mig = json.load(f)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    betas = _ordered_betas(summary, mig)
    if not betas:
        raise ValueError("No beta values found in summary/mig JSON.")

    _write_quantitative_table(output_dir, summary, mig, betas)
    _write_per_factor_table(output_dir, summary, mig, betas, args.mig_max_samples)
    _write_executed_table(output_dir, summary, mig, betas)
    _write_run_config_snippet(
        output_dir=output_dir,
        summary=summary,
        betas=betas,
        figures_rel=args.figures_rel,
        report_rel=args.report_rel,
        assets_subset=args.assets_subset,
        mig_samples=args.mig_max_samples,
    )

    print(f"Wrote report snippets to: {output_dir}")


if __name__ == "__main__":
    main()
