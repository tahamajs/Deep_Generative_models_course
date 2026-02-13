"""Generate report-ready metrics tables and plots from existing notebook outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from asset_lib import (
    build_metrics_summary,
    build_metrics_table_tex,
    ensure_output_dirs,
    expected_output_files,
    load_config,
    make_qualitative_grid,
    plot_accuracy_comparison,
    plot_prediction_pattern_breakdown,
    plot_rouge_comparison,
    write_json,
    write_text,
)


def generate_from_config(config_path: str | Path) -> Dict[str, Any]:
    """Generate all report assets from a config path."""
    config = load_config(config_path)
    ensure_output_dirs(config)

    summary = build_metrics_summary(config)

    plot_dir = Path(config["plot_output_dir"])
    style = config["style"]

    plot_rouge_comparison(
        "Part 1 (3%)",
        summary["datasets"]["p1"],
        plot_dir / "rouge_comparison_p1.png",
        style,
    )
    plot_rouge_comparison(
        "Part 2 (20%)",
        summary["datasets"]["p2"],
        plot_dir / "rouge_comparison_p2.png",
        style,
    )
    plot_accuracy_comparison(summary, plot_dir / "accuracy_comparison_p1_p2.png", style)

    plot_prediction_pattern_breakdown(
        "Part 1 (3%)",
        summary["datasets"]["p1"],
        plot_dir / "prediction_pattern_breakdown_p1.png",
        style,
    )
    plot_prediction_pattern_breakdown(
        "Part 2 (20%)",
        summary["datasets"]["p2"],
        plot_dir / "prediction_pattern_breakdown_p2.png",
        style,
    )

    make_qualitative_grid(
        config["qualitative_images"],
        plot_dir / "qualitative_examples_grid.png",
        style,
    )

    write_json(config["metrics_output_json"], summary)
    write_text(config["table_output_tex"], build_metrics_table_tex(summary))

    generated = expected_output_files(config)
    return {
        "config": config,
        "summary": summary,
        "generated_files": generated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report assets (plots + metrics table) from existing JSON results."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to report_assets_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_from_config(args.config)
    print("Generated report assets:")
    for output in result["generated_files"]:
        print(f"- {output}")


if __name__ == "__main__":
    main()
