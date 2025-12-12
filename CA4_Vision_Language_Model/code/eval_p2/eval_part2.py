"""
Wrapper to run evaluation on the 20% CLEVR-COGEN-A subset (part 2).
"""

from eval_pipeline import EvalConfig, evaluate


def main() -> None:
    cfg = EvalConfig(
        dataset_split="train[:20%]",
        output_dir="./evaluation_images_comparison_p2",
        output_json="full_evaluation_results_comparison_p2.json",
    )
    evaluate(cfg)


if __name__ == "__main__":
    main()
