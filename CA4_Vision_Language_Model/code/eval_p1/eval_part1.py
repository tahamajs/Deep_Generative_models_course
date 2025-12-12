"""
Wrapper to run evaluation on the 3% CLEVR-COGEN-A subset (part 1).
"""

from eval_pipeline import EvalConfig, evaluate


def main() -> None:
    cfg = EvalConfig(
        dataset_split="train[:3%]",
        output_dir="./evaluation_images_comparison_p1",
        output_json="full_evaluation_results_comparison_p1.json",
    )
    evaluate(cfg)


if __name__ == "__main__":
    main()
