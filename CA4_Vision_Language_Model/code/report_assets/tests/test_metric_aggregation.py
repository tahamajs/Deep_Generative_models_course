from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from asset_lib import compute_dataset_metrics  # noqa: E402


class TestMetricAggregation(unittest.TestCase):
    def test_numeric_aggregation_fixture(self) -> None:
        records = [
            {
                "ground_truth": "<answer> 3 </answer>",
                "finetuned_prediction": "3",
                "base_model_prediction": "2",
            },
            {
                "ground_truth": "<answer> 4 </answer>",
                "finetuned_prediction": "4",
                "base_model_prediction": "4",
            },
            {
                "ground_truth": "<answer> 5 </answer>",
                "finetuned_prediction": "not-a-number",
                "base_model_prediction": "still-not-a-number",
            },
        ]

        summary = compute_dataset_metrics(records)

        finetuned_numeric = summary["finetuned"]["numeric"]
        base_numeric = summary["base"]["numeric"]
        pairwise = summary["pairwise_outcomes"]

        self.assertEqual(finetuned_numeric["correct_numeric_predictions"], 2)
        self.assertEqual(finetuned_numeric["numeric_comparable_samples"], 2)
        self.assertAlmostEqual(finetuned_numeric["accuracy_overall_pct"], 66.6666666667)
        self.assertAlmostEqual(finetuned_numeric["accuracy_conditional_pct"], 100.0)
        self.assertAlmostEqual(finetuned_numeric["numeric_coverage_pct"], 66.6666666667)

        self.assertEqual(base_numeric["correct_numeric_predictions"], 1)
        self.assertEqual(base_numeric["numeric_comparable_samples"], 2)
        self.assertAlmostEqual(base_numeric["accuracy_overall_pct"], 33.3333333333)
        self.assertAlmostEqual(base_numeric["accuracy_conditional_pct"], 50.0)
        self.assertAlmostEqual(base_numeric["numeric_coverage_pct"], 66.6666666667)

        self.assertEqual(pairwise["both_correct"], 1)
        self.assertEqual(pairwise["finetuned_only_correct"], 1)
        self.assertEqual(pairwise["base_only_correct"], 0)
        self.assertEqual(pairwise["both_wrong"], 1)

    def test_rouge_exact_match_fixture(self) -> None:
        records = [
            {
                "ground_truth": "one two",
                "finetuned_prediction": "one two",
                "base_model_prediction": "one two",
            }
        ]

        summary = compute_dataset_metrics(records)

        for model in ("finetuned", "base"):
            rouge = summary[model]["rouge"]
            self.assertAlmostEqual(rouge["rouge1"], 1.0)
            self.assertAlmostEqual(rouge["rouge2"], 1.0)
            self.assertAlmostEqual(rouge["rougeL"], 1.0)


if __name__ == "__main__":
    unittest.main()
