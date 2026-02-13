from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT / "code" / "report_assets"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import asset_lib  # noqa: E402
import generate_report_assets  # noqa: E402
import validate_report_assets  # noqa: E402


class TestReportAssetIntegration(unittest.TestCase):
    def test_generation_validation_and_determinism(self) -> None:
        config_path = ROOT / "report" / "report_assets_config.json"

        result_first = generate_report_assets.generate_from_config(config_path)
        self.assertTrue(result_first["generated_files"])

        config = asset_lib.load_config(config_path)
        expected = [Path(path) for path in asset_lib.expected_output_files(config)]
        for output in expected:
            self.assertTrue(output.exists(), msg=f"Missing expected output: {output}")

        metrics_path = Path(config["metrics_output_json"])
        first_metrics = metrics_path.read_bytes()

        result_second = generate_report_assets.generate_from_config(config_path)
        self.assertEqual(result_first["generated_files"], result_second["generated_files"])
        second_metrics = metrics_path.read_bytes()
        self.assertEqual(first_metrics, second_metrics)

        errors = validate_report_assets.validate_from_config(config_path)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
