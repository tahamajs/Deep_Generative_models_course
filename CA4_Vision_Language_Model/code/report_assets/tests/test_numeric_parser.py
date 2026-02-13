from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from asset_lib import extract_first_number  # noqa: E402


class TestNumericParser(unittest.TestCase):
    def test_plain_number(self) -> None:
        self.assertEqual(extract_first_number("9"), 9)

    def test_xml_like_answer(self) -> None:
        self.assertEqual(extract_first_number("<answer> 14 </answer>"), 14)

    def test_malformed_text_with_number(self) -> None:
        self.assertEqual(extract_first_number("value=00>"), 0)

    def test_non_numeric_text(self) -> None:
        self.assertIsNone(extract_first_number("answering does not require reading text"))

    def test_none_input(self) -> None:
        self.assertIsNone(extract_first_number(None))


if __name__ == "__main__":
    unittest.main()
