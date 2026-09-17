import copy
import unittest

from common import ROOT
from paragraph_inputs import paragraph_spans
from publish import portable_paths
from verify import boundary


class ContractTests(unittest.TestCase):
    def sample(self):
        return {
            "cutoff": "2018-01-01T14:25:00+00:00",
            "interval_start": "2018-01-01T14:30:00+00:00",
            "interval_end": "2018-01-01T18:30:00+00:00",
            "price_rows": [{"end": "2017-12-29T21:00:00+00:00"}],
            "news": [{"available_at": "2018-01-01T14:20:00+00:00"}],
        }

    def test_valid_boundary(self):
        boundary(self.sample())

    def test_future_price_rejected(self):
        row = self.sample()
        row["price_rows"][0]["end"] = "2018-01-01T14:30:00+00:00"
        with self.assertRaises(ValueError):
            boundary(row)

    def test_future_news_rejected(self):
        row = self.sample()
        row["news"][0]["available_at"] = "2018-01-01T14:30:00+00:00"
        with self.assertRaises(ValueError):
            boundary(row)

    def test_target_paragraph_exact_and_complete(self):
        text = "Promo without target\nApple Inc. raised guidance from 10 to 12. It cited services growth.\n"
        units = paragraph_spans(text, "AAPL")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["text"], text[units[0]["start"] : units[0]["end"]])
        self.assertTrue(units[0]["text"].endswith("growth."))

    def test_truncated_target_tail_excluded(self):
        text = "Amazon advanced after reports revealed its"
        self.assertEqual(paragraph_spans(text, "AMZN"), [])

    def test_promotional_target_text_excluded(self):
        text = "InvestorsObserver issues critical PriceWatch Alerts for AAPL."
        self.assertEqual(paragraph_spans(text, "AAPL"), [])

    def test_public_metadata_paths_are_portable(self):
        absolute = str(ROOT / "outputs" / "example.csv")
        self.assertEqual(
            portable_paths({absolute: [absolute, "unchanged"]}),
            {"outputs/example.csv": ["outputs/example.csv", "unchanged"]},
        )


if __name__ == "__main__":
    unittest.main()
