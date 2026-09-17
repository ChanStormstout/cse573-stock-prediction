"""Focused checks for causal eligibility and strict residual fallback."""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from common import paired_block_interval
from downstream import apply_fit, count_unique_events, eligibility_counts, prepare_design
from events import FEATURES, extract_rating_values, extract_target_price_values, merge_compatible_events


def identity(members, signature=("rating", "raise", None, None)):
    return json.dumps([{"members": members, "signature": list(signature)}], sort_keys=True)


class ResidualProtocolTest(unittest.TestCase):
    def test_partial_pooling_adds_one_stock_offset(self):
        base = {f"z_{name}": 0.0 for name in FEATURES}
        train = pd.DataFrame([{**base, "symbol": "AAPL", "event_gate": 1}, {**base, "symbol": "AMZN", "event_gate": 1}])
        design = prepare_design(train, train.copy(), "C2_partial_shared")
        self.assertEqual(design[0].shape[1], len(FEATURES) + 1)

    def test_paired_interval_is_zero_for_identical_probabilities(self):
        frame = pd.DataFrame(
            {
                "day": ["2018-01-01", "2018-01-01", "2018-01-02", "2018-01-02"],
                "key": ["a", "b", "c", "d"],
                "label": [0, 1, 0, 1],
                "left": [0.2, 0.8, 0.4, 0.6],
                "right": [0.2, 0.8, 0.4, 0.6],
            }
        )
        result = paired_block_interval(frame, "left", "right", 1)
        self.assertEqual((result["BA_low"], result["BA_high"], result["Brier_low"], result["Brier_high"]), (0.0, 0.0, 0.0, 0.0))

    def test_rating_values_are_target_specific_and_literal(self):
        text = "Stifel upgraded Fitbit from Sell to Hold. BMO downgraded Apple (AAPL) from Outperform to Market Perform. Citi downgraded Aetna from Buy to Neutral."
        self.assertEqual(extract_rating_values("lower", text, "AAPL"), ("Outperform", "Market Perform"))
        self.assertEqual(
            extract_rating_values("maintain", "Amazon (AMZN) was reiterated as Outperform and its target was raised.", "AMZN"),
            (None, "Outperform"),
        )

    def test_target_price_values_are_target_specific(self):
        sentence = "Apple (AAPL) was raised from $180 to $200, while Amazon (AMZN) was lowered to $170 from $190."
        self.assertEqual(extract_target_price_values("raise", sentence, "AAPL"), ("180", "200"))
        self.assertEqual(extract_target_price_values("lower", sentence, "AMZN"), ("190", "170"))

    def test_partial_title_and_complete_body_facts_merge(self):
        events = [
            {"kind": "rating", "action": "lower", "old": None, "new": "hold", "unit": "rating", "evidence_ids": ["S0"]},
            {"kind": "rating", "action": "lower", "old": "buy", "new": "hold", "unit": "rating", "evidence_ids": ["S1"]},
        ]
        merged = merge_compatible_events(events)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["old"], merged[0]["new"]), ("buy", "hold"))
        self.assertEqual(merged[0]["evidence_ids"], ["S1", "S0"])

    def test_overlapping_duplicate_groups_are_one_event(self):
        frame = pd.DataFrame(
            [
                {"symbol": "AAPL", "event_gate": 1, "event_identities": identity(["a", "b"])},
                {"symbol": "AAPL", "event_gate": 1, "event_identities": identity(["b", "c"])},
                {"symbol": "AAPL", "event_gate": 1, "event_identities": identity(["a"], ("target_price", "raise", "100", "120"))},
                {"symbol": "AMZN", "event_gate": 1, "event_identities": identity(["x"])},
            ]
        )
        self.assertEqual(count_unique_events(frame, "AAPL"), 2)
        self.assertEqual(count_unique_events(frame, "AMZN"), 1)

    def test_many_repeated_windows_do_not_satisfy_unique_event_minimum(self):
        frame = pd.DataFrame(
            [{"symbol": "AAPL", "event_gate": 1, "event_identities": identity(["same"])} for _ in range(60)]
            + [{"symbol": "AMZN", "event_gate": 0, "event_identities": "[]"}]
        )
        counts = eligibility_counts(frame)
        self.assertEqual(counts["event_windows"]["AAPL"], 60)
        self.assertEqual(counts["unique_event_groups"]["AAPL"], 1)
        self.assertFalse(counts["independent_eligible"]["AAPL"])

    def test_g_zero_is_exact_base_probability(self):
        base = {f"z_{name}": 0.0 for name in FEATURES}
        train_rows = []
        for index, label in enumerate((0, 1, 0, 1)):
            row = {**base, "symbol": "AAPL", "event_gate": 1, "label": label, "F1": 0.4 + 0.1 * label}
            row["z_rating_raise"] = float(index + 1)
            train_rows.append(row)
        evaluation = pd.DataFrame(
            [
                {**base, "symbol": "AAPL", "event_gate": 0, "label": 0, "F1": 0.123456789},
                {**base, "symbol": "AAPL", "event_gate": 1, "label": 1, "F1": 0.6, "z_rating_raise": 3.0},
            ]
        )
        probability, _ = apply_fit(pd.DataFrame(train_rows), evaluation, "C0_shared", 0.1, {"AAPL": False, "AMZN": False})
        self.assertTrue(np.array_equal(probability[:1], evaluation.F1.to_numpy()[:1]))


if __name__ == "__main__":
    unittest.main()
