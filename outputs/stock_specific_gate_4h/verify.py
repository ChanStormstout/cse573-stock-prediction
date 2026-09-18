"""Static contract tests and, after approval, reconstruction checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import STATE_COLUMNS, design_matrix, map_advantage

HERE = Path(__file__).resolve().parent


def static_checks():
    rng = np.random.default_rng(573); state = rng.normal(size=(8, len(STATE_COLUMNS))); stock = np.array([0, 1] * 4, float)
    g2 = design_matrix(state, stock, "G2"); g3 = design_matrix(state, stock, "G3"); g3_zero = design_matrix(state, stock, "G3", interaction_zero=True)
    checks = {
        "g3_zero_nested_design": bool(np.array_equal(g3[:, :g2.shape[1]], g2) and np.all(g3_zero[:, g2.shape[1]:] == 0)),
        "mapping_bounds": bool(np.all((map_advantage(np.linspace(-2, 2, 8), np.full(8, .5), np.full(8, .6), np.ones(8))[0] >= 0) & (map_advantage(np.linspace(-2, 2, 8), np.full(8, .5), np.full(8, .6), np.ones(8))[0] <= 1))),
    }
    p, w = map_advantage(np.zeros(2), np.array([.61, .4]), np.array([np.nan, .8]), np.array([0, 1]))
    checks["no_news_exact_price"] = bool(p[0] == .61 and w[0] == 0.0)
    return checks


def result_checks(root: Path):
    pred = pd.read_csv(root / "predictions.csv"); evidence = json.loads((root / "controller_training_evidence.json").read_text())
    no = pred.has_news == 0
    checks = {
        "no_news_exact": bool(np.array_equal(pred.loc[no, "p_final"].to_numpy(float), pred.loc[no, "p_price"].to_numpy(float))),
        "weights_bounded": bool(((pred.w_text >= 0) & (pred.w_text <= 1)).all()),
        "no_news_zero_weight": bool((pred.loc[no, "w_text"] == 0).all()),
        "time_safe_fits": bool(all((not x["time_safe"]) is False for x in evidence["fits"])),
        "frozen_exposed": bool(all(x.get("frozen_no_exposed_label_update", False) for x in evidence["fits"] if x["fold"] == "august_freeze")),
    }
    return checks


def main():
    checks = static_checks(); root = HERE / "v1"
    if root.exists() and (root / "predictions.csv").exists():
        checks.update(result_checks(root))
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "mode": "static_only" if not root.exists() else "static_plus_results"}
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
