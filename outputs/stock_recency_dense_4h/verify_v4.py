"""Verify the public v4 recency/dense artifacts without loading private data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "v4"


def main() -> None:
    recency = json.loads((ROOT / "recency_selection.json").read_text())
    parity = pd.read_csv(ROOT / "recency_refit_parity.csv")
    dense = json.loads((ROOT / "dense_gate_corrected.json").read_text())
    checks = {
        "recency_status_complete": recency.get("status") == "COMPLETE",
        "recency_parity_pass": bool(recency.get("parity_pass")),
        "recency_max_probability_error_le_1e-10": bool(parity["max_probability_abs_diff"].max() <= 1e-10),
        "registered_half_lives": recency.get("half_lives") == ["infinity", "80", "40", "20"],
        "dense_status_complete": dense.get("status") == "COMPLETE",
        "dense_gate_has_six_month_rows": dense.get("dense_gate_corrected", {}).get("n_month_rows") == 6,
        "dense_gate_months_are_jun_to_aug": dense.get("dense_gate_corrected", {}).get("months") == ["2018-06", "2018-07", "2018-08"],
        "dense_gate_stops_d2": dense.get("D2_status") == "STOPPED_BY_GATE",
        "official_rows_1607": dense.get("official_rows") == 1607,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "parity_rows": int(len(parity)),
        "max_abs_probability_error": float(parity["max_probability_abs_diff"].max()),
        "recency_gate": recency.get("gate", []),
        "dense_gate": dense.get("dense_gate_corrected", {}),
        "note": "Public artifact verification only; development and later periods remain exposed exploratory backtests.",
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
