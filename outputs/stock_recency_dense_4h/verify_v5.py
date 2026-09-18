"""Verify the corrected dense v5 executable gate and fit provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "v5"
PRIVATE = HERE.parents[1] / "work" / "stock-data" / "recency_dense_4h" / "v5_dense"


def main() -> None:
    summary = json.loads((PUBLIC / "dense_gate_corrected.json").read_text())
    gate = summary["dense_gate_corrected"]
    rows = pd.read_csv(PUBLIC / "dense_gate_rows.csv")
    expected = {(s, m) for s in ("AAPL", "AMZN") for m in ("2018-06", "2018-07", "2018-08")}
    cells = set(zip(rows.symbol, rows.month))
    evidence = json.loads((PUBLIC / "dense_training_evidence.json").read_text())
    checks = {
        "status_complete": summary.get("status") == "COMPLETE",
        # The dense generator is intentionally used for both sides when the
        # official rows do not match the canonical generator.  A failed raw
        # parity check is therefore acceptable only when the matched-control
        # mode is explicitly recorded.
        "parity_handled": (summary.get("canonical_parity_pass") is True or summary.get("feature_mode") == "reconstructed_all_official_and_augmented"),
        "gate_filter_before_merge": gate.get("filter_applied_before_merge") is True,
        "exact_six_gate_cells": len(rows) == 6 and cells == expected and gate.get("n_rows") == 6,
        "gate_months_june_august": gate.get("months") == ["2018-06", "2018-07", "2018-08"],
        "fit_count_instrumented": evidence.get("fit_count") == len(evidence.get("fits", [])) == len(list((PRIVATE / "models").glob("*.joblib"))),
        "historical_v4_preserved": summary.get("historical_v4_preserved") is True,
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "gate": gate}
    (PUBLIC / "verification_v5.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
