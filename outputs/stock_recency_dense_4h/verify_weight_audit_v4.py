"""Audit the v4 recency formula and monotonicity without refitting models.

The detailed rows are private because they contain sample identifiers and
timestamps.  The public JSON contains only aggregate verification evidence.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from run_recency_v4 import HALF_LIVES, HALF_NAME, age_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = HERE / "v4"
PRIVATE = ROOT / "work" / "stock-data" / "recency_dense_4h" / "v4"


def main() -> None:
    with (PRIVATE / "canonical" / "inputs.pkl").open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        frame = obj
    elif isinstance(obj, dict):
        frame = next((obj[k] for k in ("d", "data", "frame") if k in obj), None)
    else:
        frame = obj[0]
    if frame is None:
        raise RuntimeError("canonical inputs did not contain the prepared frame")
    frame = frame.reset_index(drop=True)
    schedule = pd.read_csv(ROOT / "work" / "stock-data" / "audit" / "xnys_schedule.csv")
    schedule["open"] = pd.to_datetime(schedule["open"], utc=True)
    opens = schedule.open.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    rows = []
    end_by_key = dict(zip(frame.key.astype(str), frame.end_utc.astype(str)))
    formula_errors = []
    monotonic_violations = 0
    infinity_violations = 0
    for tag in [f"2018-{m:02d}" for m in range(3, 9)] + ["final"]:
        boundary = pd.Timestamp("2018-09-01", tz="UTC") if tag == "final" else pd.Timestamp(tag + "-01", tz="UTC")
        for symbol in ("AAPL", "AMZN"):
            train = frame[(frame.symbol == symbol) & (frame.month < ("2018-09" if tag == "final" else tag))]
            if train.empty:
                continue
            for half in HALF_LIVES:
                weights, ages = age_weights(train, boundary, half, opens)
                if half is None:
                    expected = np.ones_like(weights)
                    infinity_violations += int(np.any(weights != expected))
                else:
                    expected = np.power(2.0, -ages / float(half))
                    formula_errors.append(float(np.max(np.abs(weights - expected))))
                    order = np.argsort(ages, kind="stable")
                    monotonic_violations += int(np.sum(np.diff(weights[order]) > 1e-12))
                for key, age, weight in zip(train.key.astype(str), ages, weights):
                    rows.append({"fold": tag, "symbol": symbol, "sample_key": key,
                                 "target_end_utc": end_by_key[key],
                                 "boundary_utc": boundary.isoformat(),
                                 "half_life": HALF_NAME[half], "session_age": int(age),
                                 "weight": float(weight)})
    private_out = PRIVATE / "weight_audit.csv"
    private_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(private_out, index=False)
    result = {
        "status": "PASS" if not monotonic_violations and not infinity_violations and (not formula_errors or max(formula_errors) <= 1e-15) else "FAIL",
        "fold_count": len(set((r["fold"], r["symbol"]) for r in rows)),
        "row_count": len(rows),
        "max_formula_abs_error": max(formula_errors) if formula_errors else 0.0,
        "monotonic_violations": monotonic_violations,
        "infinity_unit_weight_violations": infinity_violations,
        "formula": "2**(-session_age/half_life); infinity=1",
        "detail_path_private": str(private_out),
        "note": "This verifies the v4 registered weighting function and saved-input fold boundaries; it does not refit models.",
    }
    (PUBLIC / "recency_weight_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
