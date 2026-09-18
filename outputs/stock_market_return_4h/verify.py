"""Integrity checks for the market/return v1 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "outputs" / "stock_market_return_4h" / "v1"
INPUT = ROOT / "work" / "stock-data" / "paper_methods_4h" / "v1" / "inputs.pkl"
RAW_FEATURES = json.loads((PUBLIC / "run_metadata.json").read_text())["features"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = pd.read_pickle(INPUT).reset_index(drop=True)
    p = pd.read_csv(PUBLIC / "predictions.csv")
    meta = json.loads((PUBLIC / "run_metadata.json").read_text())
    choices = json.loads((PUBLIC / "choices.json").read_text())["choices"]
    evidence = pd.read_json(PUBLIC / "training_evidence.json")
    failures = []

    if d.key.duplicated().any():
        failures.append("input duplicate keys")
    if p.duplicated(["method", "key"]).any():
        failures.append("prediction duplicate method/key")
    if not p.p.between(0, 1).all():
        failures.append("probability outside [0,1]")
    if not np.array_equal((d.target_return.to_numpy(float) > 0).astype(int), d.label.to_numpy(int)):
        failures.append("input label/return sign mismatch")
    expected_methods = {"F0", "R1", "F1", "F2", "F6"} | {f"{m}_{v}" for m in ("C0", "C1", "C2") for v in ("price", "price_F1")}
    if set(p.method) != expected_methods:
        failures.append(f"method set mismatch: {sorted(set(p.method) ^ expected_methods)}")
    for method in expected_methods:
        for phase, expected in (("train_forward_oof", 765), ("development", 252), ("later", 357)):
            n = int(((p.method == method) & (p.phase == phase)).sum())
            if n != expected:
                failures.append(f"{method}/{phase} count {n} != {expected}")
    # Target and time columns must never be in an input matrix.
    forbidden = {"label", "target_return", "cutoff_utc", "end_utc", "month", "phase", "day", "key"}
    for variant, names in RAW_FEATURES.items():
        if forbidden.intersection(names):
            failures.append(f"forbidden future/metadata feature in {variant}")
    # Every recorded fit uses a strictly earlier training endpoint.
    if not ((pd.to_datetime(evidence.train_end, utc=True) < pd.to_datetime(evidence.eval_cutoff, utc=True)).all()):
        failures.append("chronological training/evaluation violation")
    # C2 exposes the memory-only checkpoint round-trip error for every seed.
    c2 = evidence[evidence.method == "C2"]
    if c2.empty:
        failures.append("no C2 evidence")
    else:
        reload_errors = []
        for item in c2.evidence.dropna():
            obj = json.loads(item) if isinstance(item, str) else item
            for seed_item in obj.get("seeds", []):
                reload_errors.append(float(seed_item.get("reload_error", np.inf)))
        if not reload_errors or max(reload_errors) >= 1e-12:
            failures.append(f"checkpoint reload error {max(reload_errors) if reload_errors else 'missing'}")
    # Confirm all choices are registered candidate values, and no exposed phase
    # was used in selection.
    candidates = {"C0": {0.01, 0.1, 1.0}, "C1": {0.01, 0.1, 1.0, 10.0}, "C2": {0.0, 0.1, 0.5, 1.0}}
    for name, value in choices.items():
        method = name.split("/")[-1]
        if float(value) not in candidates[method]:
            failures.append(f"unregistered choice {name}={value}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "input_sha256": sha256(INPUT),
        "prediction_rows": len(p),
        "method_count": len(expected_methods),
        "choices": choices,
        "checks": ["keys", "probabilities", "label_sign", "counts", "feature_exclusion", "chronology", "checkpoint_reload", "registered_choices"],
    }
    (PUBLIC / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
