"""Verify the public full-corpus reaction audit and probe artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "v1"


def main() -> None:
    gate = json.loads((ROOT / "reaction_gate.json").read_text())
    probe = json.loads((ROOT / "reaction_probe_gate.json").read_text())
    downstream = pd.read_csv(ROOT / "reaction_downstream_predictions.csv")
    public_columns = set(pd.read_csv(ROOT / "reaction_predictions.csv", nrows=1).columns)
    rows = {x.get("symbol"): x for x in gate.get("gate", [])}
    checks = {
        "feasibility_pass": gate.get("status") == "PASS",
        "raw_rows_78055": gate.get("counts", {}).get("raw_rows_seen") == 78055,
        "AAPL_gate_pass": rows.get("AAPL", {}).get("passes") is True,
        "AMZN_gate_pass": rows.get("AMZN", {}).get("passes") is True,
        "probe_complete": probe.get("status") == "COMPLETE",
        "downstream_complete": probe.get("downstream_W0_W3") == "COMPLETE",
        "downstream_keys_unique": not downstream.duplicated("key").any(),
        "raw_text_not_published": not bool(public_columns & {"body", "title", "text"}),
        "AR2_scope_declared": probe.get("ar2_scope") == "coverage_limited_precomputed_frozen_vectors",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_counts": gate.get("counts", {}),
        "probe_embedding": probe.get("embedding", {}),
        "promoted_horizons": probe.get("downstream", {}).get("promoted_horizons", []),
        "downstream_rows": int(len(downstream)),
        "note": "Article-level reaction probes passed a limited horizon gate; downstream W0-W3 did not yield a stable two-stock improvement.",
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
