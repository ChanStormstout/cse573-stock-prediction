"""Independent checks for the TabPFN provenance/configuration probe."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIVATE = ROOT / "work" / "stock-data" / "tabpfn_4h" / "v2"


def metric(y, p):
    y = np.asarray(y); p = np.asarray(p); q = p >= .5
    return dict(BA=float(balanced_accuracy_score(y, q)), MCC=float(matthews_corrcoef(y, q)), Brier=float(brier_score_loss(y, p)))


def main() -> None:
    provenance = json.loads((HERE / "provenance.json").read_text())
    checks = {"provenance_pass": provenance.get("status") == "PASS", "not_synthetic_only": provenance.get("synthetic_only_claim") is False}
    outputs = {}
    fit_count = 0
    max_reload = 0.0
    for name in ("own_n8", "cross_n8"):
        p = pd.read_csv(PRIVATE / name / "predictions.csv")
        assert p.key.is_unique and len(p) == 1607
        outputs[name] = p
        e = json.loads((PRIVATE / name / "training_evidence.json").read_text())
        fit_count += e["fit_count"]
        max_reload = max(max_reload, max(float(r["reload_error"]) for r in e["fits"]))
        checks[f"{name}_fixed_n8"] = e["n_estimators"] == 8
        checks[f"{name}_fit_count_14"] = e["fit_count"] == 14
        checks[f"{name}_no_gradient_training"] = all(not r["gradient_training"] for r in e["fits"])
        checks[f"{name}_finite_predictions"] = bool(np.isfinite(p.loc[p.phase != "warmup", "p"]).all())
        checks[f"{name}_time_order"] = all(pd.Timestamp(r["train_end"]) < pd.Timestamp(r["eval_cutoff"]) for r in e["fits"])
    d = pd.read_csv(HERE / "comparison_predictions.csv")
    checks["comparison_keys_unique"] = d.key.is_unique and len(d) == 1607
    checks["comparison_recompute"] = True
    saved = pd.read_csv(HERE / "comparison_metrics.csv")
    rows = []
    for (phase, symbol), g in d[d.phase != "warmup"].groupby(["phase", "symbol"]):
        for method in ("own_n8", "cross_n8", "P_own_tabpfn", "P_cross_tabpfn"):
            got = metric(g.label, g[method])
            row = saved[(saved.phase == phase) & (saved.symbol == symbol) & (saved.method == method)]
            if len(row) != 1 or any(abs(float(row.iloc[0][k]) - got[k]) > 1e-12 for k in ("BA", "MCC", "Brier")):
                checks["comparison_recompute"] = False
            rows.append({"phase": phase, "symbol": symbol, "method": method, **got})
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fit_count": fit_count,
        "max_reload_error": max_reload,
        "note": "n=8 is a corrected configuration probe. The old n=1 run is a matched historical control; all exposed periods remain exploratory backtests.",
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
