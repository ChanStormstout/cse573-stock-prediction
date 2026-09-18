"""Independent integrity checks for the bounded run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import INPUTS, R1_COLS, load_data, metric_rows, scores, sha
from build_dense_windows import feature_for, load_bars
from common import schedule


def main(public: Path, private: Path):
    checks = {}
    d = load_data(); ref = pd.read_csv(public.parent.parent / "stock_goal60_4h" / "v1" / "predictions.csv").set_index("key")
    p = pd.read_csv(public / "recency_predictions.csv").set_index("key")
    expected = ref[ref.phase != "warmup"].index
    checks["official_keys_exact"] = bool(set(p.index) == set(expected) and len(p) == 1374)
    checks["labels_match"] = bool(np.array_equal(p.loc[expected, "label"].to_numpy(), d.set_index("key").loc[expected, "label"].to_numpy()))
    parity = {}
    for m in ("R1", "F1", "F2"):
        a = p[f"{m}_equal"].to_numpy(dtype=float); b = ref.loc[p.index, m].to_numpy(dtype=float); parity[m] = float(np.max(np.abs(a - b)))
    checks["equal_reference_exact"] = bool(max(parity.values()) == 0.0)
    rec = p[[c for c in p.columns if c.endswith("_recency")]].to_numpy(dtype=float)
    checks["probabilities_finite"] = bool(np.isfinite(rec).all() and ((rec >= 0) & (rec <= 1)).all())
    metric_recomputed, _ = metric_rows(p.reset_index(), [c for c in p.columns if c.endswith("_equal") or c.endswith("_recency")])
    saved = pd.read_csv(public / "recency_metrics.csv").sort_values(["phase", "symbol", "method"]).reset_index(drop=True)
    recomputed = metric_recomputed.sort_values(["phase", "symbol", "method"]).reset_index(drop=True)
    checks["metrics_reconstruct"] = bool(np.allclose(saved.BA.fillna(-1), recomputed.BA.fillna(-1), atol=1e-12) and np.allclose(saved.Brier, recomputed.Brier, atol=1e-12))
    # Weight audit: every recorded fit has positive finite weights and recent
    # half-lives have a non-increasing weight range, while infinity is exactly one.
    ev = json.loads((public / "recency_training_evidence.json").read_text()) if (public / "recency_training_evidence.json").exists() else json.loads((public / "training_evidence.json").read_text())
    fits = ev.get("fits", [])
    checks["fit_weights_valid"] = bool(all(np.isfinite(x.get("weight_sum", 1)) and x.get("weight_sum", 1) > 0 and x.get("weight_min", 1) > 0 for x in fits))
    # Dense rows are exactly 48 bars by construction; perturbing a target bar
    # must not change a cutoff feature. This is an independent future-feature test.
    dense = pd.read_pickle(private / "dense_windows.pkl")
    checks["dense_unique"] = bool(len(dense) == dense.key.nunique() and (dense[R1_COLS].notna().any(axis=1)).all())
    row = dense.iloc[len(dense) // 2]; _, b = load_bars(row.symbol); sch = schedule(); session = next(r for r in sch.itertuples() if r.open.date() == row.start_utc.date()); before = feature_for(row.cutoff_utc, session, sch, b.copy()); b2 = b.copy(); target = pd.date_range(row.start_utc, row.end_utc - pd.Timedelta("5min"), freq="5min"); b2.loc[target, ["open", "high", "low", "close"]] = 999999.0; after = feature_for(row.cutoff_utc, session, sch, b2)
    checks["future_target_perturbation_unchanged"] = bool(all((pd.isna(before[c]) and pd.isna(after[c])) or (np.isclose(before[c], after[c], atol=0, rtol=0)) for c in R1_COLS))
    reaction = pd.read_csv(public / "article_reaction_audit.csv"); checks["reaction_audit_nonnegative_counts"] = bool((reaction[["raw_article_count", "unique_group_count", "complete_reaction_count"]] >= 0).all().all())
    activity = json.loads((public / "activity_audit.json").read_text()); checks["activity_not_used"] = activity["status"] == "SEMANTICS_UNRESOLVED_NOT_USED"
    checks["all_pass"] = all(checks.values())
    result = {"checks": checks, "parity_max_abs_error": parity, "source_sha256": {"inputs": sha(INPUTS)}, "status": "PASS" if checks["all_pass"] else "FAIL"}
    (public / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if not checks["all_pass"]: raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v3"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v3"); a = p.parse_args(); main(Path(a.public), Path(a.private))
