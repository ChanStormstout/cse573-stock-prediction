"""Run the preregistered TabPFN-2.5 n_estimators=8 control.

The old n=1 run is preserved in stock_goal60_4h/v1.  This script changes only
the ensemble size and writes a separate private/public run family.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from threadpoolctl import threadpool_limits

from core import MONTHS, OLD, RECENT, data, metric, sha, split

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PRIVATE = ROOT / "work" / "stock-data" / "tabpfn_4h" / "v2"
CHECKPOINT = ROOT / "work" / "stock-data" / "goal60_4h" / "tabpfn_model" / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
OLD_OUT = ROOT / "outputs" / "stock_goal60_4h" / "v1" / "predictions.csv"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x), allow_nan=False) + "\n")


def create_model() -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=8,
        device="cpu",
        model_path=str(CHECKPOINT),
        random_state=573,
        n_preprocessing_jobs=1,
    )


def run_branch(name: str, x: np.ndarray, d: pd.DataFrame) -> pd.DataFrame:
    dest = PRIVATE / name
    if (dest / "done.json").exists():
        raise FileExistsError(f"{name} already complete; preserving old run")
    dest.mkdir(parents=True, exist_ok=False)
    pred = np.full(len(d), np.nan, dtype=float)
    fits = []
    cv = []
    started = time.monotonic()
    for month in MONTHS:
        for symbol in ("AAPL", "AMZN"):
            tr, ev = split(d, symbol, month)
            model = create_model()
            tick = time.monotonic()
            model.fit(x[tr], d.iloc[tr].label.to_numpy())
            p = model.predict_proba(x[ev])[:, 1].astype(float)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            np.savez_compressed(dest / f"{symbol}_{month}_n8.npz", train_indices=tr, eval_indices=ev, p=p)
            # A fresh model reload is deliberately tested on a fixed first
            # three evaluation rows.  The model itself is a frozen prior; fit
            # here means conditioning on the past rows, not gradient training.
            reload_model = create_model()
            reload_model.fit(x[tr], d.iloc[tr].label.to_numpy())
            reload_p = reload_model.predict_proba(x[ev[:3]])[:, 1]
            reload_error = float(np.max(np.abs(reload_p - p[:3])))
            assert reload_error < 1e-5
            pred[ev] = p
            row = dict(
                symbol=symbol,
                month=month,
                train_n=len(tr),
                eval_n=len(ev),
                train_end=str(d.iloc[tr].end_utc.max()),
                eval_cutoff=str(d.iloc[ev].cutoff_utc.min()),
                seconds=time.monotonic() - tick,
                n_estimators=8,
                checkpoint_sha256=sha(CHECKPOINT),
                reload_error=reload_error,
                gradient_training=False,
                device="cpu",
            )
            fits.append(row)
            if month != "final":
                cv.append({**row, **metric(d.iloc[ev].label, p)})
        print(name, month, "fits", len(fits), "elapsed", round(time.monotonic() - started, 1), flush=True)
    out = d[["key", "symbol", "month", "phase", "label"]].copy()
    out["p"] = pred
    out.to_csv(dest / "predictions.csv", index=False)
    pd.DataFrame(cv).to_csv(dest / "cv.csv", index=False)
    dump(dest / "training_evidence.json", {
        "status": "COMPLETE",
        "name": name,
        "fits": fits,
        "fit_count": len(fits),
        "n_estimators": 8,
        "checkpoint_sha256": sha(CHECKPOINT),
        "runtime_seconds": time.monotonic() - started,
        "gradient_training": False,
        "selection": "no TabPFN hyperparameter search; n=8 fixed before evaluation",
    })
    dump(dest / "done.json", {"status": "COMPLETE", "fit_count": len(fits), "checkpoint_sha256": sha(CHECKPOINT)})
    return out


def summarize(d: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, out in outputs.items():
        for (phase, symbol), g in out[out.phase != "warmup"].groupby(["phase", "symbol"]):
            rows.append({"method": name, "phase": phase, "symbol": symbol, **metric(g.label, g.p)})
            for month, m in g.groupby("month"):
                rows.append({"method": name, "phase": phase, "symbol": symbol, "month": month, **metric(m.label, m.p)})
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    old = pd.read_csv(OLD_OUT).set_index("key")
    joined = d[["key", "symbol", "month", "phase", "label"]].set_index("key").join(old[["P_own_tabpfn", "P_cross_tabpfn", "F1_new", "R1"]])
    for name, out in outputs.items():
        joined[name] = out.set_index("key").p
    comparison = []
    for (phase, symbol), g in joined[joined.phase != "warmup"].groupby(["phase", "symbol"]):
        for method in ["R1", "F1_new", "P_own_tabpfn", "P_cross_tabpfn", "own_n8", "cross_n8"]:
            comparison.append({"phase": phase, "symbol": symbol, "method": method, **metric(g.label, g[method])})
    pd.DataFrame(comparison).to_csv(OUT / "comparison_metrics.csv", index=False)
    joined.reset_index().to_csv(OUT / "comparison_predictions.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    d = data()
    x_own = d[OLD + RECENT].to_numpy(float)
    cross_path = ROOT / "work" / "stock-data" / "goal60_4h" / "v1" / "cross.pkl"
    x_cross = np.c_[x_own, pd.read_pickle(cross_path).to_numpy(float)]
    assert len(d) == len(x_own) == len(x_cross) == 1607
    with threadpool_limits(limits=4):
        own = run_branch("own_n8", x_own, d)
        cross = run_branch("cross_n8", x_cross, d)
    summarize(d, {"own_n8": own, "cross_n8": cross})
    dump(OUT / "status.json", {
        "status": "COMPLETE", "rows": len(d), "n_estimators": 8,
        "checkpoint_sha256": sha(CHECKPOINT), "runtime": "local TabPFN 6.3.0 bundle",
        "source_protocol_sha256": sha(OUT / "PRE_REGISTRATION.md"),
    })


if __name__ == "__main__":
    main()
