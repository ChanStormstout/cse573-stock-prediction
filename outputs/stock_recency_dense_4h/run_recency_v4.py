"""Corrected canonical F1/F2 recency experiment.

This file deliberately lives beside, rather than inside, the historical v3
implementation.  It uses the canonical ``Transform`` from
``stock_paper_methods_4h/run.py`` and records a true equal-weight refit before
testing recency weights.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PAPER_DIR = ROOT / "outputs" / "stock_paper_methods_4h"
MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]
INNER = MONTHS[:3]
OUTER = MONTHS[3:]
HALF_LIVES = (None, 80, 40, 20)
HALF_NAME = {None: "infinity", 80: "80", 40: "40", 20: "20"}
R1_COLS = [f"{v}_{i}" for i in range(1, 7) for v in ("return", "range")] + [
    "history_age_hours", "return_mean", "return_std", "ny_hour"
]
R1_COLS += [f"recent_{v}_{n}" for n in (5, 15, 30, 60) for v in ("return", "range", "rv", "missing")]
R1_COLS += ["overnight_gap", "overnight_gap_missing", "minutes_from_open", "minutes_to_close"]
CS = (0.01, 0.1, 1.0)


def load_paper_module():
    spec = importlib.util.spec_from_file_location("paper_run_v4_canonical", PAPER_DIR / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical paper run")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); q = p >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, q)),
        "Brier": float(brier_score_loss(y, p)),
        "pred_up": float(q.mean()),
        "constant": bool(len(np.unique(q)) == 1),
    }


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if hasattr(x, "item"):
            return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def standardize_fit(frame, cols):
    x = frame[cols].to_numpy(float)
    mean = np.nan_to_num(np.nanmean(x, axis=0))
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return mean, scale


def standardize(frame, cols, mean, scale):
    x = (frame[cols].to_numpy(float) - mean) / scale
    return np.nan_to_num(x)


def c_schedule():
    path = PAPER_DIR / "v1" / "selection.json"
    raw = json.loads(path.read_text())
    price_path = WORK / "nextgen_4h" / "price_v1" / "selection.json"
    price = json.loads(price_path.read_text())
    out = {}
    for row in price:
        if row.get("method") == "R1":
            out[(row["symbol"], "R1", row["month"])] = float(row["C"])
    for s in ("AAPL", "AMZN"):
        out[(s, "R1", "final")] = next(float(x["C"]) for x in price if x.get("symbol") == s and x.get("method") == "R1" and x.get("month") == "final")
    for row in raw["joint"]:
        if row["method"] == "J0":
            out[(row["symbol"], "F1", row["month"])] = float(row["C"])
        elif row["method"] == "J2":
            out[(row["symbol"], "F2", row["month"])] = float(row["C"])
    # The paper-methods final C is recorded in the final selection.  Fall back
    # to 1.0 only if an old selection file predates final records.
    for s in ("AAPL", "AMZN"):
        out.setdefault((s, "F1", "final"), 1.0)
        out.setdefault((s, "F2", "final"), 1.0)
    return out, sha(path)


def age_weights(train, boundary, half, opens):
    end = pd.to_datetime(train.end_utc, utc=True).to_numpy(dtype="datetime64[ns]")
    b = np.datetime64(pd.Timestamp(boundary).tz_convert("UTC").tz_localize(None), "ns")
    ages = np.asarray([int(((opens > e) & (opens < b)).sum()) for e in end], dtype=int)
    if half is None:
        weights = np.ones(len(train), dtype=float)
    else:
        weights = np.power(2.0, -ages / float(half))
    return weights, ages


def fit_r1(train, ev, C, half, boundary, opens, path):
    mean, scale = standardize_fit(train, R1_COLS)
    x = standardize(train, R1_COLS, mean, scale); z = standardize(ev, R1_COLS, mean, scale)
    weights, ages = age_weights(train, boundary, half, opens)
    model = LogisticRegression(C=C, solver="liblinear", tol=1e-8, max_iter=3000, random_state=573)
    model.fit(x, train.label.to_numpy(int), sample_weight=weights)
    p = model.predict_proba(z)[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"columns": R1_COLS, "mean": mean, "scale": scale, "model": model}, path)
    re = joblib.load(path)["model"].predict_proba(z)[:, 1]
    err = float(np.max(np.abs(p - re))) if len(p) else 0.0
    return p, {"method": "R1", "half_life": HALF_NAME[half], "C": C, "train_n": len(train), "eval_n": len(ev), "weight_sum": float(weights.sum()), "weight_min": float(weights.min()), "weight_max": float(weights.max()), "age_min": int(ages.min()) if len(ages) else None, "age_max": int(ages.max()) if len(ages) else None, "reload_max_abs_error": err, "model_sha256": sha(path)}


def fit_text(paper, full, train_idx, ev_idx, method, C, half, boundary, opens, emb, ids, means, events, path):
    # J0 and J2 are the canonical F1/F2 definitions.  The returned objects
    # are intentionally not reimplemented here, preventing a silent F2 drift.
    canonical = "J0" if method == "F1" else "J2"
    train = full.iloc[train_idx]; ev = full.iloc[ev_idx]
    tf = paper.Transform(canonical).fit(full, train_idx, emb, ids, means, events)
    x = tf.transform(full, train_idx, means, events)
    z = tf.transform(full, ev_idx, means, events)
    weights, ages = age_weights(train, boundary, half, opens)
    model = LogisticRegression(C=C, solver="liblinear", tol=1e-8, max_iter=3000, random_state=573)
    model.fit(x, train.label.to_numpy(int), sample_weight=weights)
    p = model.predict_proba(z)[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"transform": tf, "model": model, "method": method, "C": C, "half_life": HALF_NAME[half]}, path)
    loaded = joblib.load(path); re = loaded["model"].predict_proba(z)[:, 1]
    err = float(np.max(np.abs(p - re))) if len(p) else 0.0
    return p, {"method": method, "half_life": HALF_NAME[half], "C": C, "train_n": len(train), "eval_n": len(ev), "weight_sum": float(weights.sum()), "weight_min": float(weights.min()), "weight_max": float(weights.max()), "age_min": int(ages.min()) if len(ages) else None, "age_max": int(ages.max()) if len(ages) else None, "reload_max_abs_error": err, "model_sha256": sha(path)}


def main(run_name="v4"):
    public = HERE / run_name; private = WORK / "recency_dense_4h" / run_name
    if public.exists():
        allowed = {"PRE_REGISTRATION.md", "CORRECTIONS.md"}
        if any(p.name not in allowed for p in public.iterdir()):
            raise FileExistsError("new run directory required")
    else:
        public.mkdir(parents=True)
    if private.exists():
        raise FileExistsError("private run directory already exists")
    private.mkdir(parents=True); (private / "models").mkdir()
    t0 = time.monotonic(); paper = load_paper_module()
    # prepare() is the canonical paper-method input builder.  It also records
    # source fingerprints in the private directory; no raw text is published.
    canonical_private = private / "canonical"; canonical_private.mkdir(parents=True, exist_ok=True)
    d, emb, ids, means, events, fingerprints = paper.prepare(canonical_private)
    d = d.reset_index(drop=True)
    # Views retain original integer positions required by canonical Transform.
    d["_pos"] = np.arange(len(d), dtype=int)
    choices, c_sha = c_schedule()
    schedule = pd.read_csv(WORK / "audit" / "xnys_schedule.csv")
    schedule["open"] = pd.to_datetime(schedule["open"], utc=True)
    opens = schedule.open.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    ref = d[["key", "F1", "F2", "R1"]].copy()
    # All historical non-warmup references are available; warmup values are NaN.
    records = []; fits = []; parity = []; predictions = {}

    def fit_one(symbol, tag, method, half, train, ev, fallback_mode):
        C = choices[(symbol, method, tag)]
        tr = d.iloc[train]; ee = d.iloc[ev]
        boundary = pd.Timestamp(ee.cutoff_utc.min()).tz_convert("UTC") if tag != "final" else pd.Timestamp("2018-09-01", tz="UTC")
        if method == "R1":
            p, evidence = fit_r1(tr, ee, C, half, boundary, opens, private / "models" / f"{symbol}_{tag}_R1_{HALF_NAME[half]}.joblib")
        else:
            p, evidence = fit_text(paper, d, train, ev, method, C, half, boundary, opens, emb, ids, means, events, private / "models" / f"{symbol}_{tag}_{method}_{HALF_NAME[half]}.joblib")
        fits.append({**evidence, "symbol": symbol, "month": tag, "role": "main"})
        return p

    # True infinity refits and all half-life folds.  For each fold, cache R1
    # fallbacks so branch/system policies use the exact same prediction.
    for tag in MONTHS[:3]:
        for symbol in ("AAPL", "AMZN"):
            train_mask = (d.symbol == symbol) & (d.month < tag)
            ev_mask = (d.symbol == symbol) & (d.month == tag)
            train = np.flatnonzero(train_mask); ev = np.flatnonzero(ev_mask)
            r1 = {}
            for half in HALF_LIVES:
                r1[HALF_NAME[half]] = fit_one(symbol, tag, "R1", half, train, ev, "none")
            for method in ("F1", "F2"):
                for half in HALF_LIVES:
                    mainp = fit_one(symbol, tag, method, half, train, ev, "none")
                    hn = HALF_NAME[half]
                    no = d.loc[ev, "has_original_news"].to_numpy(dtype=int) == 0
                    branch = mainp.copy(); branch[no] = r1["infinity"][no]
                    system = mainp.copy(); system[no] = r1[hn][no]
                    for variant, p in (("infinity_refit", branch if half is None else None), ("branch_recency", branch), ("system_recency", system)):
                        if p is None: continue
                        met = metric(d.loc[ev, "label"], p)
                        records.append({"fold": "inner", "method": method, "variant": variant, "half_life": hn, "symbol": symbol, "month": tag, **met})
                        predictions[("inner", symbol, tag, method, variant, hn)] = (ev.copy(), p.copy())
            for variant, p in [("infinity_refit", r1["infinity"]), ("R1_recency", r1[HALF_NAME[None]])]:
                if variant == "R1_recency": continue
    # Global half-life selection uses branch_recency only for F1/F2; R1 uses its
    # own recency prediction.  Longer half-life wins exact ties.
    selection = {}
    rank = {"infinity": 0, "80": 1, "40": 2, "20": 3}
    for method in ("R1", "F1", "F2"):
        summaries = []
        for hn in ("infinity", "80", "40", "20"):
            if method == "R1":
                rows = []
                for tag in INNER:
                    for symbol in ("AAPL", "AMZN"):
                        ev = np.flatnonzero((d.symbol == symbol) & (d.month == tag));
                        p = fit_one(symbol, tag, "R1", next(h for h in HALF_LIVES if HALF_NAME[h] == hn), np.flatnonzero((d.symbol == symbol) & (d.month < tag)), ev, "none")
                        rows.append((symbol, tag, metric(d.loc[ev, "label"], p)))
            else:
                rows = [(r["symbol"], r["month"], r) for r in records if r["fold"] == "inner" and r["method"] == method and r["variant"] == "branch_recency" and r["half_life"] == hn]
            by_stock = {s: float(np.mean([x[2]["BA"] for x in rows if x[0] == s])) for s in ("AAPL", "AMZN")}
            summaries.append({"half_life": hn, "AAPL_mean_BA": by_stock["AAPL"], "AMZN_mean_BA": by_stock["AMZN"], "weaker_stock_mean_BA": min(by_stock.values()), "macro_mean_BA": float(np.mean(list(by_stock.values())))})
        selection[method] = sorted(summaries, key=lambda x: (-x["weaker_stock_mean_BA"], -x["macro_mean_BA"], rank[x["half_life"]]))[0] | {"all": summaries}
    # Parity is assessed on each chronological fold and the final freeze.  The
    # comparison is after the registered no-news policy, not just raw logits.
    for tag in INNER + ["final"]:
        for symbol in ("AAPL", "AMZN"):
            train = np.flatnonzero((d.symbol == symbol) & (d.month < ("2018-09" if tag == "final" else tag)))
            ev = np.flatnonzero((d.symbol == symbol) & ((d.month >= "2018-09") if tag == "final" else (d.month == tag)))
            if not len(ev): continue
            rinf = fit_one(symbol, tag, "R1", None, train, ev, "none")
            for method in ("R1", "F1", "F2"):
                if method == "R1": p = rinf
                else:
                    mainp = fit_one(symbol, tag, method, None, train, ev, "none"); p = mainp.copy(); p[d.loc[ev, "has_original_news"].to_numpy(dtype=int) == 0] = rinf[d.loc[ev, "has_original_news"].to_numpy(dtype=int) == 0]
                refp = d.loc[ev, method].to_numpy(float)
                valid = np.isfinite(refp)
                dif = float(np.max(np.abs(p[valid] - refp[valid]))) if valid.any() else 0.0
                a = metric(d.loc[ev, "label"].to_numpy()[valid], p[valid]); b = metric(d.loc[ev, "label"].to_numpy()[valid], refp[valid])
                parity.append({"fold": tag, "symbol": symbol, "method": method, "max_probability_abs_diff": dif, "BA_refit": a["BA"], "BA_reference": b["BA"], "BA_diff": None if a["BA"] is None or b["BA"] is None else a["BA"]-b["BA"], "MCC_diff": a["MCC"]-b["MCC"], "Brier_diff": a["Brier"]-b["Brier"], "n": int(valid.sum())})
    parity_df = pd.DataFrame(parity); parity_df.to_csv(public / "recency_refit_parity.csv", index=False)
    parity_pass = bool(len(parity_df) and parity_df.max_probability_abs_diff.max() <= 1e-10)
    if not parity_pass:
        status = "STOPPED_INFINITY_REFIT_PARITY"
        dump(public / "recency_selection.json", {"status": status, "parity_tolerance": 1e-10, "selection": selection})
        dump(public / "recency_training_evidence.json", {"status": status, "fits": fits, "parity_pass": False, "runtime_seconds": time.monotonic()-t0})
        raise SystemExit("v4 recency stopped: infinity refit parity failed")
    # Outer corrected evidence: selected branch/system against infinity refit.
    outer_rows = []; pred_rows = []
    for tag in OUTER:
        for symbol in ("AAPL", "AMZN"):
            train = np.flatnonzero((d.symbol == symbol) & (d.month < tag)); ev = np.flatnonzero((d.symbol == symbol) & (d.month == tag))
            rinf = fit_one(symbol, tag, "R1", None, train, ev, "none")
            rsel = fit_one(symbol, tag, "R1", next(h for h in HALF_LIVES if HALF_NAME[h] == selection["R1"]["half_life"]), train, ev, "none")
            row = d.loc[ev, ["key","symbol","month","phase","label","has_original_news"]].copy().reset_index(drop=True)
            row["R1_infinity_refit"] = rinf; row["R1_recency"] = rsel
            for method in ("F1", "F2"):
                inf_main = fit_one(symbol, tag, method, None, train, ev, "none")
                sel_half = next(h for h in HALF_LIVES if HALF_NAME[h] == selection[method]["half_life"])
                sel_main = fit_one(symbol, tag, method, sel_half, train, ev, "none")
                no = d.loc[ev, "has_original_news"].to_numpy(dtype=int) == 0
                inf = inf_main.copy(); inf[no] = rinf[no]
                branch = sel_main.copy(); branch[no] = rinf[no]
                system = sel_main.copy(); system[no] = rsel[no]
                row[f"{method}_infinity_refit"] = inf; row[f"{method}_branch_recency"] = branch; row[f"{method}_system_recency"] = system
            pred_rows.append(row)
            for method in ("R1","F1","F2"):
                base = row[f"{method}_infinity_refit"].to_numpy(float)
                variants = [("infinity_refit", base), ("branch_recency", row["R1_recency"])] if method == "R1" else [("infinity_refit",base),("branch_recency",row[f"{method}_branch_recency"]),("system_recency",row[f"{method}_system_recency"])]
                for variant, p in variants:
                    outer_rows.append({"method":method,"variant":variant,"symbol":symbol,"month":tag,"half_life":selection[method]["half_life"] if variant != "infinity_refit" else "infinity",**metric(row.label,p)})
    # Final freeze and exposed dev/later output.
    for symbol in ("AAPL", "AMZN"):
        train = np.flatnonzero((d.symbol == symbol) & (d.month < "2018-09")); ev = np.flatnonzero((d.symbol == symbol) & (d.month >= "2018-09"))
        if not len(ev): continue
        rinf = fit_one(symbol, "final", "R1", None, train, ev, "none"); rsel = fit_one(symbol, "final", "R1", next(h for h in HALF_LIVES if HALF_NAME[h] == selection["R1"]["half_life"]), train, ev, "none")
        row = d.loc[ev, ["key","symbol","month","phase","label","has_original_news"]].copy().reset_index(drop=True); row["R1_infinity_refit"] = rinf; row["R1_recency"] = rsel
        no = d.loc[ev, "has_original_news"].to_numpy(dtype=int) == 0
        for method in ("F1","F2"):
            a=fit_one(symbol,"final",method,None,train,ev,"none"); b=fit_one(symbol,"final",method,next(h for h in HALF_LIVES if HALF_NAME[h]==selection[method]["half_life"]),train,ev,"none")
            inf=a.copy();inf[no]=rinf[no];branch=b.copy();branch[no]=rinf[no];system=b.copy();system[no]=rsel[no]
            row[f"{method}_infinity_refit"]=inf;row[f"{method}_branch_recency"]=branch;row[f"{method}_system_recency"]=system
        pred_rows.append(row)
    preds = pd.concat(pred_rows, ignore_index=True); preds.to_csv(public / "recency_predictions.csv", index=False)
    outer_df = pd.DataFrame(outer_rows); outer_df.to_csv(public / "recency_outer_scores.csv", index=False)
    metric_rows=[]; monthly=[]
    methods=["R1_infinity_refit","R1_recency","F1_infinity_refit","F1_branch_recency","F1_system_recency","F2_infinity_refit","F2_branch_recency","F2_system_recency"]
    for (phase,symbol), g in preds.groupby(["phase","symbol"]):
        for m in methods: metric_rows.append({"phase":phase,"symbol":symbol,"method":m,**metric(g.label,g[m])})
        for (month,m_symbol), x in g.groupby(["month","symbol"]):
            for m in methods: monthly.append({"phase":phase,"symbol":symbol,"month":month,"method":m,**metric(x.label,x[m])})
    pd.DataFrame(metric_rows).to_csv(public / "recency_metrics.csv", index=False);pd.DataFrame(monthly).to_csv(public / "recency_monthly_metrics.csv", index=False)
    outer_df2=outer_df[outer_df.variant=="branch_recency"].copy(); inf=outer_df[outer_df.variant=="infinity_refit"].copy(); gate=[]
    for method in ("R1","F1","F2"):
        a=outer_df2[outer_df2.method==method].merge(inf[inf.method==method],on=["symbol","month"],suffixes=("_branch","_inf"))
        deltas=a.BA_branch-a.BA_inf; stock=a.assign(delta=deltas).groupby("symbol").delta.mean(); mon=a.assign(delta=deltas).groupby("month").delta.mean()
        gate.append({"method":method,"selected_half_life":selection[method]["half_life"],"AAPL_delta_BA":float(stock.get("AAPL",np.nan)),"AMZN_delta_BA":float(stock.get("AMZN",np.nan)),"macro_delta_BA":float(deltas.mean()),"positive_outer_months":int((mon>0).sum()),"passes":bool(stock.min()>=.01 and stock.max()>=-.01 and (mon>0).sum()>=2)})
    dump(public/"recency_selection.json",{"status":"COMPLETE","half_lives":[HALF_NAME[h] for h in HALF_LIVES],"selection":selection,"gate":gate,"parity_tolerance":1e-10,"parity_pass":parity_pass,"c_schedule_sha256":c_sha,"canonical_transform":"stock_paper_methods_4h/run.py"})
    dump(public/"recency_training_evidence.json",{"status":"COMPLETE","fits":fits,"fit_count":len(fits),"parity_pass":parity_pass,"runtime_seconds":time.monotonic()-t0,"source_fingerprints":fingerprints,"weight_formula":"2**(-session_age/half_life), infinity=1"})
    print(json.dumps({"status":"COMPLETE","gate":gate,"selection":selection,"parity_max":float(parity_df.max_probability_abs_diff.max())},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="v4");a=parser.parse_args();main(a.run)
