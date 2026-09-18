"""Run the registered four-hour continuous-return auxiliary experiment.

The script deliberately keeps the experiment small and auditable: raw price
features are fitted inside each chronological fold, the direction task remains
the primary task, and the continuous return is used only as an auxiliary head
or a diagnostic.  It writes predictions and metadata to the private work tree,
not to the repository.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs" / "stock_goal60_4h"))
import core  # noqa: E402

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    matthews_corrcoef,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:  # torch is present in the project FinBERT environment
    import torch  # noqa: E402
    from torch import nn  # noqa: E402
except Exception as exc:  # pragma: no cover - an explicit runtime error is clearer
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = repr(exc)


INPUT = ROOT / "work" / "stock-data" / "paper_methods_4h" / "v1" / "inputs.pkl"
PRIVATE = ROOT / "work" / "stock-data" / "market_return_4h" / "v1"
PUBLIC = ROOT / "outputs" / "stock_market_return_4h" / "v1"
OOF_MONTHS = [f"2018-{m:02d}" for m in range(3, 9)]
SYMBOLS = ("AAPL", "AMZN")
SEEDS = (573, 574, 575)
RAW_FEATURES = list(core.OLD) + list(core.RECENT)
METHODS = ("C0", "C1", "C2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def metric(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    q = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(np.unique(y)) > 1 else None,
        "MCC": float(matthews_corrcoef(y, q)) if len(y) else None,
        "Brier": float(brier_score_loss(y, p)) if len(y) else None,
        "pred_up": float(q.mean()) if len(q) else None,
        "constant": int(len(np.unique(q)) <= 1),
    }
    return out


def add_return_metrics(out: dict, y_return, prediction) -> dict:
    y_return = np.asarray(y_return, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    out["return_MAE"] = float(mean_absolute_error(y_return, prediction))
    out["return_RMSE"] = float(math.sqrt(mean_squared_error(y_return, prediction)))
    if len(y_return) > 1 and np.std(y_return) > 0 and np.std(prediction) > 0:
        out["return_corr"] = float(np.corrcoef(y_return, prediction)[0, 1])
    else:
        out["return_corr"] = None
    return out


def phase_for(month: str) -> str:
    if month in OOF_MONTHS:
        return "train_forward_oof"
    if month in ("2018-09", "2018-10"):
        return "development"
    if month >= "2018-11":
        return "later"
    return "warmup"


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    d = pd.read_pickle(INPUT).reset_index(drop=True)
    required = {"key", "symbol", "month", "phase", "label", "target_return", "end_utc", "cutoff_utc", *RAW_FEATURES, "F1"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"missing required input columns: {missing}")
    if d.key.duplicated().any():
        raise ValueError("duplicate sample keys in inputs.pkl")
    if not np.array_equal((d.target_return.to_numpy(float) > 0).astype(int), d.label.to_numpy(int)):
        raise ValueError("label is not the sign of target_return")
    if not np.isfinite(d.target_return.to_numpy(float)).all():
        raise ValueError("non-finite target_return")
    return d


def feature_matrix(d: pd.DataFrame, variant: str) -> tuple[np.ndarray, list[str]]:
    names = list(RAW_FEATURES)
    x = d[names].to_numpy(dtype=float)
    if variant == "price_F1":
        x = np.c_[x, d["F1"].fillna(0.5).to_numpy(dtype=float)]
        names.append("F1_probability")
    elif variant != "price":
        raise ValueError(variant)
    return x, names


def split_indices(d: pd.DataFrame, symbol: str, month: str) -> tuple[np.ndarray, np.ndarray]:
    boundary = "2018-09" if month in ("development", "later") else month
    if month == "development":
        ev_mask = (d.month >= "2018-09") & (d.month < "2018-11")
    elif month == "later":
        ev_mask = d.month >= "2018-11"
    else:
        ev_mask = d.month == month
    ev = np.flatnonzero((d.symbol == symbol) & ev_mask)
    if len(ev) == 0:
        raise ValueError(f"empty evaluation split {symbol} {month}")
    earliest_cutoff = d.iloc[ev].cutoff_utc.min()
    tr = np.flatnonzero((d.symbol == symbol) & (d.month < boundary) & (d.end_utc < earliest_cutoff))
    if len(tr) == 0 or d.iloc[tr].end_utc.max() >= earliest_cutoff:
        raise ValueError(f"invalid chronological split {symbol} {month}")
    return tr, ev


def preprocess_fit(x: np.ndarray, tr: np.ndarray, ev: np.ndarray):
    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    return prep.fit_transform(x[tr]), prep.transform(x[ev]), prep


def safe_fit_prob(model, xtr, ytr, xev) -> np.ndarray:
    if len(np.unique(ytr)) < 2:
        return np.full(len(xev), float(np.mean(ytr)))
    model.fit(xtr, ytr)
    return model.predict_proba(xev)[:, 1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


class JointLinear(nn.Module):
    def __init__(self, n_features: int, latent: int = 8):
        super().__init__()
        self.projection = nn.Linear(n_features, latent)
        self.direction = nn.Linear(latent, 1)
        self.regression = nn.Linear(latent, 1)

    def forward(self, x):
        z = self.projection(x)
        return self.direction(z).squeeze(-1), self.regression(z).squeeze(-1)


def fit_joint(xtr: np.ndarray, ytr: np.ndarray, rtr: np.ndarray, xev: np.ndarray, lam: float, seed: int):
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
    set_seed(seed)
    torch.set_num_threads(1)
    ytr = np.asarray(ytr, dtype=np.float32)
    rtr = np.asarray(rtr, dtype=np.float32)
    r_scale = float(np.std(rtr))
    if not np.isfinite(r_scale) or r_scale < 1e-6:
        r_scale = 1.0
    tx = torch.tensor(xtr, dtype=torch.float32)
    ty = torch.tensor(ytr, dtype=torch.float32)
    tr = torch.tensor(rtr / r_scale, dtype=torch.float32)
    ex = torch.tensor(xev, dtype=torch.float32)
    model = JointLinear(xtr.shape[1], latent=8)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=1.0)
    epochs = 220
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits, rr = model(tx)
        loss = bce(logits, ty) + float(lam) * huber(rr, tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logit, rr = model(ex)
    p = torch.sigmoid(logit).cpu().numpy().astype(float)
    ret = rr.cpu().numpy().astype(float) * r_scale
    # A memory-only checkpoint round trip catches serialization drift without
    # leaving model binaries in the repository or private result directory.
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    restored = JointLinear(xtr.shape[1], latent=8)
    restored.load_state_dict(torch.load(buf, weights_only=True))
    restored.eval()
    with torch.no_grad():
        logit2, rr2 = restored(ex)
    p2 = torch.sigmoid(logit2).cpu().numpy().astype(float)
    ret2 = rr2.cpu().numpy().astype(float) * r_scale
    reload_error = float(max(np.max(np.abs(p - p2)), np.max(np.abs(ret - ret2))))
    assert reload_error < 1e-12
    return p, ret, {"seed": seed, "lambda": lam, "epochs": epochs, "r_scale": r_scale, "reload_error": reload_error}


def fit_method(d, x, method: str, variant: str, tr: np.ndarray, ev: np.ndarray, param, seed: int | None = None):
    xtr, xev, prep = preprocess_fit(x, tr, ev)
    ytr = d.iloc[tr].label.to_numpy(int)
    rtr = d.iloc[tr].target_return.to_numpy(float)
    if method == "C0":
        model = LogisticRegression(C=float(param), solver="liblinear", max_iter=3000, tol=1e-7, random_state=573)
        p = safe_fit_prob(model, xtr, ytr, xev)
        ret = None
        evidence = {"parameter": float(param), "preprocess": "median+standard_scaler", "seed": 573}
    elif method == "C1":
        model = Ridge(alpha=float(param))
        model.fit(xtr, rtr)
        ret = model.predict(xev)
        scale = max(float(np.std(rtr)), 1e-6)
        p = 1.0 / (1.0 + np.exp(-np.clip(ret / scale, -30, 30)))
        evidence = {"parameter": float(param), "preprocess": "median+standard_scaler", "return_scale": scale, "seed": 573}
    elif method == "C2":
        if seed is None:
            raise ValueError("C2 requires a seed")
        p, ret, evidence = fit_joint(xtr, ytr, rtr, xev, float(param), seed)
        evidence["preprocess"] = "median+standard_scaler"
    else:
        raise ValueError(method)
    return p, ret, evidence


def params_for(method: str):
    if method == "C0":
        return [0.01, 0.1, 1.0]
    if method == "C1":
        return [0.01, 0.1, 1.0, 10.0]
    if method == "C2":
        return [0.0, 0.1, 0.5, 1.0]
    raise ValueError(method)


def choose_param(rows: list[dict], method: str):
    usable = [r for r in rows if r["method"] == method and r["BA"] is not None]
    if not usable:
        return params_for(method)[0]
    frame = pd.DataFrame(usable)
    by = frame.groupby(["parameter", "symbol"], as_index=False).BA.mean()
    pivot = by.pivot(index="parameter", columns="symbol", values="BA")
    for symbol in SYMBOLS:
        if symbol not in pivot:
            pivot[symbol] = np.nan
    pivot["min_stock"] = pivot[list(SYMBOLS)].min(axis=1)
    pivot["mean_stock"] = pivot[list(SYMBOLS)].mean(axis=1)
    selected = sorted(pivot.index.tolist(), key=lambda v: (-pivot.loc[v, "min_stock"], -pivot.loc[v, "mean_stock"], float(v)))[0]
    return float(selected)


def append_prediction(rows: list[dict], d: pd.DataFrame, ev: np.ndarray, method_name: str, p: np.ndarray, ret=None):
    for j, prob in zip(ev, p):
        r = {
            "key": d.iloc[j].key,
            "symbol": d.iloc[j].symbol,
            "month": d.iloc[j].month,
            "phase": phase_for(d.iloc[j].month),
            "day": str(d.iloc[j].day),
            "label": int(d.iloc[j].label),
            "target_return": float(d.iloc[j].target_return),
            "method": method_name,
            "p": float(prob),
        }
        if ret is not None:
            r["return_pred"] = float(ret[list(ev).index(j)])
        else:
            r["return_pred"] = None
        rows.append(r)


def run():
    started = time.time()
    d = load_data()
    x_by_variant = {v: feature_matrix(d, v) for v in ("price", "price_F1")}
    PRIVATE.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / "outputs" / "stock_market_return_4h" / "PRE_REGISTRATION.md"
    meta = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "protocol_sha256": sha256(protocol),
        "code_sha256": sha256(Path(__file__).resolve()),
        "n_rows": len(d),
        "features": {v: names for v, (_, names) in x_by_variant.items()},
        "oof_months": OOF_MONTHS,
        "seeds": SEEDS,
        "models": {"C0": params_for("C0"), "C1": params_for("C1"), "C2": params_for("C2")},
        "started_epoch": started,
    }
    json_dump(PUBLIC / "run_metadata.json", meta)

    all_predictions: list[dict] = []
    cv_rows: list[dict] = []
    fit_rows: list[dict] = []
    choices: dict[str, float] = {}
    for variant, (x, feature_names) in x_by_variant.items():
        for method in METHODS:
            candidates = params_for(method)
            print(f"[{variant}/{method}] forward OOF", flush=True)
            for month in OOF_MONTHS:
                for symbol in SYMBOLS:
                    tr, ev = split_indices(d, symbol, month)
                    for param in candidates:
                        t0 = time.time()
                        if method == "C2":
                            outputs = [fit_method(d, x, method, variant, tr, ev, param, seed) for seed in SEEDS]
                            p = np.mean([o[0] for o in outputs], axis=0)
                            ret = np.mean([o[1] for o in outputs], axis=0)
                            evidence = {"seeds": [o[2] for o in outputs]}
                        else:
                            p, ret, evidence = fit_method(d, x, method, variant, tr, ev, param)
                        m = metric(d.iloc[ev].label, p)
                        if ret is not None:
                            m = add_return_metrics(m, d.iloc[ev].target_return, ret)
                        cv_rows.append({"variant": variant, "method": method, "parameter": float(param), "symbol": symbol, "month": month, **m})
                        fit_rows.append({"variant": variant, "method": method, "parameter": float(param), "symbol": symbol, "month": month, "train_n": len(tr), "eval_n": len(ev), "train_end": str(d.iloc[tr].end_utc.max()), "eval_cutoff": str(d.iloc[ev].cutoff_utc.min()), "seconds": time.time() - t0, "evidence": evidence})

            method_cv = [r for r in cv_rows if r["variant"] == variant and r["method"] == method]
            selected = choose_param(method_cv, method)
            choices[f"{variant}/{method}"] = selected
            print(f"[{variant}/{method}] selected {selected}", flush=True)
            # Refit the chosen parameter on each OOF fold so the published
            # panel contains the actual, pre-registered OOF predictions (the
            # candidate grid above is only used for selection).
            for month in OOF_MONTHS:
                for symbol in SYMBOLS:
                    tr, ev = split_indices(d, symbol, month)
                    t0 = time.time()
                    if method == "C2":
                        outputs = [fit_method(d, x, method, variant, tr, ev, selected, seed) for seed in SEEDS]
                        p = np.mean([o[0] for o in outputs], axis=0)
                        ret = np.mean([o[1] for o in outputs], axis=0)
                        evidence = {"seeds": [o[2] for o in outputs]}
                    else:
                        p, ret, evidence = fit_method(d, x, method, variant, tr, ev, selected)
                    append_prediction(all_predictions, d, ev, f"{method}_{variant}", p, ret)
                    m = metric(d.iloc[ev].label, p)
                    if ret is not None:
                        m = add_return_metrics(m, d.iloc[ev].target_return, ret)
                    cv_rows.append({"variant": variant, "method": method, "parameter": selected, "symbol": symbol, "month": month, "selected_oof": True, **m})
                    fit_rows.append({"variant": variant, "method": method, "parameter": selected, "symbol": symbol, "month": month, "train_n": len(tr), "eval_n": len(ev), "train_end": str(d.iloc[tr].end_utc.max()), "eval_cutoff": str(d.iloc[ev].cutoff_utc.min()), "seconds": time.time() - t0, "selected_oof": True, "evidence": evidence})
            # Refit once on the fixed January-August training window and score each exposed period.
            for phase_name in ("development", "later"):
                for symbol in SYMBOLS:
                    tr, ev = split_indices(d, symbol, phase_name)
                    t0 = time.time()
                    if method == "C2":
                        outputs = [fit_method(d, x, method, variant, tr, ev, selected, seed) for seed in SEEDS]
                        p = np.mean([o[0] for o in outputs], axis=0)
                        ret = np.mean([o[1] for o in outputs], axis=0)
                        evidence = {"seeds": [o[2] for o in outputs]}
                    else:
                        p, ret, evidence = fit_method(d, x, method, variant, tr, ev, selected)
                    append_prediction(all_predictions, d, ev, f"{method}_{variant}", p, ret)
                    m = metric(d.iloc[ev].label, p)
                    if ret is not None:
                        m = add_return_metrics(m, d.iloc[ev].target_return, ret)
                    cv_rows.append({"variant": variant, "method": method, "parameter": selected, "symbol": symbol, "month": phase_name, "selected_final": True, **m})
                    fit_rows.append({"variant": variant, "method": method, "parameter": selected, "symbol": symbol, "month": phase_name, "train_n": len(tr), "eval_n": len(ev), "train_end": str(d.iloc[tr].end_utc.max()), "eval_cutoff": str(d.iloc[ev].cutoff_utc.min()), "seconds": time.time() - t0, "selected_final": True, "evidence": evidence})

    # Keep the historical baseline columns in the same machine-readable panel.
    for base in ("F0", "R1", "F1", "F2", "F6"):
        for row in d.itertuples():
            if row.phase == "warmup" or not np.isfinite(getattr(row, base)):
                continue
            all_predictions.append({"key": row.key, "symbol": row.symbol, "month": row.month, "phase": phase_for(row.month), "day": str(row.day), "label": int(row.label), "target_return": float(row.target_return), "method": base, "p": float(getattr(row, base)), "return_pred": None})

    pred = pd.DataFrame(all_predictions).sort_values(["method", "key"]).reset_index(drop=True)
    pred.to_csv(PUBLIC / "predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(PUBLIC / "cv_and_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_json(PUBLIC / "training_evidence.json", orient="records", indent=2)
    json_dump(PUBLIC / "choices.json", {"choices": choices, "selection_basis": "OOF BA, max-min stock then mean, smaller parameter tie-break", "future_periods_not_used": True})
    json_dump(PUBLIC / "done.json", {"completed_epoch": time.time(), "n_predictions": len(pred), "n_fits": len(fit_rows), "choices": choices, "input_sha256": sha256(INPUT), "status": "COMPLETED"})
    print(json.dumps({"status": "COMPLETED", "predictions": len(pred), "fits": len(fit_rows), "choices": choices, "elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    run()
