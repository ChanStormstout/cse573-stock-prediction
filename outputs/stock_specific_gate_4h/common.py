"""Shared fixed protocol helpers for the preregistered stock gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TAU = 0.05
STATE_COLUMNS = [
    "recent_return_15", "recent_return_60", "recent_rv_15", "recent_rv_60",
    "minutes_from_open", "overnight_gap", "overnight_gap_missing",
    "log_news_count", "abs_expert_gap", "abs_price_centered",
]
INTERACTION_COLUMNS = [f"stock_centered:{c}" for c in STATE_COLUMNS]
CONTROLLER_NAMES = ("G0", "G1", "G2", "G3")
RIDGE_JITTER = 1e-8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def weighted_mean(y, weights):
    y = np.asarray(y, float); weights = np.asarray(weights, float)
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(weights).all() or weights.sum() <= 0:
        return 0.0
    return float(np.sum(y * weights) / np.sum(weights))


def weighted_ridge(X, y, weights, penalty):
    X = np.asarray(X, float); y = np.asarray(y, float); weights = np.asarray(weights, float); penalty = np.asarray(penalty, float)
    if X.ndim != 2 or X.shape[0] != len(y) or len(penalty) != X.shape[1]:
        raise ValueError("weighted ridge dimensions do not match")
    sw = np.sqrt(np.clip(weights, 0.0, None)); Xw = X * sw[:, None]; yw = y * sw
    A = Xw.T @ Xw + np.diag(penalty) + np.eye(X.shape[1]) * RIDGE_JITTER
    b = Xw.T @ yw
    return np.linalg.solve(A, b)


def design_matrix(state, stock, variant, interaction_zero=False):
    state = np.asarray(state, float); stock = np.asarray(stock, float)
    if state.ndim != 2 or state.shape[1] != len(STATE_COLUMNS):
        raise ValueError("state vector does not match preregistration")
    z = np.c_[np.ones(len(state)), stock]
    if variant in ("G2", "G3"):
        z = np.c_[z, state]
    if variant == "G3":
        inter = (stock - 0.5)[:, None] * state
        if interaction_zero:
            inter = np.zeros_like(inter)
        z = np.c_[z, inter]
    return z


def penalty_for(variant):
    if variant == "G2":
        return np.array([0.0, 10.0] + [10.0] * len(STATE_COLUMNS))
    if variant == "G3":
        return np.array([0.0, 10.0] + [10.0] * len(STATE_COLUMNS) + [50.0] * len(STATE_COLUMNS))
    raise ValueError(variant)


def map_advantage(d_hat, p_price, p_text, has_news):
    d_hat = np.asarray(d_hat, float); p_price = np.asarray(p_price, float); p_text = np.asarray(p_text, float); has_news = np.asarray(has_news, int)
    exponent = np.clip(d_hat / TAU, -60.0, 60.0)
    w = 1.0 / (1.0 + np.exp(exponent)); w = np.where(has_news == 1, w, 0.0)
    p = (1.0 - w) * p_price + w * p_text
    p = np.where(has_news == 1, p, p_price)
    return p, w
