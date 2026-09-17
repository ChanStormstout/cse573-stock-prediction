"""Shared contracts for the four-hour next-generation experiments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
W = ROOT / "work" / "stock-data"
INTEGRATED = ROOT / "outputs" / "stock_integrated_4h"
DIRECT = ROOT / "outputs" / "stock_llm_direct_4h"
CS = (0.01, 0.1, 1.0)
PRICE0 = [
    *(f"{kind}_{i}" for i in range(1, 7) for kind in ("return", "range")),
    "history_age_hours", "return_mean", "return_std", "ny_hour",
]


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def dump(path: Path | str, value) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n"
    )


def jsonl(path: Path | str):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ece(y, p, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    ids = np.maximum(ids, 0)
    result = 0.0
    for i in range(bins):
        mask = ids == i
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def metrics(y, p) -> dict:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid prediction vector")
    q = p >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, q)) if len(set(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, q)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE10": ece(y, p),
        "pred_up": float(q.mean()),
        "up_rate": float(y.mean()),
        "constant": bool(np.all(q == q[0])) if len(q) else True,
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def source_hashes(paths):
    return {str(Path(path)): sha(path) for path in paths}


def assert_four_hour_rows(frame) -> None:
    import pandas as pd

    if len(frame) != 1607:
        raise AssertionError(f"expected 1607 rows, got {len(frame)}")
    for column in ("start_utc", "end_utc", "cutoff_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if not frame["horizon"].eq("4h").all():
        raise AssertionError("mixed horizon")
    if not ((frame.end_utc - frame.start_utc) == pd.Timedelta("4h")).all():
        raise AssertionError("wrong target length")
    if not ((frame.start_utc - frame.cutoff_utc) == pd.Timedelta("5min")).all():
        raise AssertionError("wrong cutoff boundary")


def stable_choice(rows, score_key, brier_key, c_key="C"):
    """Mean monthly BA, then Brier, then smaller C."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row[c_key], []).append(row)
    candidates = []
    for value, values in grouped.items():
        candidates.append(
            (
                -float(np.mean([row[score_key] for row in values])),
                float(np.mean([row[brier_key] for row in values])),
                float(value),
                value,
            )
        )
    return sorted(candidates)[0][-1]
