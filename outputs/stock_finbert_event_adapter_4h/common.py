"""Shared helpers for the finite FinBERT event-adapter experiment."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "v1"
WORK = ROOT / "work/stock-data/finbert_event_adapter_4h/v1"
ANNOTATION = ROOT / "work/stock-data/annotation/student_v3"
ANNOTATION_SOURCES = ROOT / "work/stock-data/annotation/v3/sources.jsonl"
PARAGRAPHS = ROOT / "work/stock-data/nextgen_4h/paragraphs_v4"
NEXTGEN = ROOT / "outputs/stock_nextgen_4h/runs/v1"
FINBERT_CACHE = ROOT / "work/stock-data/finbert-cache"
MODEL_ID = "ProsusAI/finbert"
TYPES = ("rating", "target_price")
ACTIONS = ("raise", "lower", "maintain", "initiate", "unknown")
SEEDS = (573, 574, 575)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def clip_probability(values):
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def logit(values):
    values = clip_probability(values)
    return np.log(values / (1 - values))


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def classification_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = clip_probability(p)
    pred = p >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
        "MCC": float(matthews_corrcoef(y, pred)) if len(set(y)) == 2 else None,
        "Brier": float(brier_score_loss(y, p)),
        "pred_up": float(pred.mean()),
        "up_rate": float(y.mean()),
        "constant": bool(len(set(pred.tolist())) == 1),
    }


def paired_block_interval(frame: pd.DataFrame, challenger: str, baseline: str, block_days: int, seed: int = 573) -> dict:
    """Paired day/block bootstrap for BA and Brier differences."""
    ordered = frame.sort_values(["day", "key"]).reset_index(drop=True)
    daily = [np.asarray(indices, dtype=int) for indices in ordered.groupby("day", sort=True).indices.values()]
    if not daily:
        return {"block_days": block_days, "samples": 0, "BA_low": None, "BA_high": None, "Brier_low": None, "Brier_high": None}
    blocks = [np.concatenate(daily[i : i + block_days]) for i in range(0, len(daily), block_days)]
    labels = ordered.label.to_numpy(dtype=int)
    challenger_probability = clip_probability(ordered[challenger])
    baseline_probability = clip_probability(ordered[baseline])
    challenger_direction = challenger_probability >= 0.5
    baseline_direction = baseline_probability >= 0.5

    def balanced(labels_sample, direction_sample):
        positive = labels_sample == 1
        negative = ~positive
        if not positive.any() or not negative.any():
            return None
        return 0.5 * (direction_sample[positive].mean() + (~direction_sample[negative]).mean())

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(2000):
        indices = np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        labels_sample = labels[indices]
        challenger_ba = balanced(labels_sample, challenger_direction[indices])
        baseline_ba = balanced(labels_sample, baseline_direction[indices])
        if challenger_ba is None or baseline_ba is None:
            continue
        challenger_brier = np.mean((challenger_probability[indices] - labels_sample) ** 2)
        baseline_brier = np.mean((baseline_probability[indices] - labels_sample) ** 2)
        deltas.append((challenger_ba - baseline_ba, challenger_brier - baseline_brier))
    if not deltas:
        return {"block_days": block_days, "samples": 0, "BA_low": None, "BA_high": None, "Brier_low": None, "Brier_high": None}
    values = np.asarray(deltas)
    return {
        "block_days": block_days,
        "samples": int(len(values)),
        "BA_low": float(np.quantile(values[:, 0], 0.025)),
        "BA_high": float(np.quantile(values[:, 0], 0.975)),
        "Brier_low": float(np.quantile(values[:, 1], 0.025)),
        "Brier_high": float(np.quantile(values[:, 1], 0.975)),
    }


def percent(value) -> str:
    return "NA" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.2f}%"
