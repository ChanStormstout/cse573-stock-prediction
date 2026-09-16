"""Read-only diagnostics of existing development data/predictions. No model fits.

Run from any directory using the existing finbert-env Python. Final-test files
are not inputs. All correlations are descriptive, not significance/causal tests.
"""
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sources = {}


def read_csv(path):
    sources[str(path.relative_to(OUT))] = sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


frames = {}
summary = {
    "date": "2026-09-15",
    "status": "posthoc_development_diagnostics_no_new_training",
    "test_evaluated": False,
    "scope": "Original nonflat eligible-hour population; Jan-Aug train, Sep-Oct validation.",
    "stocks": {},
}
monthly, peer_rows, error_rows = [], [], []
for symbol in ("AAPL", "AMZN"):
    f = read_csv(OUT / "stock_finbert/results" / symbol / "development_features.csv")
    assert set(f.split) == {"train", "validation"}
    for col in ("start_utc", "end_utc", "cutoff_utc"):
        f[col] = pd.to_datetime(f[col], utc=True)
    assert (f.cutoff_utc < f.start_utc).all()
    assert (f.end_utc > f.start_utc).all()
    assert f.start_utc.max() < pd.Timestamp("2018-11-01", tz="UTC")
    assert not f.start_utc.duplicated().any()
    assert np.array_equal((f.target_return > 0).astype(int), f.label)
    f["day"] = f.start_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    f["month"] = f.day.str[:7]
    f["keys"] = f.news_record_keys.fillna("").map(lambda s: s.split("|") if s else [])
    assert (f["keys"].map(len) == f.news_count).all()
    frames[symbol] = f
    per_stock = {}
    for split, g in f.groupby("split"):
        labels_by_key = defaultdict(list)
        for row in g.itertuples():
            for key in row.keys:
                labels_by_key[key].append(int(row.label))
        repeated = [ys for ys in labels_by_key.values() if len(ys) > 1]
        mixed = [ys for ys in labels_by_key.values() if len(set(ys)) > 1]
        per_stock[split] = {
            "windows": len(g), "trading_days": g.day.nunique(),
            "up_fraction": float(g.label.mean()),
            "news_coverage": float(g.has_news.mean()),
            "unique_records": len(labels_by_key),
            "article_window_occurrences": sum(map(len, labels_by_key.values())),
            "records_reused_across_windows": len(repeated),
            "records_seen_with_both_window_labels": len(mixed),
            "mixed_fraction_of_reused_records": len(mixed) / len(repeated) if repeated else None,
            "median_abs_return_bps": float(g.target_return.abs().median() * 10000),
        }
    summary["stocks"][symbol] = per_stock
    for month, g in f.groupby("month"):
        monthly.append({
            "symbol": symbol, "month": month, "split": g.split.iloc[0],
            "windows": len(g), "trading_days": g.day.nunique(),
            "up_fraction": g.label.mean(), "news_coverage": g.has_news.mean(),
            "mean_news_count": g.news_count.mean(),
            "median_abs_return_bps": g.target_return.abs().median() * 10000,
            "return_sd_bps": g.target_return.std() * 10000,
            "fraction_abs_return_lt_10bps": (g.target_return.abs() < 0.001).mean(),
        })
    early = read_csv(OUT / "stock_improvement/results" / symbol / "validation_predictions.csv")
    temporal = read_csv(OUT / "stock_temporal/results" / symbol / "validation_predictions.csv")
    preds = early.merge(temporal, on=["start_utc", "label"], validate="one_to_one")
    assert len(preds) == len(early) == len(temporal) == 252
    assert np.allclose(preds.price_sentiment, preds.current_lr)
    selected = ["price", "price_sentiment", "paper_price_bow", "sequence_lr", "gru_573"]
    # Fixed seed 573 is a diagnostic representative, not selection of a best seed.
    for a, b in combinations(selected, 2):
        ea = ((preds[a] >= .5).astype(int) != preds.label).astype(int)
        eb = ((preds[b] >= .5).astype(int) != preds.label).astype(int)
        error_rows.append({
            "symbol": symbol, "model_a": a, "model_b": b, "windows": len(preds),
            "error_indicator_correlation": pearson(ea, eb),
            "both_wrong": int(((ea == 1) & (eb == 1)).sum()),
            "b_fixes_a": int(((ea == 1) & (eb == 0)).sum()),
            "b_breaks_a": int(((ea == 0) & (eb == 1)).sum()),
        })
    y = preds.label.to_numpy()
    train_up = float(f.loc[f.split == "train", "label"].mean())
    summary["stocks"][symbol]["probability_baselines"] = {
        "constant_0_5_validation_brier": float(np.mean((y - .5) ** 2)),
        "constant_train_up_probability": train_up,
        "constant_train_up_validation_brier": float(np.mean((y - train_up) ** 2)),
        "note": "BA of an all-up or all-down prediction is 0.5 when both classes exist.",
    }

both = frames["AAPL"].merge(frames["AMZN"], on="start_utc", suffixes=("_a", "_b"), validate="one_to_one")
assert (both.split_a == both.split_b).all()
summary["contemporaneous_returns"] = {}
for split, g in both.groupby("split_a"):
    summary["contemporaneous_returns"][split] = {
        "paired_hours": len(g),
        "pearson": pearson(g.target_return_a, g.target_return_b),
        "warning": "Same future hour of peer stock; NOT available for predicting this hour.",
    }

# Past peer returns from existing eligible hours only, known by current cutoff.
# This is a diagnostic, not an engineered feature claiming complete price coverage.
for symbol, peer in (("AAPL", "AMZN"), ("AMZN", "AAPL")):
    left = frames[symbol][["cutoff_utc", "target_return", "split"]].sort_values("cutoff_utc")
    right = frames[peer][["end_utc", "target_return"]].rename(columns={"target_return": "past_peer_return"}).sort_values("end_utc")
    joined = pd.merge_asof(left, right, left_on="cutoff_utc", right_on="end_utc", direction="backward")
    joined = joined.dropna(subset=["end_utc"])
    assert (joined.end_utc <= joined.cutoff_utc).all()
    joined["age_hours"] = (joined.cutoff_utc - joined.end_utc).dt.total_seconds() / 3600
    for split, g in joined.groupby("split"):
        peer_rows.append({
            "target": symbol, "peer": peer, "split": split, "windows": len(g),
            "past_peer_return_vs_target_return_pearson": pearson(g.past_peer_return, g.target_return),
            "median_peer_age_hours": g.age_hours.median(),
            "note": "Latest completed eligible hour, includes overnight/weekend gaps; not a causal or conditional predictability test.",
        })

summary["available_past_peer_correlations"] = peer_rows
summary["source_sha256"] = sources
summary["script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
pd.DataFrame(monthly).to_csv(HERE / "monthly_data.csv", index=False)
pd.DataFrame(error_rows).to_csv(HERE / "prediction_overlap.csv", index=False)
with (HERE / "audit.json").open("w") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in summary.items() if "sha256" not in k}, ensure_ascii=False, indent=2))
print(pd.DataFrame(monthly).query("split == 'validation'").to_string(index=False))
print("Checks passed: development-only rows, labels/counts/joins, saved baseline consistency and past-peer cutoffs.")
