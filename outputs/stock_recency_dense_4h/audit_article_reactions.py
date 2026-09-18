"""Audit article reactions after availability; this script never trains a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import NEWS_INDEX, PUBLIC, RAW, WORK, dump, load_data, schedule, sha


def bars(symbol):
    path = RAW / ("APPLE5.csv" if symbol == "AAPL" else "AMAZON5.csv")
    b = pd.read_csv(path, header=None, names=["date", "time", "open", "high", "low", "close", "activity"])
    b.index = pd.to_datetime(b.date + " " + b.time, format="%Y.%m.%d %H:%M", utc=True)
    return path, b.sort_index()


def regular_index(b, sch):
    mask = np.zeros(len(b), dtype=bool)
    idx = b.index
    for r in sch.itertuples():
        mask |= (idx >= r.open) & (idx + pd.Timedelta("5min") <= r.close)
    return b.loc[mask].sort_index()


def audit(public: Path, private: Path):
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=False)
    d = load_data(); raw = pd.read_pickle(NEWS_INDEX).copy()
    raw["key"] = raw.archive.astype(str) + "::" + raw.member.astype(str)
    for c in ("available_utc", "published_utc", "crawled_utc"):
        raw[c] = pd.to_datetime(raw[c], utc=True, errors="coerce")
    associations = {}
    raw_key_set = set(raw.key)
    for row in d.itertuples():
        for key in str(row.news_record_keys or "").split("|"):
            if key and key in raw_key_set:
                associations.setdefault(key, set()).add(row.symbol)
    raw = raw[raw.key.isin(associations)].copy()
    raw["symbol"] = raw.key.map(lambda k: sorted(associations[k])[0] if len(associations[k]) == 1 else None)
    raw = raw[raw.symbol.notna()].copy()
    raw["group_key"] = raw["normalized_hash"].fillna(raw["exact_hash"]).fillna(raw["title_norm"]).fillna(raw.key).astype(str)
    raw["online_group"] = raw.symbol.astype(str) + "|" + raw.group_key
    raw["period"] = np.where(raw.available_utc < pd.Timestamp("2018-09-01", tz="UTC"), "jan_aug", "sep_onward")
    sch = schedule(); horizons = {30: 6, 60: 12, 120: 24, 240: 48}; all_rows = []
    for symbol in ("AAPL", "AMZN"):
        path, b = bars(symbol); rb = regular_index(b, sch); global_idx = rb.index; pos = {x: i for i, x in enumerate(global_idx)}
        sessions = {r.open.date(): r for r in sch.itertuples()}
        for art in raw[raw.symbol == symbol].itertuples():
            if pd.isna(art.available_utc):
                continue
            first_pos = int(global_idx.searchsorted(art.available_utc, side="left"))
            if first_pos >= len(global_idx):
                continue
            first = global_idx[first_pos]; first_session = sessions.get(first.date())
            for horizon, n in horizons.items():
                # Trading-time bars may cross a regular-session boundary, but all
                # bars must be present in the raw regular-session index.
                seq = global_idx[first_pos:first_pos + n]
                valid_trading = len(seq) == n
                valid_same = valid_trading and all(x.date() == first.date() for x in seq)
                for definition, valid in (("TRADING_TIME", valid_trading), ("SAME_SESSION", valid_same)):
                    ret = np.nan
                    if valid:
                        ret = float(np.log(rb.loc[seq[-1], "close"] / rb.loc[seq[0], "open"]))
                    all_rows.append({"article_key": art.key, "symbol": symbol, "online_group": art.online_group, "available_utc": art.available_utc, "published_utc": art.published_utc, "crawled_utc": art.crawled_utc, "site": art.site, "period": art.period, "horizon_min": horizon, "definition": definition, "valid_timestamp": 1, "complete_reaction": int(valid), "crosses_session": int(valid_trading and len({x.date() for x in seq}) > 1), "reaction_return": ret})
    rows = pd.DataFrame(all_rows)
    if rows.empty: raise AssertionError("no accepted article reactions")
    rows.to_csv(private / "article_reaction_rows.csv", index=False)
    agg = []
    for (symbol, definition, horizon, period), g in rows.groupby(["symbol", "definition", "horizon_min", "period"]):
        valid = g[g.complete_reaction == 1]
        unique_groups = g.online_group.nunique()
        agg.append({"symbol": symbol, "definition": definition, "horizon_min": int(horizon), "period": period, "raw_article_count": int(len(g)), "unique_group_count": int(unique_groups), "valid_timestamp_count": int(g.valid_timestamp.sum()), "complete_reaction_count": int(len(valid)), "crossing_session_count": int(g.loc[g.complete_reaction == 1, "crosses_session"].sum()), "positive_count": int((valid.reaction_return > 0).sum()), "negative_count": int((valid.reaction_return < 0).sum()), "zero_count": int((valid.reaction_return == 0).sum()), "sign_balance": float((valid.reaction_return > 0).mean()) if len(valid) else None, "median_abs_return": float(valid.reaction_return.abs().median()) if len(valid) else None, "duplicate_concentration": float(len(g) / unique_groups) if unique_groups else None, "unique_publication_dates": int(g.published_utc.dt.date.nunique()), "source_count": int(g.site.nunique()), "month_count": int(g.available_utc.dt.strftime('%Y-%m').nunique())})
    audit_df = pd.DataFrame(agg).sort_values(["symbol", "definition", "horizon_min", "period"])
    audit_df.to_csv(public / "article_reaction_audit.csv", index=False)
    train = audit_df[(audit_df.period == "jan_aug") & (audit_df.definition == "SAME_SESSION")]
    # Operational feasibility line: at least 100 unique accepted groups per stock
    # at one horizon. It is not a significance test and does not authorize training.
    feasible = bool(all(train[train.symbol == s].unique_group_count.max() >= 100 for s in ("AAPL", "AMZN")))
    summary = {"status": "AUDIT_ONLY", "association": "official_accepted_article_ids_only", "rows": int(len(rows)), "unique_articles": int(rows.article_key.nunique()), "unique_groups": int(rows.online_group.nunique()), "feasible_for_next_reaction_experiment": feasible, "feasibility_threshold": 100, "source_sha256": sha(NEWS_INDEX), "bar_sources": {"AAPL": sha(RAW / "APPLE5.csv"), "AMZN": sha(RAW / "AMAZON5.csv")}, "note": "Reaction returns are descriptive after-availability measurements, not causal estimates; no reaction model was trained."}
    dump(public / "article_reaction_summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--public", default="outputs/stock_recency_dense_4h/v3"); p.add_argument("--private", default="work/stock-data/recency_dense_4h/v3/reactions"); a = p.parse_args(); print(json.dumps(audit(Path(a.public), Path(a.private)), indent=2))
