#!/usr/bin/env python3
"""Build an outcome-blind FNSPID Amazon candidate-coverage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

AMZN_CIK = "1018724"
HIGH = ("DIRECT_TARGET_HIGH_CONFIDENCE", "MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE")
ALL_REL = HIGH + ("INDIRECT_OR_COMPETITOR",)
SEED = "FNSPID_AMZN_4H_V1|2026-09-20"
FORBIDDEN = {"label", "target_return", "return", "ba", "mcc", "brier", "probability", "prediction"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def stable_hash(value: str) -> str:
    return hashlib.sha256((SEED + "|" + value).encode()).hexdigest()


def load_windows(source_repo: Path) -> pd.DataFrame:
    source = source_repo / "work/stock-data/paper_methods_4h/v1/inputs.pkl"
    frame = pd.read_pickle(source)
    # Project immediately onto outcome-free columns. No outcome value is read,
    # selected, summarized or written by this lane.
    cols = ["key", "symbol", "day", "start_utc", "end_utc", "cutoff_utc", "split", "has_original_news"]
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise RuntimeError(f"canonical input missing columns: {missing}")
    out = frame.loc[frame["symbol"].eq("AMZN"), cols].copy()
    if len(out) != 804 or out["key"].nunique() != 804:
        raise RuntimeError(f"expected 804 unique AMZN windows, got {len(out)} / {out['key'].nunique()}")
    for c in ("start_utc", "end_utc", "cutoff_utc"):
        out[c] = pd.to_datetime(out[c], utc=True)
    out["day"] = pd.to_datetime(out["day"]).dt.date.astype(str)
    out["has_original_news"] = out["has_original_news"].astype(int)
    return out.sort_values("start_utc").reset_index(drop=True)


def build_schedule(source_repo: Path, windows: pd.DataFrame) -> pd.DataFrame:
    import exchange_calendars as xcals

    start = (pd.Timestamp(windows["day"].min()) - pd.Timedelta(days=45)).date().isoformat()
    end = (pd.Timestamp(windows["day"].max()) + pd.Timedelta(days=7)).date().isoformat()
    schedule = xcals.get_calendar("XNYS").schedule.loc[start:end].reset_index()
    date_col = schedule.columns[0]
    schedule = schedule.rename(columns={date_col: "session_date"})
    schedule["session_date"] = pd.to_datetime(schedule["session_date"]).dt.date.astype(str)
    schedule["open_utc"] = pd.to_datetime(schedule["open"], utc=True)
    schedule["close_utc"] = pd.to_datetime(schedule["close"], utc=True)
    schedule = schedule[["session_date", "open_utc", "close_utc"]].copy()
    schedule["session_index"] = np.arange(len(schedule), dtype=int)

    accepted_path = source_repo / "work/stock-data/audit/xnys_schedule.csv"
    accepted = pd.read_csv(accepted_path)
    accepted = accepted.rename(columns={accepted.columns[0]: "session_date"})
    accepted["session_date"] = pd.to_datetime(accepted["session_date"]).dt.date.astype(str)
    accepted["open_utc"] = pd.to_datetime(accepted["open"], utc=True)
    accepted["close_utc"] = pd.to_datetime(accepted["close"], utc=True)
    overlap = schedule.merge(accepted[["session_date", "open_utc", "close_utc"]], on="session_date", suffixes=("_new", "_accepted"))
    if len(overlap) != len(accepted):
        raise RuntimeError("generated XNYS schedule does not cover every accepted schedule row")
    for c in ("open_utc", "close_utc"):
        if not overlap[f"{c}_new"].equals(overlap[f"{c}_accepted"]):
            raise RuntimeError(f"generated XNYS {c} differs from accepted schedule")
    return schedule


def load_amzn_edges(source_repo: Path) -> pd.DataFrame:
    edge = source_repo / "outputs/stock_ecni_e1r/FNSPID_RELINKED_ENTITY_EDGES.parquet"
    meta = source_repo / "work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet"
    con = duckdb.connect()
    query = f"""
    SELECT e.record_id_hash,e.native_ticker,e.resolved_ticker,e.relation_type,
           e.matched_alias,e.matched_field,e.confidence_rule,e.available_time_class,
           e.recorded_time,e.source_file,e.publisher_hash,e.body_available,e.title_hash,e.url_hash,
           m.row_number,m.normalized_title_hash,m.near_title_fingerprint,m.body_length,m.body_hash
    FROM read_parquet('{edge.as_posix()}') e
    JOIN read_parquet('{meta.as_posix()}') m USING(record_id_hash)
    WHERE cast(e.CIK as varchar)='{AMZN_CIK}'
      AND e.relation_type IN {ALL_REL}
      AND cast(e.recorded_time as timestamp) >= timestamp '2017-12-01'
      AND cast(e.recorded_time as timestamp) < timestamp '2019-02-02'
    """
    out = con.execute(query).fetchdf()
    con.close()
    if out.empty:
        raise RuntimeError("no Amazon edges found")
    if set(out["available_time_class"].dropna().unique()) != {"DATE_ONLY_CONSERVATIVE"}:
        raise RuntimeError("v1 expects every relevant Amazon FNSPID row to be date-only")
    out["recorded_time"] = pd.to_datetime(out["recorded_time"], utc=True)
    out["recorded_date"] = out["recorded_time"].dt.date.astype(str)
    for c in ("url_hash", "normalized_title_hash", "near_title_fingerprint", "record_id_hash"):
        out[c] = out[c].fillna("").astype(str)

    def group_key(row) -> str:
        for prefix, col in (("U", "url_hash"), ("T", "normalized_title_hash"), ("N", "near_title_fingerprint")):
            if row[col]:
                return f"{prefix}:{row[col]}"
        return f"R:{row['record_id_hash']}"

    out["article_group_key"] = out.apply(group_key, axis=1)
    out["article_group_hash"] = out["article_group_key"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return out


def attach_availability(edges: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    session_dates = np.array(schedule["session_date"], dtype=str)
    recorded = edges["recorded_date"].to_numpy(dtype=str)
    positions = np.searchsorted(session_dates, recorded, side="right")
    valid = positions < len(schedule)
    edges = edges.loc[valid].copy()
    positions = positions[valid]
    edges["available_session_date"] = session_dates[positions]
    edges["available_session_index"] = schedule.iloc[positions]["session_index"].to_numpy()
    edges["available_at_utc"] = schedule.iloc[positions]["open_utc"].to_numpy()
    if not (pd.to_datetime(edges["available_at_utc"], utc=True) > pd.to_datetime(edges["recorded_time"], utc=True)).all():
        raise RuntimeError("date-only availability did not move strictly past recorded date")
    return edges


def collapse_groups(edges: pd.DataFrame) -> pd.DataFrame:
    relation_rank = {
        "INDIRECT_OR_COMPETITOR": 0,
        "DIRECT_TARGET_HIGH_CONFIDENCE": 1,
        "MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE": 2,
    }
    rows = []
    for key, g in edges.groupby("article_group_hash", sort=True):
        rel = max(g["relation_type"], key=lambda x: relation_rank[x])
        first = g.sort_values(["available_at_utc", "record_id_hash"]).iloc[0]
        rows.append({
            "article_group_hash": key,
            "relation_stratum": rel,
            "recorded_date_first": g["recorded_date"].min(),
            "recorded_date_last": g["recorded_date"].max(),
            "available_session_date": first["available_session_date"],
            "available_session_index": int(first["available_session_index"]),
            "available_at_utc": first["available_at_utc"],
            "row_count": int(len(g)),
            "record_count": int(g["record_id_hash"].nunique()),
            "source_count": int(g["source_file"].nunique()),
            "publisher_count": int(g.loc[g["publisher_hash"].ne(""), "publisher_hash"].nunique()),
            "body_available": int(g["body_available"].fillna(False).any()),
            "body_length_max": int(g["body_length"].fillna(0).max()),
            "matched_in_title": int(g["matched_field"].eq("title").any()),
            "matched_in_body": int(g["matched_field"].eq("body").any()),
            "native_ticker_agrees": int(g["native_ticker"].eq("AMZN").any()),
            "source_files": "|".join(sorted(set(g["source_file"].astype(str)))),
            "record_id_hashes": "|".join(sorted(set(g["record_id_hash"].astype(str)))),
        })
    out = pd.DataFrame(rows)
    out["available_at_utc"] = pd.to_datetime(out["available_at_utc"], utc=True)
    return out.sort_values(["available_at_utc", "article_group_hash"]).reset_index(drop=True)


def map_coverage(windows: pd.DataFrame, groups: pd.DataFrame, schedule: pd.DataFrame):
    session_idx = dict(zip(schedule["session_date"], schedule["session_index"]))
    rows, links = [], []
    for w in windows.itertuples(index=False):
        cutoff = pd.Timestamp(w.cutoff_utc)
        current_idx = int(session_idx[w.day])
        available = groups.loc[groups["available_at_utc"].le(cutoff)]
        h24 = available.loc[available["available_at_utc"].gt(cutoff - pd.Timedelta(hours=24))]
        s3 = available.loc[available["available_session_index"].between(current_idx - 2, current_idx)]
        row = {
            "key": w.key,
            "day": w.day,
            "start_utc": pd.Timestamp(w.start_utc).isoformat(),
            "cutoff_utc": cutoff.isoformat(),
            "split": w.split,
            "has_original_news": int(w.has_original_news),
        }
        for view, subset in (("24h", h24), ("3s", s3)):
            direct = subset["relation_stratum"].eq("DIRECT_TARGET_HIGH_CONFIDENCE")
            multi = subset["relation_stratum"].eq("MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE")
            indirect = subset["relation_stratum"].eq("INDIRECT_OR_COMPETITOR")
            row[f"fnspid_direct_groups_{view}"] = int(direct.sum())
            row[f"fnspid_multi_groups_{view}"] = int(multi.sum())
            row[f"fnspid_indirect_groups_{view}"] = int(indirect.sum())
            row[f"fnspid_direct_or_multi_groups_{view}"] = int((direct | multi).sum())
            row[f"fnspid_body_groups_{view}"] = int(subset.loc[direct | multi, "body_available"].sum())
            for g in subset.loc[direct | multi].itertuples(index=False):
                links.append({"key": w.key, "view": view, "article_group_hash": g.article_group_hash,
                              "relation_stratum": g.relation_stratum, "available_at_utc": g.available_at_utc})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(links)


def candidate_census(edges: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    e["year_month"] = e["recorded_time"].dt.strftime("%Y-%m")
    agg = e.groupby(["year_month", "source_file", "relation_type"], dropna=False).agg(
        edge_rows=("record_id_hash", "size"),
        records=("record_id_hash", "nunique"),
        article_groups=("article_group_hash", "nunique"),
        body_rows=("body_available", "sum"),
        publishers=("publisher_hash", "nunique"),
    ).reset_index()
    return agg.sort_values(["year_month", "source_file", "relation_type"])


def sample_review(groups: pd.DataFrame) -> pd.DataFrame:
    eligible = groups[groups["relation_stratum"].isin(HIGH)].copy()
    eligible["year"] = eligible["recorded_date_first"].str[:4]
    eligible["sample_hash"] = eligible["article_group_hash"].map(stable_hash)
    eligible["body_class"] = np.where(eligible["body_available"].eq(1), "BODY", "NO_BODY")
    eligible["field_class"] = np.where(eligible["matched_in_title"].eq(1), "TITLE", "BODY_ONLY")
    strata = ["relation_stratum", "year", "source_files", "body_class", "field_class"]
    eligible = eligible.sort_values("sample_hash")
    selected = []
    # Round-robin over observed strata, then fill by deterministic hash.
    pools = [g.copy() for _, g in eligible.groupby(strata, sort=True)]
    while len(selected) < 120 and pools:
        remaining = []
        for pool in pools:
            if len(selected) >= 120:
                break
            if not pool.empty:
                selected.append(pool.iloc[0])
                pool = pool.iloc[1:]
            if not pool.empty:
                remaining.append(pool)
        pools = remaining
    used = {x["article_group_hash"] for x in selected}
    if len(selected) < 120:
        for _, row in eligible.loc[~eligible["article_group_hash"].isin(used)].sort_values("sample_hash").iterrows():
            selected.append(row)
            if len(selected) == 120:
                break
    out = pd.DataFrame(selected).copy()
    keep = ["article_group_hash", "sample_hash", "relation_stratum", "year", "source_files",
            "body_class", "field_class", "row_count", "record_count", "native_ticker_agrees"]
    out = out[keep].sort_values("sample_hash").reset_index(drop=True)
    out.insert(0, "review_id", [f"AMZN-FNSPID-{i:03d}" for i in range(1, len(out) + 1)])
    out["semantic_review_decision"] = ""
    out["independent_reviewer_id"] = ""
    return out


def coverage_summary(coverage: pd.DataFrame, edges: pd.DataFrame, groups: pd.DataFrame) -> dict:
    old_gap = coverage["has_original_news"].eq(0)
    result = {
        "status": "OUTCOME_BLIND_CANDIDATE_COVERAGE_COMPLETE",
        "canonical_amzn_windows": int(len(coverage)),
        "original_news_windows": int(coverage["has_original_news"].sum()),
        "original_no_news_windows": int(old_gap.sum()),
        "source_edge_rows": int(len(edges)),
        "source_unique_records": int(edges["record_id_hash"].nunique()),
        "deduplicated_article_groups": int(len(groups)),
        "native_ticker_agreeing_edge_rows": int(edges["native_ticker"].eq("AMZN").sum()),
        "available_time_classes": sorted(edges["available_time_class"].dropna().unique().tolist()),
        "human_validated": False,
        "predictive_models_fitted": 0,
        "outcomes_used": 0,
    }
    for view in ("24h", "3s"):
        for scope in ("direct", "direct_or_multi"):
            present = coverage[f"fnspid_{scope}_groups_{view}"].gt(0)
            result[f"windows_with_{scope}_{view}"] = int(present.sum())
            result[f"old_no_news_windows_with_{scope}_{view}"] = int((old_gap & present).sum())
            result[f"old_no_news_recovery_rate_{scope}_{view}"] = float((old_gap & present).sum() / old_gap.sum())
    result["warning"] = "Coverage counts are candidates, not independently validated Amazon-event precision and not prediction improvement."
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)

    paths = {
        "canonical_inputs": args.source_repo / "work/stock-data/paper_methods_4h/v1/inputs.pkl",
        "accepted_schedule": args.source_repo / "work/stock-data/audit/xnys_schedule.csv",
        "entity_edges": args.source_repo / "outputs/stock_ecni_e1r/FNSPID_RELINKED_ENTITY_EDGES.parquet",
        "complete_metadata": args.source_repo / "work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    windows = load_windows(args.source_repo)
    schedule = build_schedule(args.source_repo, windows)
    edges = attach_availability(load_amzn_edges(args.source_repo), schedule)
    groups = collapse_groups(edges)
    coverage, links = map_coverage(windows, groups, schedule)
    census = candidate_census(edges, groups)
    review = sample_review(groups)
    summary = coverage_summary(coverage, edges, groups)

    public_schedule = schedule.loc[schedule["session_date"].between("2017-12-01", "2019-02-05")].copy()
    public_schedule.to_csv(args.output / "XNYS_SESSION_SCHEDULE.csv", index=False)
    census.to_csv(args.output / "AMZN_CANDIDATE_CENSUS.csv", index=False)
    coverage.to_csv(args.output / "WINDOW_COVERAGE.csv", index=False)
    review.to_csv(args.output / "REVIEW_SAMPLE_MANIFEST.csv", index=False)
    write_json(args.output / "COVERAGE_SUMMARY.json", summary)

    edges.to_parquet(args.private_output / "amzn_edges_with_availability.parquet", index=False)
    groups.to_parquet(args.private_output / "amzn_article_groups.parquet", index=False)
    links.to_parquet(args.private_output / "window_candidate_links.parquet", index=False)

    manifest = {
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                         for name, path in paths.items()},
        "public_files": {},
        "private_files": {},
        "forbidden_outcome_fields": sorted(FORBIDDEN),
        "code_version": "FNSPID_AMZN_4H_V1",
    }
    for path in sorted(args.output.iterdir()):
        if path.name != "INPUT_AUDIT.json" and path.is_file():
            manifest["public_files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for path in sorted(args.private_output.iterdir()):
        if path.is_file():
            manifest["private_files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(args.output / "INPUT_AUDIT.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
