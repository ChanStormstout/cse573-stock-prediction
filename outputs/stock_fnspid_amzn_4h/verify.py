#!/usr/bin/env python3
"""Independent, fit-free verification for the FNSPID AMZN v1 coverage lane."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FORBIDDEN_EXACT = {"label", "target_return", "return", "ba", "mcc", "brier", "probability", "prediction"}
HIGH = {"DIRECT_TARGET_HIGH_CONFIDENCE", "MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(name, condition, details, checks):
    checks.append({"name": name, "passes": bool(condition), "details": details})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    args = ap.parse_args()
    checks = []
    audit = json.loads((args.output / "INPUT_AUDIT.json").read_text())
    summary = json.loads((args.output / "COVERAGE_SUMMARY.json").read_text())
    coverage = pd.read_csv(args.output / "WINDOW_COVERAGE.csv")
    review = pd.read_csv(args.output / "REVIEW_SAMPLE_MANIFEST.csv")
    schedule = pd.read_csv(args.output / "XNYS_SESSION_SCHEDULE.csv")
    edges = pd.read_parquet(args.private_output / "amzn_edges_with_availability.parquet")
    groups = pd.read_parquet(args.private_output / "amzn_article_groups.parquet")
    links = pd.read_parquet(args.private_output / "window_candidate_links.parquet")
    review_audit_path = args.output / "REVIEW_CARD_AUDIT.json"

    source_ok = True
    bad_sources = []
    for name, record in audit["source_files"].items():
        path = Path(record["path"])
        actual = sha256_file(path) if path.exists() else None
        if actual != record["sha256"]:
            source_ok = False
            bad_sources.append(name)
    check("source_hashes", source_ok, {"mismatches": bad_sources}, checks)

    public_ok = True
    bad_public = []
    for name, record in audit["public_files"].items():
        path = args.output / name
        if not path.exists() or sha256_file(path) != record["sha256"]:
            public_ok = False
            bad_public.append(name)
    check("public_artifact_hashes", public_ok, {"mismatches": bad_public}, checks)

    cols = [c.lower() for c in coverage.columns] + [c.lower() for c in review.columns]
    forbidden_found = sorted(set(cols) & FORBIDDEN_EXACT)
    check("outcome_free_public_schema", not forbidden_found, {"forbidden_matches": forbidden_found}, checks)
    check("canonical_amzn_windows", len(coverage) == 804 and coverage["key"].nunique() == 804,
          {"rows": len(coverage), "unique_keys": coverage["key"].nunique()}, checks)
    check("canonical_original_gap", int((coverage["has_original_news"] == 0).sum()) == 389,
          {"no_original_news": int((coverage["has_original_news"] == 0).sum())}, checks)
    check("date_only_source", set(edges["available_time_class"].unique()) == {"DATE_ONLY_CONSERVATIVE"},
          {"classes": sorted(edges["available_time_class"].unique().tolist())}, checks)

    recorded = pd.to_datetime(edges["recorded_time"], utc=True)
    available = pd.to_datetime(edges["available_at_utc"], utc=True)
    next_date = available.dt.date.astype(str) > recorded.dt.date.astype(str)
    opens = set(pd.to_datetime(schedule["open_utc"], utc=True).astype("int64"))
    on_open = available.astype("int64").isin(opens)
    check("next_session_open_availability", bool(next_date.all() and on_open.all()),
          {"not_strictly_next_date": int((~next_date).sum()), "not_session_open": int((~on_open).sum())}, checks)

    merged = links.merge(coverage[["key", "cutoff_utc"]], on="key", how="left", validate="many_to_one")
    time_safe = pd.to_datetime(merged["available_at_utc"], utc=True) <= pd.to_datetime(merged["cutoff_utc"], utc=True)
    check("window_links_time_safe", bool(time_safe.all()), {"violations": int((~time_safe).sum())}, checks)
    check("links_only_direct_or_multi", set(links["relation_stratum"].unique()).issubset(HIGH),
          {"relations": sorted(links["relation_stratum"].unique().tolist())}, checks)

    expected_group_cols = {"article_group_hash", "relation_stratum", "available_at_utc", "record_count"}
    check("deduplicated_group_contract", expected_group_cols.issubset(groups.columns) and groups["article_group_hash"].is_unique,
          {"groups": len(groups), "unique": groups["article_group_hash"].nunique()}, checks)
    check("review_sample_frozen", len(review) == 120 and review["article_group_hash"].is_unique,
          {"rows": len(review), "unique_groups": review["article_group_hash"].nunique()}, checks)
    check("review_sample_high_relations", set(review["relation_stratum"]).issubset(HIGH),
          {"relations": sorted(review["relation_stratum"].unique().tolist())}, checks)
    if review_audit_path.exists():
        review_audit = json.loads(review_audit_path.read_text())
        private_cards = args.private_output / "AMZN_ENTITY_REVIEW_CARDS.jsonl"
        card_hash = sha256_file(private_cards) if private_cards.exists() else None
        check("private_review_pack_hash", card_hash == review_audit.get("private_file_sha256"),
              {"expected": review_audit.get("private_file_sha256"), "actual": card_hash}, checks)
        check("private_review_pack_counts",
              review_audit.get("review_cards") == 120 and review_audit.get("unique_review_ids") == 120,
              {"review_cards": review_audit.get("review_cards"),
               "unique_review_ids": review_audit.get("unique_review_ids"),
               "evidence_alias_found": review_audit.get("evidence_alias_found")}, checks)
    check("zero_prediction_activity", summary.get("predictive_models_fitted") == 0 and summary.get("outcomes_used") == 0,
          {"predictive_models_fitted": summary.get("predictive_models_fitted"), "outcomes_used": summary.get("outcomes_used")}, checks)
    check("native_ticker_not_used_as_truth", summary.get("native_ticker_agreeing_edge_rows") == 0,
          {"native_ticker_agreeing_edge_rows": summary.get("native_ticker_agreeing_edge_rows")}, checks)

    result = {"status": "PASS" if all(x["passes"] for x in checks) else "FAIL", "checks": checks}
    (args.output / "VERIFICATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
