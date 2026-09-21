#!/usr/bin/env python3
"""Extract private source evidence for the frozen Amazon review sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def representative_rows(edges: pd.DataFrame, groups: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    wanted = review[["review_id", "article_group_hash", "relation_stratum"]]
    rows = edges.merge(wanted, on="article_group_hash", how="inner", validate="many_to_one")
    rows["relation_match"] = rows["relation_type"].eq(rows["relation_stratum"]).astype(int)
    rows["body_rank"] = rows["body_available"].fillna(False).astype(int)
    rows = rows.sort_values(
        ["review_id", "relation_match", "body_rank", "matched_field", "source_file", "row_number", "record_id_hash"],
        ascending=[True, False, False, True, True, True, True],
    )
    reps = rows.groupby("review_id", as_index=False).first()
    if len(reps) != len(review):
        raise RuntimeError(f"expected {len(review)} representatives, got {len(reps)}")
    return reps


def scan_file(path: Path, source_file: str, targets: dict[int, list[dict]]) -> list[dict]:
    found = []
    if not targets:
        return found
    max_row = max(targets)
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, 1):
            if row_number in targets:
                title = row.get("Article_title") or ""
                body = row.get("Article") or ""
                publisher = row.get("Publisher") or ""
                date = row.get("Date") or ""
                for target in targets[row_number]:
                    text = title if target["matched_field"] == "title" else body
                    nt = norm(text)
                    alias = norm(target["matched_alias"])
                    pos = nt.find(alias)
                    if pos < 0:
                        context = nt[:900]
                        evidence_found = False
                    else:
                        context = nt[max(0, pos - 360): pos + len(alias) + 540]
                        evidence_found = True
                    found.append({
                        "review_id": target["review_id"],
                        "article_group_hash": target["article_group_hash"],
                        "record_id_hash": target["record_id_hash"],
                        "source_file": source_file,
                        "source_row_number": row_number,
                        "recorded_date": date,
                        "publisher": publisher,
                        "title": title,
                        "evidence_context_normalized": context,
                        "matched_alias": target["matched_alias"],
                        "matched_field": target["matched_field"],
                        "mechanical_relation_claim": target["relation_stratum"],
                        "evidence_alias_found": evidence_found,
                        "semantic_review_decision": None,
                        "independent_reviewer_id": None,
                    })
            if row_number % 1_000_000 == 0:
                print(f"{source_file}: scanned {row_number:,}/{max_row:,}; found {len(found)}", flush=True)
            if row_number >= max_row:
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    args = ap.parse_args()

    review = pd.read_csv(args.output / "REVIEW_SAMPLE_MANIFEST.csv")
    edges = pd.read_parquet(args.private_output / "amzn_edges_with_availability.parquet")
    groups = pd.read_parquet(args.private_output / "amzn_article_groups.parquet")
    reps = representative_rows(edges, groups, review)
    targets = defaultdict(lambda: defaultdict(list))
    for row in reps.to_dict("records"):
        targets[str(row["source_file"])][int(row["row_number"])].append(row)

    raw = args.source_repo / "work/stock-data/ecni_e1r/raw"
    paths = {"all_external": raw / "All_external.csv", "nasdaq": raw / "nasdaq_exteral_data.csv"}
    cards = []
    for source_file, path in paths.items():
        cards.extend(scan_file(path, source_file, targets[source_file]))
    cards = sorted(cards, key=lambda x: x["review_id"])
    if len(cards) != 120 or len({x["review_id"] for x in cards}) != 120:
        raise RuntimeError(f"review-card extraction incomplete: {len(cards)} rows")
    private_path = args.private_output / "AMZN_ENTITY_REVIEW_CARDS.jsonl"
    with private_path.open("w", encoding="utf-8") as f:
        for row in cards:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    audit = {
        "status": "PRIVATE_REVIEW_PACK_READY",
        "review_cards": len(cards),
        "unique_review_ids": len({x["review_id"] for x in cards}),
        "evidence_alias_found": sum(bool(x["evidence_alias_found"]) for x in cards),
        "semantic_labels_completed": 0,
        "independent_reviewers": 0,
        "private_file_sha256": digest(private_path),
        "private_text_committed": False,
    }
    (args.output / "REVIEW_CARD_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
