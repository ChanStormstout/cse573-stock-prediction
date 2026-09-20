#!/usr/bin/env python3
"""Fit-free, label-free verifier for the public E1R-V2 GPT review pack."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stock_ecni_e1rv2_gpt_review"
PRIVATE = ROOT / "work/stock-data/ecni_e1rv2/private/ENTITY_SEMANTIC_REVIEW_CARDS.jsonl"
PRIVATE_SHA256 = "41a906f9822b52539f76cb61636954b7c9ff4e996e7bb472984a08712e8a9654"
SAFE_FIELDS = {
    "review_index", "card_id", "target_company", "target_ticker", "matched_alias",
    "matched_field", "publisher", "date", "native_ticker", "source_url",
    "source_file", "source_record_locator", "source_access_status", "CIK",
}
PROHIBITED_FIELDS = {
    "relation_claim_under_review", "relation_type", "sample_roles", "risk_flags",
    "risk_flag", "semantic_label", "label", "expected_answer", "precision_gate_result",
    "ndaq_sentinel", "panel_base", "risk_enriched", "mechanical_diagnosis",
    "article", "article_body", "body", "title", "article_title", "evidence",
    "evidence_context", "evidence_context_normalized", "evidence_excerpt",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    frozen = read_jsonl(PRIVATE)
    frozen_ids = [row["card_id"] for row in frozen]
    index_path = OUT / "ENTITY_REVIEW_INDEX.jsonl"
    index = read_jsonl(index_path)
    batches = []
    batch_counts = {}
    for number in range(1, 6):
        path = OUT / "review_batches" / f"BATCH_{number:02d}.jsonl"
        rows = read_jsonl(path)
        batches.extend(rows)
        batch_counts[path.name] = len(rows)

    order = json.load((OUT / "BLINDED_DISPLAY_ORDER.json").open())
    mapping = json.load((OUT / "SOURCE_URL_MAPPING_AUDIT.json").open())
    manifest = json.load((OUT / "GPT_REVIEW_PACK_MANIFEST.json").open())
    index_ids = [row.get("card_id") for row in index]
    batch_ids = [row.get("card_id") for row in batches]
    ordered_text = "\n".join(index_ids) + "\n"
    ordered_hash = hashlib.sha256(ordered_text.encode()).hexdigest()

    prohibited_present = sorted({key for row in index + batches for key in row if key.lower() in PROHIBITED_FIELDS})
    unexpected_fields = sorted({key for row in index + batches for key in row if key not in SAFE_FIELDS})
    data_text = "\n".join(
        (OUT / name).read_text(encoding="utf-8")
        for name in ["ENTITY_REVIEW_INDEX.jsonl"]
    ) + "\n" + "\n".join(
        (OUT / "review_batches" / f"BATCH_{number:02d}.jsonl").read_text(encoding="utf-8")
        for number in range(1, 6)
    )

    checks = {
        "private_frozen_sha256": digest(PRIVATE) == PRIVATE_SHA256,
        "private_sample_515_unique": len(frozen_ids) == len(set(frozen_ids)) == 515,
        "index_515_unique": len(index_ids) == len(set(index_ids)) == 515,
        "index_exact_frozen_ids": set(index_ids) == set(frozen_ids),
        "review_indices_exact_1_to_515": [row.get("review_index") for row in index] == list(range(1, 516)),
        "five_batches_each_103": batch_counts == {f"BATCH_{number:02d}.jsonl": 103 for number in range(1, 6)},
        "batches_partition_frozen_sample": len(batch_ids) == len(set(batch_ids)) == 515 and set(batch_ids) == set(frozen_ids),
        "batch_rows_equal_index_rows_in_order": batches == index,
        "display_order_equals_index": order.get("ordered_card_ids") == index_ids,
        "display_order_hash_recomputed": order.get("ordered_card_ids_sha256") == ordered_hash,
        "reviewer_safe_fields_only": not unexpected_fields and all(set(row) == SAFE_FIELDS for row in index + batches),
        "no_prohibited_answer_or_text_fields": not prohibited_present,
        "no_prefilled_semantic_labels": all("label" not in row and "semantic_label" not in row for row in index + batches),
        "no_private_article_body_or_evidence_excerpt": not any(
            token in data_text for token in ('"article_body":', '"body":', '"evidence_context":', '"evidence_excerpt":', '"title":')
        ),
        "URL_mapping_pass_zero_failures": mapping.get("status") == "PASS" and mapping.get("mapping_failures") == 0,
        "all_source_statuses_allowed": all(row.get("source_access_status") in {"PUBLIC_URL_PRESENT", "NO_PUBLIC_URL", "INVALID_SOURCE_URL"} for row in index),
        "manifest_source_identity": manifest.get("source_frozen_card_sha256") == PRIVATE_SHA256 and manifest.get("source_frozen_card_count") == 515,
        "manifest_index_hash": manifest.get("public_index_sha256") == digest(index_path),
        "manifest_order_hash": manifest.get("display_order_file_sha256") == digest(OUT / "BLINDED_DISPLAY_ORDER.json"),
        "manifest_batch_hashes": all(
            manifest.get("batch_file_sha256", {}).get(f"review_batches/BATCH_{number:02d}.jsonl")
            == digest(OUT / "review_batches" / f"BATCH_{number:02d}.jsonl")
            for number in range(1, 6)
        ),
        "manifest_no_labels_no_outcomes_no_copyrighted_text": (
            manifest.get("semantic_labels_generated_by_codex") == 0
            and manifest.get("returns_or_outcomes_accessed") == 0
            and manifest.get("copyrighted_text_committed") is False
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "status": status,
        "checks": checks,
        "details": {
            "batch_counts": batch_counts,
            "prohibited_fields_found": prohibited_present,
            "unexpected_fields_found": unexpected_fields,
            "frozen_cards": len(frozen_ids),
            "index_rows": len(index_ids),
            "batch_rows": len(batch_ids),
            "semantic_labels_generated": 0,
            "returns_or_outcomes_accessed": 0,
        },
    }
    (OUT / "GPT_REVIEW_PACK_VERIFICATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(status != "PASS")


if __name__ == "__main__":
    main()
