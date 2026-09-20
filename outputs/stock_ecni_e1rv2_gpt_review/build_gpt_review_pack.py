#!/usr/bin/env python3
"""Build the public, answer-blind ECNI E1R-V2 GPT review pack.

This program uses the frozen private cards only to preserve their IDs and safe
metadata.  It reads the original FNSPID rows only to recover and verify the
stored source URL.  Article text is never written to the public pack.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.csv as pacsv
import pyarrow.dataset as pads


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stock_ecni_e1rv2_gpt_review"
PRIVATE_CARDS = ROOT / "work/stock-data/ecni_e1rv2/private/ENTITY_SEMANTIC_REVIEW_CARDS.jsonl"
METADATA = ROOT / "work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet"
RAW = {
    "all_external": ROOT / "work/stock-data/ecni_e1r/raw/All_external.csv",
    "nasdaq": ROOT / "work/stock-data/ecni_e1r/raw/nasdaq_exteral_data.csv",
}
SOURCE_CHECKPOINT = "ad0c7af2415d436a05bf346d352e2bbbd0edcbef"
PRIVATE_SHA256 = "41a906f9822b52539f76cb61636954b7c9ff4e996e7bb472984a08712e8a9654"
DISPLAY_SEED = "ECNI_E1RV2_GPT_REVIEW_DISPLAY_V1"
SAFE_FIELDS = [
    "review_index", "card_id", "target_company", "target_ticker",
    "matched_alias", "matched_field", "publisher", "date", "native_ticker",
    "source_url", "source_file", "source_record_locator", "source_access_status",
    "CIK",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_cards() -> list[dict]:
    if file_sha256(PRIVATE_CARDS) != PRIVATE_SHA256:
        raise RuntimeError("frozen private-card SHA-256 mismatch")
    cards = [json.loads(line) for line in PRIVATE_CARDS.open(encoding="utf-8")]
    ids = [row["card_id"] for row in cards]
    if len(cards) != 515 or len(set(ids)) != 515:
        raise RuntimeError("frozen private sample is not exactly 515 unique cards")
    return cards


def metadata_rows(cards: list[dict]) -> dict[str, dict]:
    record_ids = sorted({row["record_id_hash"] for row in cards})
    dataset = pads.dataset(METADATA, format="parquet")
    table = dataset.to_table(
        columns=["source_file", "row_number", "record_id_hash", "url_hash"],
        filter=pads.field("record_id_hash").isin(record_ids),
    )
    rows = table.to_pylist()
    counts = Counter(row["record_id_hash"] for row in rows)
    if set(counts) != set(record_ids) or any(value != 1 for value in counts.values()):
        raise RuntimeError("record provenance does not map one-to-one to frozen record IDs")
    return {row["record_id_hash"]: row for row in rows}


def recover_urls(meta: dict[str, dict]) -> tuple[dict[str, str], list[dict]]:
    wanted: dict[str, dict[int, str]] = {name: {} for name in RAW}
    for record_id, row in meta.items():
        source = row["source_file"]
        if source not in RAW:
            raise RuntimeError(f"unknown source file tag: {source}")
        row_number = int(row["row_number"])
        if row_number in wanted[source]:
            raise RuntimeError(f"duplicate provenance row: {source}:{row_number}")
        wanted[source][row_number] = record_id

    urls: dict[str, str] = {}
    failures: list[dict] = []
    for source, path in RAW.items():
        targets = wanted[source]
        if not targets:
            continue
        remaining = set(targets)
        offset = 0
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=64 << 20, use_threads=True),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(include_columns=["Date", "Article_title", "Url"]),
        )
        for batch in reader:
            first = offset + 1
            last = offset + batch.num_rows
            hit_rows = sorted(number for number in remaining if first <= number <= last)
            if hit_rows:
                dates = batch.column(batch.schema.get_field_index("Date"))
                titles = batch.column(batch.schema.get_field_index("Article_title"))
                raw_urls = batch.column(batch.schema.get_field_index("Url"))
                for row_number in hit_rows:
                    index = row_number - first
                    date = dates[index].as_py() or ""
                    title = titles[index].as_py() or ""
                    url = raw_urls[index].as_py() or ""
                    record_id = targets[row_number]
                    expected_record = sha256_bytes(
                        f"{source}|{row_number}|{url}|{date}|{title}".encode("utf-8", "replace")
                    )
                    expected_url_hash = sha256_bytes(norm(url).encode("utf-8", "replace")) if url else ""
                    reasons = []
                    if expected_record != record_id:
                        reasons.append("record_id_hash_mismatch")
                    if expected_url_hash != (meta[record_id]["url_hash"] or ""):
                        reasons.append("url_hash_mismatch")
                    if reasons:
                        failures.append({
                            "record_id_hash": record_id,
                            "source_file": source,
                            "row_number": row_number,
                            "reasons": reasons,
                        })
                    else:
                        urls[record_id] = url
                    remaining.remove(row_number)
            offset = last
            if not remaining:
                break
        for row_number in sorted(remaining):
            failures.append({
                "record_id_hash": targets[row_number],
                "source_file": source,
                "row_number": row_number,
                "reasons": ["source_row_not_found"],
            })
    return urls, failures


def valid_public_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_batches").mkdir(exist_ok=True)
    cards = load_cards()
    meta = metadata_rows(cards)
    urls, failures = recover_urls(meta)
    if failures or len(urls) != len({row["record_id_hash"] for row in cards}):
        write_json(OUT / "SOURCE_URL_MAPPING_AUDIT.json", {
            "status": "FAIL", "cards_total": 515, "mapping_failures": len(failures),
            "failure_records": failures,
        })
        raise RuntimeError(f"source URL mapping failed for {len(failures)} records")

    ordered_cards = sorted(
        cards,
        key=lambda row: (sha256_bytes(f"{DISPLAY_SEED}|{row['card_id']}".encode()), row["card_id"]),
    )
    rows = []
    for index, card in enumerate(ordered_cards, 1):
        metadata = meta[card["record_id_hash"]]
        url = urls[card["record_id_hash"]]
        if valid_public_url(url):
            access = "PUBLIC_URL_PRESENT"
        elif not url.strip():
            access = "NO_PUBLIC_URL"
        else:
            access = "INVALID_SOURCE_URL"
        row = {
            "review_index": index,
            "card_id": card["card_id"],
            "target_company": card["target_company"],
            "target_ticker": card["target_ticker"],
            "matched_alias": card["matched_alias"],
            "matched_field": card["matched_field"],
            "publisher": card["publisher"],
            "date": card["date"],
            "native_ticker": card["native_ticker"],
            "source_url": url,
            "source_file": card["source_file"],
            "source_record_locator": f"{metadata['source_file']}:{int(metadata['row_number'])}",
            "source_access_status": access,
            "CIK": int(card["CIK"]),
        }
        if list(row) != SAFE_FIELDS:
            raise RuntimeError("review index field contract changed")
        rows.append(row)

    index_path = OUT / "ENTITY_REVIEW_INDEX.jsonl"
    write_jsonl(index_path, rows)
    ordered_ids = [row["card_id"] for row in rows]
    ordered_text = "\n".join(ordered_ids) + "\n"
    order_path = OUT / "BLINDED_DISPLAY_ORDER.json"
    write_json(order_path, {
        "display_seed": DISPLAY_SEED,
        "ordering_method": "ascending SHA256(display_seed + '|' + card_id), then card_id",
        "ordered_card_ids": ordered_ids,
        "ordered_card_ids_sha256": sha256_bytes(ordered_text.encode()),
        "card_count": len(ordered_ids),
    })

    batch_paths = []
    for batch_index in range(5):
        batch = rows[batch_index * 103:(batch_index + 1) * 103]
        path = OUT / "review_batches" / f"BATCH_{batch_index + 1:02d}.jsonl"
        write_jsonl(path, batch)
        batch_paths.append(path)

    valid_urls = [row["source_url"] for row in rows if row["source_access_status"] == "PUBLIC_URL_PRESENT"]
    duplicate_counts = Counter(valid_urls)
    duplicated_values = {url: count for url, count in duplicate_counts.items() if count > 1}
    audit = {
        "status": "PASS",
        "source_checkpoint": SOURCE_CHECKPOINT,
        "cards_total": len(rows),
        "cards_with_valid_nonempty_source_url": len(valid_urls),
        "cards_without_valid_public_source_url": len(rows) - len(valid_urls),
        "cards_with_no_source_url": sum(row["source_access_status"] == "NO_PUBLIC_URL" for row in rows),
        "cards_with_invalid_source_url": sum(row["source_access_status"] == "INVALID_SOURCE_URL" for row in rows),
        "unique_valid_source_urls": len(duplicate_counts),
        "duplicate_url_values": len(duplicated_values),
        "cards_in_duplicate_url_groups": sum(duplicated_values.values()),
        "duplicate_card_occurrences_beyond_first": sum(value - 1 for value in duplicated_values.values()),
        "mapping_failures": 0,
        "provenance_checks": {
            "record_id_hash_recomputed_from_exact_source_row": True,
            "url_hash_matches_frozen_metadata": True,
        },
        "semantic_content_reviewed_by_builder": False,
    }
    write_json(OUT / "SOURCE_URL_MAPPING_AUDIT.json", audit)

    manifest_path = OUT / "GPT_REVIEW_PACK_MANIFEST.json"
    manifest = {
        "status": "BUILT_PENDING_INDEPENDENT_PACK_VERIFICATION",
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_frozen_card_sha256": PRIVATE_SHA256,
        "source_frozen_card_count": len(cards),
        "public_index_sha256": file_sha256(index_path),
        "display_order_file_sha256": file_sha256(order_path),
        "ordered_card_ids_sha256": sha256_bytes(ordered_text.encode()),
        "batch_file_sha256": {str(path.relative_to(OUT)): file_sha256(path) for path in batch_paths},
        "batch_counts": {str(path.relative_to(OUT)): sum(1 for _ in path.open()) for path in batch_paths},
        "url_mapping_counts": {
            "cards_with_valid_nonempty_source_url": audit["cards_with_valid_nonempty_source_url"],
            "cards_without_valid_public_source_url": audit["cards_without_valid_public_source_url"],
            "duplicate_url_values": audit["duplicate_url_values"],
            "mapping_failures": 0,
        },
        "copyrighted_text_committed": False,
        "semantic_labels_generated_by_codex": 0,
        "returns_or_outcomes_accessed": 0,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
