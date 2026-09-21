#!/usr/bin/env python3
"""Build course-news + direct FNSPID inputs and frozen semantic vectors."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRIVATE = ROOT / "work/stock-data/fnspid_augmented_4h/v1"
PUBLIC = HERE / "v1"
RAW_NAMES = {"all_external": "All_external.csv", "nasdaq": "nasdaq_exteral_data.csv"}
CIKS = {"AAPL": "320193", "AMZN": "1018724"}


def sha_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def load_direct(source: Path) -> pd.DataFrame:
    edges = source / "outputs/stock_ecni_e1r/FNSPID_RELINKED_ENTITY_EDGES.parquet"
    meta = source / "work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet"
    con = duckdb.connect()
    query = f"""
    SELECT e.record_id_hash,e.resolved_ticker AS symbol,e.recorded_time,e.source_file,
           e.publisher_hash,e.body_available,e.title_hash,e.url_hash,
           m.row_number,m.normalized_title_hash,m.near_title_fingerprint,m.body_length
    FROM read_parquet('{edges.as_posix()}') e
    JOIN read_parquet('{meta.as_posix()}') m USING(record_id_hash)
    WHERE cast(e.CIK as varchar) IN {tuple(CIKS.values())}
      AND e.relation_type='DIRECT_TARGET_HIGH_CONFIDENCE'
      AND cast(e.recorded_time as timestamp) >= timestamp '2017-12-01'
      AND cast(e.recorded_time as timestamp) < timestamp '2019-02-02'
    """
    d = con.execute(query).fetchdf()
    con.close()
    if set(d.symbol) != set(CIKS):
        raise RuntimeError("missing one target symbol")
    d["recorded_time"] = pd.to_datetime(d.recorded_time, utc=True)
    for c in ("url_hash", "normalized_title_hash", "near_title_fingerprint", "record_id_hash"):
        d[c] = d[c].fillna("").astype(str)

    def group_key(r):
        for prefix, col in (("U", "url_hash"), ("T", "normalized_title_hash"), ("N", "near_title_fingerprint")):
            if r[col]:
                return prefix + ":" + r[col]
        return "R:" + r.record_id_hash

    d["group_key"] = d.apply(group_key, axis=1)
    d["group_hash"] = d.group_key.map(sha_bytes)
    d = d.sort_values(["symbol", "group_hash", "body_available", "body_length", "recorded_time", "record_id_hash"],
                      ascending=[True, True, False, False, True, True])
    reps = d.groupby(["symbol", "group_hash"], as_index=False).first()
    reps["group_record_count"] = reps.set_index(["symbol", "group_hash"]).index.map(
        d.groupby(["symbol", "group_hash"]).size())
    return reps


def schedule(source: Path, start: str, end: str) -> pd.DataFrame:
    accepted = pd.read_csv(source / "work/stock-data/audit/xnys_schedule.csv")
    accepted = accepted.rename(columns={accepted.columns[0]: "session_date"})
    accepted["session_date"] = pd.to_datetime(accepted.session_date).dt.date.astype(str)
    accepted["open_utc"] = pd.to_datetime(accepted.open, utc=True)
    accepted["close_utc"] = pd.to_datetime(accepted.close, utc=True)
    s = accepted[(accepted.session_date >= start) & (accepted.session_date <= end)][["session_date", "open_utc", "close_utc"]].copy()
    if s.empty or s.session_date.max() < "2019-02-01":
        raise RuntimeError("accepted XNYS schedule does not cover the experiment")
    s["session_index"] = np.arange(len(s))
    return s.reset_index(drop=True)


def attach_availability(reps: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    dates = sessions.session_date.to_numpy(str)
    recorded = reps.recorded_time.dt.date.astype(str).to_numpy()
    pos = np.searchsorted(dates, recorded, side="right")
    valid = pos < len(sessions)
    reps = reps.loc[valid].copy().reset_index(drop=True)
    pos = pos[valid]
    reps["available_session_index"] = sessions.iloc[pos].session_index.to_numpy()
    reps["available_at_utc"] = sessions.iloc[pos].open_utc.to_numpy()
    if not (pd.to_datetime(reps.available_at_utc, utc=True) > reps.recorded_time).all():
        raise RuntimeError("date-only record did not move to next open")
    return reps


def recover_text(reps: pd.DataFrame, source: Path) -> pd.DataFrame:
    raw_dir = source / "work/stock-data/ecni_e1r/raw"
    wanted = {tag: {} for tag in RAW_NAMES}
    for r in reps.itertuples(index=False):
        wanted[r.source_file][int(r.row_number)] = r
    rows = []
    for tag, name in RAW_NAMES.items():
        targets = wanted[tag]
        if not targets:
            continue
        remaining = set(targets)
        offset = 0
        reader = pacsv.open_csv(
            raw_dir / name,
            read_options=pacsv.ReadOptions(block_size=64 << 20, use_threads=True),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=["Date", "Article_title", "Article", "Publisher", "Url"],
                column_types={x: pa.string() for x in ["Date", "Article_title", "Article", "Publisher", "Url"]},
            ),
        )
        for batch in reader:
            first, last = offset + 1, offset + batch.num_rows
            hits = sorted(x for x in remaining if first <= x <= last)
            if hits:
                names = {n: batch.column(batch.schema.get_field_index(n)) for n in ["Date", "Article_title", "Article", "Publisher", "Url"]}
                for number in hits:
                    i = number - first
                    vals = {n: names[n][i].as_py() or "" for n in names}
                    r = targets[number]
                    actual = sha_bytes(f"{tag}|{number}|{vals['Url']}|{vals['Date']}|{vals['Article_title']}")
                    if actual != r.record_id_hash:
                        raise RuntimeError(f"raw provenance mismatch {tag}:{number}")
                    rows.append({
                        "symbol": r.symbol, "group_hash": r.group_hash,
                        "article_key": f"FNSPID::{r.symbol}::{r.group_hash}",
                        "record_id_hash": r.record_id_hash, "source_file": tag,
                        "row_number": number, "title": vals["Article_title"],
                        "body": vals["Article"], "publisher": vals["Publisher"],
                        "normalized_title_hash": sha_bytes(norm(vals["Article_title"])) if vals["Article_title"] else "",
                        "available_at_utc": r.available_at_utc,
                        "available_session_index": int(r.available_session_index),
                        "group_record_count": int(r.group_record_count),
                    })
                    remaining.remove(number)
            offset = last
            if not remaining:
                break
        if remaining:
            raise RuntimeError(f"missing {len(remaining)} raw rows from {tag}")
        print("raw", tag, "recovered", len(targets), flush=True)
    out = pd.DataFrame(rows)
    if len(out) != len(reps) or out.article_key.nunique() != len(out):
        raise RuntimeError("representative text cardinality mismatch")
    return out


def text_tokens(articles: pd.DataFrame) -> dict[str, str]:
    stemmer = EnglishStemmer()

    @lru_cache(maxsize=150000)
    def stem(w):
        return stemmer.stem(w)

    bad = re.compile(r"newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here", re.I)
    result = {}
    for r in articles.itertuples(index=False):
        text = html.unescape(re.sub(r"<[^>]+>", " ", (r.title or "") + "\n" + (r.body or "")))
        text = " ".join(line for line in text.splitlines() if not bad.search(line))
        text = re.sub(r"https?://\S+", " ", text)
        toks = {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS}
        result[r.article_key] = " ".join(sorted(toks))
    return result


def encode_external(articles: pd.DataFrame, source: Path):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    import sys
    sys.path.insert(0, str(source / "outputs/stock_foundation_4h"))
    from pooling import special_token_excluded_mean

    texts = articles.title.fillna("").astype(str).tolist()
    keys = articles.article_key.to_numpy(str)
    torch.manual_seed(573)
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    fin_kw = dict(revision="4556d13015211d73dccd3fdd39d39232506f3e43", cache_dir=str(source / "work/stock-data/finbert-cache"), local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", **fin_kw)
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", **fin_kw).eval().to(device)
    model.requires_grad_(False)
    fin = []
    for start in range(0, len(texts), 32):
        x = tok(texts[start:start + 32], padding=True, truncation=True, max_length=256, return_special_tokens_mask=True, return_tensors="pt").to(device)
        special = x.pop("special_tokens_mask")
        mask = (x["attention_mask"] * (1 - special)).unsqueeze(-1)
        with torch.inference_mode():
            hidden = model.bert(**x).last_hidden_state
            fin.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
        if start % 640 == 0:
            print("finbert", min(start + 32, len(texts)), "/", len(texts), flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()

    modern_path = source / "work/stock-data/foundation_4h/models/modern"
    tok = AutoTokenizer.from_pretrained(modern_path, local_files_only=True)
    model = AutoModel.from_pretrained(modern_path, local_files_only=True, torch_dtype=torch.float32,
                                      attn_implementation="eager", reference_compile=False).eval().to(device)
    model.requires_grad_(False)
    modern = []
    for start in range(0, len(texts), 32):
        x = tok(texts[start:start + 32], padding=True, truncation=True, max_length=256,
                return_special_tokens_mask=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state
            modern.append(special_token_excluded_mean(hidden, x["attention_mask"], x["special_tokens_mask"]).cpu().numpy())
        if start % 640 == 0:
            print("modern", min(start + 32, len(texts)), "/", len(texts), flush=True)
    return keys, np.concatenate(fin).astype(np.float32), np.concatenate(modern).astype(np.float32), device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-repo", type=Path, required=True)
    args = ap.parse_args()
    source = args.source_repo.resolve()
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("v1 exists; runs are immutable")
    PRIVATE.mkdir(parents=True)
    PUBLIC.mkdir(parents=True)
    tick = time.monotonic()

    d = pd.read_pickle(source / "work/stock-data/paper_methods_4h/v3/inputs.pkl").sort_values(["start_utc", "symbol"]).reset_index(drop=True)
    for c in ("start_utc", "end_utc", "cutoff_utc"):
        d[c] = pd.to_datetime(d[c], utc=True)
    if len(d) != 1607 or d.key.nunique() != 1607:
        raise RuntimeError("canonical rows mismatch")
    sessions = schedule(source, "2017-11-01", "2019-02-08")
    reps = attach_availability(load_direct(source), sessions)
    articles = recover_text(reps, source)
    articles.to_parquet(PRIVATE / "fnspid_articles.parquet", index=False)
    tokens = text_tokens(articles)

    original_npz = np.load(source / "outputs/stock_integrated_4h/prepared/articles.npz")
    original_keys = original_npz["keys"].astype(str)
    original_fin = original_npz["embeddings"].astype(np.float32)
    original_modern_npz = np.load(source / "work/stock-data/foundation_4h/v2/modern_embeddings.npz")
    if not np.array_equal(original_keys, original_modern_npz["keys"].astype(str)):
        raise RuntimeError("original encoder keys mismatch")
    ext_keys, ext_fin, ext_modern, device = encode_external(articles, source)
    all_keys = np.r_[original_keys, ext_keys]
    if len(set(all_keys)) != len(all_keys):
        raise RuntimeError("combined article key collision")
    np.savez_compressed(PRIVATE / "combined_embeddings.npz", keys=all_keys,
                        finbert=np.vstack([original_fin, ext_fin]),
                        modern=np.vstack([original_modern_npz["embeddings"].astype(np.float32), ext_modern]))

    manifest = json.loads((source / "outputs/stock_integrated_4h/prepared/embedding_inputs.json").read_text())
    original_titles = dict(manifest["keys_and_titles"])
    raw_course = pd.read_pickle(source / "work/stock-data/audit/news_index.pkl")
    raw_course["key"] = raw_course.archive + "::" + raw_course.member
    raw_course = raw_course.set_index("key")
    raw_course["available_utc"] = pd.to_datetime(raw_course.available_utc, utc=True)
    sessions_by_day = dict(zip(sessions.session_date, sessions.session_index))
    by_symbol = {s: g.sort_values(["available_at_utc", "article_key"]) for s, g in articles.groupby("symbol")}
    ext_title_by_key = dict(zip(articles.article_key, articles.title.fillna("").astype(str)))
    ext_pub_by_key = dict(zip(articles.article_key, articles.publisher.fillna("").astype(str)))
    ext_avail_by_key = dict(zip(articles.article_key, pd.to_datetime(articles.available_at_utc, utc=True)))

    aug_keys, fnspid_keys, added_counts, removed_duplicates = [], [], [], []
    aug_text, aug_articles, aug_clusters, aug_sources, aug_newest, aug_median, aug_only_dup = [], [], [], [], [], [], []
    for r in d.itertuples(index=False):
        course = [k for k in str(r.news_record_keys or "").split("|") if k]
        course_hashes = {sha_bytes(norm(original_titles[k])) for k in course if original_titles.get(k)}
        current_idx = int(sessions_by_day[str(r.day)])
        candidates = by_symbol[r.symbol]
        candidates = candidates[(pd.to_datetime(candidates.available_at_utc, utc=True) <= r.cutoff_utc) &
                                candidates.available_session_index.between(current_idx - 2, current_idx)]
        duplicate = candidates.normalized_title_hash.isin(course_hashes)
        selected = candidates.loc[~duplicate]
        external = selected.article_key.tolist()
        combined = course + external
        aug_keys.append("|".join(combined)); fnspid_keys.append("|".join(external))
        added_counts.append(len(external)); removed_duplicates.append(int(duplicate.sum()))
        toks = set(str(r.stem_body or "").split())
        for k in external:
            toks.update(tokens[k].split())
        aug_text.append(" ".join(sorted(toks)))
        ages = [(r.cutoff_utc - raw_course.at[k, "available_utc"]).total_seconds() / 3600 for k in course]
        ages += [(r.cutoff_utc - ext_avail_by_key[k]).total_seconds() / 3600 for k in external]
        pubs = {str(raw_course.at[k, "site"]).strip().lower() for k in course if pd.notna(raw_course.at[k, "site"]) and str(raw_course.at[k, "site"]).strip()}
        pubs |= {ext_pub_by_key[k].strip().lower() for k in external if ext_pub_by_key[k].strip()}
        aug_articles.append(len(combined))
        aug_clusters.append(int(r.clusters) + len(external))
        aug_sources.append(len(pubs))
        aug_newest.append(min(ages) if ages else 0.0)
        aug_median.append(float(np.median(ages)) if ages else 0.0)
        aug_only_dup.append(int(bool(combined) and int(r.only_duplicates) == 1 and not external))
    d["aug_news_record_keys"] = aug_keys
    d["fnspid_record_keys"] = fnspid_keys
    d["fnspid_added_groups"] = added_counts
    d["fnspid_removed_course_duplicates"] = removed_duplicates
    d["aug_stem_body"] = aug_text
    d["aug_articles"] = aug_articles
    d["aug_clusters"] = aug_clusters
    d["aug_sources"] = aug_sources
    d["aug_newest_age"] = aug_newest
    d["aug_median_age"] = aug_median
    d["aug_only_duplicates"] = aug_only_dup
    d["has_aug_news"] = (d.aug_articles > 0).astype(int)
    d.to_pickle(PRIVATE / "inputs.pkl")
    sessions.to_csv(PRIVATE / "xnys_schedule.csv", index=False)

    coverage = d.groupby(["symbol", "phase"], dropna=False).agg(
        windows=("key", "size"), original_news_windows=("has_original_news", "sum"),
        augmented_news_windows=("has_aug_news", "sum"), windows_with_fnspid=("fnspid_added_groups", lambda x: int((x > 0).sum())),
        added_group_occurrences=("fnspid_added_groups", "sum"), removed_course_duplicates=("fnspid_removed_course_duplicates", "sum"),
    ).reset_index()
    coverage.to_csv(PUBLIC / "coverage.csv", index=False)
    source_files = {
        "canonical_inputs": source / "work/stock-data/paper_methods_4h/v3/inputs.pkl",
        "entity_edges": source / "outputs/stock_ecni_e1r/FNSPID_RELINKED_ENTITY_EDGES.parquet",
        "complete_metadata": source / "work/stock-data/ecni_e1r/private/fnspid_complete_metadata.parquet",
        "original_finbert": source / "outputs/stock_integrated_4h/prepared/articles.npz",
        "original_modern": source / "work/stock-data/foundation_4h/v2/modern_embeddings.npz",
        "protocol": HERE / "PRE_REGISTRATION.md",
    }
    evidence = {
        "status": "PREPARED", "canonical_rows": len(d), "representative_fnspid_groups": len(articles),
        "representative_groups_by_stock": articles.groupby("symbol").size().to_dict(),
        "raw_record_hashes_verified": len(articles), "semantic_device": device,
        "finbert_external_vectors": len(ext_fin), "modern_external_vectors": len(ext_modern),
        "encoder_training": False, "seconds": time.monotonic() - tick,
        "source_hashes": {str(p.relative_to(source if p.is_relative_to(source) else ROOT)): sha_file(p) for p in source_files.values()},
        "combined_embedding_sha256": sha_file(PRIVATE / "combined_embeddings.npz"),
        "independent_entity_review_complete": False,
    }
    dump(PUBLIC / "preparation_evidence.json", evidence)
    print(json.dumps(evidence, indent=2, default=str), flush=True)
    print(coverage.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
