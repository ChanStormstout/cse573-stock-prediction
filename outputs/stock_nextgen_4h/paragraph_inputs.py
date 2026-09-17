"""Create P0--P3 target-company passage packs without loading outcome labels into prompts."""
from __future__ import annotations

import argparse
import ast
import copy
import html
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import B, DIRECT, INTEGRATED, W, dump, jsonl, sha, text_sha

MODEL = W / "model_compare/model"
OLD_INPUTS = W / "direct_4h/inputs_v1"
VARIANTS = ("P0", "P1", "P2", "P3")
BAD = re.compile(
    r"newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here|sponsored financial content|view original content|copy this link|press the \[enter\]|investorsobserver issues critical|get [a-z ]+ alerts",
    re.I,
)
ALIASES = {
    "AAPL": re.compile(r"\b(?:Apple(?:\s+Inc\.?)?|AAPL)\b", re.I),
    "AMZN": re.compile(r"\b(?:Amazon(?:\.com)?(?:\s+Inc\.?)?|AMZN)\b", re.I),
}


def old_system_prompt() -> str:
    source = (DIRECT / "choice.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "SYSTEM" for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("old SYSTEM prompt not found")


SYSTEM = old_system_prompt()


def messages(row: dict):
    payload = {key: row[key] for key in ("symbol", "cutoff", "interval_start", "interval_end", "timezone")}
    payload.update(price_rows=row["price_rows"], price_summary=row["price_summary"])
    payload.update(
        news=[{key: value for key, value in article.items() if key not in {"record_key", "cluster_members", "eligible_target_units", "included_target_units"}} for article in row["news"]],
        news_summary=row["news_summary"],
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def prompt_tokens(tokenizer, row: dict) -> int:
    rendered = tokenizer.apply_chat_template(
        messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def paragraph_spans(body: str, symbol: str):
    """Return exact raw spans; a long line is reduced to complete sentence neighborhoods."""
    alias = ALIASES[symbol]
    paragraphs = []
    for match in re.finditer(r"[^\r\n]+", body):
        text = match.group(0)
        if not alias.search(text):
            continue
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        start, end = match.start() + left, match.start() + right
        if end <= start:
            continue
        terminal = bool(re.search(r"[.!?][\"'”’)]?\s*$", body[start:end]))
        if end - start <= 1200 and terminal and not BAD.search(body[start:end]):
            paragraphs.append((start, end, "paragraph"))
            continue
        local = body[start:end]
        # Boundaries require terminal punctuation followed by likely new-sentence text.
        sentences = []
        cursor = 0
        boundary = re.compile(r"[.!?][\"'”’)]*(?=\s+[A-Z0-9\"'])")
        for stop in boundary.finditer(local):
            finish = stop.end()
            prefix = local[cursor:finish]
            final_word = re.search(r"([A-Za-z]{1,8})\.[\"'”’)]*$", prefix)
            if final_word and final_word.group(1).lower() in {
                "inc", "corp", "co", "ltd", "mr", "mrs", "ms", "dr", "prof", "st", "vs", "u", "s",
            }:
                continue
            if finish > cursor:
                sentences.append((cursor, finish))
            cursor = finish
            while cursor < len(local) and local[cursor].isspace():
                cursor += 1
        # Do not turn a source-truncated tail into a sentence.
        if cursor < len(local) and re.search(r"[.!?][\"'”’)]?\s*$", local[cursor:]):
            sentences.append((cursor, len(local)))
        hits = [i for i, (a, b) in enumerate(sentences) if alias.search(local[a:b])]
        groups = []
        for hit in hits:
            a, b = max(0, hit - 1), min(len(sentences), hit + 2)
            if groups and a <= groups[-1][1]:
                groups[-1] = (groups[-1][0], max(groups[-1][1], b))
            else:
                groups.append((a, b))
        for a, b in groups:
            unit_start = start + sentences[a][0]
            unit_end = start + sentences[b - 1][1]
            unit_text = body[unit_start:unit_end]
            if BAD.search(unit_text):
                continue
            if "http" in unit_text.lower() and len(re.findall(r"[A-Za-z]+", unit_text)) < 50:
                continue
            paragraphs.append((unit_start, unit_end, "sentence_neighborhood"))
    unique = []
    seen = set()
    for start, end, kind in paragraphs:
        if (start, end) not in seen:
            unique.append({"start": start, "end": end, "text": body[start:end], "unit": kind})
            seen.add((start, end))
    return unique


def normalized_title(title: str):
    return set(re.findall(r"[a-z0-9]+", title.lower())) - {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for"}


def title_jaccard(a: str, b: str) -> float:
    x, y = normalized_title(a), normalized_title(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def online_clusters(keys, raw):
    ordered = sorted(keys, key=lambda key: (raw.loc[key, "available_utc"], key))
    clusters = []
    for key in ordered:
        row = raw.loc[key]
        match = None
        for i, members in enumerate(clusters):
            for prior in members:
                other = raw.loc[prior]
                same_hash = row.normalized_hash == other.normalized_hash or row.exact_hash == other.exact_hash
                if same_hash or title_jaccard(str(row.title), str(other.title)) >= 0.65:
                    match = i
                    break
            if match is not None:
                break
        if match is None:
            clusters.append([key])
        else:
            clusters[match].append(key)
    # At this cutoff the latest received report represents an already observed group.
    representatives = [max(members, key=lambda key: (raw.loc[key, "available_utc"], key)) for members in clusters]
    return clusters, sorted(representatives, key=lambda key: (raw.loc[key, "available_utc"], key), reverse=True)


def article_shell(key, raw, units, cluster_members=None):
    row = raw.loc[key]
    published = None if pd.isna(row.published_utc) else row.published_utc.isoformat()
    return {
        "id": "",
        "record_key": key,
        "title": (str(row.title) if pd.notna(row.title) else "")[:240],
        "published_at": published,
        "available_at": row.available_utc.isoformat(),
        "publication_age_hours": None,
        "collection_delay_hours": None,
        "passages": units,
        "body_chars_omitted": 0,
        "title_chars_omitted": max(0, len(str(row.title) if pd.notna(row.title) else "") - 240),
        "cluster_members": cluster_members or [key],
    }


def finalize_times(article, cutoff):
    available = pd.Timestamp(article["available_at"])
    published = pd.Timestamp(article["published_at"]) if article["published_at"] else None
    article["publication_age_hours"] = round((cutoff - published).total_seconds() / 3600, 6) if published is not None else None
    article["collection_delay_hours"] = round((available - published).total_seconds() / 3600, 6) if published is not None else None


def pack(row, keys, raw, bodies, tokenizer, ceiling, clusters=None):
    result = copy.deepcopy(row)
    result["news"] = []
    cutoff = pd.Timestamp(row["cutoff"])
    if clusters is None:
        clusters = [[key] for key in keys]
    by_rep = {members[-1]: members for members in clusters}
    candidates = []
    for key in keys:
        units = paragraph_spans(bodies[key], row["symbol"])
        article = article_shell(key, raw, [], by_rep.get(key, [key]))
        article["body_chars_omitted"] = len(bodies[key])
        finalize_times(article, cutoff)
        candidates.append((article, units))
    # Include the same article metadata first. Passage text is the only greedy part.
    for article, _ in candidates:
        result["news"].append(article)
    for i, article in enumerate(result["news"], 1):
        article["id"] = f"N{i}"
    reports = sum(len(members) for members in clusters)
    result["news_summary"] = {
        "eligible_articles": reports,
        "included_articles": len(keys),
        "omitted_articles": reports - len(keys),
    }
    if prompt_tokens(tokenizer, result) > ceiling:
        raise ValueError("metadata alone exceeds token ceiling")
    for index, (article, units) in enumerate(candidates):
        for unit in units:
            trial = copy.deepcopy(result)
            trial["news"][index]["passages"].append(unit)
            trial["news"][index]["body_chars_omitted"] = len(bodies[article["record_key"]]) - sum(
                len(value["text"]) for value in trial["news"][index]["passages"]
            )
            if prompt_tokens(tokenizer, trial) <= ceiling:
                result = trial
    for article, (_, units) in zip(result["news"], candidates):
        included = sum(len(unit["text"]) for unit in article["passages"])
        article["body_chars_omitted"] = len(bodies[article["record_key"]]) - included
        article["eligible_target_units"] = len(units)
        article["included_target_units"] = len(article["passages"])
    return result


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    old_manifest = json.loads((OLD_INPUTS / "manifest.json").read_text())
    if sha(OLD_INPUTS / "inputs.jsonl") != old_manifest["input_sha"]:
        raise ValueError("old direct input fingerprint mismatch")
    p0 = jsonl(OLD_INPUTS / "inputs.jsonl")
    if len(p0) != 1607:
        raise ValueError("wrong P0 row count")
    data = pd.read_pickle(INTEGRATED / "prepared/data.pkl").copy()
    data["key"] = data.symbol + "|" + pd.to_datetime(data.start_utc, utc=True).astype(str)
    labels = data[["key", "symbol", "split", "day", "start_utc", "end_utc", "cutoff_utc", "label", "has_news", "news_count"]].copy()
    order = [row["key"] for row in p0]
    labels = labels.set_index("key").loc[order].reset_index()
    labels.to_pickle(out / "labels.pkl")
    meta = labels.drop(columns=["label"]).copy()
    meta.to_pickle(out / "rows.pkl")
    if order != labels.key.tolist():
        raise ValueError("P0 key order differs from fixed rows")

    raw = pd.read_pickle(W / "audit/news_index.pkl")
    raw["key"] = raw.archive + "::" + raw.member
    raw = raw.set_index("key")
    all_keys = sorted({article["record_key"] for row in p0 for article in row["news"]})
    bodies = {}
    archive_hashes = {}
    for archive, group in raw.loc[all_keys].groupby("archive"):
        path = W / "raw/news" / archive
        archive_hashes[str(path)] = sha(path)
        with zipfile.ZipFile(path) as bundle:
            for record in group.itertuples():
                bodies[record.Index] = json.loads(bundle.read(record.member)).get("text", "") or ""

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    outputs = {variant: [] for variant in VARIANTS}
    unit_records, cluster_records, token_records = [], [], []
    for number, row in enumerate(p0):
        p0_count = prompt_tokens(tokenizer, row)
        outputs["P0"].append(row)
        keys = [article["record_key"] for article in row["news"]]
        p1 = pack(row, keys, raw, bodies, tokenizer, p0_count)
        p2 = pack(row, keys, raw, bodies, tokenizer, 6000)
        clusters, representatives = online_clusters(keys, raw) if keys else ([], [])
        # Reorder each group so representative is last, matching pack's mapping.
        normalized_clusters = [sorted(group, key=lambda key: (raw.loc[key, "available_utc"], key)) for group in clusters]
        p3 = pack(row, representatives, raw, bodies, tokenizer, 6000, normalized_clusters)
        for variant, item in (("P1", p1), ("P2", p2), ("P3", p3)):
            count = prompt_tokens(tokenizer, item)
            if count > (p0_count if variant == "P1" else 6000):
                raise AssertionError("token limit")
            outputs[variant].append(item)
            token_records.append({"key": row["key"], "symbol": row["symbol"], "variant": variant, "prompt_tokens": count, "P0_prompt_tokens": p0_count, "articles": len(item["news"]), "passages": sum(len(a["passages"]) for a in item["news"])})
            for article in item["news"]:
                body = bodies[article["record_key"]]
                for passage in article["passages"]:
                    if body[passage["start"]:passage["end"]] != passage["text"]:
                        raise AssertionError("raw span mismatch")
                    if not ALIASES[row["symbol"]].search(passage["text"]):
                        raise AssertionError("selected unit lacks explicit target")
                    unit_records.append({"key": row["key"], "symbol": row["symbol"], "variant": variant, "record_key": article["record_key"], "start": passage["start"], "end": passage["end"], "unit": passage["unit"], "text_sha256": text_sha(passage["text"]), "chars": len(passage["text"])})
        for index, group in enumerate(normalized_clusters):
            cluster_records.append({"key": row["key"], "symbol": row["symbol"], "cluster": index, "members": "|".join(group), "reports": len(group), "representative": group[-1]})
        if (number + 1) % 200 == 0:
            print("PACKED", number + 1, "/ 1607", flush=True)

    for variant, rows in outputs.items():
        path = out / f"{variant}.jsonl"
        with path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(unit_records).to_csv(out / "units.csv", index=False)
    pd.DataFrame(cluster_records).to_csv(out / "clusters.csv", index=False)
    pd.DataFrame(token_records).to_csv(out / "tokens.csv", index=False)
    manifest = {
        "status": "SEALED",
        "rows": 1607,
        "variants": list(VARIANTS),
        "system_prompt_sha256": text_sha(SYSTEM),
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "model_revision": json.loads((W / "model_compare/comparison_v1/manifest.json").read_text())["revision"],
        "input_sha256": {variant: sha(out / f"{variant}.jsonl") for variant in VARIANTS},
        "labels_sha256": sha(out / "labels.pkl"),
        "rows_sha256": sha(out / "rows.pkl"),
        "units_sha256": sha(out / "units.csv"),
        "clusters_sha256": sha(out / "clusters.csv"),
        "tokens_sha256": sha(out / "tokens.csv"),
        "old_input_sha256": old_manifest["input_sha"],
        "archive_hashes": archive_hashes,
        "code": {path.name: sha(path) for path in (Path(__file__), B / "common.py", B / "PROTOCOL.md")},
        "labels_absent_from_prompt_files": True,
    }
    dump(out / "manifest.json", manifest)
    print(json.dumps({"status": "SEALED", "rows": 1607, "input_sha256": manifest["input_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.out)
