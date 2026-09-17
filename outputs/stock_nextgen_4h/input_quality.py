"""Training-only paragraph quality gate; never reads a future outcome to select cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from common import B, W, dump, jsonl, sha
from paragraph_inputs import ALIASES, prompt_tokens
from transformers import AutoTokenizer

MODEL = W / "model_compare/model"


def stable_rank(namespace: str, key: str):
    return hashlib.sha256(f"{namespace}|{key}".encode()).hexdigest()


def is_fragment(passage):
    text = passage["text"].strip()
    if not text:
        return True
    # Character excerpts often begin/end inside a sentence; whole paragraph units are accepted.
    if passage.get("unit") == "paragraph":
        return False
    return not bool(re.search(r"[.!?][\"'”’)]*$", text))


def main(inputs: Path):
    products = [inputs / "quality.json", inputs / "quality_cases.csv", inputs / "QUALITY_CARDS.md"]
    if any(path.exists() for path in products):
        raise FileExistsError("quality outputs already exist; use a new paragraph-input directory")
    manifest = json.loads((inputs / "manifest.json").read_text())
    for variant, fingerprint in manifest["input_sha256"].items():
        if sha(inputs / f"{variant}.jsonl") != fingerprint:
            raise ValueError("sealed paragraph input changed")
    labels = pd.read_pickle(inputs / "labels.pkl")
    train_keys = set(labels.loc[labels.split.eq("train"), "key"])
    variants = {name: {row["key"]: row for row in jsonl(inputs / f"{name}.jsonl")} for name in ("P0", "P1", "P2", "P3")}
    raw = pd.read_pickle(W / "audit/news_index.pkl")
    raw["key"] = raw.archive + "::" + raw.member
    raw = raw.set_index("key")
    needed = sorted({article["record_key"] for key in train_keys for article in variants["P2"][key]["news"]})
    bodies = {}
    for archive, group in raw.loc[needed].groupby("archive"):
        with zipfile.ZipFile(W / "raw/news" / archive) as bundle:
            for record in group.itertuples():
                bodies[record.Index] = json.loads(bundle.read(record.member)).get("text", "") or ""

    rows = []
    row_tags = {}
    duplicate_keys = set(pd.read_csv(inputs / "clusters.csv").query("reports > 1").key)
    for key in sorted(train_keys):
        item = variants["P2"][key]
        text = " ".join(
            [article["title"] + " " + " ".join(passage["text"] for passage in article["passages"]) for article in item["news"]]
        ).lower()
        count = item["news_summary"]["eligible_articles"]
        tags = {
            "no_news" if count == 0 else "few_news" if count <= 6 else "many_news",
        }
        if re.search(r"\b(?:apple|aapl)\b", text) and re.search(r"\b(?:amazon|amzn)\b", text):
            tags.add("multi_company")
        if re.search(r"\b(?:last|previous|formerly|history|historical|year ago|quarter ago)\b", text):
            tags.add("background")
        if re.search(r"\b(?:rating|target price|price target|analyst|upgrade|downgrade|buy|sell)\b", text):
            tags.add("rating")
        if re.search(r"\b(?:earnings|revenue|guidance|quarterly|eps)\b", text):
            tags.add("earnings")
        if key in duplicate_keys:
            tags.add("duplicate")
        if any((article.get("collection_delay_hours") or 0) > 24 for article in item["news"]):
            tags.add("delay")
        row_tags[key] = tags

    selected = []
    for symbol in ("AAPL", "AMZN"):
        symbol_keys = [key for key in train_keys if variants["P2"][key]["symbol"] == symbol]
        for tag in ("no_news", "few_news", "many_news", "multi_company", "background", "rating", "earnings", "duplicate", "delay"):
            candidates = [key for key in symbol_keys if tag in row_tags[key]]
            if candidates:
                key = min(candidates, key=lambda value: stable_rank(f"{symbol}|{tag}", value))
                selected.append((symbol, tag, key))
    selected_keys = sorted({key for _, _, key in selected})

    exact, aliases, boundaries, passages = 0, 0, 0, 0
    per_variant = []
    for variant in ("P1", "P2", "P3"):
        for key in train_keys:
            item = variants[variant][key]
            for article in item["news"]:
                body = bodies[article["record_key"]]
                for passage in article["passages"]:
                    passages += 1
                    exact += body[passage["start"] : passage["end"]] == passage["text"]
                    aliases += bool(ALIASES[item["symbol"]].search(passage["text"]))
                    boundaries += passage["unit"] in {"paragraph", "sentence_neighborhood"} and not is_fragment(passage)
        subset = [variants[variant][key] for key in train_keys]
        per_variant.append(
            {
                "variant": variant,
                "rows": len(subset),
                "no_news_rows": sum(not item["news"] for item in subset),
                "zero_passage_rows": sum(sum(len(article["passages"]) for article in item["news"]) == 0 for item in subset),
                "included_passages": sum(sum(len(article["passages"]) for article in item["news"]) for item in subset),
                "eligible_units": sum(sum(article.get("eligible_target_units", 0) for article in item["news"]) for item in subset),
                "included_units": sum(sum(article.get("included_target_units", 0) for article in item["news"]) for item in subset),
            }
        )

    tokens = pd.read_csv(inputs / "tokens.csv")
    train_token_keys = set(train_keys)
    p1_tokens = tokens[(tokens.variant == "P1") & tokens.key.isin(train_token_keys)]
    token_violations = int((p1_tokens.prompt_tokens > p1_tokens.P0_prompt_tokens).sum())
    cluster = pd.read_csv(inputs / "clusters.csv")
    train_cluster = cluster[cluster.key.isin(train_keys)]
    deduplicated_reports = int((train_cluster.reports - 1).clip(lower=0).sum())

    p0_fragments = 0
    p1_fragments = 0
    cards = []
    lines = ["# Blind training-only input quality cards", "", "No labels or future returns are shown in this file.", ""]
    for key in selected_keys:
        p0, p1, p2, p3 = (variants[name][key] for name in ("P0", "P1", "P2", "P3"))
        p0_passages = [passage for article in p0["news"] for passage in article["passages"]]
        p1_passages = [passage for article in p1["news"] for passage in article["passages"]]
        p0_fragment_count = sum(
            not bool(re.search(r"[.!?][\"'”’)]*$", passage["text"].strip())) for passage in p0_passages
        )
        p1_fragment_count = sum(is_fragment(passage) for passage in p1_passages)
        p0_fragments += p0_fragment_count
        p1_fragments += p1_fragment_count
        tags = sorted(row_tags[key])
        card = {
            "key": key,
            "symbol": p0["symbol"],
            "tags": tags,
            "P0_passages": len(p0_passages),
            "P1_passages": len(p1_passages),
            "P2_passages": sum(len(article["passages"]) for article in p2["news"]),
            "P3_articles": len(p3["news"]),
            "P0_fragments": p0_fragment_count,
            "P1_fragments": p1_fragment_count,
        }
        cards.append(card)
        lines.extend([f"## {key}", "", f"Tags: {', '.join(tags)}", "", "### P0 excerpts", ""])
        for passage in p0_passages[:4]:
            lines.append("> " + passage["text"].replace("\n", " ")[:650])
        lines.extend(["", "### P1 target units", ""])
        for passage in p1_passages[:4]:
            lines.append("> " + passage["text"].replace("\n", " ")[:650])
        lines.append("")
    (inputs / "QUALITY_CARDS.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(cards).to_csv(inputs / "quality_cases.csv", index=False)
    mechanical_pass = (
        exact == passages
        and aliases == passages
        and boundaries == passages
        and token_violations == 0
        and p1_fragments < p0_fragments
    )
    result = {
        "status": "MECHANICAL_PASS_MANUAL_REVIEW_REQUIRED" if mechanical_pass else "FAIL",
        "training_only": True,
        "independent_review": False,
        "selected_cases": len(selected_keys),
        "strata_requests": [{"symbol": symbol, "tag": tag, "key": key} for symbol, tag, key in selected],
        "passages_checked": passages,
        "exact_span_rate": exact / passages if passages else 1,
        "explicit_target_rate": aliases / passages if passages else 1,
        "complete_unit_rate": boundaries / passages if passages else 1,
        "P1_token_violations": token_violations,
        "blind_case_P0_fragments": p0_fragments,
        "blind_case_P1_fragments": p1_fragments,
        "deduplicated_training_reports": deduplicated_reports,
        "variant_counts": per_variant,
        "manual_review": "PENDING",
        "gate": bool(mechanical_pass),
    }
    dump(inputs / "quality.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    main(args.inputs)
