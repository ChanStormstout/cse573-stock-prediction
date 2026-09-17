"""Run rules, frozen FinBERT heads, and the selected adapter on fixed-window news."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from adapter import ACTIONS, EncodedDataset, EventAdapter, TYPES, article_examples, infer, load_trainable, tokenizer
from common import PARAGRAPHS, PUBLIC, SEEDS, dump, jsonl, sha, write_jsonl
from events import RATING_PATTERN, TARGET_PATTERN, extract_rating_values, extract_target_price_values, merge_compatible_events, rule_events, validate_numeric_event

CANDIDATE = re.compile(r"price\s+target|target\s+price|price\s+objective|\brating\b|\brated\b|coverage|upgrad|downgrad|reiterat|reaffirm|maintain|\braised\b|\blowered\b|\bboosted\b|\bcut\b|\bset\b.*?target", re.I)
ALIASES = {
    "AAPL": re.compile(r"\b(?:Apple(?:\s+Inc\.?)?|AAPL)\b", re.I),
    "AMZN": re.compile(r"\b(?:Amazon(?:\.com)?(?:\s+Inc\.?)?|AMZN)\b", re.I),
}


def split_sentences(text: str) -> list[str]:
    boundaries = list(re.finditer(r"[.!?][\"'”’)]*(?=\s+[A-Z0-9\"'])", text))
    result, cursor = [], 0
    for match in boundaries:
        end = match.end()
        value = text[cursor:end].strip()
        if value:
            result.append(value)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    tail = text[cursor:].strip()
    if tail:
        result.append(tail)
    return result


def corpus_articles() -> tuple[list[dict], dict]:
    records = {}
    occurrence = collections.Counter()
    for window in jsonl(PARAGRAPHS / "P2.jsonl"):
        for article in window["news"]:
            key = (window["symbol"], article["record_key"])
            occurrence[key] += 1
            item = records.setdefault(
                key,
                {
                    "id": hashlib.sha256((window["symbol"] + "|" + article["record_key"]).encode()).hexdigest()[:24],
                    "symbol": window["symbol"],
                    "title": article["title"],
                    "record_key": article["record_key"],
                    "available_utc": article["available_at"],
                    "event_group": article["record_key"],
                    "passages": {},
                },
            )
            for passage in article["passages"]:
                item["passages"][(passage["start"], passage["end"])] = passage["text"]
    articles = []
    candidate_count = 0
    for item in records.values():
        sentences = collections.OrderedDict()
        seen = set()
        for passage_number, (_, text) in enumerate(sorted(item.pop("passages").items())):
            for sentence_number, sentence in enumerate(split_sentences(text)):
                normalized = " ".join(sentence.lower().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                sentences[f"P{passage_number}S{sentence_number}"] = sentence
        title_id = "T0"
        combined = collections.OrderedDict([(title_id, item["title"]), *sentences.items()])
        candidate_ids = [key for key, value in combined.items() if CANDIDATE.search(value) and ALIASES[item["symbol"]].search(value)]
        if not candidate_ids:
            continue
        candidate_count += len(candidate_ids)
        articles.append({**item, "sentences": combined, "candidate_ids": candidate_ids})
    stats = {
        "unique_article_target_pairs": len(records),
        "retrieved_article_target_pairs": len(articles),
        "retrieved_candidate_sentences": candidate_count,
        "window_occurrences": int(sum(occurrence.values())),
    }
    return articles, stats


def target_event(action: str, sentence_id: str, sentence: str, symbol: str):
    old, new = extract_target_price_values(action, sentence, symbol)
    event = {"kind": "target_price", "action": action, "old": old, "new": new, "unit": "USD", "evidence_ids": [sentence_id]}
    event = validate_numeric_event(event, sentence)
    return event if event["numeric_valid"] and event["new"] is not None else None


def facts_from_predictions(articles: list[dict], predictions: list[dict], threshold: float):
    source = {row["id"]: row for row in articles}
    by_article = collections.defaultdict(list)
    for row in predictions:
        by_article[row["article_id"]].append(row)
    facts = []
    for article_id, rows in by_article.items():
        article = source[article_id]
        events, seen = [], set()
        for row in rows:
            sentence = article["sentences"][row["sentence_id"]]
            if not ALIASES[article["symbol"]].search(sentence):
                continue
            for type_index, kind in enumerate(TYPES):
                if row["evidence_probability"][type_index] < threshold:
                    continue
                action = ACTIONS[int(np.argmax(row["action_probability"][type_index]))]
                if kind == "rating":
                    if action == "unknown" or not RATING_PATTERN.search(sentence):
                        continue
                    old, new = extract_rating_values(action, sentence, article["symbol"])
                    event = {"kind": kind, "action": action, "old": old, "new": new, "unit": "rating", "evidence_ids": [row["sentence_id"]], "numeric_valid": True, "target_change": None}
                else:
                    if not TARGET_PATTERN.search(sentence):
                        continue
                    event = target_event(action, row["sentence_id"], sentence, article["symbol"])
                    if event is None:
                        continue
                signature = (event["kind"], event["action"], event.get("old"), event.get("new"), row["sentence_id"])
                if signature in seen:
                    continue
                seen.add(signature)
                event["evidence_probability"] = float(row["evidence_probability"][type_index])
                events.append(event)
        events = merge_compatible_events(events)
        if events:
            facts.append({"article_id": article_id, "symbol": article["symbol"], "record_key": article["record_key"], "available_utc": article["available_utc"], "event_group": article["event_group"], "events": events})
    return facts


def rule_facts(articles: list[dict]):
    facts = []
    for article in articles:
        events, seen = [], set()
        for sentence_id in article["candidate_ids"]:
            sentence = article["sentences"][sentence_id]
            for event in rule_events(article["symbol"], sentence_id, sentence):
                signature = (event["kind"], event["action"], event.get("old"), event.get("new"), sentence_id)
                if signature not in seen:
                    seen.add(signature)
                    events.append(event)
        events = merge_compatible_events(events)
        if events:
            facts.append({"article_id": article["id"], "symbol": article["symbol"], "record_key": article["record_key"], "available_utc": article["available_utc"], "event_group": article["event_group"], "events": events})
    return facts


def main(adapter_run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    selection = json.loads((adapter_run / "selection.json").read_text())
    articles, corpus_stats = corpus_articles()
    examples = article_examples(articles, labels=None)
    tok = tokenizer()
    dataset = EncodedDataset(examples, tok)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    configs = sorted({"A0", selection["selected_config"]})
    all_predictions = []
    runtimes = []
    for config in configs:
        for seed in SEEDS:
            checkpoint = torch.load(adapter_run / "checkpoints" / f"{config}_seed{seed}.pt", map_location="cpu", weights_only=True)
            model = EventAdapter(config).to(device)
            load_trainable(model, checkpoint["state"])
            started = time.time()
            rows = infer(model, dataset, tok, device, batch_size=32)
            runtimes.append({"config": config, "seed": seed, "seconds": time.time() - started, "sentences": len(rows)})
            for row in rows:
                row.update(config=config, seed=seed)
                all_predictions.append(row)
            print(json.dumps(runtimes[-1]), flush=True)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
    write_jsonl(out / "sentence_probabilities.jsonl", all_predictions)

    facts = {"rules": rule_facts(articles)}
    for config in configs:
        threshold = selection["selected_thresholds"][config]
        config_rows = [row for row in all_predictions if row["config"] == config]
        for seed in SEEDS:
            facts[f"{config}_seed{seed}"] = facts_from_predictions(articles, [row for row in config_rows if row["seed"] == seed], threshold)
        lookup = collections.defaultdict(list)
        for row in config_rows:
            lookup[(row["article_id"], row.get("model_unit_id", row["sentence_id"]))].append(row)
        ensemble = []
        for key, group in lookup.items():
            exemplar = group[0]
            ensemble.append({**exemplar, "seed": "ensemble", "evidence_probability": np.mean([r["evidence_probability"] for r in group], axis=0).tolist(), "action_probability": np.mean([r["action_probability"] for r in group], axis=0).tolist()})
        facts[f"{config}_ensemble"] = facts_from_predictions(articles, ensemble, threshold)
    for name, rows in facts.items():
        write_jsonl(out / f"facts_{name}.jsonl", rows)
    summary = {
        "status": "COMPLETE",
        "corpus": corpus_stats,
        "candidate_input": "TARGET + title + protected complete candidate sentence + all numbered complete sentences from the target-company passages; only context token truncation at 512",
        "candidate_retrieval": CANDIDATE.pattern,
        "configs": configs,
        "selected_config": selection["selected_config"],
        "thresholds": {name: selection["selected_thresholds"][name] for name in configs},
        "fact_articles": {name: len(rows) for name, rows in facts.items()},
        "fact_counts": {name: sum(len(row["events"]) for row in rows) for name, rows in facts.items()},
        "runtimes": runtimes,
        "source_hashes": {"P2": sha(PARAGRAPHS / "P2.jsonl"), "selection": sha(adapter_run / "selection.json")},
    }
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.adapter_run, args.out)
