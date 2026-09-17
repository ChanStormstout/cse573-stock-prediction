"""Evaluate adapter facts under the same exact signature used by prior extractors."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd

from common import ANNOTATION, SEEDS, jsonl, write_jsonl
from extract_corpus import facts_from_predictions, rule_facts


FIELDS = {"kind", "action", "old", "new", "unit", "evidence_ids"}


def normalized(value):
    if value is None:
        return None
    try:
        return str(float(str(value).replace(",", "")))
    except ValueError:
        return " ".join(str(value).lower().split())


def signature(event):
    return (event["kind"], event["action"], normalized(event.get("old")), normalized(event.get("new")), event["unit"])


def literal_valid(events: list[dict], article: dict) -> bool:
    for event in events:
        if set(event) != FIELDS or event["kind"] not in {"rating", "target_price"} or event["action"] not in {"raise", "lower", "maintain", "initiate", "unknown"}:
            return False
        if event["unit"] != ("USD" if event["kind"] == "target_price" else "rating"):
            return False
        ids = event["evidence_ids"]
        if not isinstance(ids, list) or not ids or any(sentence_id not in article["sentences"] for sentence_id in ids):
            return False
        evidence = " ".join(article["sentences"][sentence_id] for sentence_id in ids)
        for name in ("old", "new"):
            value = event.get(name)
            if value is not None and not re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", evidence, re.I):
                return False
    return True


def clean_events(events: list[dict]) -> list[dict]:
    return [{name: event.get(name) for name in ("kind", "action", "old", "new", "unit", "evidence_ids")} for event in events]


def metrics(predictions: dict, labels: list[dict], symbol: str) -> dict:
    selected = [row for row in labels if symbol == "ALL" or row["symbol"] == symbol]
    tp = fp = fn = exact = valid = no_event = no_event_fp = 0
    gold_groups, predicted_groups = set(), set()
    gold_group_facts, predicted_group_facts = set(), set()
    for gold in selected:
        prediction = predictions[gold["id"]]
        gold_set = {signature(event) for event in gold["answer"]["events"]}
        pred_set = {signature(event) for event in prediction["events"]} if prediction["valid"] else set()
        group = (gold["symbol"], gold["event_group"])
        if gold_set:
            gold_groups.add(group)
            gold_group_facts.update((group, value) for value in gold_set)
        if pred_set:
            predicted_groups.add(group)
            predicted_group_facts.update((group, value) for value in pred_set)
        valid += int(prediction["valid"])
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        exact += int(prediction["valid"] and pred_set == gold_set)
        if not gold_set:
            no_event += 1
            no_event_fp += int(bool(pred_set))
    denominator = 2 * tp + fp + fn
    return {
        "symbol": symbol,
        "n": len(selected),
        "schema_and_literal_evidence_valid": valid,
        "exact_fact_set": exact,
        "fact_tp": tp,
        "fact_fp": fp,
        "fact_fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / denominator if denominator else None,
        "no_event_n": no_event,
        "no_event_false_positive": no_event_fp,
        "gold_nonduplicate_event_groups": len(gold_groups),
        "predicted_nonduplicate_event_groups": len(predicted_groups),
        "gold_nonduplicate_group_facts": len(gold_group_facts),
        "predicted_nonduplicate_group_facts": len(predicted_group_facts),
    }


def main(adapter_run: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    inputs = jsonl(ANNOTATION / "inputs.jsonl")
    labels = jsonl(ANNOTATION / "labels.jsonl")
    articles = [{**row, "event_group": row["group"], "candidate_ids": list(row["sentences"])} for row in inputs]
    article_lookup = {row["id"]: row for row in articles}
    selection = json.loads((adapter_run / "selection.json").read_text())
    sentence_rows = jsonl(adapter_run / "evaluation_sentence_predictions.jsonl")
    ensemble_rows = jsonl(adapter_run / "ensemble_sentence_predictions.jsonl")
    output_rows, metric_rows = [], []
    def evaluate_facts(config, seed, threshold, facts):
        by_article = {row["article_id"]: clean_events(row["events"]) for row in facts}
        predictions = {}
        for article in articles:
            events = by_article.get(article["id"], [])
            valid = literal_valid(events, article)
            predictions[article["id"]] = {"events": events, "valid": valid}
            if article["split"] in {"valid", "check"}:
                output_rows.append({"id": article["id"], "split": article["split"], "symbol": article["symbol"], "config": config, "seed": seed, "valid": valid, "events": events})
        for split in ("valid", "check"):
            split_labels = [row for row in labels if row["split"] == split]
            for symbol in ("ALL", "AAPL", "AMZN"):
                if not any(symbol == "ALL" or row["symbol"] == symbol for row in split_labels):
                    continue
                metric_rows.append({"config": config, "seed": seed, "split": "development" if split == "valid" else split, "threshold": threshold, **metrics(predictions, split_labels, symbol)})

    evaluate_facts("RULES_CURRENT", "deterministic", None, rule_facts(articles))
    for config in ("A0", "A1", "A2"):
        threshold = selection["selected_thresholds"][config]
        variants = [(str(seed), [row for row in sentence_rows if row["config"] == config and row["seed"] == seed]) for seed in SEEDS]
        variants.append(("ensemble", [row for row in ensemble_rows if row["config"] == config]))
        for seed, rows in variants:
            facts = facts_from_predictions(articles, rows, threshold)
            evaluate_facts(config, seed, threshold, facts)
    write_jsonl(out / "exact_predictions.jsonl", output_rows)
    pd.DataFrame(metric_rows).to_csv(out / "exact_metrics.csv", index=False)
    print(json.dumps({"status": "COMPLETE", "metric_rows": len(metric_rows), "prediction_rows": len(output_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.adapter_run, args.out)
