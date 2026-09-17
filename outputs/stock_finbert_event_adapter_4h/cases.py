"""Apply the pre-fixed case rules after extraction and downstream results exist."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ACTIONS, ANNOTATION, PARAGRAPHS, PUBLIC, TYPES, dump, jsonl
from evaluate_adapter import signature


def fact_set(rows: list[dict], threshold: float):
    facts = set()
    for row in rows:
        for type_index, kind in enumerate(TYPES):
            if row["evidence_probability"][type_index] >= threshold:
                facts.add((kind, ACTIONS[int(np.argmax(row["action_probability"][type_index]))]))
    return facts


def main(adapter_run: Path, extraction: Path, downstream: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    case_protocol = json.loads((PUBLIC / "case_selection.json").read_text())
    selection = json.loads((adapter_run / "selection.json").read_text())
    selected = selection["selected_config"]
    threshold = selection["selected_thresholds"][selected]
    ensemble = [row for row in jsonl(adapter_run / "ensemble_sentence_predictions.jsonl") if row["config"] == selected]
    ensemble_by_article = collections.defaultdict(list)
    for row in ensemble:
        ensemble_by_article[row["article_id"]].append(row)
    labels = {row["id"]: row for row in jsonl(ANNOTATION / "labels.jsonl")}
    inputs = {row["id"]: row for row in jsonl(ANNOTATION / "inputs.jsonl")}
    deployed_facts = {
        (row["symbol"], row["record_key"]): row["events"]
        for row in jsonl(extraction / f"facts_{selected}_ensemble.jsonl")
    }
    qwen_frozen = {row["id"]: row for row in jsonl(Path("work/stock-data/annotation/runs/frozen_v3/predictions.jsonl"))}
    qwen_tuned = {row["id"]: row for row in jsonl(Path("work/stock-data/annotation/runs/tuned_v3/predictions.jsonl"))}

    article_cases = []
    for case in case_protocol["semantic_cases"]:
        article_id = case.get("article_id")
        if article_id is None:
            continue
        gold = {(event["kind"], event["action"]) for event in labels[article_id]["answer"]["events"]}
        adapter = fact_set(ensemble_by_article[article_id], threshold)
        old = qwen_frozen[article_id]
        tuned = qwen_tuned[article_id]
        old_facts = {(event["kind"], event["action"]) for event in old.get("parsed", {}).get("events", [])} if old.get("valid") else set()
        tuned_facts = {(event["kind"], event["action"]) for event in tuned.get("parsed", {}).get("events", [])} if tuned.get("valid") else set()
        article_cases.append(
            {
                "cohort": case["cohort"],
                "article_id": article_id,
                "symbol": inputs[article_id]["symbol"],
                "gold_type_action": sorted("|".join(value) for value in gold),
                "frozen_qwen_type_action": sorted("|".join(value) for value in old_facts),
                "qlora_qwen_type_action": sorted("|".join(value) for value in tuned_facts),
                "adapter_type_action": sorted("|".join(value) for value in adapter),
                "adapter_exact": adapter == gold,
                "adapter_repairs_frozen_qwen": adapter == gold and old_facts != gold,
            }
        )

    predictions = pd.read_csv(downstream / "predictions.csv")
    p2 = {row["key"]: row for row in jsonl(PARAGRAPHS / "P2.jsonl")}
    p3 = {row["key"]: row for row in jsonl(PARAGRAPHS / "P3.jsonl")}
    id_by_record = {(row["symbol"], row["record_key"]): row["id"] for row in inputs.values()}
    exact_by_record = {}
    for (symbol, record_key), article_id in id_by_record.items():
        if article_id in ensemble_by_article:
            gold = {signature(event) for event in labels[article_id]["answer"]["events"]}
            predicted = {signature(event) for event in deployed_facts.get((symbol, record_key), [])}
            exact_by_record[(symbol, record_key)] = predicted == gold

    outcome_cases = []
    changed_wrong = predictions[((predictions.D4 >= 0.5) != (predictions.D1 >= 0.5)) & ((predictions.D1 >= 0.5) == predictions.label) & ((predictions.D4 >= 0.5) != predictions.label)].sort_values(["day", "symbol", "key"])
    for desired in (True, False):
        chosen = None
        for row in changed_wrong.to_dict("records"):
            article_status = []
            for article in p2[row["key"]]["news"]:
                record = (row["symbol"], article["record_key"])
                article_id = id_by_record.get(record)
                if article_id and (labels[article_id]["answer"]["events"] or deployed_facts.get(record)):
                    article_status.append((article_id, exact_by_record.get(record)))
            matches = [value for value in article_status if value[1] is desired]
            if matches:
                chosen = {"window_key": row["key"], "symbol": row["symbol"], "article_id": matches[0][0], "base_probability": row["D1"], "adapter_probability": row["D4"], "label": int(row["label"]), "adapter_full_fact_exact": desired}
                break
        outcome_cases.append({"cohort": "extraction_correct_prediction_wrong" if desired else "extraction_error_prediction_wrong", "available": chosen is not None, "case": chosen})

    duplicate_key = next(case["window_key"] for case in case_protocol["semantic_cases"] if case["cohort"] == "duplicate_reports")
    duplicate_clusters = [article for article in p3[duplicate_key]["news"] if len(article.get("cluster_members", [])) > 1]
    no_event_key = next(case["window_key"] for case in case_protocol["semantic_cases"] if case["cohort"] == "no_event_exact_fallback")
    fallback = predictions[predictions.key.eq(no_event_key)].iloc[0]
    window_cases = {
        "duplicate_reports": {"window_key": duplicate_key, "duplicate_clusters": len(duplicate_clusters), "reports_in_duplicate_clusters": sum(len(row["cluster_members"]) for row in duplicate_clusters)},
        "no_event_exact_fallback": {"window_key": no_event_key, "D1": float(fallback.D1), "D2": float(fallback.D2), "D3": float(fallback.D3), "D4": float(fallback.D4), "all_exact": bool(fallback.D1 == fallback.D2 == fallback.D3 == fallback.D4)},
    }
    result = {"selected_adapter": selected, "threshold": threshold, "article_cases": article_cases, "window_cases": window_cases, "outcome_cases": outcome_cases, "selection_was_fixed_before_results": True}
    dump(out / "case_results.json", result)

    lines = [
        "# Case notes",
        "",
        "Cases and outcome-selection rules were fixed in `case_selection.json` before adapter and downstream results were revealed. Labels remain model-provisional.",
        "",
        "## Extraction cases",
        "",
        "| Cohort | ID | Gold type/action | Frozen Qwen | QLoRA Qwen | Selected FinBERT | Exact? |",
        "|---|---|---|---|---|---|---:|",
    ]
    for row in article_cases:
        render = lambda values: ", ".join(values) if values else "no-event"
        lines.append(f"| {row['cohort']} | {row['article_id']} | {render(row['gold_type_action'])} | {render(row['frozen_qwen_type_action'])} | {render(row['qlora_qwen_type_action'])} | {render(row['adapter_type_action'])} | {'yes' if row['adapter_exact'] else 'no'} |")
    lines += [
        "",
        "These rows test extraction only. A correct fact does not imply that the next four-hour direction is predictable.",
        "",
        "## Window cases",
        "",
        f"- Duplicate cohort `{duplicate_key}` contains {window_cases['duplicate_reports']['duplicate_clusters']} multi-report clusters ({window_cases['duplicate_reports']['reports_in_duplicate_clusters']} reports). The window feature builder counts each cluster once and keeps report count separately.",
        f"- No-event fallback `{no_event_key}`: D1={fallback.D1:.12f}, D2={fallback.D2:.12f}, D3={fallback.D3:.12f}, D4={fallback.D4:.12f}; exact equality is {'verified' if window_cases['no_event_exact_fallback']['all_exact'] else 'NOT verified'}.",
        "",
        "## Prediction-error cases",
        "",
    ]
    for row in outcome_cases:
        if not row["available"]:
            lines.append(f"- `{row['cohort']}`: no window met the pre-fixed rule; no replacement case was cherry-picked.")
        else:
            value = row["case"]
            lines.append(f"- `{row['cohort']}`: `{value['window_key']}` / article `{value['article_id']}`. The base was correct, D4 changed it to an error. Full provisional fact signature exact={value['adapter_full_fact_exact']}. This separates a market-response failure from an extraction failure.")
    (PUBLIC / "CASE_NOTES.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-run", required=True, type=Path)
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--downstream", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.adapter_run, args.extraction, args.downstream, args.out)
