"""Audit deterministic old/new extraction on explicit-target provisional evidence."""
from __future__ import annotations

import json

from common import ANNOTATION, PUBLIC, dump, jsonl, sha
from events import company_alias, extract_rating_values, extract_target_price_values


def normalize(value):
    if value is None:
        return None
    return " ".join(str(value).lower().replace(",", "").replace("-", " ").split())


def main() -> None:
    inputs = {row["id"]: row for row in jsonl(ANNOTATION / "inputs.jsonl")}
    labels = jsonl(ANNOTATION / "labels.jsonl")
    rows = []
    for label in labels:
        source = inputs[label["id"]]
        for event in label["answer"]["events"]:
            evidence = " ".join(source["sentences"][sentence_id] for sentence_id in event["evidence_ids"])
            explicit_target = bool(company_alias(label["symbol"]).search(evidence))
            if event["kind"] == "rating":
                predicted = extract_rating_values(event["action"], evidence, label["symbol"])
            else:
                predicted = extract_target_price_values(event["action"], evidence, label["symbol"])
            gold = (event.get("old"), event.get("new"))
            exact = tuple(map(normalize, predicted)) == tuple(map(normalize, gold))
            rows.append(
                {
                    "id": label["id"],
                    "symbol": label["symbol"],
                    "split": label["split"],
                    "kind": event["kind"],
                    "explicit_target": explicit_target,
                    "exact_old_new": exact,
                }
            )
    summary = {}
    for kind in ("rating", "target_price"):
        selected = [row for row in rows if row["kind"] == kind and row["explicit_target"]]
        summary[kind] = {
            "explicit_target_facts": len(selected),
            "exact_old_new": sum(row["exact_old_new"] for row in selected),
            "failed_ids": sorted({row["id"] for row in selected if not row["exact_old_new"]}),
        }
    by_split_kind = {}
    for split in ("train", "valid", "check"):
        for kind in ("rating", "target_price"):
            selected = [row for row in rows if row["split"] == split and row["kind"] == kind and row["explicit_target"]]
            by_split_kind[f"{split}|{kind}"] = {
                "explicit_target_facts": len(selected),
                "exact_old_new": sum(row["exact_old_new"] for row in selected),
            }
    result = {
        "status": "PASS_COMPONENT_DIAGNOSTIC_WITH_RECORDED_FAILURES",
        "scope": "deterministic_value_copy_on_model_provisional_evidence_not_independent_gold",
        "summary": summary,
        "by_split_kind": by_split_kind,
        "development_status": "The deterministic parser was developed against known provisional examples across this exposed panel; these counts are regression diagnostics, not held-out generalization estimates.",
        "all_facts": len(rows),
        "facts_without_explicit_target_in_combined_evidence": sum(not row["explicit_target"] for row in rows),
        "source_hashes": {
            "inputs": sha(ANNOTATION / "inputs.jsonl"),
            "labels": sha(ANNOTATION / "labels.jsonl"),
        },
    }
    dump(PUBLIC / "value_extraction_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
