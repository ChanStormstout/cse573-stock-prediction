"""Fail closed if the public event-adapter run violates its registered protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ANNOTATION, PUBLIC, sha


REQUIRED = (
    "REPORT.md",
    "DATA_AUDIT.md",
    "data_audit.json",
    "CASE_NOTES.md",
    "PROTOCOL_AMENDMENT.md",
    "protocol.json",
    "input_length_audit.json",
    "cache_equivalence.json",
    "value_extraction_audit.json",
    "metrics.csv",
    "predictions.csv",
    "training_evidence.json",
    "training_summary.csv",
    "adapter_selection.json",
    "residual_selection.json",
    "adapter_extraction_metrics.csv",
    "adapter_exact_metrics.csv",
    "extraction_comparison.csv",
    "monthly_metrics.csv",
    "subgroup_metrics.csv",
    "transitions.csv",
    "paired_intervals.csv",
    "seed_metrics.csv",
    "seed_variation.csv",
    "summary.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main(root: Path) -> None:
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    require(not missing, f"missing public artifacts: {missing}")

    protocol = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "data_audit.json").read_text())
    adapter = json.loads((root / "adapter_selection.json").read_text())
    residual = json.loads((root / "residual_selection.json").read_text())
    training = json.loads((root / "training_evidence.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    input_audit = json.loads((root / "input_length_audit.json").read_text())
    cache_audit = json.loads((root / "cache_equivalence.json").read_text())
    value_audit = json.loads((root / "value_extraction_audit.json").read_text())

    require(protocol["saved_before_new_results"] is True, "protocol was not marked pre-registered")
    require(protocol["target"] == "AAPL_AMZN_four_hour_direction", "wrong target")
    require(protocol["development_and_later_pooled_for_selection"] is False, "evaluation periods entered selection")
    require(protocol["per_stock_later_winner_selection"] is False, "per-stock later selection allowed")
    require(protocol["independent_human_review_passed"] is False, "human review status is overstated")
    require(input_audit["status"] == "PASS" and input_audit["candidate_protected"] is True, "input length audit failed")
    require(all(row["units_over_512"] == 0 for row in input_audit["sets"]), "an encoded unit exceeds FinBERT length")
    require(cache_audit["status"] == "PASS" and cache_audit["saved_before_official_A1_A2_results"] is True, "cache equivalence audit failed")
    require(max(cache_audit["maximum_absolute_probability_difference"].values()) <= cache_audit["registered_cache_tolerance"], "cache quantization exceeds tolerance")
    require(value_audit["source_hashes"]["inputs"] == audit["hashes"]["inputs"] and value_audit["source_hashes"]["labels"] == audit["hashes"]["labels"], "value audit panel drift")
    require(value_audit["summary"]["rating"]["exact_old_new"] == value_audit["summary"]["rating"]["explicit_target_facts"], "rating value-copy regression")

    require(audit["rows"] == 458, "annotation row count drift")
    require(audit["split_counts"] == {"check": 122, "valid": 87, "train": 249}, "annotation split drift")
    require(audit["positive_article_counts"]["train|AAPL"] == 47, "AAPL train-positive drift")
    require(audit["positive_article_counts"]["train|AMZN"] == 13, "AMZN train-positive drift")
    require(audit["event_counts"]["train|AAPL"] == 57, "AAPL train-fact drift")
    require(audit["event_counts"]["train|AMZN"] == 18, "AMZN train-fact drift")
    require(audit["event_fields"] == ["action", "evidence_ids", "kind", "new", "old", "unit"], "invented or missing event fields")
    require(audit["grouping"]["cross_split_components"] == 0, "known duplicate component crosses splits")
    require(audit["grouping"]["near_duplicate_pairs_crossing_train_forward_segments"] == 0, "near duplicate crosses a train-forward segment")
    require(audit["mapping"]["failures"] == 0, "source mapping failure")
    require(audit["hashes"]["inputs"] == sha(ANNOTATION / "inputs.jsonl"), "annotation inputs changed")
    require(audit["hashes"]["labels"] == sha(ANNOTATION / "labels.jsonl"), "annotation labels changed")
    require(audit["hashes"]["protocol"] == sha(root / "protocol.json"), "audit predates current protocol")

    require(adapter["selection_source"] == "January-February chronological forward folds only", "adapter selection source drift")
    require(adapter["development_or_check_used_for_selection"] is False, "adapter used evaluation labels")
    require(residual["selection_period"] == "March-August train-forward OOF only", "residual selection source drift")
    require(residual["development_and_later_used"] is False, "residual used development/later")
    require(training["status"] == "COMPLETE", "training not complete")
    require(training["runs"] == 36, "expected 27 forward folds and 9 full fits")
    require(training["maximum_reload_difference"] <= training["checkpoint_reload_tolerance"], "checkpoint reload mismatch")
    require(set(training["trainable_parameters"]) == {"A0", "A1", "A2"}, "missing adapter config")
    source = root.parent
    for name, expected_hash in training["code_hashes"].items():
        require(expected_hash == sha(source / name), f"training source changed after execution: {name}")
    require(training["protocol_amendment_sha256"] == sha(root / "PROTOCOL_AMENDMENT.md"), "training predates current protocol amendment")
    require(training["cache_equivalence_sha256"] == sha(root / "cache_equivalence.json"), "training cache audit mismatch")
    require(bool(training["resume_provenance_sha256"]), "missing resume provenance fingerprint")
    require(training["source_hashes"]["inputs"] == audit["hashes"]["inputs"], "training input fingerprint mismatch")
    require(training["source_hashes"]["labels"] == audit["hashes"]["labels"], "training label fingerprint mismatch")
    require(summary["independent_human_review_passed"] is False, "summary overstates human review")
    require(summary["exact_fallback_verified"] is True, "downstream did not verify fallback")
    training_summary = pd.read_csv(root / "training_summary.csv")
    forward = training_summary[training_summary.stage.eq("forward_fold")]
    full = training_summary[training_summary.stage.eq("full_train")]
    require(len(forward) == 27 and not forward.duplicated(["config", "seed", "fold"]).any(), "forward-fit identity mismatch")
    require(len(full) == 9 and not full.duplicated(["config", "seed"]).any(), "full-fit identity mismatch")

    predictions = pd.read_csv(root / "predictions.csv")
    methods = ["F0", "R1", "F1", "F2", "F6", "D0", "D1", "D2", "D3", "D4"]
    gates = ["D2_event_gate", "D3_event_gate", "D4_event_gate"]
    require(len(predictions) == 1374, "prediction row count drift")
    require(not predictions.duplicated(["key", "phase"]).any(), "duplicate prediction key/phase")
    require(predictions.phase.value_counts().to_dict() == {"train_forward_oof": 765, "later": 357, "development": 252}, "phase row count drift")
    require(set(methods + gates).issubset(predictions.columns), "missing probability or gate column")
    values = predictions[methods].to_numpy(float)
    require(np.isfinite(values).all(), "non-finite probability")
    require(((values >= 0) & (values <= 1)).all(), "probability outside [0,1]")
    no_news = predictions.has_news.eq(0).to_numpy()
    require(np.array_equal(predictions.loc[no_news, "F1"].to_numpy(), predictions.loc[no_news, "R1"].to_numpy()), "F1 did not use exact R1 fallback on a no-news row")
    for method in ("D2", "D3", "D4"):
        mask = predictions[f"{method}_event_gate"].eq(0).to_numpy()
        require(np.array_equal(predictions.loc[mask, method].to_numpy(), predictions.loc[mask, "F1"].to_numpy()), f"{method} changed a g=0 row")

    metrics = pd.read_csv(root / "metrics.csv")
    expected = {(phase, symbol, method) for phase in ("train_forward_oof", "development", "later") for symbol in ("AAPL", "AMZN") for method in methods}
    observed = set(metrics[["phase", "symbol", "method"]].itertuples(index=False, name=None))
    require(observed == expected, "primary result table is incomplete or duplicated")
    require(metrics[["BA", "MCC", "Brier"]].notna().all().all(), "missing primary metric")

    extraction = pd.read_csv(root / "adapter_extraction_metrics.csv")
    ensemble_check = extraction[(extraction.seed.astype(str) == "ensemble") & extraction.split.eq("check")]
    require(set(ensemble_check.config) == {"A0", "A1", "A2"}, "missing check adapter")
    require(set(ensemble_check.symbol) == {"ALL", "AAPL", "AMZN"}, "missing per-stock extraction metric")
    exact = pd.read_csv(root / "adapter_exact_metrics.csv")
    exact_check = exact[(exact.seed.astype(str) == "ensemble") & exact.split.eq("check")]
    require(set(exact_check.config) == {"A0", "A1", "A2"}, "missing comparable exact adapter metric")
    require(set(exact_check.symbol) == {"ALL", "AAPL", "AMZN"}, "missing per-stock exact adapter metric")
    require({"gold_nonduplicate_event_groups", "predicted_nonduplicate_event_groups", "gold_nonduplicate_group_facts", "predicted_nonduplicate_group_facts"}.issubset(exact.columns), "missing nonduplicate extraction counts")
    comparison = pd.read_csv(root / "extraction_comparison.csv")
    require(comparison.metric_family.eq("exact_full_fact_signature").all(), "incomparable extraction metrics were pooled")

    seed_metrics = pd.read_csv(root / "seed_metrics.csv")
    require(set(seed_metrics.seed.astype(int)) == {573, 574, 575}, "seed result drift")
    require(set(seed_metrics.method) == {"D3", "D4"}, "unexpected seed methods")
    require(len(seed_metrics) == 36, "seed metric row count drift")

    forbidden_suffixes = {".pt", ".bin", ".safetensors", ".npy", ".npz", ".jsonl"}
    forbidden = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.suffix in forbidden_suffixes]
    require(not forbidden, f"private cache/raw/model artifact in public run: {forbidden}")

    result = {
        "status": "PASS",
        "public_files": len([path for path in root.rglob("*") if path.is_file()]),
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "checkpoint_reload_max_abs": training["maximum_reload_difference"],
        "selection_uses_evaluation": False,
        "exact_no_event_fallback": True,
        "independent_human_review_passed": False,
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PUBLIC)
    args = parser.parse_args()
    main(args.root)
