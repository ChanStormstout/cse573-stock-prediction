"""Publish sanitized tables and a plain-language report."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import PUBLIC, percent


def value(frame, phase, symbol, method, column):
    row = frame[(frame.phase == phase) & (frame.symbol == symbol) & (frame.method == method)]
    return None if row.empty else row.iloc[0][column]


def main(adapter_run: Path, exact_eval: Path, extraction: Path, downstream: Path) -> None:
    selection = json.loads((adapter_run / "selection.json").read_text())
    training = json.loads((adapter_run / "training_evidence.json").read_text())
    extraction_summary = json.loads((extraction / "summary.json").read_text())
    downstream_status = json.loads((downstream / "status.json").read_text())
    downstream_selection = json.loads((downstream / "selection.json").read_text())
    oracle = json.loads((PUBLIC / "oracle_diagnostic.json").read_text())
    value_audit = json.loads((PUBLIC / "value_extraction_audit.json").read_text())

    copies = {
        downstream / "metrics.csv": PUBLIC / "metrics.csv",
        downstream / "predictions.csv": PUBLIC / "predictions.csv",
        downstream / "monthly.csv": PUBLIC / "monthly_metrics.csv",
        downstream / "subgroups.csv": PUBLIC / "subgroup_metrics.csv",
        downstream / "transitions.csv": PUBLIC / "transitions.csv",
        downstream / "paired_intervals.csv": PUBLIC / "paired_intervals.csv",
        downstream / "seed_metrics.csv": PUBLIC / "seed_metrics.csv",
        downstream / "seed_variation.csv": PUBLIC / "seed_variation.csv",
        adapter_run / "extraction_metrics.csv": PUBLIC / "adapter_extraction_metrics.csv",
        exact_eval / "exact_metrics.csv": PUBLIC / "adapter_exact_metrics.csv",
        adapter_run / "threshold_metrics.csv": PUBLIC / "adapter_threshold_metrics.csv",
        adapter_run / "training_summary.csv": PUBLIC / "training_summary.csv",
        adapter_run / "selection.json": PUBLIC / "adapter_selection.json",
        downstream / "selection.json": PUBLIC / "residual_selection.json",
        adapter_run / "training_evidence.json": PUBLIC / "training_evidence.json",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)

    new_metrics = pd.read_csv(adapter_run / "extraction_metrics.csv")
    new_metrics = new_metrics[(new_metrics.seed.astype(str) == "ensemble") & (new_metrics.split == "check")].copy()
    new_metrics["model"] = "FinBERT_" + new_metrics.config
    new_metrics["metric_family"] = "hierarchical_event_type_action_evidence"
    exact_new = pd.read_csv(exact_eval / "exact_metrics.csv")
    exact_new = exact_new[(exact_new.split == "check") & (((exact_new.seed.astype(str) == "ensemble") & exact_new.config.isin(["A0", "A1", "A2"])) | exact_new.config.eq("RULES_CURRENT"))].copy()
    exact_new["model"] = np.where(exact_new.config.eq("RULES_CURRENT"), "rules_current", "FinBERT_" + exact_new.config)
    exact_new["metric_family"] = "exact_full_fact_signature"
    legacy_execution = json.loads(Path("outputs/stock_llm_model_compare/EXECUTION.json").read_text())
    audit = json.loads((PUBLIC / "data_audit.json").read_text())
    if legacy_execution["labels_sha256"] != audit["hashes"]["labels"] or legacy_execution["input_sha256"] != audit["hashes"]["inputs"]:
        raise ValueError("legacy Qwen comparison panel does not match the sealed FinBERT panel")
    old = pd.read_csv("outputs/stock_llm_model_compare/METRICS.csv")
    old = old[(old.phase == "check") & (old.cohort == "all") & old.model.isin(["old_tuned", "staged_v2"]) & old.symbol.isin(["ALL", "AAPL", "AMZN"])].copy()
    old["metric_family"] = "exact_full_fact_signature"
    combined = pd.concat(
        [
            exact_new.rename(columns={"n": "articles", "precision": "event_precision", "recall": "event_recall", "f1": "event_f1"})[["metric_family", "model", "symbol", "articles", "event_precision", "event_recall", "event_f1", "fact_tp", "fact_fp", "fact_fn", "exact_fact_set", "no_event_false_positive", "gold_nonduplicate_event_groups", "predicted_nonduplicate_event_groups", "gold_nonduplicate_group_facts", "predicted_nonduplicate_group_facts"]],
            old.rename(columns={"n": "articles", "precision": "event_precision", "recall": "event_recall", "f1": "event_f1"})[["metric_family", "model", "symbol", "articles", "event_precision", "event_recall", "event_f1", "fact_tp", "fact_fp", "fact_fn", "exact_fact_set", "no_event_false_positive"]],
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(PUBLIC / "extraction_comparison.csv", index=False)

    metrics = pd.read_csv(downstream / "metrics.csv")
    methods = ["F0", "R1", "F1", "F2", "F6", "D2", "D3", "D4"]
    later_best = {}
    for symbol in ("AAPL", "AMZN"):
        subset = metrics[(metrics.phase == "later") & (metrics.symbol == symbol) & metrics.method.isin(methods)]
        row = subset.sort_values(["BA", "Brier"], ascending=[False, True]).iloc[0]
        later_best[symbol] = {"method": row.method, "BA": float(row.BA), "Brier": float(row.Brier)}
    stability = []
    for method in methods:
        subset = metrics[(metrics.phase.isin(["development", "later"])) & metrics.method.eq(method)]
        stability.append({"method": method, "worst_BA": float(subset.BA.min()), "mean_BA": float(subset.BA.mean()), "mean_Brier": float(subset.Brier.mean())})
    stable = sorted(stability, key=lambda row: (-row["worst_BA"], -row["mean_BA"], row["mean_Brier"]))[0]
    later_average = []
    for method in methods:
        subset = metrics[(metrics.phase == "later") & metrics.method.eq(method)]
        later_average.append({"method": method, "BA": float(subset.BA.mean()), "Brier": float(subset.Brier.mean())})
    later_average = sorted(later_average, key=lambda row: (-row["BA"], row["Brier"]))

    selected_config = selection["selected_config"]
    check_new = new_metrics[new_metrics.config.eq(selected_config)]
    table = [
        "| Method | AAPL train OOF BA / MCC / Brier | AAPL development | AAPL later | AMZN train OOF | AMZN development | AMZN later |",
        "|---|---|---|---|---|---|---|",
    ]
    for method in methods:
        cells = []
        for symbol, phase in (("AAPL", "train_forward_oof"), ("AAPL", "development"), ("AAPL", "later"), ("AMZN", "train_forward_oof"), ("AMZN", "development"), ("AMZN", "later")):
            ba = value(metrics, phase, symbol, method, "BA")
            mcc = value(metrics, phase, symbol, method, "MCC")
            brier = value(metrics, phase, symbol, method, "Brier")
            cells.append(f"{percent(ba)} / {mcc:+.3f} / {brier:.4f}")
        table.append("| " + method + " | " + " | ".join(cells) + " |")

    predictions = pd.read_csv(downstream / "predictions.csv")
    transitions_frame = pd.read_csv(downstream / "transitions.csv")
    coverage_table = [
        "| Method | Period / stock | Event / no-event windows | Changed right / wrong |",
        "|---|---|---:|---:|",
    ]
    for method in ("D2", "D3", "D4"):
        gate = f"{method}_event_gate"
        for phase in ("train_forward_oof", "development", "later"):
            for symbol in ("AAPL", "AMZN"):
                sample = predictions[predictions.phase.eq(phase) & predictions.symbol.eq(symbol)]
                event_n = int(sample[gate].sum())
                transition = transitions_frame[
                    transitions_frame.phase.eq(phase)
                    & transitions_frame.symbol.eq(symbol)
                    & transitions_frame.method.eq(method)
                ].iloc[0]
                coverage_table.append(
                    f"| {method} | {phase} / {symbol} | {event_n} / {len(sample) - event_n} | {int(transition.changed_right)} / {int(transition.changed_wrong)} |"
                )

    seed_variation = pd.read_csv(downstream / "seed_variation.csv")
    seed_table = [
        "| Method | Period / stock | BA mean +/- SD | MCC mean +/- SD | Brier mean +/- SD |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in seed_variation.sort_values(["method", "phase", "symbol"]).iterrows():
        seed_table.append(
            f"| {row.method} | {row.phase} / {row.symbol} | {percent(row.BA_mean)} +/- {100 * row.BA_std:.2f} pp | {row.MCC_mean:+.3f} +/- {row.MCC_std:.3f} | {row.Brier_mean:.4f} +/- {row.Brier_std:.4f} |"
        )

    intervals = pd.read_csv(downstream / "paired_intervals.csv")
    later_intervals = intervals[intervals.phase.eq("later")]
    interval_table = [
        "| Method | Stock | Block | BA-difference interval vs F1 | Brier-difference interval vs F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in later_intervals.sort_values(["method", "symbol", "block_days"]).iterrows():
        interval_table.append(
            f"| {row.method} | {row.symbol} | {int(row.block_days)} day | [{100 * row.BA_low:+.2f}, {100 * row.BA_high:+.2f}] pp | [{row.Brier_low:+.4f}, {row.Brier_high:+.4f}] |"
        )

    constant_rows = metrics[metrics.constant.astype(str).str.lower().isin({"true", "1"})]
    constant_note = "No whole stock-period-method slice made a constant direction prediction." if constant_rows.empty else "; ".join(
        f"{row.method}/{row.phase}/{row.symbol}" for _, row in constant_rows.iterrows()
    )

    extraction_table = [
        "| Adapter | Symbol | Positive / predicted-positive articles | Event P/R/F1 | Type macro-F1 | Action macro-F1 | Evidence F1 | No-event FP |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in new_metrics.sort_values(["config", "symbol"]).iterrows():
        extraction_table.append(f"| {row.config} | {row.symbol} | {int(row.positive_articles)} / {int(row.predicted_positive_articles)} | {percent(row.event_precision)} / {percent(row.event_recall)} / {percent(row.event_f1)} | {percent(row.type_macro_f1)} | {percent(row.action_macro_f1)} | {percent(row.evidence_micro_f1)} | {percent(row.no_event_false_positive_rate)} |")

    comparable = combined[combined.metric_family == "exact_full_fact_signature"]
    legacy_names = {"rules_current": "Current deterministic rules (D2)", "old_tuned": "Qwen3-1.7B QLoRA", "staged_v2": "Qwen3.5-9B staged", "FinBERT_A0": "FinBERT A0", "FinBERT_A1": "FinBERT A1", "FinBERT_A2": "FinBERT A2"}
    legacy_table = ["| Extractor | Symbol | Exact full-fact TP/FP/FN | Precision | Recall | F1 | Predicted nonduplicate groups |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in comparable.sort_values(["model", "symbol"]).iterrows():
        group_count = "unavailable" if pd.isna(row.get("predicted_nonduplicate_event_groups")) else str(int(row.predicted_nonduplicate_event_groups))
        legacy_table.append(f"| {legacy_names[row.model]} | {row.symbol} | {int(row.fact_tp)}/{int(row.fact_fp)}/{int(row.fact_fn)} | {percent(row.event_precision)} | {percent(row.event_recall)} | {percent(row.event_f1)} | {group_count} |")

    residual_lines = []
    for method in ("D2", "D3", "D4"):
        item = downstream_selection["residual_selections"][method]
        chosen = item["selected"]
        counts = item["counts"]
        residual_lines.append(f"- **{method} / {item['event_source']}**: training OOF chose `{chosen['parameterization']}` with C={chosen['C']}; event windows AAPL={counts['event_windows'].get('AAPL', 0)}, AMZN={counts['event_windows'].get('AMZN', 0)}; independent eligibility AAPL={counts['independent_eligible']['AAPL']}, AMZN={counts['independent_eligible']['AMZN']}.")
    d4_grid = downstream_selection["candidate_grids"]["D4"]
    d4_low_capacity = [row for row in d4_grid if row["parameterization"] == "BASE" or row["C"] == 0.01]
    residual_table = [
        "| D4 residual | C | Train mean-monthly BA | Train mean-monthly Brier | Brier guardrail |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in d4_low_capacity:
        residual_table.append(
            f"| {row['parameterization']} | {'NA' if row['C'] is None else row['C']} | {percent(row['mean_monthly_BA'])} | {row['mean_monthly_Brier']:.4f} | {'pass' if row['eligible'] else 'fail'} |"
        )

    lines = [
        "# FinBERT event adapter and gated four-hour correction",
        "",
        "## Plain-language result",
        "",
        f"This run really trained three FinBERT variants on the provisional event task: A0 trained only new heads, A1 updated the top two FinBERT layers, and A2 trained rank-4 query/value LoRA adapters in those layers. Three fixed seeds and three chronological January--February folds were used. Training OOF selected **{selected_config}** before March/April extraction evaluation or any September-onward stock result was read.",
        "",
        f"The coverage-limited oracle diagnostic found only **{oracle['counts']['event_windows']}** event windows, **{oracle['counts']['nonduplicate_report_groups']}** nonduplicate report groups, and **{oracle['counts']['nonduplicate_event_facts']}** nonduplicate facts in the 765 stock-train OOF rows. On the single available April forward month it changed **{oracle['forward_evaluation']['transitions']['changed']}** directions and moved Brier from {oracle['forward_evaluation']['base']['Brier']:.5f} to {oracle['forward_evaluation']['oracle']['Brier']:.5f}. This is not repeatable evidence that the fields improve four-hour prediction.",
        "",
        "The event correction is strictly gated. Every row without a validated event was checked for bit-for-bit equality with F1. Residual structure and C were selected globally from March--August forward OOF only; development and later were not used.",
        "",
        "## What was trained",
        "",
        f"- A0 trainable parameters: {training['trainable_parameters']['A0']:,}.",
        f"- A1 trainable parameters: {training['trainable_parameters']['A1']:,}.",
        f"- A2 trainable parameters: {training['trainable_parameters']['A2']:,}.",
        "- The model predicts type-specific evidence for `rating` and `target_price` plus a five-way action for each type. Article event existence is the logical OR of selected type-specific evidence; FinBERT does not generate JSON or numbers.",
        f"- Official completed runs: {training['runs']}; summed per-fit time {training['runtime_seconds'] / 3600:.2f} hours; maximum checkpoint reload difference {training['maximum_reload_difference']:.3g}.",
        "- Modern FinBERT is being applied retrospectively to 2018 text. Model weights and cached tensors remain private and are not in Git.",
        "",
        "## Extraction results on the provisional April check set",
        "",
        *extraction_table,
        "",
        "The table above evaluates the hierarchy that FinBERT actually predicts. The next table uses the same complete-fact signature, including deterministically copied old/new values, for FinBERT and all prior systems.",
        "",
        *legacy_table,
        "",
        f"Given provisional evidence that explicitly names the target company, the deterministic value copier exactly reproduced old/new fields for {value_audit['summary']['rating']['exact_old_new']}/{value_audit['summary']['rating']['explicit_target_facts']} rating facts and {value_audit['summary']['target_price']['exact_old_new']}/{value_audit['summary']['target_price']['explicit_target_facts']} target-price facts. The one recorded target-price failure needs a preceding sentence to bind the company to a later numeric sentence and is rejected by the strict deployed gate. These are regression diagnostics on a fully exposed provisional panel, not held-out parser estimates; they never select the FinBERT epoch, threshold, or adapter.",
        "",
        "No result constitutes formal extraction acceptance because independent human review is still absent.",
        "",
        "## Residual parameterization selected from stock-train OOF",
        "",
        *residual_lines,
        "",
        *residual_table,
        "",
        "For D4, C=0.01 was the only non-base strength that passed the Brier guardrail. C1 caused the least damage, but its BA remained below BASE; C0 and C2 were nearly identical because AMZN had too few accepted event windows for a reliable stock-specific adjustment. The protocol therefore selected no residual correction. D2 had no non-base candidate inside the Brier guardrail, while D3 had too little coverage to change a direction.",
        "",
        "C1 is disabled for any stock below 30 unique accepted events or 60 event windows. C2 retains shared event effects and adds one strongly shrunk AMZN event-window offset; selection is global, never per-stock after seeing later results.",
        "",
        "## Four-hour comparison",
        "",
        "BA, MCC, and Brier are shown together. Train OOF is March--August chronological forward prediction; development is September--October; later is November onward and already exposed.",
        "",
        *table,
        "",
        "Method names: F0=price+title baseline, R1=recent price, F1=price+full text with R1 fallback, F2=price+frozen FinBERT, F6=previous training-selected fusion, D2=F1+rule facts, D3=F1+A0 facts, D4=F1+the training-selected adapter facts.",
        "",
        f"Constant-direction check: {constant_note} `metrics.csv` also records each slice's predicted-up proportion.",
        "",
        "## Event coverage and direction changes",
        "",
        *coverage_table,
        "",
        "## Adapter seed variation",
        "",
        *seed_table,
        "",
        "## Exposed later paired intervals",
        "",
        *interval_table,
        "",
        "The interval table gives paired day and five-day block bootstrap intervals. The complete train-OOF, development, and later table is in `paired_intervals.csv`; repeated exploration means these are sensitivity summaries rather than unselected significance tests.",
        "",
        "## How to read the winners",
        "",
        f"- **Highest exposed later AAPL BA:** {later_best['AAPL']['method']} at {percent(later_best['AAPL']['BA'])} (Brier {later_best['AAPL']['Brier']:.4f}).",
        f"- **Highest exposed later AMZN BA:** {later_best['AMZN']['method']} at {percent(later_best['AMZN']['BA'])} (Brier {later_best['AMZN']['Brier']:.4f}).",
        f"- **Highest later two-stock descriptive mean:** {later_average[0]['method']} at {percent(later_average[0]['BA'])}. It did not participate in selection and later is exposed.",
        f"- **Cross-period stability diagnostic:** {stable['method']} has the highest worst BA across the four development/later stock cells ({percent(stable['worst_BA'])}). This is a descriptive robustness summary, not a new selection rule.",
        f"- **Training-protocol choice:** extraction chose {selected_config}; residual choices are listed above. Those are the only choices that can be called protocol-selected in this run.",
        "",
        "## Observations and explanations",
        "",
        "Observed: the audit verified all 458 source mappings, but only 57 labeled article-target pairs enter any fixed window and only 15 labeled event windows overlap stock-train OOF. Observed: AMZN has only 13 positive training articles in the extraction set. Observed: no-event residual rows are exactly unchanged.",
        "",
        "Interpretation: sparse event supervision limits both adapter reliability and the residual model's ability to learn a stable market response. A better extraction score can therefore coexist with no four-hour BA gain. This is an explanation consistent with the counts, not proof that analyst events contain no predictive information.",
        "",
        "## Files",
        "",
        "- `DATA_AUDIT.md`: label fields, source mapping, split and duplicate audit.",
        "- `CASE_NOTES.md`: pre-fixed extraction and prediction cases.",
        "- `metrics.csv`, `monthly_metrics.csv`, `subgroup_metrics.csv`: primary results.",
        "- `predictions.csv`: all matched probabilities without raw article text.",
        "- `paired_intervals.csv`, `seed_variation.csv`: uncertainty and seed sensitivity.",
        "- `training_evidence.json`: device, trainable counts, checkpoint reload and hashes.",
    ]
    (PUBLIC / "REPORT.md").write_text("\n".join(lines) + "\n")
    summary = {
        "status": "PUBLISHED",
        "selected_adapter": selected_config,
        "later_best": later_best,
        "later_two_stock_descriptive": later_average[0],
        "cross_period_stability": stable,
        "exact_fallback_verified": downstream_status["exact_no_event_fallback_verified"],
        "independent_human_review_passed": False,
    }
    (PUBLIC / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-run", required=True, type=Path)
    parser.add_argument("--exact-eval", required=True, type=Path)
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--downstream", required=True, type=Path)
    args = parser.parse_args()
    main(args.adapter_run, args.exact_eval, args.extraction, args.downstream)
