"""Coverage-limited oracle diagnostic using provisional facts and stock-train OOF base probabilities."""
from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import ANNOTATION, NEXTGEN, PARAGRAPHS, PUBLIC, WORK, classification_metrics, dump, jsonl, logit, sha, sigmoid
from events import FEATURES, aggregate_events, validate_numeric_event


def fit_offset(x, y, offset, c):
    x, y, offset = np.asarray(x, float), np.asarray(y, float), np.asarray(offset, float)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return np.zeros(x.shape[1], dtype=float), False

    def objective(beta):
        score = offset + x @ beta
        loss = np.mean(np.logaddexp(0, score) - y * score) + np.sum(beta * beta) / (2 * c * len(y))
        gradient = x.T @ (sigmoid(score) - y) / len(y) + beta / (c * len(y))
        return loss, gradient

    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True)
    return result.x, bool(result.success)


def main() -> None:
    audit = json.loads((PUBLIC / "data_audit.json").read_text())
    if audit["status"] != "PASS_PROVISIONAL_DATA_AUDIT":
        raise ValueError("audit not passed")
    inputs = {row["id"]: row for row in jsonl(ANNOTATION / "inputs.jsonl")}
    labels = jsonl(ANNOTATION / "labels.jsonl")
    label_articles = collections.defaultdict(list)
    for row in labels:
        if not row["answer"]["events"]:
            continue
        source = inputs[row["id"]]
        evidence = source["sentences"]
        events = []
        for event in row["answer"]["events"]:
            text = " ".join(evidence[key] for key in event["evidence_ids"])
            events.append(validate_numeric_event(event, text))
        label_articles[(row["symbol"], source["record_key"])].append(
            {
                "record_key": source["record_key"],
                "available_utc": source["available_utc"],
                "event_group": row["event_group"],
                "events": events,
            }
        )

    windows = {row["key"]: row for row in jsonl(PARAGRAPHS / "P2.jsonl")}
    oof = pd.read_csv(NEXTGEN / "OOF_PREDICTIONS.csv")
    rows = []
    distinct_report_groups = set()
    distinct_facts = set()
    for record in oof.to_dict("records"):
        window = windows[record["key"]]
        matched = []
        for article in window["news"]:
            matched.extend(label_articles.get((window["symbol"], article["record_key"]), []))
        for article in matched:
            distinct_report_groups.add((window["symbol"], article["event_group"]))
            for event in article["events"]:
                distinct_facts.add(
                    (
                        window["symbol"],
                        article["event_group"],
                        event.get("kind"),
                        event.get("action"),
                        event.get("old"),
                        event.get("new"),
                        event.get("unit"),
                    )
                )
        vector, gate, details = aggregate_events(matched, pd.Timestamp(window["cutoff"]))
        rows.append({**record, **{f"z_{name}": float(value) for name, value in zip(FEATURES, vector)}, "event_gate": gate, "unique_events": details["unique_groups"]})
    frame = pd.DataFrame(rows)
    event = frame[frame.event_gate.eq(1)].copy()
    counts = {
        "oof_windows": int(len(frame)),
        "event_windows": int(len(event)),
        "window_event_group_occurrences": int(event.unique_events.sum()),
        "nonduplicate_report_groups": len(distinct_report_groups),
        "nonduplicate_event_facts": len(distinct_facts),
        "by_month_symbol": {"|".join(map(str, key)): int(value) for key, value in event.groupby(["month", "symbol"]).size().items()},
    }

    # March provides the only prior labeled month for an April forward evaluation.
    march = frame[frame.month.eq("2018-03")]
    april = frame[frame.month.eq("2018-04")]
    early = march[pd.to_datetime(march.day).dt.day.le(15) & march.event_gate.eq(1)]
    late = march[pd.to_datetime(march.day).dt.day.gt(15)]
    feature_columns = [f"z_{name}" for name in FEATURES]
    candidates = []
    for c in (0.01, 0.1, 1.0):
        beta, ok = fit_offset(early[feature_columns], early.label, logit(early.F1), c)
        probability = late.F1.to_numpy(float).copy()
        active = late.event_gate.to_numpy(bool)
        probability[active] = sigmoid(logit(probability[active]) + late.loc[active, feature_columns].to_numpy(float) @ beta)
        score = classification_metrics(late.label, probability)
        candidates.append({"C": c, "fit_ok": ok, **score})
    eligible = [row for row in candidates if row["fit_ok"]]
    selected = sorted(eligible, key=lambda row: (-(-1 if row["BA"] is None else row["BA"]), row["Brier"], row["C"]))[0] if eligible else None
    april_probability = april.F1.to_numpy(float).copy()
    fit_ok = False
    if selected is not None:
        march_event = march[march.event_gate.eq(1)]
        beta, fit_ok = fit_offset(march_event[feature_columns], march_event.label, logit(march_event.F1), selected["C"])
        active = april.event_gate.to_numpy(bool)
        april_probability[active] = sigmoid(logit(april_probability[active]) + april.loc[active, feature_columns].to_numpy(float) @ beta)
    base = classification_metrics(april.label, april.F1)
    corrected = classification_metrics(april.label, april_probability)
    transitions = {
        "changed": int(np.sum((april_probability >= 0.5) != (april.F1.to_numpy() >= 0.5))),
        "right": int(np.sum(((april_probability >= 0.5) == april.label.to_numpy()) & ((april.F1.to_numpy() >= 0.5) != april.label.to_numpy()))),
        "wrong": int(np.sum(((april_probability >= 0.5) != april.label.to_numpy()) & ((april.F1.to_numpy() >= 0.5) == april.label.to_numpy()))),
    }
    repeatable = False  # Only one forward evaluation month has mapped provisional event facts.
    result = {
        "status": "COMPLETE_STOP_FOR_SPARSE_FORWARD_EVIDENCE",
        "scope": "coverage_limited_oracle_diagnostic_not_deployable",
        "unknown_policy": "unlabelled articles never become negative; their gate is zero and base probability is unchanged",
        "counts": counts,
        "feature_count": len(FEATURES),
        "C_selection": {"source": "March 1-15 fit, March 16-31 validation", "candidates": candidates, "selected": selected},
        "forward_evaluation": {"period": "2018-04", "fit_ok": fit_ok, "base": base, "oracle": corrected, "transitions": transitions},
        "repeatable_train_oof_increment": repeatable,
        "stop_reason": "Only March and April provisional labels overlap stock-train OOF windows, leaving one forward evaluation month; this cannot demonstrate repeatable increment in at least two months.",
        "independent_human_review_passed": False,
        "source_hashes": {
            "labels": sha(ANNOTATION / "labels.jsonl"),
            "oof": sha(NEXTGEN / "OOF_PREDICTIONS.csv"),
            "paragraphs": sha(PARAGRAPHS / "P2.jsonl"),
        },
    }
    dump(WORK / "oracle.json", result)
    dump(PUBLIC / "oracle_diagnostic.json", result)
    frame[["key", "symbol", "day", "month", "label", "F1", "event_gate", "unique_events", *feature_columns]].to_csv(WORK / "oracle_windows.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
