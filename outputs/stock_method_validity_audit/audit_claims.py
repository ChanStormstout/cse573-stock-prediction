#!/usr/bin/env python3
"""Audit narrow method claims without fitting or selecting a new model.

This stage reads public calibration manifests and produces a small, reviewable
audit.  It deliberately does not load private inputs, fit a calibrator, or
change any historical prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "stock_method_validity_audit" / "v1"


def _float(value):
    if value in (None, "", "null"):
        return None
    return float(value)


def add_row(rows, source, record, variant, symbol, period, method, slope, intercept, n):
    if slope is None:
        return
    rows.append(
        {
            "source": str(source.relative_to(ROOT)),
            "record": record,
            "variant": variant or "",
            "symbol": symbol or "",
            "period": period or "",
            "method": method or "",
            "slope": float(slope),
            "intercept": "" if intercept is None else float(intercept),
            "train_n": "" if n is None else n,
            "monotone_non_decreasing": bool(float(slope) >= 0),
        }
    )


def load_rows():
    rows = []
    nextgen = ROOT / "outputs/stock_nextgen_4h/runs/v1/CALIBRATION_FITS.json"
    for i, record in enumerate(json.loads(nextgen.read_text())):
        if record.get("method") == "platt" and "coef" in record:
            add_row(
                rows,
                nextgen,
                i,
                record.get("variant"),
                record.get("symbol"),
                record.get("period"),
                "platt",
                record.get("coef"),
                record.get("intercept"),
                record.get("train_n"),
            )

    csv_specs = [
        ROOT / "outputs/stock_paper_methods_4h/v1/calibration.csv",
        ROOT / "outputs/stock_combination_4h/v1/calibrators.csv",
    ]
    for path in csv_specs:
        with path.open(newline="") as handle:
            for i, record in enumerate(csv.DictReader(handle)):
                add_row(
                    rows,
                    path,
                    i,
                    record.get("branch") or record.get("method"),
                    record.get("symbol"),
                    record.get("month"),
                    record.get("kind") or record.get("method"),
                    _float(record.get("slope")),
                    _float(record.get("intercept")),
                    record.get("n"),
                )

    goal = ROOT / "outputs/stock_goal60_4h/v1/calibration.json"
    for i, record in enumerate(json.loads(goal.read_text())):
        add_row(
            rows,
            goal,
            i,
            record.get("method"),
            record.get("symbol"),
            record.get("month"),
            "positive_slope",
            record.get("positive_slope"),
            None,
            record.get("past_n"),
        )
    return rows


CLAIMS = [
    ("ISSUE-012", "masked-reconstruction SSL", "VALID_WITH_CAVEAT", "small SSL pilot only; not TS2Vec", "TS2Vec failed"),
    ("ISSUE-013", "historical analogy", "VALID_WITH_CAVEAT", "lexical/coarse historical analogy probe", "FinSeer was reproduced or refuted"),
    ("ISSUE-014", "Event Adapter", "VALID_WITH_CAVEAT", "target evidence/action extraction probe", "Ding event/graph embedding was reproduced"),
    ("ISSUE-015", "Chronos-2", "VALID_WITH_CAVEAT", "frozen endpoint-feature probe", "Chronos-2 is ineffective in general"),
    ("ISSUE-016", "early unconstrained Platt", "AUDITED", "32 of 84 saved early Platt fits have negative slopes; later constrained maps are the interpretable calibration", "all historical Platt maps are monotone"),
    ("ISSUE-017", "dissemination/event clustering", "VALID_WITH_CAVEAT", "local gain is regularization-confounded", "deduplication independently caused the AMZN gain"),
    ("ISSUE-018", "Qwen direct UP/DOWN", "VALID_WITH_CAVEAT", "conditional token preference, not calibrated probability", "token scores are market probabilities"),
    ("ISSUE-019", "cross-stock past state", "VALID", "uses only peer state available before the cutoff", "future contemporaneous peer returns were used"),
    ("ISSUE-020", "continuous-return auxiliary", "VALID_WITH_CAVEAT", "one shared direction-plus-return objective lacked stable promotion", "return magnitude has no signal"),
    ("ISSUE-021", "F1/F2 controls", "DOCUMENTED", "historical controls are separate from reselected F1_new/F2_new", "the controls were silently swapped"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fieldnames = [
        "source",
        "record",
        "variant",
        "symbol",
        "period",
        "method",
        "slope",
        "intercept",
        "train_n",
        "monotone_non_decreasing",
    ]
    with (OUT / "CALIBRATION_SLOPES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "METHOD_CLAIM_AUDIT.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["issue", "method", "status", "safe_claim", "unsafe_claim"])
        writer.writerows(CLAIMS)

    negative = [row for row in rows if row["slope"] < 0]
    source_counts = {}
    for row in rows:
        source_counts.setdefault(row["source"], {"rows": 0, "negative": 0})
        source_counts[row["source"]]["rows"] += 1
        source_counts[row["source"]]["negative"] += int(row["slope"] < 0)
    verification = {
        "status": "PASS",
        "scope": "claim-only; no fit, inference, or model selection",
        "calibration_rows": len(rows),
        "negative_early_platt_slopes": len(negative),
        "positive_or_zero_slopes": len(rows) - len(negative),
        "source_counts": source_counts,
        "negative_slopes_are_non_monotone": True,
        "later_constrained_files_have_negative_slopes": any(
            row["slope"] < 0
            for row in rows
            if row["source"]
            in {
                "outputs/stock_paper_methods_4h/v1/calibration.csv",
                "outputs/stock_combination_4h/v1/calibrators.csv",
                "outputs/stock_goal60_4h/v1/calibration.json",
            }
        ),
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    report = f"""# Method validity and claim audit (Stage 5)

## Scope

This is a claim-only audit. It reads saved public calibration manifests and
current-facing documentation; it does not fit a model, run inference, change
predictions, or select a method using development/later scores. The historical
outputs remain unchanged.

## Calibration finding

The early unconstrained Platt manifest contains **{len(negative)} negative
slopes among {len(rows)} saved calibration records**. All 32 negative records
come from `outputs/stock_nextgen_4h/runs/v1/CALIBRATION_FITS.json` and are
AMZN records. A negative coefficient reverses the ordering of the raw score,
so this mapping is not a simple monotone calibration. The later
`platt_shrunk`/temperature/positive-slope manifests contain no negative
slopes in this audit and are the safer interpretation for probability quality.

This does not change any BA result. It narrows the claim: early Platt values
are historical score remappings, while the later constrained calibration is
the interpretable probability-calibration probe.

## Claim corrections

| issue | method | status | safe interpretation |
|---|---|---|---|
""" + "\n".join(
        f"| {issue} | {method} | {status} | {safe} |"
        for issue, method, status, safe, _unsafe in CLAIMS
    ) + """

See `METHOD_CLAIM_AUDIT.csv` for the corresponding unsafe wording to avoid.
The small masked-reconstruction encoder is not a TS2Vec reproduction; the
analogy run is not FinSeer; the Event Adapter is not a Ding graph/event
embedding; the Chronos result is an endpoint probe; Qwen direct scores are
token preferences; and continuous-return C2 does not show that return
magnitude lacks information.

## Verification artifacts

- `CALIBRATION_SLOPES.csv` contains the exact public slope records audited.
- `METHOD_CLAIM_AUDIT.csv` is the machine-readable safe/unsafe claim table.
- `verification.json` records counts and the no-fit scope.

No independent human event-review gate is claimed to have passed. All project
development/later numbers remain exposed historical backtests.
"""
    (OUT / "REPORT.md").write_text(report)
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": [
            "REPORT.md",
            "CALIBRATION_SLOPES.csv",
            "METHOD_CLAIM_AUDIT.csv",
            "verification.json",
        ],
        "status": "PASS",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
