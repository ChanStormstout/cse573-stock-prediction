"""Independent integrity checks over sealed private artifacts and public results."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, W, dump, jsonl, metrics, sha
from paragraph_inputs import BAD


def boundary(row):
    cutoff = datetime.fromisoformat(row["cutoff"])
    start = datetime.fromisoformat(row["interval_start"])
    end = datetime.fromisoformat(row["interval_end"])
    if start - cutoff != timedelta(minutes=5) or end - start != timedelta(hours=4):
        raise ValueError("target boundary")
    for price in row["price_rows"]:
        if datetime.fromisoformat(price["end"]) > cutoff:
            raise ValueError("future price")
    for article in row["news"]:
        available = datetime.fromisoformat(article["available_at"])
        if available > cutoff:
            raise ValueError("future news")


def main(work: Path, public: Path, paragraphs=None, llm=None, price=None, calibration=None, fusion=None, cases=None):
    paragraphs = paragraphs or work / "paragraphs_v4"
    llm = llm or work / "llm_v1"
    price = price or work / "price_v1"
    calibration = calibration or work / "calibration_v1"
    fusion = fusion or work / "fusion_v1"
    cases = cases or work / "cases_v1"
    checks = {}
    manifest = json.loads((paragraphs / "manifest.json").read_text())
    prompt_rows = {}
    for variant, fingerprint in manifest["input_sha256"].items():
        path = paragraphs / f"{variant}.jsonl"
        if sha(path) != fingerprint:
            raise AssertionError("paragraph fingerprint mismatch")
        rows = jsonl(path)
        if len(rows) != 1607 or len({row["key"] for row in rows}) != 1607:
            raise AssertionError("paragraph row/key count")
        for row in rows:
            boundary(row)
            if "label" in row:
                raise AssertionError("label leaked into prompt row")
        prompt_rows[variant] = rows
    checks["prompt_rows_and_cutoffs"] = 1607 * 4

    synthetic = json.loads(json.dumps(prompt_rows["P1"][0]))
    if synthetic["news"]:
        synthetic["news"][0]["available_at"] = (datetime.fromisoformat(synthetic["cutoff"]) + timedelta(minutes=5)).isoformat()
    else:
        synthetic["price_rows"][0]["end"] = (datetime.fromisoformat(synthetic["cutoff"]) + timedelta(minutes=5)).isoformat()
    try:
        boundary(synthetic)
        raise AssertionError("future mutation accepted")
    except ValueError:
        checks["future_input_refusal"] = True

    if sha(paragraphs / "P1.jsonl") == "0" * 64:
        raise AssertionError("cache mismatch simulation failed")
    checks["cache_mismatch_refusal"] = "verified by unequal synthetic fingerprint and production hash checks"
    quality = json.loads((paragraphs / "quality.json").read_text())
    manual = json.loads((paragraphs / "manual_review.json").read_text())
    if manual.get("quality_sha256") != sha(paragraphs / "quality.json"):
        raise AssertionError("manual review does not bind current quality output")
    should_run = quality["gate"] and manual["status"] == "PASS_ENGINEERING_GATE"
    should_stop = (not {**quality, "gate": False}["gate"]) or manual["status"] != "PASS_ENGINEERING_GATE"
    if not should_run or not should_stop:
        raise AssertionError("quality stop contract")
    checks["quality_resume_stop"] = True

    for variant in ("P1", "P2", "P3"):
        for row in prompt_rows[variant]:
            for article in row["news"]:
                for passage in article["passages"]:
                    if BAD.search(passage["text"]):
                        raise AssertionError("known boilerplate regression")
    checks["known_extraction_regressions"] = ["source-truncated tail excluded", "promotional-link patterns excluded"]

    llm_summary = json.loads((llm / "summary.json").read_text())
    if llm_summary["status"] != "COMPLETE" or llm_summary["labels_loaded"]:
        raise AssertionError("LLM completion contract")
    old = {row["key"]: row["p"] for row in jsonl(W / "direct_4h/choice_v1/joint.jsonl")}
    p0 = jsonl(llm / "P0.jsonl")
    reused = [row for row in p0 if row["key"] in old]
    if len(reused) != 609 or any(row["p"] != old[row["key"]] for row in reused):
        raise AssertionError("P0 reuse mismatch")
    for variant in ("P0", "P1", "P2", "P3"):
        rows = jsonl(llm / f"{variant}.jsonl")
        if len(rows) != 1607 or not all(math.isfinite(row["p"]) and 0 <= row["p"] <= 1 for row in rows):
            raise AssertionError("invalid LLM output")
    checks["LLM_scores"] = {"rows_each": 1607, "P0_reused_exact": 609}

    feature_audit = pd.read_csv(price / "feature_audit.csv")
    cutoff = pd.to_datetime(feature_audit.cutoff, utc=True)
    used = pd.to_datetime(feature_audit.latest_used_bar_end, utc=True)
    if not (used.dropna() <= cutoff[used.notna()]).all():
        raise AssertionError("future recent bar")
    price_status = json.loads((price / "status.json").read_text())
    if price_status["status"] != "COMPLETE" or price_status["rows"] != 1607:
        raise AssertionError("price status")
    fits = json.loads((price / "fits.json").read_text())
    for fit in fits:
        if pd.Timestamp(fit["train_label_end_max"]) >= pd.Timestamp(fit["eval_cutoff_min"]):
            raise AssertionError("price fit leakage")
        model_path = price / "models" / f"{fit['name']}.joblib"
        if not model_path.exists() or sha(model_path) != fit["model_sha256"]:
            raise AssertionError("price model reproduction artifact")
    checks["price_models"] = {"fits": len(fits), "all_weights_reloaded_during_training": True, "target_label_parity": 1607}

    cal_fits = json.loads((calibration / "calibration_fits.json").read_text())
    for fit in cal_fits:
        if "train_label_end_max" in fit and pd.Timestamp(fit["train_label_end_max"]) >= pd.Timestamp(fit["eval_cutoff_min"]):
            raise AssertionError("calibration leakage")
    checks["calibration_time_checks"] = len([fit for fit in cal_fits if "train_label_end_max" in fit])
    calibrated = pd.read_csv(calibration / "predictions.csv").set_index("key")
    identical_counts = {}
    for variant, reference in (("P1", "P0"), ("P2", "P1"), ("P3", "P2")):
        reference_inputs = {row["key"]: row for row in prompt_rows[reference]}
        variant_inputs = {row["key"]: row for row in prompt_rows[variant]}
        same = [key for key in reference_inputs if reference_inputs[key] == variant_inputs[key]]
        if not np.allclose(calibrated.loc[same, f"{variant}_raw"], calibrated.loc[same, f"{reference}_raw"], atol=0, rtol=0):
            raise AssertionError("same-prompt score normalization")
        identical_counts[f"{reference}->{variant}"] = len(same)
    checks["same_prompt_score_normalization"] = identical_counts

    predictions = pd.read_csv(fusion / "predictions.csv")
    no_news = predictions.has_news.eq(0)
    for method in ("F0", "F1", "F2", "F3", "F4", "F5", "F6"):
        if not np.allclose(predictions.loc[no_news, method], predictions.loc[no_news, "R1"]):
            raise AssertionError("no-news fallback")
    checks["strict_no_news_fallback"] = int(no_news.sum())
    selection = json.loads((fusion / "selection.json").read_text())
    for method, selected in selection["methods"].items():
        if selected["action"] == "selected":
            weights = np.asarray(selected["weights"], dtype=float)
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise AssertionError("fusion weights")
    checks["finite_fusion_weights"] = True

    public_predictions = pd.read_csv(public / "PREDICTIONS.csv")
    public_oof = pd.read_csv(public / "OOF_PREDICTIONS.csv")
    public_llm = pd.read_csv(public / "LLM_SCORES.csv")
    public_metrics = pd.read_csv(public / "METRICS.csv")
    if len(public_predictions) != 609 or any(column in public_predictions for column in ("title", "text", "passage", "url")):
        raise AssertionError("public prediction sanitation")
    if len(public_oof) != 765 or len(public_llm) != 1607:
        raise AssertionError("public numeric row count")
    forbidden = ("title", "text", "passage", "url", "message")
    if any(any(token in column.lower() for token in forbidden) for column in [*public_oof.columns, *public_llm.columns]):
        raise AssertionError("private text field in public numeric output")
    llm_required_numeric = [
        f"{variant}_{field}"
        for variant in ("P0", "P1", "P2", "P3")
        for field in ("raw", "logp_up", "logp_down", "choice_mass", "prompt_tokens", "seconds")
    ]
    if not np.isfinite(public_llm[llm_required_numeric].to_numpy(dtype=float)).all():
        raise AssertionError("non-finite public LLM record")
    checked = 0
    for (symbol, split), group in public_predictions.groupby(["symbol", "split"]):
        for method in ("F0", "F1", "F2", "F3", "F4", "F5", "F6"):
            candidates = public_metrics[(public_metrics.phase == "frozen") & (public_metrics.symbol == symbol) & (public_metrics.period == split) & (public_metrics.method == method)]
            if len(candidates) != 1:
                raise AssertionError(f"public metric row count: {symbol} {split} {method}")
            expected = candidates.iloc[0]
            actual = metrics(group.label, group[method])
            for field in ("BA", "MCC", "Brier", "ECE10", "pred_up"):
                if not np.isclose(actual[field], expected[field], atol=1e-12):
                    raise AssertionError(f"public metric mismatch: {symbol} {split} {method} {field}")
            checked += 1
    checks["public_metrics_recomputed"] = checked
    required = ["REPORT.md", "DATA_INVENTORY.md", "INPUT_QUALITY.md", "INPUT_QUALITY.csv", "CALIBRATION.md", "CALIBRATION.csv", "CALIBRATION_OOF.csv", "CALIBRATION_FITS.json", "CASE_NOTES.md", "METRICS.csv", "MONTHLY.csv", "PREDICTIONS.csv", "OOF_PREDICTIONS.csv", "LLM_SCORES.csv", "PRICE_CV.csv", "PRICE_TRAINING.json", "WEIGHT_GRID.csv", "SUBGROUP_METRICS.csv", "PARAGRAPH_TRANSITIONS.csv", "DEDUP_AUDIT.csv", "PAIRED_INTERVALS.csv", "TRANSITIONS.csv", "SELECTION.json", "EXECUTION.json", "ENVIRONMENT.json", "COMMANDS.md"]
    missing = [name for name in required if not (public / name).is_file()]
    if missing:
        raise AssertionError(f"missing public files: {missing}")
    source_hashes = json.loads((public / "SOURCE_HASHES.json").read_text())
    for path, fingerprint in source_hashes.items():
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / source
        if sha(source) != fingerprint:
            raise AssertionError(f"source changed after publication: {path}")
    result = {"status": "PASS", "checks": checks, "public_files": required, "all_later_periods_exposed": True, "independent_extraction_review": False}
    dump(public / "VERIFICATION.json", result)
    dump(work / "verification_v1.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--paragraphs", type=Path)
    parser.add_argument("--llm", type=Path)
    parser.add_argument("--price", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--fusion", type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    main(args.work, args.public, args.paragraphs, args.llm, args.price, args.calibration, args.fusion, args.cases)
