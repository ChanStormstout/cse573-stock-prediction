"""Verifier-only repair for V10 Stage A2 final artifacts.

This module performs no estimator fitting and writes no private-model file. It
checks quarantined-grid isolation and reloads the exact manifest-hashed joblib
artifacts that remain after the original Stage A2 replay.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work/stock-data"
BASE = ROOT / "outputs/stock_priorwork_repro"
V9 = BASE / "v9"
V10 = BASE / "v10"
PRIVATE = WORK / "priorwork_v10/models/news_only"
V9_COMMIT = "b7ee2af9829143707ec309e0ec6809935524324b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def expected_selection_months(month: str) -> list[str]:
    if month == "2018-03":
        return []
    if month <= "2018-08":
        return pd.period_range("2018-03", str(pd.Period(month) - 1), freq="M").astype(str).tolist()
    return pd.period_range("2018-03", "2018-08", freq="M").astype(str).tolist()


def quarantine_isolation() -> dict:
    issued = pd.concat([
        pd.read_csv(V10 / "ISSUED_PARAMS_4H.csv"),
        pd.read_csv(V10 / "ISSUED_PARAMS_1D.csv"),
    ], ignore_index=True)
    grids = pd.concat([
        pd.read_csv(V9 / "GRID_OOF_4H.csv"),
        pd.read_csv(V9 / "GRID_OOF_1D.csv"),
    ], ignore_index=True)
    authorized = grids[grids.month <= "2018-08"].copy()
    quarantined = grids[grids.month >= "2018-09"].copy()
    checked = 0
    references = 0
    quarantined_references = 0
    bad = []
    for row in issued.itertuples(index=False):
        checked += 1
        actual_months = json.loads(row.selection_months_used)
        expected = expected_selection_months(row.prediction_month)
        horizon = row.horizon
        candidate_rows = authorized[(authorized.stock == row.stock) &
                                    (authorized.horizon == horizon) &
                                    (authorized.method == row.method) &
                                    (authorized.month.isin(actual_months))]
        quarantine_rows = quarantined[(quarantined.stock == row.stock) &
                                      (quarantined.horizon == horizon) &
                                      (quarantined.method == row.method) &
                                      (quarantined.month.isin(actual_months))]
        references += len(candidate_rows)
        quarantined_references += len(quarantine_rows)
        correct_months = actual_months == expected and not any(m >= "2018-09" for m in actual_months)
        row_count_correct = int(row.authorized_candidate_rows_used) == len(candidate_rows)
        every_reference_authorized = candidate_rows.month.le("2018-08").all()
        if not (correct_months and row_count_correct and every_reference_authorized and len(quarantine_rows) == 0):
            bad.append({"stock": row.stock, "horizon": horizon, "method": row.method,
                        "prediction_month": row.prediction_month, "selection_months_used": actual_months,
                        "expected_selection_months": expected,
                        "recorded_authorized_candidate_rows": int(row.authorized_candidate_rows_used),
                        "found_authorized_candidate_rows": len(candidate_rows),
                        "found_quarantined_candidate_rows": len(quarantine_rows)})
    # The historical selection audit is inspected as a second independent input:
    # it must still record zero parameter/freeze failures.
    selection_audit = json.loads((V10 / "SELECTION_RECONSTRUCTION_AUDIT.json").read_text())
    source_counts = {"authorized_grid_rows": len(authorized), "quarantined_grid_rows": len(quarantined)}
    passed = (checked == 576 and source_counts == {"authorized_grid_rows": 1044, "quarantined_grid_rows": 1044}
              and quarantined_references == 0 and not bad
              and selection_audit.get("4h_mismatches") == 0 and selection_audit.get("1d_mismatches") == 0)
    return {"title": "Computed quarantine selection isolation audit", "status": "PASS" if passed else "FAIL",
            "issued_decisions_checked": checked, "authorized_candidate_references_checked": references,
            "quarantined_candidate_references": quarantined_references,
            "bad_selection_month_records": bad, "source_counts": source_counts,
            "selection_reconstruction_status": selection_audit.get("status")}


def read_canonical_evaluation_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read canonical sources only; never invoke a fit or mutate a model.

    The full raw-source reconstruction remains in the frozen Stage A2 verifier.
    Here these canonical source artifacts are paired with its stored raw-source
    parity facts, then used to reconstruct each exact model evaluation slice.
    """
    four = pd.read_pickle(WORK / "nextgen_4h/price_v1/features.pkl").copy()
    four["news_window"] = "4H"
    daily = pd.read_pickle(BASE / "v8/private_daily.pkl").copy()
    source_audit = json.loads((V10 / "STAGE_A_FINAL_AUDIT.json").read_text())["canonical_4h_row_contract"]
    contract = {"four_hour_source_rows": len(four), "daily_source_rows": len(daily),
                "inherited_raw_source_article_key_parity": source_audit["article_key_parity"],
                "inherited_raw_source_stem_body_parity": source_audit["stem_body_parity"],
                "pass": len(four) == 1607 and len(daily) == 1072 and
                        source_audit["article_key_parity"] == "1607/1607" and
                        source_audit["stem_body_parity"] == "1607/1607"}
    return four, daily, contract


def evaluation_slice(data: pd.DataFrame, stock: str, horizon: str, month: str) -> pd.DataFrame:
    window = "4H" if horizon == "4h" else horizon.split(":", 1)[1]
    group = data[(data.symbol == stock) & (data.news_window == window)].sort_values("cutoff_utc")
    return group[group.start_utc.dt.strftime("%Y-%m") == month].sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)


def predict_loaded(bundle: dict, ev: pd.DataFrame) -> np.ndarray:
    matrix = bundle["vectorizer"].transform(ev.stem_body.fillna("").astype(str))
    if bundle["selector"] is not None:
        matrix = bundle["selector"].transform(matrix)
    if bundle["decision_only"]:
        score = bundle["classifier"].decision_function(matrix)
        fallback = bundle["training_class_prior"] - 0.5
    else:
        score = bundle["classifier"].predict_proba(matrix)[:, 1]
        fallback = bundle["training_class_prior"]
    score = np.asarray(score, dtype=float)
    score[ev.has_news.to_numpy(dtype=int) == 0] = fallback
    return score


def final_artifact_reload() -> dict:
    manifest = json.loads((V10 / "NEWS_ONLY_MODEL_MANIFEST.json").read_text())
    before = {path.name: sha(path) for path in sorted(PRIVATE.glob("*.joblib"))}
    four, daily, source_contract = read_canonical_evaluation_rows()
    pred4 = pd.read_csv(V9 / "PREDICTIONS_4H.csv", parse_dates=["start_utc", "cutoff_utc"])
    pred1 = pd.read_csv(V9 / "PREDICTIONS_1D.csv", parse_dates=["start_utc", "cutoff_utc"])
    hash_mismatches = []
    missing = []
    probability_error = 0.0
    svm_error = 0.0
    direction_mismatches = 0
    for entry in manifest:
        path = PRIVATE / f"{entry['logical_model_id']}.joblib"
        if not path.is_file():
            missing.append(entry["logical_model_id"])
            continue
        file_sha = sha(path)
        if file_sha != entry["sha256"]:
            hash_mismatches.append({"logical_model_id": entry["logical_model_id"],
                                    "manifest_sha256": entry["sha256"], "actual_sha256": file_sha})
        bundle = joblib.load(path)
        data = four if entry["horizon"] == "4h" else daily
        ev = evaluation_slice(data, entry["stock"], entry["horizon"], entry["month"])
        frozen = pred4 if entry["horizon"] == "4h" else pred1
        expected = frozen[(frozen.symbol == entry["stock"]) &
                          (frozen.horizon == entry["horizon"]) &
                          (frozen.method == entry["method"]) &
                          (frozen.month == entry["month"])].sort_values(["start_utc", "cutoff_utc"]).reset_index(drop=True)
        if len(ev) != len(expected) or not ev[["symbol", "start_utc", "cutoff_utc", "label"]].equals(expected[["symbol", "start_utc", "cutoff_utc", "label"]]):
            raise RuntimeError(f"canonical evaluation mapping mismatch: {entry['logical_model_id']}")
        score = predict_loaded(bundle, ev)
        error = float(np.max(np.abs(score - expected.p.to_numpy(dtype=float)))) if len(score) else 0.0
        if bundle["decision_only"]:
            svm_error = max(svm_error, error)
            predicted = score >= 0
            reference = expected.p.to_numpy(dtype=float) >= 0
        else:
            probability_error = max(probability_error, error)
            predicted = score >= 0.5
            reference = expected.p.to_numpy(dtype=float) >= 0.5
        direction_mismatches += int(np.sum(predicted != reference))
    after = {path.name: sha(path) for path in sorted(PRIVATE.glob("*.joblib"))}
    passed = (len(manifest) == 576 and len(before) == 576 and len(after) == 576 and len(missing) == 0 and
              len(hash_mismatches) == 0 and probability_error <= 1e-10 and svm_error <= 1e-10 and
              direction_mismatches == 0 and before == after and source_contract["pass"])
    return {"title": "Final manifest-hashed serialized artifacts reload audit", "status": "PASS" if passed else "FAIL",
            "model_files_found": len(before), "expected_model_files": 576, "manifest_entries": len(manifest),
            "missing_model_files": missing, "manifest_hash_mismatches": hash_mismatches,
            "max_probability_error": probability_error, "max_svm_decision_score_error": svm_error,
            "direction_mismatches": direction_mismatches, "bytes_unchanged_during_repair": before == after,
            "before_hashes": before, "after_hashes": after, "canonical_evaluation_source_contract": source_contract}


def tracked_priorwork_fingerprint() -> dict:
    tracked = subprocess.check_output(["git", "ls-files", "outputs/stock_priorwork_repro/v6", "outputs/stock_priorwork_repro/v7", "outputs/stock_priorwork_repro/v8", "outputs/stock_priorwork_repro/v9"], cwd=ROOT, text=True).splitlines()
    initial = {name: sha(ROOT / name) for name in tracked}
    return {"tracked_file_count": len(tracked), "before": initial}


def main() -> None:
    original = json.loads((V10 / "STAGE_A_FINAL_AUDIT.json").read_text())
    original_hash = sha(V10 / "STAGE_A_FINAL_AUDIT.json")
    priorwork_before = tracked_priorwork_fingerprint()
    isolation = quarantine_isolation()
    dump(V10 / "QUARANTINE_SELECTION_ISOLATION_AUDIT.json", isolation)
    reload_audit = final_artifact_reload()
    dump(V10 / "FINAL_SERIALIZED_MODEL_RELOAD_AUDIT.json", reload_audit)
    priorwork_after = {name: sha(ROOT / name) for name in priorwork_before["before"]}
    tracked_unchanged = priorwork_before["before"] == priorwork_after
    source_diff = subprocess.run(["git", "diff", "--exit-code", V9_COMMIT, "--", "outputs/stock_priorwork_repro/v9/"], cwd=ROOT)
    checks = {"original_stage_a_scientific_replay": original.get("status") == "PASS",
              "original_stage_a_audit_unchanged": sha(V10 / "STAGE_A_FINAL_AUDIT.json") == original_hash,
              "quarantine_selection_isolation": isolation["status"] == "PASS",
              "final_serialized_model_reload": reload_audit["status"] == "PASS",
              "all_576_private_model_hashes_unchanged": reload_audit["bytes_unchanged_during_repair"],
              "v6_v7_v8_v9_tracked_artifacts_unchanged": tracked_unchanged and source_diff.returncode == 0}
    final = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "historical_stage_a_final_audit_sha256": original_hash,
             "quarantine_selection_isolation": isolation,
             "final_serialized_model_reload": reload_audit,
             "v6_to_v9_tracked_artifact_count": priorwork_before["tracked_file_count"],
             "v6_to_v9_before_hashes": priorwork_before["before"],
             "v6_to_v9_after_hashes": priorwork_after,
             "scope": {"refit_occurred": False, "candidate_grid_rerun": False, "dprice_run": False,
                       "method_family_selection_run": False, "news_price_run": False,
                       "market_context_changed": False, "relation_reader_changed": False}}
    dump(V10 / "STAGE_A_FINAL_AUDIT_V2.json", final)
    print("V10_STAGE_A_FULLY_VERIFIED_AWAITING_STAGE_B" if final["status"] == "PASS" else "V10_STAGE_A_FINAL_VERIFIER_REPAIR_FAILED")


if __name__ == "__main__":
    main()
