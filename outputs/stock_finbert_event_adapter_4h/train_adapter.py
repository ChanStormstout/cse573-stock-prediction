"""Train A0/A1/A2 under the registered chronological extraction protocol."""
from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapter import EncodedDataset, adapter_score, article_examples, extraction_metrics, fit_once, infer, reload_check, tokenizer
from common import ANNOTATION, PUBLIC, SEEDS, WORK, dump, jsonl, sha, write_jsonl

FOLDS = (
    ("2018-01-15", "2018-01-16", "2018-01-31"),
    ("2018-01-31", "2018-02-01", "2018-02-14"),
    ("2018-02-14", "2018-02-15", "2018-02-28"),
)
CONFIGS = ("A0", "A1", "A2")
THRESHOLDS = (0.35, 0.5, 0.65)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_jsonl(path: Path, rows) -> None:
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(out: Path, resume: bool = False) -> None:
    if out.exists() and not resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=resume)
    protocol = json.loads((PUBLIC / "protocol.json").read_text())
    if protocol["seeds"] != list(SEEDS):
        raise ValueError("protocol seed mismatch")
    inputs = jsonl(ANNOTATION / "inputs.jsonl")
    labels = {row["id"]: row for row in jsonl(ANNOTATION / "labels.jsonl")}
    train_inputs = [row for row in inputs if row["split"] == "train"]
    examples = article_examples(inputs, labels)
    examples_by_article = collections.defaultdict(list)
    for row in examples:
        examples_by_article[row["article_id"]].append(row)
    tok = tokenizer()
    device = choose_device()
    progress_training = out / "progress_training.jsonl"
    progress_oof = out / "progress_oof_predictions.jsonl"
    progress_eval = out / "progress_evaluation_predictions.jsonl"
    training_records = jsonl(progress_training) if progress_training.exists() else []
    oof_predictions = jsonl(progress_oof) if progress_oof.exists() else []
    completed_folds = {(row["config"], row["seed"], row["fold"]) for row in training_records if row["stage"] == "forward_fold"}
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=resume)
    started = time.time()

    for config_name in CONFIGS:
        for seed in SEEDS:
            for fold_number, (train_end, eval_start, eval_end) in enumerate(FOLDS, 1):
                if (config_name, seed, fold_number) in completed_folds:
                    print(json.dumps({"stage": "resume_skip_forward_fold", "config": config_name, "seed": seed, "fold": fold_number}), flush=True)
                    continue
                train_ids = {row["id"] for row in train_inputs if row["available_utc"][:10] <= train_end}
                eval_ids = {row["id"] for row in train_inputs if eval_start <= row["available_utc"][:10] <= eval_end}
                train_examples = [example for key in train_ids for example in examples_by_article[key]]
                eval_examples = [example for key in eval_ids for example in examples_by_article[key]]
                model, state, predictions, evidence = fit_once(config_name, seed, train_examples, eval_examples, tok, device)
                reload_max = reload_check(config_name, state, predictions, EncodedDataset(eval_examples, tok), tok, device)
                evidence.update(stage="forward_fold", fold=fold_number, train_articles=len(train_ids), eval_articles=len(eval_ids), reload_max_abs=reload_max)
                training_records.append(evidence)
                print(json.dumps({"stage": "forward_fold", "config": config_name, "seed": seed, "fold": fold_number, "best_epoch": evidence["best_epoch"], "epochs_ran": evidence["epochs_ran"], "seconds": round(evidence["elapsed_seconds"], 2)}), flush=True)
                for row in predictions:
                    row.update(config=config_name, seed=seed, fold=fold_number)
                    oof_predictions.append(row)
                append_jsonl(progress_training, [evidence])
                append_jsonl(progress_oof, predictions)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()

    threshold_rows, config_rows = [], []
    selected_thresholds = {}
    for config_name in CONFIGS:
        best = None
        for threshold in THRESHOLDS:
            seed_scores = []
            for seed in SEEDS:
                subset = [row for row in oof_predictions if row["config"] == config_name and row["seed"] == seed]
                metric = extraction_metrics(subset, threshold)
                score = adapter_score(metric)
                seed_scores.append(score)
                threshold_rows.append({"config": config_name, "threshold": threshold, "seed": seed, "score": score, **metric})
            candidate = {"config": config_name, "threshold": threshold, "mean_score": float(np.mean(seed_scores)), "std_score": float(np.std(seed_scores, ddof=1))}
            if best is None or (-candidate["mean_score"], candidate["threshold"]) < (-best["mean_score"], best["threshold"]):
                best = candidate
        selected_thresholds[config_name] = best["threshold"]
        config_rows.append(best)
    order = {"A0": 0, "A2": 1, "A1": 2}
    top = max(row["mean_score"] for row in config_rows)
    selected_config = sorted([row for row in config_rows if top - row["mean_score"] <= 0.002], key=lambda row: order[row["config"]])[0]["config"]

    full_predictions = jsonl(progress_eval) if progress_eval.exists() else []
    completed_full = {(row["config"], row["seed"]) for row in training_records if row["stage"] == "full_train"}
    full_train_examples = [example for row in train_inputs for example in examples_by_article[row["id"]]]
    evaluation_examples = [example for row in inputs if row["split"] in {"valid", "check"} for example in examples_by_article[row["id"]]]
    full_epochs = {}
    for config_name in CONFIGS:
        epochs = [row["best_epoch"] for row in training_records if row["config"] == config_name and row["stage"] == "forward_fold"]
        full_epochs[config_name] = max(1, int(round(statistics.mean(epochs))))
        for seed in SEEDS:
            if (config_name, seed) in completed_full:
                print(json.dumps({"stage": "resume_skip_full_train", "config": config_name, "seed": seed}), flush=True)
                continue
            model, state, _, evidence = fit_once(config_name, seed, full_train_examples, [], tok, device, fixed_epochs=full_epochs[config_name])
            evaluation_data = EncodedDataset(evaluation_examples, tok)
            predictions = infer(model, evaluation_data, tok, device)
            path = checkpoint_dir / f"{config_name}_seed{seed}.pt"
            torch.save({"state": state, "config": config_name, "seed": seed, "epochs": full_epochs[config_name]}, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
            reload_max = reload_check(config_name, reloaded["state"], predictions, evaluation_data, tok, device)
            evidence.update(stage="full_train", train_articles=len(train_inputs), eval_articles=len({row["article_id"] for row in predictions}), reload_max_abs=reload_max, checkpoint_sha256=sha(path))
            training_records.append(evidence)
            print(json.dumps({"stage": "full_train", "config": config_name, "seed": seed, "epochs": full_epochs[config_name], "seconds": round(evidence["elapsed_seconds"], 2)}), flush=True)
            for row in predictions:
                row.update(config=config_name, seed=seed)
                full_predictions.append(row)
            append_jsonl(progress_training, [evidence])
            append_jsonl(progress_eval, predictions)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    metrics_rows = []
    for config_name in CONFIGS:
        threshold = selected_thresholds[config_name]
        for seed in SEEDS:
            for split in ("valid", "check"):
                subset = [row for row in full_predictions if row["config"] == config_name and row["seed"] == seed and row["split"] == split]
                for symbol in ("ALL", "AAPL", "AMZN"):
                    metric = extraction_metrics(subset, threshold, symbol)
                    metrics_rows.append({"config": config_name, "seed": seed, "split": "development" if split == "valid" else split, "threshold": threshold, "score": adapter_score(metric), **metric})

    # Seed-ensemble sentence probabilities are the deployable extractors.
    ensemble_predictions = []
    for config_name in CONFIGS:
        lookup = collections.defaultdict(list)
        for row in full_predictions:
            if row["config"] == config_name:
                lookup[(row["split"], row["article_id"], row.get("model_unit_id", row["sentence_id"]))].append(row)
        for key, group in lookup.items():
            if len(group) != len(SEEDS):
                raise ValueError({"incomplete_seed_group": key, "n": len(group)})
            exemplar = group[0]
            ensemble_predictions.append(
                {
                    **{name: exemplar[name] for name in ("article_id", "symbol", "split", "available_utc", "record_key", "event_group", "sentence_id", "gold_evidence", "gold_actions")},
                    "model_unit_id": exemplar.get("model_unit_id", exemplar["sentence_id"]),
                    "model_chunk_index": exemplar.get("model_chunk_index", 0),
                    "model_chunk_count": exemplar.get("model_chunk_count", 1),
                    "config": config_name,
                    "evidence_probability": np.mean([row["evidence_probability"] for row in group], axis=0).tolist(),
                    "action_probability": np.mean([row["action_probability"] for row in group], axis=0).tolist(),
                }
            )
    ensemble_metrics = []
    for config_name in CONFIGS:
        threshold = selected_thresholds[config_name]
        for split in ("valid", "check"):
            subset = [row for row in ensemble_predictions if row["config"] == config_name and row["split"] == split]
            for symbol in ("ALL", "AAPL", "AMZN"):
                metric = extraction_metrics(subset, threshold, symbol)
                ensemble_metrics.append({"config": config_name, "seed": "ensemble", "split": "development" if split == "valid" else split, "threshold": threshold, "score": adapter_score(metric), **metric})

    write_jsonl(out / "oof_sentence_predictions.jsonl", oof_predictions)
    write_jsonl(out / "evaluation_sentence_predictions.jsonl", full_predictions)
    write_jsonl(out / "ensemble_sentence_predictions.jsonl", ensemble_predictions)
    pd.DataFrame(training_records).drop(columns="history").to_csv(out / "training_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "threshold_metrics.csv", index=False)
    pd.DataFrame(metrics_rows + ensemble_metrics).to_csv(out / "extraction_metrics.csv", index=False)
    dump(out / "training_curves.json", training_records)
    selection = {
        "selected_config": selected_config,
        "selected_thresholds": selected_thresholds,
        "config_forward_scores": config_rows,
        "full_epochs": full_epochs,
        "selection_source": "January-February chronological forward folds only",
        "development_or_check_used_for_selection": False,
    }
    dump(out / "selection.json", selection)
    evidence = {
        "status": "COMPLETE",
        "model": "ProsusAI/finbert",
        "revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
        "retrospective_2018_experiment": True,
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "trainable_parameters": {config: next(row["trainable_parameters"] for row in training_records if row["config"] == config) for config in CONFIGS},
        "runtime_seconds": sum(row["elapsed_seconds"] for row in training_records),
        "driver_wall_seconds_current_resume": time.time() - started,
        "runs": len(training_records),
        "checkpoint_reload_tolerance": 1e-6,
        "maximum_reload_difference": max(row["reload_max_abs"] for row in training_records),
        "source_hashes": {"inputs": sha(ANNOTATION / "inputs.jsonl"), "labels": sha(ANNOTATION / "labels.jsonl"), "protocol": sha(PUBLIC / "protocol.json")},
        "selection": selection,
    }
    dump(out / "training_evidence.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.out, args.resume)
