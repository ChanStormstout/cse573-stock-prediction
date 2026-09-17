"""Pinned Qwen binary-choice inference for P0--P3 with resumable immutable logs."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

from common import B, W, dump, jsonl, sha
from paragraph_inputs import MODEL, SYSTEM, VARIANTS, messages

os.environ["HF_HOME"] = str(W / "model_compare/hf-home")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PINNED = W / "model_compare/comparison_v1/manifest.json"
OLD_EVAL = W / "direct_4h/choice_v1/joint.jsonl"


def check_inputs(directory: Path, seal: dict):
    for variant, fingerprint in seal["input_sha256"].items():
        if sha(directory / f"{variant}.jsonl") != fingerprint:
            raise ValueError(f"input fingerprint mismatch: {variant}")


def main(inputs: Path, out: Path, variants, smoke: bool, resume: bool, batch_size: int, smoke_count: int):
    seal = json.loads((inputs / "manifest.json").read_text())
    quality = json.loads((inputs / "quality.json").read_text())
    manual = json.loads((inputs / "manual_review.json").read_text())
    if not quality["gate"] or manual["status"] != "PASS_ENGINEERING_GATE":
        raise ValueError("paragraph quality gate not passed")
    check_inputs(inputs, seal)
    pinned = json.loads(PINNED.read_text())
    for filename, fingerprint in pinned["model_files"].items():
        if sha(MODEL / filename) != fingerprint:
            raise ValueError(f"pinned model mismatch: {filename}")
    if seal["model_revision"] != pinned["revision"]:
        raise ValueError("revision mismatch")
    source_eval = jsonl(OLD_EVAL)
    if len(source_eval) != 609 or any(row["variant"] != "joint" for row in source_eval):
        raise ValueError("old P0 evaluation source mismatch")

    run_manifest = {
        "model": pinned["model"],
        "revision": pinned["revision"],
        "model_files": pinned["model_files"],
        "input_sha256": seal["input_sha256"],
        "quality_sha256": sha(inputs / "quality.json"),
        "manual_review_sha256": sha(inputs / "manual_review.json"),
        "variants": variants,
        "temperature": 0,
        "seed": 573,
        "thinking": False,
        "max_output_tokens": 1,
        "max_prompt_tokens": 6000,
        "answer_interface": "conditional_UP_DOWN_token_logprob",
        "batch_order": "prompt_token_count_then_key; outcome labels never loaded",
        "system_prompt_sha256": seal["system_prompt_sha256"],
        "training": False,
        "smoke": smoke,
        "batch_size": batch_size,
        "smoke_count": smoke_count if smoke else None,
        "old_P0_eval_sha256": sha(OLD_EVAL),
        "code": {path.name: sha(path) for path in (Path(__file__), B / "paragraph_inputs.py", B / "common.py", B / "PROTOCOL.md")},
    }
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        if json.loads((out / "manifest.json").read_text()) != run_manifest:
            raise ValueError("resume fingerprint mismatch")
        if (out / "summary.json").exists():
            raise ValueError("run is already complete")
    else:
        out.mkdir(parents=True)
        dump(out / "manifest.json", run_manifest)
        (out / "code").mkdir()
        for name in run_manifest["code"]:
            shutil.copy2(B / name, out / "code" / name)

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(str(MODEL))
    model.eval()
    mx.random.seed(573)
    answer_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("UP", "DOWN")]
    if any(len(value) != 1 for value in answer_ids) or answer_ids[0] == answer_ids[1]:
        raise ValueError("UP/DOWN must be distinct single tokens")
    answer_ids = [value[0] for value in answer_ids]
    if [tokenizer.decode([value]) for value in answer_ids] != ["UP", "DOWN"]:
        raise ValueError("answer token decode mismatch")

    summary = {}
    total_start = time.time()
    for variant in variants:
        source_rows = jsonl(inputs / f"{variant}.jsonl")
        if len(source_rows) != 1607:
            raise ValueError("wrong input row count")
        if smoke:
            source_rows = source_rows[:smoke_count]
        reused = {}
        if variant == "P0" and not smoke:
            reused = {row["key"]: {**row, "variant": "P0", "reused_from": str(OLD_EVAL)} for row in source_eval}
        infer_rows = [row for row in source_rows if row["key"] not in reused]
        prepared = []
        for row in infer_rows:
            msg = messages(row)
            rendered = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            limit = 6000 if variant in {"P2", "P3"} else None
            if limit is not None and len(tokens) > limit:
                raise ValueError("prompt exceeds registered ceiling")
            if len(tokens) + 1 > 6144:
                raise ValueError("prompt would truncate")
            prepared.append((row, msg, tokens))
        prepared.sort(key=lambda value: (len(value[2]), value[0]["key"]))
        log_path = out / f"{variant}.generated.jsonl"
        done = jsonl(log_path)
        expected_prefix = [row[0]["key"] for row in prepared[: len(done)]]
        if [row["key"] for row in done] != expected_prefix:
            raise ValueError("resume prefix mismatch")
        with log_path.open("a") as stream:
            for offset in range(len(done), len(prepared), batch_size):
                check_inputs(inputs, seal)
                batch = prepared[offset : offset + batch_size]
                began = time.time()
                generator = BatchGenerator(
                    model,
                    stop_tokens=[[token] for token in tokenizer.eos_token_ids],
                    sampler=make_sampler(temp=0),
                    completion_batch_size=batch_size,
                    prefill_batch_size=batch_size,
                    prefill_step_size=512,
                )
                uids = generator.insert([tokens for _, _, tokens in batch], [1] * len(batch))
                scores = {}
                try:
                    while responses := generator.next_generated():
                        for response in responses:
                            full = response.logprobs.astype(mx.float32)
                            full = full - mx.logsumexp(full)
                            selected = full[mx.array(answer_ids)]
                            mx.eval(selected)
                            scores[response.uid] = (
                                selected.tolist(),
                                int(response.token),
                                response.finish_reason,
                            )
                finally:
                    generator.close()
                elapsed = time.time() - began
                for (row, msg, tokens), uid in zip(batch, uids):
                    logps, top_token, finish = scores[uid]
                    if not all(math.isfinite(value) for value in logps):
                        raise ValueError("non-finite token score")
                    maximum = max(logps)
                    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logps))
                    p_up = math.exp(logps[0] - normalizer)
                    record = {
                        "key": row["key"],
                        "variant": variant,
                        "p": p_up,
                        "valid": True,
                        "parsed": {"direction": "UP" if p_up >= 0.5 else "DOWN", "p_up": p_up},
                        "logp_up": logps[0],
                        "logp_down": logps[1],
                        "choice_mass": math.exp(normalizer),
                        "unconstrained_token": tokenizer.decode([top_token]),
                        "unconstrained_token_id": top_token,
                        "seconds": elapsed / len(batch),
                        "batch_seconds": elapsed,
                        "prompt_tokens": len(tokens),
                        "output_tokens": 1,
                        "finish_reason": "scored",
                        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
                        "messages": msg,
                        "reused_from": None,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    done.append(record)
                mx.clear_cache()
                if len(done) % 40 == 0 or len(done) == len(prepared):
                    print(variant, len(done), "/", len(prepared), "new; reused", len(reused), "seconds", round(sum(row["seconds"] for row in done), 1), flush=True)
        generated = {row["key"]: row for row in done}
        complete = []
        for row in source_rows:
            record = reused.get(row["key"], generated.get(row["key"]))
            if record is None:
                raise AssertionError("incomplete variant")
            complete.append(record)
        final_path = out / f"{variant}.jsonl"
        with final_path.open("w") as stream:
            for record in complete:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary[variant] = {
            "rows": len(complete),
            "generated": len(done),
            "reused": len(reused),
            "seconds_generated": sum(row["seconds"] for row in done),
            "peak_mlx_gb": max((row["peak_mlx_gb"] for row in done), default=None),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in complete),
            "output_sha256": sha(final_path),
        }
        dump(out / "progress.json", summary)
    check_inputs(inputs, seal)
    dump(out / "summary.json", {"status": "COMPLETE", "variants": summary, "process_seconds": time.time() - total_start, "training": False, "labels_loaded": False, "source_hashes_unchanged": True})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke-count", type=int, default=1)
    args = parser.parse_args()
    main(args.inputs, args.out, args.variants, args.smoke, args.resume, args.batch_size, args.smoke_count)
