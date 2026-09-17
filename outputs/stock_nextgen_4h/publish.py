"""Create sanitized public results, Chinese report and reproducibility record."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import B, DIRECT, ROOT, dump, jsonl, metrics, sha


def percent(value):
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:.2f}%"


def metric_table(frame, methods):
    labels = {
        "F0": "F0 价格＋标题 baseline（无新闻回退 R1）",
        "F1": "F1 价格＋全文词特征",
        "F2": "F2 价格＋FinBERT",
        "F3": "F3 全文／FinBERT 概率融合",
        "F4": "F4 全文／段落 LLM 概率融合",
        "F5": "F5 全文／FinBERT／LLM 概率融合",
        "F6": "F6 F5＋近期价格概率",
        "R0": "R0 原六小时价格",
        "R1": "R1 加近期五分钟特征",
        "R2": "R2 仅近期价格",
        "S2": "S2 两股完全共享",
        "S3": "S3 部分共享",
        "llm": "训练期选出的价格＋段落 LLM",
    }
    lines = ["| 方法 | AAPL validation BA / Brier | AAPL later BA / Brier | AMZN validation BA / Brier | AMZN later BA / Brier |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        values = []
        for symbol in ("AAPL", "AMZN"):
            for period in ("validation", "test"):
                row = frame[(frame.symbol == symbol) & (frame.period == period) & (frame.method == method)].iloc[0]
                values.append(f"{percent(row.BA)} / {row.Brier:.4f}")
        lines.append(f"| {labels[method]} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    return "\n".join(lines)


def portable_paths(value):
    """Replace repository-root prefixes in nested public metadata."""
    root_prefix = str(ROOT) + "/"
    if isinstance(value, dict):
        return {
            (key[len(root_prefix):] if isinstance(key, str) and key.startswith(root_prefix) else key): portable_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(root_prefix):
        return value[len(root_prefix):]
    return value


def bootstrap_difference(frame, challenger, block, draws=5000):
    rng = np.random.default_rng(573 + block)
    days = sorted(frame.day.unique())
    daily = []
    for day in days:
        group = frame[frame.day == day]
        y = group.label.to_numpy(dtype=int)
        base = group.F0.to_numpy() >= 0.5
        test = group[challenger].to_numpy() >= 0.5
        daily.append(
            [
                int((y == 1).sum()),
                int((y == 0).sum()),
                int(((y == 1) & base).sum()),
                int(((y == 0) & ~base).sum()),
                int(((y == 1) & test).sum()),
                int(((y == 0) & ~test).sum()),
            ]
        )
    daily = np.asarray(daily, dtype=float)
    values = []
    for _ in range(draws):
        if block == 1:
            sampled = rng.integers(0, len(days), size=len(days)).tolist()
        else:
            sampled = []
            while len(sampled) < len(days):
                start = int(rng.integers(0, len(days)))
                sampled.extend((start + offset) % len(days) for offset in range(block))
            sampled = sampled[: len(days)]
        weights = np.bincount(sampled, minlength=len(days))
        totals = weights @ daily
        if totals[0] and totals[1]:
            baseline_ba = 0.5 * (totals[2] / totals[0] + totals[3] / totals[1])
            challenger_ba = 0.5 * (totals[4] / totals[0] + totals[5] / totals[1])
            values.append(challenger_ba - baseline_ba)
    observed = metrics(frame.label, frame[challenger])["BA"] - metrics(frame.label, frame.F0)["BA"]
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values)


def main(inventory: Path, paragraphs: Path, llm: Path, price: Path, calibration: Path, fusion: Path, cases: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    implementation_files = [
        B / name
        for name in (
            "PROTOCOL.md", "config.json", "common.py", "inventory.py",
            "paragraph_inputs.py", "input_quality.py", "recent_price.py",
            "llm_choice.py", "calibrate.py", "fusion.py", "cases.py",
            "publish.py", "verify.py", "run_all.py", "test_contract.py",
        )
    ]
    required = [
        inventory / "inventory.json",
        paragraphs / "quality.json",
        paragraphs / "manual_review.json",
        llm / "summary.json",
        price / "status.json",
        calibration / "status.json",
        fusion / "status.json",
        cases / "status.json",
    ]
    for path in required:
        if json.loads(path.read_text())["status"] not in {"COMPLETE", "PASS_ENGINEERING_GATE", "MECHANICAL_PASS_MANUAL_REVIEW_REQUIRED"}:
            raise ValueError(f"incomplete prerequisite: {path}")
    inventory_data = json.loads((inventory / "inventory.json").read_text())
    quality = json.loads((paragraphs / "quality.json").read_text())
    review = json.loads((paragraphs / "manual_review.json").read_text())
    llm_summary = json.loads((llm / "summary.json").read_text())
    calibration_status = json.loads((calibration / "status.json").read_text())
    calibration_selection = json.loads((calibration / "selection.json").read_text())
    fusion_selection = json.loads((fusion / "selection.json").read_text())
    price_selection = json.loads((price / "selection.json").read_text())
    paragraph_choice = calibration_selection["paragraph"]["chosen"]["variant"]

    fusion_metrics = pd.read_csv(fusion / "metrics.csv")
    fusion_predictions = pd.read_csv(fusion / "predictions.csv")
    calibration_predictions = pd.read_csv(calibration / "predictions.csv")
    calibration_metrics = pd.read_csv(calibration / "metrics.csv")
    price_metrics = pd.read_csv(price / "metrics.csv")
    public_metrics = fusion_metrics.copy()
    public_metrics["source"] = "nextgen_fusion"
    # Preserve calibrated paragraph variants as explicit single-method results.
    extra = calibration_metrics[calibration_metrics.method.astype(str).str.match(r"^P[0-3]_")].copy()
    extra["phase"] = "frozen"
    extra["source"] = "paragraph_calibration"
    public_metrics = pd.concat([public_metrics, extra], ignore_index=True, sort=False)
    historical = pd.read_csv(DIRECT / "METRICS.csv")
    historical = historical[historical.method.isin(["legacy_price", "legacy_title", "legacy_body", "legacy_semantic", "legacy_integrated", "llm_price", "llm_news", "llm_joint"]) & historical.period.isin(["validation", "test"])].copy()
    historical["phase"] = "historical_frozen"
    historical["source"] = "prior_matched_experiment"
    public_metrics = pd.concat([public_metrics, historical], ignore_index=True, sort=False)
    public_metrics.to_csv(out / "METRICS.csv", index=False)
    public_metrics[public_metrics.period.astype(str).str.match(r"^2018-|^2019-")].to_csv(out / "MONTHLY.csv", index=False)

    keep = ["key", "symbol", "day", "month", "split", "label", "has_news", "R0", "R1", "R2", "S2", "S3", "llm", "F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    eval_predictions = fusion_predictions[fusion_predictions.phase.eq("frozen")][keep].copy()
    eval_predictions.to_csv(out / "PREDICTIONS.csv", index=False)
    fusion_predictions[fusion_predictions.phase.eq("outer")][keep].to_csv(out / "OOF_PREDICTIONS.csv", index=False)
    shutil.copy2(price / "cv.csv", out / "PRICE_CV.csv")
    shutil.copy2(price / "fits.json", out / "PRICE_TRAINING.json")
    shutil.copy2(calibration / "calibration_fits.json", out / "CALIBRATION_FITS.json")
    shutil.copy2(fusion / "weight_grid.csv", out / "WEIGHT_GRID.csv")

    # Publish all numeric LLM outputs while keeping source messages, news text
    # and URLs in the private work directory.
    llm_public = calibration_predictions[[
        "key", "symbol", "start_utc", "end_utc", "cutoff_utc", "split", "month",
        "label", "has_news", "R1",
        *[f"{variant}_{suffix}" for variant in ("P0", "P1", "P2", "P3") for suffix in ("observed", "raw", "cal", "strict")],
        "paragraph_selected", "phase",
    ]].copy()
    for variant in ("P0", "P1", "P2", "P3"):
        records = {row["key"]: row for row in jsonl(llm / f"{variant}.jsonl")}
        for field in ("logp_up", "logp_down", "choice_mass", "prompt_tokens", "seconds"):
            llm_public[f"{variant}_{field}"] = llm_public.key.map({key: row.get(field) for key, row in records.items()})
    llm_public.to_csv(out / "LLM_SCORES.csv", index=False)

    shutil.copy2(fusion / "transitions.csv", out / "TRANSITIONS.csv")
    shutil.copy2(calibration / "paragraph_transitions.csv", out / "PARAGRAPH_TRANSITIONS.csv")
    shutil.copy2(cases / "CASE_NOTES.md", out / "CASE_NOTES.md")
    shutil.copy2(inventory / "DATA_INVENTORY.md", out / "DATA_INVENTORY.md")
    with (out / "DATA_INVENTORY.md").open("a") as stream:
        stream.write("\n## External feasibility sources\n\n- [Alpha Vantage official documentation](https://www.alphavantage.co/documentation/)\n- [Polygon minute aggregate documentation](https://polygon.io/docs/flat-files/stocks/minute-aggregates/2009/03)\n- [Twelve Data historical data guidance](https://support.twelvedata.com/en/articles/5214728-getting-historical-data)\n")

    intervals = []
    for (symbol, split), frame in eval_predictions.groupby(["symbol", "split"]):
        for challenger in ("R1", "S2", "S3", "llm", "F1", "F2", "F3", "F4", "F5", "F6"):
            for block in (1, 5):
                observed, lower, upper, draws = bootstrap_difference(frame, challenger, block)
                intervals.append({"symbol": symbol, "period": split, "challenger": challenger, "baseline": "F0", "block_days": block, "BA_difference": observed, "lower_2.5": lower, "upper_97.5": upper, "draws": draws, "interpretation": "descriptive after repeated exploration"})
    pd.DataFrame(intervals).to_csv(out / "PAIRED_INTERVALS.csv", index=False)

    subgroup_source = eval_predictions.merge(
        calibration_predictions[["key", "P0_strict", "P1_strict", "P2_strict", "P3_strict"]],
        on="key",
        validate="one_to_one",
    )
    subgroup_rows = []
    subgroup_methods = ["R0", "R1", "R2", "S2", "S3", "llm", "F0", "F1", "F2", "F3", "F4", "F5", "F6", "P0_strict", "P1_strict", "P2_strict", "P3_strict"]
    for (symbol, split, has_news), sample in subgroup_source.groupby(["symbol", "split", "has_news"]):
        for method in subgroup_methods:
            subgroup_rows.append({"symbol": symbol, "period": split, "news_scope": "news" if has_news else "no_news", "method": method, **metrics(sample.label, sample[method])})
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    subgroup_metrics.to_csv(out / "SUBGROUP_METRICS.csv", index=False)

    token_rows = pd.read_csv(paragraphs / "tokens.csv")
    quality_rows = []
    for (symbol, variant), group in token_rows.groupby(["symbol", "variant"]):
        quality_rows.append({"symbol": symbol, "variant": variant, "rows": len(group), "mean_prompt_tokens": group.prompt_tokens.mean(), "max_prompt_tokens": group.prompt_tokens.max(), "mean_passages": group.passages.mean(), "zero_passage_rows": int(group.passages.eq(0).sum())})
    pd.DataFrame(quality_rows).to_csv(out / "INPUT_QUALITY.csv", index=False)
    raw_news = pd.read_pickle(ROOT / "work" / "stock-data" / "audit" / "news_index.pkl")
    raw_news["record_key"] = raw_news.archive.astype(str) + "::" + raw_news.member.astype(str)
    source_by_key = raw_news.set_index("record_key").site.astype(str).to_dict()
    cluster_occurrences = pd.read_csv(paragraphs / "clusters.csv")
    cluster_occurrences["source_count"] = cluster_occurrences.members.map(
        lambda value: len({source_by_key[key] for key in str(value).split("|")})
    )
    dedup_rows = []
    for symbol, group in cluster_occurrences.groupby("symbol"):
        dedup_rows.append(
            {
                "symbol": symbol,
                "window_cluster_occurrences": len(group),
                "multi_report_occurrences": int(group.reports.gt(1).sum()),
                "reports_collapsed": int((group.reports - 1).clip(lower=0).sum()),
                "mean_reports_per_cluster": float(group.reports.mean()),
                "max_reports_in_cluster": int(group.reports.max()),
                "mean_independent_sources_per_cluster": float(group.source_count.mean()),
                "max_independent_sources_in_cluster": int(group.source_count.max()),
            }
        )
    pd.DataFrame(dedup_rows).to_csv(out / "DEDUP_AUDIT.csv", index=False)
    input_lines = [
        "# Paragraph input quality",
        "",
        f"Training-only mechanical audit checked {quality['passages_checked']:,} window-passage occurrences (repeated articles can appear in adjacent windows): exact raw-span rate {quality['exact_span_rate']:.3f}, explicit-target rate {quality['explicit_target_rate']:.3f}, complete-unit rate {quality['complete_unit_rate']:.3f}.",
        "",
        f"In the 16 fixed blind cards, P0 contained {quality['blind_case_P0_fragments']} character-cut fragments and P1 contained {quality['blind_case_P1_fragments']}. P1 exceeded its matched P0 prompt budget in {quality['P1_token_violations']} rows. P3 removed {quality['deduplicated_training_reports']} repeated training-period reports.",
        "",
        f"Assistant review status: **{review['status']}**. This was not an independent review. Background/stale facts can remain when they explicitly concern the target company.",
        "",
        "P3 keeps report counts in its sealed window input and independent-source counts in `DEDUP_AUDIT.csv`; raw member identities and article text remain private. Counts are window-cluster occurrences, so a report can recur in adjacent prediction windows.",
        "",
        "| Stock | Variant | Rows | Mean tokens | Max tokens | Mean passages | Zero-passage rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in quality_rows:
        input_lines.append(f"| {row['symbol']} | {row['variant']} | {row['rows']} | {row['mean_prompt_tokens']:.1f} | {int(row['max_prompt_tokens'])} | {row['mean_passages']:.2f} | {row['zero_passage_rows']} |")
    (out / "INPUT_QUALITY.md").write_text("\n".join(input_lines) + "\n")

    calibration_rows = []
    for variant in ("P0", "P1", "P2", "P3"):
        selected = calibration_selection[variant]
        calibration_rows.append({"variant": variant, **selected})
    pd.DataFrame(calibration_rows).to_csv(out / "CALIBRATION.csv", index=False)
    calibration_oof = pd.read_csv(calibration / "calibration_metrics.csv")
    calibration_oof.to_csv(out / "CALIBRATION_OOF.csv", index=False)
    cal_lines = ["# LLM calibration", "", "Calibration type was selected globally from March--August past-only forward predictions by Brier, then BA. Fits themselves remain stock-specific and past-only.", "", "The private inference ledger preserves every observed batch score. For the controlled sequence P0→P1→P2→P3, a byte-for-byte identical prompt carries forward the preceding variant's normalized score; this prevents batch-composition numerical drift from being counted as content selection, added context or deduplication. `LLM_SCORES.csv` contains both `*_observed` and normalized `*_raw` values.", "", "| Variant | Selected calibration | Mean OOF BA | Mean OOF Brier | Mean OOF ECE |", "|---|---|---:|---:|---:|"]
    for row in calibration_rows:
        cal_lines.append(f"| {row['variant']} | {row['method']} | {percent(row['mean_BA'])} | {row['mean_Brier']:.4f} | {row['mean_ECE10']:.4f} |")
    cal_lines += ["", f"Global paragraph choice after strict R1 fallback: **{paragraph_choice}**. The choice used the registered P0 Brier guardrail and no later-period labels.", "", "## No-news policies on training-period forward predictions", "", "`policy_raw` keeps the calibrated model on news rows and uses the raw LLM prior on no-news rows. `policy_price` uses strict R1 fallback. `policy_learned` uses a past-only no-news Platt fit when enough examples exist, otherwise R1.", "", "| Variant | Policy | Mean OOF BA | Mean OOF Brier | Mean OOF ECE |", "|---|---|---:|---:|---:|"]
    for row in calibration_oof[calibration_oof.scope.eq("no_news_policy_oof")].itertuples():
        cal_lines.append(f"| {row.variant} | {row.method} | {percent(row.mean_BA)} | {row.mean_Brier:.4f} | {row.mean_ECE10:.4f} |")
    (out / "CALIBRATION.md").write_text("\n".join(cal_lines) + "\n")

    dump(out / "SELECTION.json", {"price": price_selection, "calibration": calibration_selection, "fusion": fusion_selection})
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "mlx", "mlx-lm", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    def sysctl(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except Exception:
            return "unavailable"
    environment = {"python": sys.version, "platform": platform.platform(), "hardware_model": sysctl("hw.model"), "memory_bytes": sysctl("hw.memsize"), "packages": packages, "model": json.loads((llm / "manifest.json").read_text())["model"], "model_revision": json.loads((llm / "manifest.json").read_text())["revision"], "paid_API": False, "Sol_used": False}
    dump(out / "ENVIRONMENT.json", environment)
    (out / "ENVIRONMENT.md").write_text("# Environment\n\n```json\n" + json.dumps(environment, indent=2) + "\n```\n")
    execution = portable_paths({"status": "COMPLETE", "task": "AAPL/AMZN four-hour direction", "rows": 1607, "LLM": llm_summary, "price": json.loads((price / "status.json").read_text()), "calibration": calibration_status, "fusion": json.loads((fusion / "status.json").read_text()), "cases": json.loads((cases / "status.json").read_text()), "implementation_sha256": {str(path.relative_to(ROOT)): sha(path) for path in implementation_files}, "market_branch": inventory_data["decision"], "interpretation": "all later periods are exposed exploratory historical replays"})
    dump(out / "EXECUTION.json", execution)
    command_lines = ["# Reproduction commands", "", "Set `PYTHON` to the project Python environment, then run:", "", "```bash", "${PYTHON} outputs/stock_nextgen_4h/inventory.py --out work/stock-data/nextgen_4h/inventory_v2", "${PYTHON} outputs/stock_nextgen_4h/paragraph_inputs.py --out work/stock-data/nextgen_4h/paragraphs_v4", "${PYTHON} outputs/stock_nextgen_4h/input_quality.py --inputs work/stock-data/nextgen_4h/paragraphs_v4", "${PYTHON} outputs/stock_nextgen_4h/recent_price.py --out work/stock-data/nextgen_4h/price_v1", "${PYTHON} outputs/stock_nextgen_4h/llm_choice.py --inputs work/stock-data/nextgen_4h/paragraphs_v4 --out work/stock-data/nextgen_4h/llm_v1 --batch-size 4", "${PYTHON} outputs/stock_nextgen_4h/calibrate.py --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --llm work/stock-data/nextgen_4h/llm_v1 --price work/stock-data/nextgen_4h/price_v1 --out work/stock-data/nextgen_4h/calibration_v1", "${PYTHON} outputs/stock_nextgen_4h/fusion.py --price work/stock-data/nextgen_4h/price_v1 --calibration work/stock-data/nextgen_4h/calibration_v1 --out work/stock-data/nextgen_4h/fusion_v1", "${PYTHON} outputs/stock_nextgen_4h/cases.py --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --calibration work/stock-data/nextgen_4h/calibration_v1 --fusion work/stock-data/nextgen_4h/fusion_v1 --out work/stock-data/nextgen_4h/cases_v1", "${PYTHON} outputs/stock_nextgen_4h/publish.py --inventory work/stock-data/nextgen_4h/inventory_v2 --paragraphs work/stock-data/nextgen_4h/paragraphs_v4 --llm work/stock-data/nextgen_4h/llm_v1 --price work/stock-data/nextgen_4h/price_v1 --calibration work/stock-data/nextgen_4h/calibration_v1 --fusion work/stock-data/nextgen_4h/fusion_v1 --cases work/stock-data/nextgen_4h/cases_v1 --out outputs/stock_nextgen_4h/runs/v1", "${PYTHON} outputs/stock_nextgen_4h/verify.py --work work/stock-data/nextgen_4h --public outputs/stock_nextgen_4h/runs/v1", "```", "", "Failed inventory v1, paragraph input versions v1--v3 and unlabeled LLM smoke/benchmark runs are retained locally."]
    (out / "COMMANDS.md").write_text("\n".join(command_lines) + "\n")

    core = fusion_metrics[(fusion_metrics.phase == "frozen") & fusion_metrics.period.isin(["validation", "test"])]
    methods = ["R0", "R1", "R2", "S2", "S3", "llm", "F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    uniform_above_random = []
    uniform_matched_gain = []
    for method in methods:
        rows = core[core.method.eq(method)]
        if len(rows) == 4 and rows.BA.gt(0.5).all():
            uniform_above_random.append(method)
        if len(rows) == 4:
            passed = True
            for row in rows.itertuples():
                baseline_row = core[(core.symbol == row.symbol) & (core.period == row.period) & (core.method == "F0")].iloc[0]
                passed = passed and row.BA > baseline_row.BA and row.Brier <= baseline_row.Brier + 0.002
            if passed:
                uniform_matched_gain.append(method)
    stable = True
    for symbol in ("AAPL", "AMZN"):
        for period in ("validation", "test"):
            f0 = core[(core.symbol == symbol) & (core.period == period) & (core.method == "F0")].iloc[0]
            f6 = core[(core.symbol == symbol) & (core.period == period) & (core.method == "F6")].iloc[0]
            stable = bool(stable and f6.BA > f0.BA and f6.Brier <= f0.Brier + 0.002)
    conclusion = "F6 在两股和两个暴露时期均超过 F0，并满足概率误差护栏；仍需新时期确认。" if stable else "没有找到同时跨两只股票、两个暴露时期稳定超过 F0 且守住概率误差的统一方法。"
    old_labels = {
        "legacy_price": "旧价格 LR",
        "legacy_title": "旧价格＋标题 baseline",
        "legacy_body": "旧价格＋全文词特征",
        "legacy_semantic": "旧价格＋FinBERT",
        "legacy_integrated": "旧完整融合",
        "llm_price": "旧直接 LLM：价格",
        "llm_news": "旧直接 LLM：新闻",
        "llm_joint": "旧直接 LLM：价格＋新闻",
    }
    old_lines = ["| 已有方法 | AAPL validation BA / Brier | AAPL later BA / Brier | AMZN validation BA / Brier | AMZN later BA / Brier |", "|---|---:|---:|---:|---:|"]
    for method, label in old_labels.items():
        values = []
        for symbol in ("AAPL", "AMZN"):
            for period in ("validation", "test"):
                row = historical[(historical.symbol == symbol) & (historical.period == period) & (historical.method == method)].iloc[0]
                values.append(f"{percent(row.BA)} / {row.Brier:.4f}")
        old_lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    paragraph_lines = ["| 段落版本（校准＋无新闻R1回退） | AAPL validation | AAPL later | AMZN validation | AMZN later |", "|---|---:|---:|---:|---:|"]
    for variant in ("P0", "P1", "P2", "P3"):
        values = []
        for symbol in ("AAPL", "AMZN"):
            for period in ("validation", "test"):
                row = calibration_metrics[(calibration_metrics.symbol == symbol) & (calibration_metrics.period == period) & (calibration_metrics.method == f"{variant}_strict")].iloc[0]
                values.append(f"{percent(row.BA)} / {row.Brier:.4f}")
        paragraph_lines.append(f"| {variant} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    paragraph_transitions = pd.read_csv(calibration / "paragraph_transitions.csv")
    transition_lines = ["| P0 对比 | AAPL validation 改对/改错 | AAPL later 改对/改错 | AMZN validation 改对/改错 | AMZN later 改对/改错 |", "|---|---:|---:|---:|---:|"]
    for variant in ("P1", "P2", "P3"):
        values = []
        for symbol in ("AAPL", "AMZN"):
            for period in ("validation", "test"):
                row = paragraph_transitions[(paragraph_transitions.symbol == symbol) & (paragraph_transitions.period == period) & (paragraph_transitions.variant == variant)].iloc[0]
                values.append(f"{int(row.P0_wrong_new_right)}/{int(row.P0_right_new_wrong)}")
        transition_lines.append(f"| {variant} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    news_lines = ["| 有新闻窗口 | AAPL validation BA / n | AAPL later BA / n | AMZN validation BA / n | AMZN later BA / n |", "|---|---:|---:|---:|---:|"]
    for method, label in (("F0", "F0价格＋标题baseline"), ("llm", "训练期选择的价格＋段落LLM"), ("F6", "最终融合F6")):
        values = []
        for symbol in ("AAPL", "AMZN"):
            for period in ("validation", "test"):
                row = subgroup_metrics[(subgroup_metrics.symbol == symbol) & (subgroup_metrics.period == period) & (subgroup_metrics.news_scope == "news") & (subgroup_metrics.method == method)].iloc[0]
                values.append(f"{percent(row.BA)} / {int(row.n)}")
        news_lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} | {values[3]} |")
    price_choice_lines = ["| 价格模型 | 训练范围 | 最终 C |", "|---|---|---:|"]
    for row in price_selection:
        if row["phase"] == "frozen":
            price_choice_lines.append(f"| {row['method']} | {row['symbol']} | {row['C']} |")
    weight_lines = ["| 融合 | 非零训练期权重 | Mean OOF BA | Mean OOF Brier |", "|---|---|---:|---:|"]
    for method in ("F3", "F4", "F5", "F6"):
        selected = fusion_selection["methods"][method]
        if selected["action"] == "selected":
            terms = " + ".join(f"{component}={weight:.2f}" for component, weight in zip(selected["components"], selected["weights"]) if weight > 0)
            weight_lines.append(f"| {method} | {terms} | {percent(selected['mean_BA'])} | {selected['mean_Brier']:.4f} |")
        else:
            weight_lines.append(f"| {method} | fallback F0 | — | — |")
    report = [
        "# CSE 573 四小时股票方向：新一轮完整实验",
        "",
        "## 先说结论",
        "",
        conclusion,
        "",
        "四个股票×时期单元的 BA 都高于50%的方法：" + ("、".join(uniform_above_random) if uniform_above_random else "无") + "。同时在四个单元都超过F0且满足Brier护栏的方法：" + ("、".join(uniform_matched_gain) if uniform_matched_gain else "无") + "。这是完整表的描述，不是用后期标签重新选定部署规则。",
        "",
        "这不表示实验白做了。近期五分钟价格、跨股票共享、目标公司完整段落、去重、LLM 校准和低容量融合都已在同一 1,607 个窗口上完成可复现实验。结果能区分哪些组件只在某只股票或某个时期有效，以及哪些改善了概率但没有改善方向。",
        "",
        "所有 September 2018 之后的数据早已参与历史探索，本报告只能称探索性历史回放。",
        "",
        "## 主要结果",
        "",
        metric_table(core, methods),
        "",
        "BA 的 50% 是随机方向参照；Brier 越低越好，恒定输出 0.5 的 Brier 为 0.25。单段高分不能当作稳定泛化证据。",
        "",
        "### P0--P3 同协议对照",
        "",
        "\n".join(paragraph_lines),
        "",
        "下面的改对／改错按同一窗口与P0配对；它说明变化的净方向，不把个别案例当作因果证据。",
        "",
        "\n".join(transition_lines),
        "",
        "### 有新闻窗口",
        "",
        "\n".join(news_lines),
        "",
        "F0、F1、F2 与段落LLM都含价格信息；F3--F6 是这些概率的晚期融合。无新闻窗口的 F0--F6 和段落LLM均按预设严格回退 R1；完整分组指标见 `SUBGROUP_METRICS.csv`。",
        "",
        "### 训练期冻结的参数与融合权重",
        "",
        "\n".join(price_choice_lines),
        "",
        "\n".join(weight_lines),
        "",
        "这些参数只由训练期向前记录决定；`SELECTION.json` 保存全部校准候选、段落选择和权重网格选择结果。",
        "",
        "## 与此前方法的原始结果对照",
        "",
        "\n".join(old_lines),
        "",
        "这些历史行保留各自原协议；它们没有全部采用本轮的严格无新闻回退，因此用于追踪项目演化，不能把差值全部归因于单一组件。",
        "",
        "## 这轮具体解决了什么",
        "",
        f"1. **新闻输入**：P0 的字符切片改为完整目标公司单位。质量审计覆盖 {quality['passages_checked']:,} 个训练期“窗口—段落”出现次数（相邻窗口可重复同文）；P1 与 P0 使用相同 token 上限，P2 放宽到 6,000 tokens，P3 再做只看截止前文章的事件去重。训练期盲审中字符断句从 {quality['blind_case_P0_fragments']} 降到 {quality['blind_case_P1_fragments']}。",
        f"2. **LLM**：本地 Qwen3.5-9B 4-bit 对 P0--P3 全量给出 UP/DOWN token 分数；训练期选出的段落版本为 **{paragraph_choice}**。Platt／温度校准全部只使用过去月份。相邻版本逐字节相同输入的归一化行数为 P0→P1 {calibration_status['same_prompt_score_normalization']['P1']['rows']}、P1→P2 {calibration_status['same_prompt_score_normalization']['P2']['rows']}、P2→P3 {calibration_status['same_prompt_score_normalization']['P3']['rows']}；观测分数仍单独保留。",
        "3. **近期价格**：加入截止前 5/15/30/60 分钟收益、范围、实现波动、隔夜缺口和日内位置；行情第七列语义未知，因此没有冒充成交量。",
        "4. **共享学习**：S2 完全共享、S3 部分共享与独立 R1 使用相同样本和网格。共享能否帮助因股票和时期而异，表格中没有用后期标签挑每股赢家。",
        "5. **融合**：F3--F6 的非负 0.25 权重只从 March--August 向前预测中选择，并应用 F0 的 Brier 护栏；无新闻时所有新闻分支严格回退 R1。",
        "",
        "## 为什么仍难稳定提升",
        "",
        "- 两只股票的信息覆盖并不对称；AMZN 的很多窗口没有明确目标段落。完整抽取提高了输入正确性，却不能创造缺失事件。",
        "- 四小时方向噪声很大。同一条长期利好、历史持仓或已经反映在开盘前的消息，不一定决定接下来四小时。",
        "- 不同月份的价格和词语关系变化。共享模型增加了样本，但也把两只股票不同的反应机制混在一起。",
        "- 校准可修正过度自信；它不能保证 0.5 两侧的排序改变，因此 Brier 改善不等于 BA 改善。",
        "- 训练窗口有限且相邻窗口共享新闻，复杂模型的有效独立监督少于表面行数。",
        "",
        "## 没有执行的分支",
        "",
        "本地没有 SPY、QQQ、XLK、XLY 的 2018 分钟线，也没有 FOMC/CPI/就业 consensus 或盈利 consensus/actuals。免费来源调查未确认兼具历史深度、分钟对齐、调整语义和可复现使用条件的数据，因此 M1--M3、market PCA、HMM 和两节点 graph 按停止规则没有运行。没有对 9B 模型做收益标签微调。",
        "",
        "## 如何读案例与区间",
        "",
        "`CASE_NOTES.md` 固定覆盖改对、改错、共同正确／错误、新闻／无新闻、段落变化和近期价格变化。案例只说明具体失败方式。`PAIRED_INTERVALS.csv` 给出共同交易日的 1 日和 5 日块重采样差值；由于方法已反复探索，这些区间是敏感性描述，不是未经选择的显著性检验。",
        "",
        "## 当前项目判断",
        "",
        "工程层已经完成：输入有来源和截止检查，训练／校准只看过去，完整窗口都保留，失败版本也保留，结果可从命令重现。方法层是否成功必须由上面的统一表决定；即使某只股票某一时期超过 50%，也不能改成按股票挑赢家的规则。下一次真正的确认需要方法冻结后收集一个未参与设计的新时期。",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    run_log = ["# Run log", "", "- Inventory completed; market factor branch stopped for missing qualified minute data.", "- Price feature preparation rechecked all 1,607 labels and every recent bar cutoff.", f"- Paragraph v2 failed mechanical completeness; v3 passed mechanics but failed assistant review; v4 passed the limited engineering gate. Independent review remains absent.", f"- LLM generated {sum(row['generated'] for row in llm_summary['variants'].values()):,} new binary scores and reused {sum(row['reused'] for row in llm_summary['variants'].values()):,} sealed P0 scores; no paid API or Sol was used.", "- Calibration, no-news policies, global paragraph choice and F0--F6 weights used March--August past-only forward scores.", "- September onward results are exposed historical replays."]
    (out / "RUN_LOG.md").write_text("\n".join(run_log) + "\n")
    source_paths = required + [fusion / "predictions.csv", calibration / "predictions.csv", price / "predictions.csv", paragraphs / "manifest.json", llm / "manifest.json", *implementation_files]
    source_hashes = {}
    for path in source_paths:
        key = str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)
        source_hashes[key] = sha(path)
    dump(out / "SOURCE_HASHES.json", source_hashes)
    # Stable review entrypoints; detailed CSV files remain versioned in runs/v1.
    for name in ("REPORT.md", "DATA_INVENTORY.md", "INPUT_QUALITY.md", "CALIBRATION.md", "CASE_NOTES.md", "COMMANDS.md", "RUN_LOG.md"):
        shutil.copy2(out / name, B / name)
    print(json.dumps({"status": "PUBLISHED", "paragraph_choice": paragraph_choice, "stable_uniform_improvement": stable, "conclusion": conclusion}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--paragraphs", required=True, type=Path)
    parser.add_argument("--llm", required=True, type=Path)
    parser.add_argument("--price", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.inventory, args.paragraphs, args.llm, args.price, args.calibration, args.fusion, args.cases, args.out)
