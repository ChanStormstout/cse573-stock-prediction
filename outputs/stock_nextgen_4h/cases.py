"""Deterministic exposed-period case audit with raw text kept private."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from common import dump, jsonl, sha


def rank(namespace, key):
    return hashlib.sha256(f"{namespace}|{key}".encode()).hexdigest()


def choose(frame, namespace):
    if frame.empty:
        return None
    return frame.loc[min(frame.index, key=lambda index: rank(namespace, frame.loc[index, "key"]))]


def main(paragraphs: Path, calibration: Path, fusion: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(fusion / "predictions.csv")
    predictions = predictions[predictions.phase.eq("frozen")].copy()
    llm = pd.read_csv(calibration / "predictions.csv").set_index("key")
    choice = json.loads((calibration / "selection.json").read_text())["paragraph"]["chosen"]["variant"]
    variants = ("P0", "P1", "P2", "P3")
    packs = {variant: {row["key"]: row for row in jsonl(paragraphs / f"{variant}.jsonl")} for variant in variants}
    tokens = pd.read_csv(paragraphs / "tokens.csv")
    token_map = {(row.key, row.variant): int(row.prompt_tokens) for row in tokens.itertuples()}
    clusters = pd.read_csv(paragraphs / "clusters.csv")
    cluster_map = clusters.groupby("key").reports.agg(["sum", "count", "max"]).to_dict("index") if len(clusters) else {}
    predictions["F0_correct"] = predictions.F0.ge(0.5).astype(int).eq(predictions.label)
    predictions["F6_correct"] = predictions.F6.ge(0.5).astype(int).eq(predictions.label)
    for variant in ("P1", "P2", "P3"):
        predictions[f"P0_{variant}_changed"] = llm.loc[predictions.key, "P0_strict"].ge(0.5).to_numpy() != llm.loc[predictions.key, f"{variant}_strict"].ge(0.5).to_numpy()
    predictions["paragraph_changed"] = llm.loc[predictions.key, "P0_strict"].ge(0.5).to_numpy() != llm.loc[predictions.key, f"{choice}_strict"].ge(0.5).to_numpy()
    predictions["recent_changed"] = predictions.R0.ge(0.5) != predictions.R1.ge(0.5)

    slots, availability = [], []
    for symbol in ("AAPL", "AMZN"):
        for split in ("validation", "test"):
            base = predictions[predictions.symbol.eq(symbol) & predictions.split.eq(split)]
            categories = {
                "baseline_wrong_new_right": ~base.F0_correct & base.F6_correct,
                "baseline_right_new_wrong": base.F0_correct & ~base.F6_correct,
                "both_wrong": ~base.F0_correct & ~base.F6_correct,
                "both_right": base.F0_correct & base.F6_correct,
                "paragraph_direction_changed": base.paragraph_changed,
                "P0_P1_direction_changed": base.P0_P1_changed,
                "P0_P2_direction_changed": base.P0_P2_changed,
                "P0_P3_direction_changed": base.P0_P3_changed,
                "recent_direction_changed": base.recent_changed,
                "news": base.has_news.eq(1),
                "no_news": base.has_news.eq(0),
            }
            for category, mask in categories.items():
                row = choose(base[mask], f"{symbol}|{split}|{category}")
                availability.append({
                    "symbol": symbol,
                    "split": split,
                    "category": category,
                    "candidates": int(mask.sum()),
                    "selected_key": None if row is None else row.key,
                })
                if row is not None:
                    slots.append({"symbol": symbol, "split": split, "category": category, "key": row.key})
    selected_keys = sorted({row["key"] for row in slots})
    private, public = [], []
    for key in selected_keys:
        row = predictions[predictions.key.eq(key)].iloc[0]
        source = packs[choice][key]
        news = source["news"]
        passages = [passage for article in news for passage in article["passages"]]
        max_delay = max((article.get("collection_delay_hours") or 0 for article in news), default=0)
        llm_row = llm.loc[key]
        reasons = [slot["category"] for slot in slots if slot["key"] == key]
        record = {
            "key": key,
            "symbol": row.symbol,
            "split": row.split,
            "selection_reasons": reasons,
            "has_news": int(row.has_news),
            "selected_paragraph_variant": choice,
            "source_reports": source["news_summary"].get("eligible_articles", 0),
            "included_articles": source["news_summary"].get("included_articles", 0),
            "target_passages": len(passages),
            "max_collection_delay_hours": max_delay,
            "P0_prompt_tokens": token_map.get((key, "P1"), None) and int(tokens[(tokens.key == key) & (tokens.variant == "P1")].P0_prompt_tokens.iloc[0]),
            "selected_prompt_tokens": token_map.get((key, choice), None),
            "cluster_summary": cluster_map.get(key, {"sum": 0, "count": 0, "max": 0}),
            "paragraph_direction_changes": {
                variant: bool(row[f"P0_{variant}_changed"])
                for variant in ("P1", "P2", "P3")
            },
            "probabilities": {
                "R0": row.R0,
                "R1": row.R1,
                "S2": row.S2,
                "S3": row.S3,
                **{f"{variant}_strict": llm_row[f"{variant}_strict"] for variant in variants},
                "F0": row.F0,
                "F6": row.F6,
            },
            "label_revealed_after_input_record": int(row.label),
            "error_types": {
                "input_error": bool(row.has_news and len(passages) == 0),
                "selection_change": bool(row.paragraph_changed),
                "llm_prediction_error": int(llm_row.paragraph_selected >= 0.5) != int(row.label),
                "market_factor_status": "unavailable; branch not run",
                "final_prediction_error": int(row.F6 >= 0.5) != int(row.label),
            },
        }
        public.append(record)
        record_private = dict(record)
        record_private["variant_article_titles"] = {
            variant: [article["title"] for article in packs[variant][key]["news"]]
            for variant in variants
        }
        record_private["variant_passage_previews"] = {
            variant: [
                passage["text"][:500]
                for article in packs[variant][key]["news"]
                for passage in article["passages"][:6]
            ]
            for variant in variants
        }
        private.append(record_private)
    dump(out / "private_cases.json", private)
    dump(out / "case_selection.json", {"slots": slots, "availability": availability, "unique_cases": len(selected_keys), "selection": "minimum SHA-256 within each preregistered category", "labels_used_only_to_form_predeclared correctness categories": True})
    lines = [
        "# Case audit",
        "",
        "Cases use fixed categories and deterministic key hashing. The input fields were recorded before the outcome field was appended. Raw article text and titles remain outside Git.",
        "",
        f"Selected paragraph variant: **{choice}**. Independent extraction review: **not performed**.",
        "",
        "| Stock | Split | Selection slots | News | Target passages | R0 | R1 | P0 | Selected paragraph | F0 | F6 | Actual | Error notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in public:
        p = row["probabilities"]
        notes = ", ".join(name for name, value in row["error_types"].items() if value is True) or "none recorded"
        lines.append(
            f"| {row['symbol']} | {row['split']} | {', '.join(row['selection_reasons'])} | {row['has_news']} | {row['target_passages']} | {p['R0']:.3f} | {p['R1']:.3f} | {p['P0_strict']:.3f} | {p[f'{choice}_strict']:.3f} | {p['F0']:.3f} | {p['F6']:.3f} | {row['label_revealed_after_input_record']} | {notes} |"
        )
    lines += [
        "",
        "`input_error` means news existed but the target-paragraph selector returned no eligible passage. `selection_change` only says P0 and the selected paragraph variant crossed 0.5 differently; it is not a causal explanation. The unavailable market-factor branch is recorded separately from model error.",
    ]
    unavailable = [row for row in availability if row["candidates"] == 0]
    if unavailable:
        lines += [
            "",
            "The following fixed slots had no eligible window and therefore no case was substituted: "
            + "; ".join(f"{row['symbol']}/{row['split']}/{row['category']}" for row in unavailable)
            + ".",
        ]
    (out / "CASE_NOTES.md").write_text("\n".join(lines) + "\n")
    dump(out / "status.json", {"status": "COMPLETE", "unique_cases": len(selected_keys), "private_raw_text_in_git": False, "sources": {str(fusion / "predictions.csv"): sha(fusion / "predictions.csv"), str(calibration / "predictions.csv"): sha(calibration / "predictions.csv"), str(paragraphs / "manifest.json"): sha(paragraphs / "manifest.json")}})
    print(json.dumps({"status": "COMPLETE", "cases": len(selected_keys), "choice": choice}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--fusion", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.paragraphs, args.calibration, args.fusion, args.out)
