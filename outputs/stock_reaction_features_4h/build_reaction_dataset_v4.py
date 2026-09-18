"""Build the v4 time-safe reaction corpus without changing v2/v3 artifacts.

The raw article body and all review cards remain under ``work/stock-data``.
This wrapper deliberately reuses the reviewed v2 parser/schedule code, while
making v4-specific target disambiguation and missing-context representation
explicit and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work" / "stock-data"
PUBLIC = HERE / "v4"
PRIVATE = WORK / "reaction_features_4h" / "v4"
ASSOCIATION_PHRASE = re.compile(
    r"american\s+association\s+for\s+physician\s+leadership", re.I
)
APPLE_INDEPENDENT = re.compile(
    r"\bApple\s*(?:,?\s*(?:Inc\.?|Corporation)|stock|shares|earnings|iPhone|Mac|supplier|investors?)\b"
    r"|\b(?:NASDAQ|NYSE)\s*:\s*AAPL\b|\bApple\s*\(\s*AAPL\s*\)",
    re.I,
)


def load_base():
    spec = importlib.util.spec_from_file_location(
        "reaction_builder_v2_for_v4", HERE / "build_reaction_dataset_v2.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import reaction v2 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed_context_v4(bars, session, available, minutes):
    """Use only completed contiguous bars; represent absent context as NaN."""
    n = minutes // 5
    usable = bars[
        (bars.index >= session.open)
        & (bars.index + pd.Timedelta("5min") <= available)
        & (bars.index < session.close)
    ]
    idx = usable.index[-n:]
    valid = (
        len(idx) == n
        and len(idx) > 0
        and np.all(np.diff(idx.view("i8")) == 5 * 60 * 10**9)
    )
    if not valid:
        return {
            "valid": False,
            "used_bar_end_utc": None,
            "return": np.nan,
            "rv": np.nan,
        }
    return {
        "valid": True,
        "used_bar_end_utc": (idx[-1] + pd.Timedelta("5min")).isoformat(),
        "return": float(np.log(bars.loc[idx[-1], "close"] / bars.loc[idx[0], "open"])),
        "rv": float(
            np.sqrt(
                np.square(
                    np.log(
                        bars.loc[idx, "close"].to_numpy()
                        / bars.loc[idx, "open"].to_numpy()
                    )
                ).sum()
            )
        ),
    }


def main(public: Path = PUBLIC, private: Path = PRIVATE) -> None:
    if public.exists() and any(public.iterdir()):
        raise FileExistsError("reaction v4 public directory must be new")
    if private.exists():
        raise FileExistsError("reaction v4 private directory must be new")

    base = load_base()
    rejected_cards: list[dict] = []
    rejected_count = 0
    base_evidence_for = base.evidence_for

    def evidence_for_v4(symbol: str, title: str, body: str):
        nonlocal rejected_count
        all_text = f"{title or ''}\n{body or ''}"
        # This narrow exclusion applies only where AAPL is the apparent sole
        # association and the known non-Apple organization is spelled out.
        if (
            symbol == "AAPL"
            and ASSOCIATION_PHRASE.search(all_text)
            and base.TICKER["AAPL"].search(all_text)
            and not base.LEGAL["AAPL"].search(all_text)
            and not APPLE_INDEPENDENT.search(all_text)
        ):
            rejected_count += 1
            card_key = hashlib.sha256(all_text.encode("utf-8", "replace")).hexdigest()
            rejected_cards.append(
                {
                    "card_key": card_key,
                    "reason": "AAPL acronym expanded as American Association for Physician Leadership without independent Apple evidence",
                    "title": title or "",
                    "body_context": base.text_sentences(body or "")[:4],
                }
            )
            return None
        return base_evidence_for(symbol, title, body)

    # The imported routines resolve these names in their own module namespace.
    base.evidence_for = evidence_for_v4
    base._completed_context = _completed_context_v4
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    index = pd.read_pickle(WORK / "audit" / "news_index.pkl")
    index["article_key"] = index.archive + "::" + index.member
    cand, canonical, counts, schema_keys, sample_objs = base.build_pairs(index, private)
    counts["ambiguous_AAPL_acronym_rejected_count"] = int(rejected_count)
    reaction = base.bars_and_reactions(canonical, private)
    for minutes in (5, 15, 30, 60):
        invalid = reaction[f"pre_{minutes}m_valid"].eq(0)
        reaction.loc[invalid, [f"pre_{minutes}m_return", f"pre_{minutes}m_rv"]] = np.nan
    reaction.loc[reaction.session_day.isna(), "minutes_from_open"] = np.nan
    reaction.to_pickle(private / "article_reactions_all_time.pkl")

    # Only a deterministic, private review-card sample has article text.
    cards = sorted(rejected_cards, key=lambda x: x["card_key"])[:50]
    (private / "ambiguous_AAPL_acronym_review_cards.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in cards)
        + ("\n" if cards else "")
    )
    gate_pass, gate = base.audit(
        public, private, index, cand, canonical, reaction, counts, schema_keys, sample_objs
    )
    evidence_path = public / "build_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence.update(
        {
            "builder_version": "v4",
            "ambiguous_AAPL_acronym_rejected_count": int(rejected_count),
            "private_ambiguous_AAPL_review_card_count": int(len(cards)),
            "missing_context_representation": "invalid return/rv are NaN and validity flags are explicit model inputs",
        }
    )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    (public / "DATA_AUDIT.md").write_text(
        "# Reaction v4 data audit\n\n"
        "- All v4 reaction labels use same-session bars beginning no earlier than article availability.\n"
        "- Pre-article return and realized-volatility values are `NaN` when their contiguous completed-bar context is unavailable; each has a separate `valid` flag.\n"
        "- `AAPL` occurrences expanded as *American Association for Physician Leadership* are rejected only when no independent Apple evidence is present. "
        f"The deterministic private review-card count is {len(cards)}; this narrow rule does not resolve all entity ambiguity.\n"
        f"- Candidate-pair count: {len(cand)}; canonical group count: {len(canonical)}; Jan--August feasibility gate: {'PASS' if gate_pass else 'FAIL'}.\n"
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "gate_pass": bool(gate_pass),
                "candidate_pairs": int(len(cand)),
                "canonical_groups": int(len(canonical)),
                "ambiguous_AAPL_acronym_rejected_count": int(rejected_count),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args()
    main(args.public, args.private)
