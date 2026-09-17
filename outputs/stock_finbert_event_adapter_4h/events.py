"""Event normalization, deterministic number checks, and window features."""
from __future__ import annotations

import collections
import math
import re

import numpy as np

from common import ACTIONS, TYPES

FEATURES = [
    "rating_raise",
    "rating_lower",
    "rating_maintain",
    "rating_initiate",
    "rating_unknown",
    "target_price_raise",
    "target_price_lower",
    "target_price_maintain",
    "target_price_initiate",
    "target_price_unknown",
    "target_change_mean",
    "target_change_missing",
    "event_count",
    "article_count",
    "event_age_hours_mean",
    "duplicate_report_count",
]

ACTION_WORDS = {
    "raise": re.compile(r"\b(?:rais(?:e|ed|es|ing)|boost(?:ed|s|ing)?|increase[sd]?|lift(?:ed|s|ing)?|upgrad(?:e|ed|es|ing)|hike[sd]?)\b", re.I),
    "lower": re.compile(r"\b(?:lower(?:ed|s|ing)?|cut(?:s|ting)?|reduce[sd]?|decrease[sd]?|downgrad(?:e|ed|es|ing)|trim(?:med|s|ming)?)\b", re.I),
    "maintain": re.compile(r"\b(?:maintain(?:ed|s|ing)?|reiterat(?:e|ed|es|ing)|reaffirm(?:ed|s|ing)?|repeat(?:ed|s|ing)?|kept|retain(?:ed|s|ing)?)\b", re.I),
    "initiate": re.compile(r"\b(?:initiat(?:e|ed|es|ing)|begin|began|start(?:ed|s|ing)?|launch(?:ed|es|ing)?)\b", re.I),
}
TARGET_PATTERN = re.compile(r"(?:price\s+target|target\s+price|price\s+objective|target)", re.I)
RATING_PATTERN = re.compile(r"\b(?:rating|rated|coverage|upgrade|downgrade|reiterat|maintain|outperform|underperform|overweight|underweight|buy|sell|hold)\b", re.I)
RATING_VALUE = r"(?:strong[- ]buy|market\s+perform|sector\s+weight|equal\s+weight|outperform|underperform|overweight|underweight|positive|negative|neutral|mixed|buy|hold|sell)"
MONEY = re.compile(r"\$\s*([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]+)?)")


def company_alias(symbol: str) -> re.Pattern:
    if symbol == "AAPL":
        return re.compile(r"\b(?:Apple(?:\s+Inc\.?)?|AAPL)\b", re.I)
    if symbol == "AMZN":
        return re.compile(r"\b(?:Amazon(?:\.com)?(?:\s+Inc\.?)?|AMZN)\b", re.I)
    raise ValueError(symbol)


def span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[0] <= right[1] and right[0] <= left[1]:
        return 0
    return min(abs(left[0] - right[1]), abs(right[0] - left[1]))


def parse_number(value):
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def validate_numeric_event(event: dict, evidence: str) -> dict:
    event = dict(event)
    if event.get("kind") != "target_price":
        event["numeric_valid"] = True
        event["target_change"] = None
        return event
    mentioned = {parse_number(value) for value in MONEY.findall(evidence)}
    mentioned.discard(None)
    old, new = parse_number(event.get("old")), parse_number(event.get("new"))
    old_valid = old is None or old in mentioned
    new_valid = new is None or new in mentioned
    currency_valid = bool(MONEY.search(evidence)) and event.get("unit") == "USD"
    event["numeric_valid"] = bool(old_valid and new_valid and currency_valid)
    event["target_change"] = (new - old) / old if event["numeric_valid"] and old and new else None
    return event


def extract_rating_values(action: str, evidence: str, symbol: str | None = None) -> tuple[str | None, str | None]:
    """Copy rating words from evidence; never infer a missing value."""
    if symbol is not None:
        alias = company_alias(symbol)
        aliases = list(alias.finditer(evidence))
        all_action_matches = sorted([match for pattern in ACTION_WORDS.values() for match in pattern.finditer(evidence)], key=lambda match: match.start())
        relevant_matches = list(ACTION_WORDS.get(action, re.compile(r"$^", re.I)).finditer(evidence)) or all_action_matches
        if aliases and relevant_matches:
            candidates = []
            for target in aliases:
                for chosen in relevant_matches:
                    distance = 0 if chosen.start() <= target.end() and chosen.end() >= target.start() else min(abs(chosen.start() - target.end()), abs(target.start() - chosen.end()))
                    if distance > 180:
                        continue
                    start = min(chosen.start(), target.start())
                    following = [match for match in all_action_matches if match.start() > max(chosen.end(), target.end())]
                    end = following[0].start() if following else len(evidence)
                    segment = evidence[start:end]
                    rating_values = len(list(re.finditer(RATING_VALUE, segment, re.I)))
                    directional = int(bool(re.search(rf"\b(?:from\s+{RATING_VALUE}.*?to\s+{RATING_VALUE}|to\s+{RATING_VALUE}.*?from\s+{RATING_VALUE})", segment, re.I)))
                    candidates.append((directional, rating_values, -distance, segment))
            if candidates:
                evidence = max(candidates, key=lambda item: item[:3])[3]
    quoted = r"[\"'“”‘’]?"
    old = new = None
    forward = re.search(
        rf"\bfrom\s+(?:an?\s+)?{quoted}({RATING_VALUE}){quoted}(?:\s+rating)?\s+to\s+(?:an?\s+)?{quoted}({RATING_VALUE}){quoted}",
        evidence,
        re.I,
    )
    reverse = re.search(
        rf"\bto\s+(?:an?\s+)?{quoted}({RATING_VALUE}){quoted}(?:\s+rating)?\s+from\s+(?:an?\s+)?{quoted}({RATING_VALUE}){quoted}",
        evidence,
        re.I,
    )
    if forward:
        old, new = forward.group(1), forward.group(2)
    elif reverse:
        new, old = reverse.group(1), reverse.group(2)
    else:
        patterns = (
            rf"\b(?:reiterat\w*|reaffirm\w*|restat\w*|reissu\w*|maintain\w*|retain\w*|initiat\w*)\b[^.;:]{{0,90}}?{quoted}({RATING_VALUE}){quoted}(?:\s+rating)?",
            rf"\b(?:rating|rated)\s+(?:of\s+|at\s+|as\s+)?{quoted}({RATING_VALUE}){quoted}",
            rf"\b(?:at|as|with|to)\s+(?:an?\s+)?{quoted}({RATING_VALUE}){quoted}(?:\s+rating)?\b",
            rf"{quoted}({RATING_VALUE}){quoted}\s+rating\b",
        )
        for pattern in patterns:
            match = re.search(pattern, evidence, re.I)
            if match:
                new = match.group(1)
                break
    if action in {"raise", "lower"} and old is None and new is None:
        values = [match.group(0).strip("\"'“”‘’") for match in re.finditer(RATING_VALUE, evidence, re.I)]
        if len(values) >= 2:
            old, new = values[-2], values[-1]
    return old, new


def extract_target_price_values(action: str, evidence: str, symbol: str) -> tuple[str | None, str | None]:
    """Copy the target company's literal price pair, not a nearby company's."""
    aliases = list(company_alias(symbol).finditer(evidence))
    if not aliases:
        return None, None
    action_matches = list(ACTION_WORDS.get(action, re.compile(r"$^", re.I)).finditer(evidence))
    if not action_matches:
        action_matches = sorted(
            [match for pattern in ACTION_WORDS.values() for match in pattern.finditer(evidence)],
            key=lambda match: match.start(),
        )
    target_matches = list(TARGET_PATTERN.finditer(evidence))

    def score(start: int, end: int):
        span = (start, end)
        alias_distance = min(span_distance(span, match.span()) for match in aliases)
        action_distance = min((span_distance(span, match.span()) for match in action_matches), default=0)
        target_distance = min((span_distance(span, match.span()) for match in target_matches), default=0)
        return alias_distance + 0.5 * action_distance + 0.25 * target_distance, alias_distance

    pairs = []
    number = r"([0-9]{1,4}(?:,[0-9]{3})*(?:\.[0-9]+)?)"
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r";|,\s*(?:while|whereas|but)\b", evidence, re.I))
    boundaries.append(len(evidence))
    for start, end in zip(boundaries, boundaries[1:]):
        segment = evidence[start:end]
        for match in re.finditer(rf"from\s*\$\s*{number}[^.;]{{0,120}}?to\s*\$\s*{number}", segment, re.I):
            pairs.append((match.group(1), match.group(2), start + match.start(), start + match.end()))
        for match in re.finditer(rf"to\s*\$\s*{number}[^.;]{{0,120}}?from\s*\$\s*{number}", segment, re.I):
            pairs.append((match.group(2), match.group(1), start + match.start(), start + match.end()))
    ranked_pairs = sorted((score(start, end), old, new) for old, new, start, end in pairs)
    if ranked_pairs and ranked_pairs[0][0][1] <= 220:
        _, old, new = ranked_pairs[0]
        return old, new

    explicit_new = []
    for pattern in (
        rf"(?:price\s+targets?|target\s+prices?|price\s+objectives?|targets?)[^.;]{{0,100}}?\b(?:to|at|of)\s*\$\s*{number}",
        rf"\$\s*{number}\s+(?:price\s+target|target\s+price|price\s+objective)",
    ):
        for match in re.finditer(pattern, evidence, re.I):
            candidate_score, alias_distance = score(match.start(), match.end())
            if alias_distance <= 220:
                explicit_new.append((candidate_score, match.group(1)))
    if explicit_new:
        return None, min(explicit_new)[1]

    amounts = []
    for match in MONEY.finditer(evidence):
        candidate_score, alias_distance = score(match.start(), match.end())
        if alias_distance <= 180:
            amounts.append((candidate_score, match.group(1)))
    if amounts:
        return None, min(amounts)[1]
    return None, None


def merge_compatible_events(events: list[dict]) -> list[dict]:
    """Merge title/body copies when known values do not conflict."""
    result = []

    def normalized(value):
        if value is None:
            return None
        number = parse_number(value)
        return ("number", number) if number is not None else ("text", " ".join(str(value).lower().replace("-", " ").split()))

    for raw in sorted(events, key=lambda event: sum(event.get(name) is not None for name in ("old", "new")), reverse=True):
        event = dict(raw)
        matched = None
        for existing in result:
            if (existing.get("kind"), existing.get("action"), existing.get("unit")) != (event.get("kind"), event.get("action"), event.get("unit")):
                continue
            conflicts = any(
                existing.get(name) is not None
                and event.get(name) is not None
                and normalized(existing[name]) != normalized(event[name])
                for name in ("old", "new")
            )
            if not conflicts:
                matched = existing
                break
        if matched is None:
            event["evidence_ids"] = list(dict.fromkeys(event.get("evidence_ids", [])))
            result.append(event)
            continue
        for name in ("old", "new"):
            if matched.get(name) is None and event.get(name) is not None:
                matched[name] = event[name]
        matched["evidence_ids"] = list(dict.fromkeys([*matched.get("evidence_ids", []), *event.get("evidence_ids", [])]))
        if "evidence_probability" in event:
            matched["evidence_probability"] = max(float(matched.get("evidence_probability", 0)), float(event["evidence_probability"]))
        if matched.get("kind") == "target_price":
            old, new = parse_number(matched.get("old")), parse_number(matched.get("new"))
            matched["target_change"] = (new - old) / old if old and new else None
    return result


def rule_events(symbol: str, sentence_id: str, sentence: str) -> list[dict]:
    """Conservative deterministic facts for D2; values come only from evidence."""
    lower = sentence.lower()
    aliases = ("apple", "aapl") if symbol == "AAPL" else ("amazon", "amzn")
    if not any(re.search(rf"\b{re.escape(alias)}\b", lower) for alias in aliases):
        return []
    actions = [name for name, pattern in ACTION_WORDS.items() if pattern.search(sentence)]
    action = actions[0] if len(actions) == 1 else "unknown"
    result = []
    if RATING_PATTERN.search(sentence) and actions:
        old, new = extract_rating_values(action, sentence, symbol)
        result.append({"kind": "rating", "action": action, "old": old, "new": new, "unit": "rating", "evidence_ids": [sentence_id]})
    if TARGET_PATTERN.search(sentence) and (actions or MONEY.search(sentence)):
        old, new = extract_target_price_values(action, sentence, symbol)
        event = {"kind": "target_price", "action": action, "old": old, "new": new, "unit": "USD", "evidence_ids": [sentence_id]}
        event = validate_numeric_event(event, sentence)
        if event["numeric_valid"] and event["new"] is not None:
            result.append(event)
    return result


def aggregate_events(event_articles: list[dict], cutoff) -> tuple[np.ndarray, int, dict]:
    counts = collections.Counter()
    changes, ages, groups = [], [], set()
    duplicate_reports = 0
    for article in event_articles:
        groups.add(article.get("event_group") or article.get("record_key"))
        duplicate_reports += max(0, int(article.get("reports", 1)) - 1)
        available = article.get("available_utc") or article.get("available_at")
        if available is not None:
            age = (cutoff - __import__("pandas").Timestamp(available)).total_seconds() / 3600
            if age >= 0:
                ages.append(age)
        for raw_event in article.get("events", []):
            kind, action = raw_event.get("kind"), raw_event.get("action")
            if kind not in TYPES or action not in ACTIONS:
                continue
            counts[f"{kind}_{action}"] += 1
            change = raw_event.get("target_change")
            if change is None and kind == "target_price":
                old, new = parse_number(raw_event.get("old")), parse_number(raw_event.get("new"))
                change = (new - old) / old if old and new else None
            if change is not None and math.isfinite(float(change)):
                changes.append(float(np.clip(change, -1, 1)))
    values = {
        **{key: float(counts[key]) for key in FEATURES[:10]},
        "target_change_mean": float(np.mean(changes)) if changes else 0.0,
        "target_change_missing": float(bool(counts["target_price_raise"] + counts["target_price_lower"] + counts["target_price_maintain"] + counts["target_price_initiate"] + counts["target_price_unknown"]) and not changes),
        "event_count": float(sum(counts.values())),
        "article_count": float(len(event_articles)),
        "event_age_hours_mean": float(np.mean(ages)) if ages else 0.0,
        "duplicate_report_count": float(duplicate_reports),
    }
    vector = np.asarray([values[key] for key in FEATURES], dtype=np.float32)
    return vector, int(values["event_count"] > 0), {"unique_groups": len(groups), **values}
