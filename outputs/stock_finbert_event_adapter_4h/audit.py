"""Audit the sealed provisional extraction set before any new training."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import ANNOTATION, ANNOTATION_SOURCES, PARAGRAPHS, PUBLIC, WORK, dump, jsonl, sha


def connected_components(ids: set[str], links: list[dict]):
    graph = {value: set() for value in ids}
    for row in links:
        if row["left"] in graph and row["right"] in graph:
            graph[row["left"]].add(row["right"])
            graph[row["right"]].add(row["left"])
    result, seen = [], set()
    for start in sorted(graph):
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            stack.extend(graph[item] - component)
        seen |= component
        result.append(component)
    return result


def conservative_near_duplicates(inputs: list[dict]) -> list[dict]:
    """High-similarity, same-stock, nearby-date pairs used only for split audit."""
    titles = [row["title"] for row in inputs]
    contexts = [" ".join(row["sentences"].values()) for row in inputs]
    matrices = []
    for values in (titles, contexts):
        matrix = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(values)
        matrices.append(cosine_similarity(matrix))
    result = []
    for left in range(len(inputs)):
        for right in range(left + 1, len(inputs)):
            a, b = inputs[left], inputs[right]
            if a["symbol"] != b["symbol"]:
                continue
            day_distance = abs((pd.Timestamp(a["available_utc"]) - pd.Timestamp(b["available_utc"])).total_seconds()) / 86400
            if day_distance > 14:
                continue
            title_similarity = float(matrices[0][left, right])
            context_similarity = float(matrices[1][left, right])
            if max(title_similarity, context_similarity) >= 0.90:
                result.append({"left": a["id"], "right": b["id"], "title_similarity": title_similarity, "context_similarity": context_similarity, "day_distance": day_distance})
    return result


def forward_segment(value: str) -> str:
    day = value[:10]
    if day <= "2018-01-15":
        return "through_2018-01-15"
    if day <= "2018-01-31":
        return "2018-01-16_to_31"
    if day <= "2018-02-14":
        return "2018-02-01_to_14"
    if day <= "2018-02-28":
        return "2018-02-15_to_28"
    return "after_train"


def main() -> None:
    protocol = json.loads((PUBLIC / "protocol.json").read_text())
    if protocol.get("saved_before_new_results") is not True:
        raise ValueError("pre-registration missing")
    inputs = jsonl(ANNOTATION / "inputs.jsonl")
    labels = jsonl(ANNOTATION / "labels.jsonl")
    links = jsonl(ANNOTATION / "duplicate_links.jsonl")
    seal = json.loads((ANNOTATION / "label_seal.json").read_text())
    if sha(ANNOTATION / "labels.jsonl") != seal["sha256"]:
        raise ValueError("sealed labels changed")
    if len(inputs) != len(labels) or {r["id"] for r in inputs} != {r["id"] for r in labels}:
        raise ValueError("input/label identity mismatch")
    by_id = {row["id"]: row for row in inputs}
    label_by_id = {row["id"]: row for row in labels}
    if any(by_id[key]["input_sha256"] != label_by_id[key]["input_sha256"] for key in by_id):
        raise ValueError("input fingerprints differ from labels")

    sources = {row["record_key"]: row for row in jsonl(ANNOTATION_SOURCES)}
    mapping_failures = []
    for row in inputs:
        source = sources.get(row["record_key"])
        if source is None:
            mapping_failures.append({"id": row["id"], "reason": "missing_source"})
            continue
        body, title = source["body"], source["title"]
        if row["body_sha256"] != __import__("hashlib").sha256(body.encode()).hexdigest():
            mapping_failures.append({"id": row["id"], "reason": "body_hash"})
        for sentence_id, span in row["spans"].items():
            value = title if span["source"] == "title" else body
            if value[span["start"] : span["end"]] != row["sentences"][sentence_id]:
                mapping_failures.append({"id": row["id"], "sentence_id": sentence_id, "reason": "span"})

    accepted_links = [row for row in links if row["left"] in by_id and row["right"] in by_id]
    accepted_link_endpoints = {row[side] for row in links for side in ("left", "right") if row[side] in by_id}
    near_links = conservative_near_duplicates(inputs)
    cross_forward_near = [
        row
        for row in near_links
        if forward_segment(by_id[row["left"]]["available_utc"]) != forward_segment(by_id[row["right"]]["available_utc"])
    ]
    components = connected_components(set(by_id), [*accepted_links, *near_links])
    cross_components = []
    for component in components:
        splits = sorted({by_id[item]["split"] for item in component})
        if len(splits) > 1:
            cross_components.append({"ids": sorted(component), "splits": splits})
    exact_groups = collections.defaultdict(set)
    for row in inputs:
        exact_groups[row["group"]].add(row["split"])
    cross_exact = {key: sorted(value) for key, value in exact_groups.items() if len(value) > 1}
    if mapping_failures or cross_components or cross_exact or cross_forward_near:
        raise ValueError({"mapping_failures": mapping_failures[:5], "cross_components": cross_components[:5], "cross_exact": cross_exact, "cross_forward_near": cross_forward_near[:5]})

    window_rows = jsonl(PARAGRAPHS / "P2.jsonl")
    occurrences = collections.defaultdict(list)
    for window in window_rows:
        for article in window["news"]:
            occurrences[(window["symbol"], article["record_key"])].append(window["key"])

    all_events = [event for row in labels for event in row["answer"]["events"]]
    split_counts = collections.Counter(row["split"] for row in labels)
    positive_articles = collections.Counter((row["split"], row["symbol"]) for row in labels if row["answer"]["events"])
    event_counts = collections.Counter((row["split"], row["symbol"]) for row in labels for _ in row["answer"]["events"])
    fields = sorted({field for event in all_events for field in event})
    expected_fields = ["action", "evidence_ids", "kind", "new", "old", "unit"]
    if fields != expected_fields:
        raise ValueError({"unexpected_event_fields": fields})
    stats = {
        "status": "PASS_PROVISIONAL_DATA_AUDIT",
        "labels_status": seal["status"],
        "independent_human_review_passed": False,
        "rows": len(labels),
        "split_counts": dict(split_counts),
        "positive_article_counts": {"|".join(key): value for key, value in sorted(positive_articles.items())},
        "event_counts": {"|".join(key): value for key, value in sorted(event_counts.items())},
        "total_positive_articles": sum(bool(row["answer"]["events"]) for row in labels),
        "total_events": len(all_events),
        "event_fields": fields,
        "kind_action_counts": {"|".join(key): value for key, value in sorted(collections.Counter((e["kind"], e["action"]) for e in all_events).items())},
        "mapping": {
            "source_rows_available": len(sources),
            "accepted_rows_verified": len(inputs),
            "failures": 0,
            "article_target_pairs_seen_in_fixed_windows": sum(bool(occurrences.get((r["symbol"], r["record_key"]))) for r in inputs),
            "article_target_pairs_not_seen_in_fixed_windows": sum(not occurrences.get((r["symbol"], r["record_key"])) for r in inputs),
        },
        "grouping": {
            "accepted_exact_groups": len(exact_groups),
            "sealed_duplicate_links_total": len(links),
            "sealed_links_with_both_accepted_endpoints": len(accepted_links),
            "accepted_articles_referenced_by_sealed_links": len(accepted_link_endpoints),
            "conservative_near_duplicate_pairs": len(near_links),
            "near_duplicate_pairs_crossing_train_forward_segments": len(cross_forward_near),
            "near_duplicate_scan": {"same_symbol": True, "maximum_day_distance": 14, "title_or_context_char_ngram_cosine_minimum": 0.90},
            "connected_components": len(components),
            "non_singleton_components": sum(len(component) > 1 for component in components),
            "cross_split_components": 0,
            "cross_split_exact_groups": 0,
            "warning": "Conservative known links cannot prove that every semantic duplicate was found.",
        },
        "hashes": {
            "inputs": sha(ANNOTATION / "inputs.jsonl"),
            "labels": sha(ANNOTATION / "labels.jsonl"),
            "duplicate_links": sha(ANNOTATION / "duplicate_links.jsonl"),
            "paragraph_manifest": sha(PARAGRAPHS / "manifest.json"),
            "protocol": sha(PUBLIC / "protocol.json"),
        },
    }
    audit_work = WORK / "audit_v3"
    audit_work.mkdir(parents=True, exist_ok=False)
    dump(audit_work / "data_audit.json", stats)
    dump(audit_work / "near_duplicate_pairs.json", near_links)
    dump(PUBLIC / "data_audit.json", stats)
    lines = [
        "# Data audit",
        "",
        "The sealed extraction set passed the mechanical audit. It remains **model-provisional, not independent-human gold**.",
        "",
        "## Counts",
        "",
        "| Split | Articles | Positive articles | Facts |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "valid", "check"):
        positives = sum(value for (part, _), value in positive_articles.items() if part == split)
        facts = sum(value for (part, _), value in event_counts.items() if part == split)
        lines.append(f"| {'development' if split == 'valid' else split} | {split_counts[split]} | {positives} | {facts} |")
    lines += [
        "",
        f"Training positives are AAPL {positive_articles[('train', 'AAPL')]} and AMZN {positive_articles[('train', 'AMZN')]}; together they contain {sum(value for (part, _), value in event_counts.items() if part == 'train')} facts.",
        "",
        "## Fields actually present",
        "",
        "`kind`, `action`, `old`, `new`, `unit`, and `evidence_ids`. The adapter receives no importance, sentiment, broker, disclosure-age label, or stock-return label.",
        "",
        "## Mapping and grouping checks",
        "",
        f"- Revalidated all {len(inputs)} source-body hashes and every stored title/body span: 0 failures.",
        f"- The sealed file has {len(links)} links, but {len(accepted_links)} have both endpoints inside the accepted 458-article panel ({len(accepted_link_endpoints)} accepted articles are referenced at one endpoint).",
        f"- A separate conservative same-stock, within-14-day title/context scan found {len(near_links)} high-similarity accepted pairs and {sum(len(c) > 1 for c in components)} non-singleton accepted components: 0 cross-split components.",
        f"- None of those pairs crosses the fixed January--February forward-fold segments.",
        f"- Exact event-group IDs also have 0 cross-split groups.",
        f"- {stats['mapping']['article_target_pairs_seen_in_fixed_windows']} accepted article-target pairs occur in the fixed 1,607-window corpus; {stats['mapping']['article_target_pairs_not_seen_in_fixed_windows']} do not enter any fixed window.",
        "- The duplicate check covers known conservative links; it is not proof that all semantic reposts were discovered.",
        "",
        "## Consequence",
        "",
        "The data are usable for the requested exploratory adapter comparison. They are too small and too provisionally labeled to support a formal extraction-quality claim, especially for AMZN.",
    ]
    (PUBLIC / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
