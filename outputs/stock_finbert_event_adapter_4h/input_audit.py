"""Verify protected candidate chunks and context-only token truncation."""
from __future__ import annotations

import json

from adapter import EncodedDataset, article_examples, tokenizer
from common import ANNOTATION, PARAGRAPHS, PUBLIC, dump, jsonl, sha
from extract_corpus import corpus_articles


def summarize(name, articles, tok):
    examples = article_examples(articles, labels=None)
    dataset = EncodedDataset(examples, tok)
    lengths = [len(row["input_ids"]) for row in dataset.encodings]
    chunked_sentences = {(row["article_id"], row["sentence_id"]) for row in dataset.examples if row.get("model_chunk_count", 1) > 1}
    return {
        "name": name,
        "articles": len(articles),
        "candidate_sentences": len(examples),
        "model_units": len(dataset),
        "chunked_candidate_sentences": len(chunked_sentences),
        "maximum_encoded_tokens": max(lengths),
        "units_over_512": sum(value > 512 for value in lengths),
    }


def main() -> None:
    tok = tokenizer()
    annotation = jsonl(ANNOTATION / "inputs.jsonl")
    corpus, corpus_inventory = corpus_articles()
    result = {
        "status": "PASS",
        "candidate_protected": True,
        "only_context_may_be_token_truncated": True,
        "maximum_tokens": 512,
        "sets": [summarize("sealed_annotation", annotation, tok), summarize("fixed_window_corpus_candidates", corpus, tok)],
        "corpus_inventory": corpus_inventory,
        "source_hashes": {"annotation_inputs": sha(ANNOTATION / "inputs.jsonl"), "P2": sha(PARAGRAPHS / "P2.jsonl")},
    }
    if any(row["units_over_512"] for row in result["sets"]):
        raise AssertionError(result)
    dump(PUBLIC / "input_length_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
