# Input audit

Old sorted binary stem sets remain valid historical inputs, but discard frequency/order/numbers and stoplisted negation. New A inputs reconstruct the same article membership from raw title/body, preserving within-sentence sequence and repetitions. B reuses existing target-paragraph vectors unchanged. No model fits were performed during preparation.

```json
{
  "status": "PASS_FIT_FREE",
  "rows": 1607,
  "article_keys": 5078,
  "row_mapping_sha256": "14282f8a8835d5f51cd9115e02df8a1620a30b083144c7e498209711b030e618",
  "input_sha256": "f9233bc533d5198070b75a535a2db3d7b384812bf9f987d53bc0030176f97dde",
  "canonical_sha256": "a59a8ca7ff3b13815c6c52f6b5b8de76718b5a114886558e7e4036fd11f1a932",
  "protected_files": 24230,
  "protected_ledger_sha256": "655ae614a53be43dbbe0f38955cc8f313ae56113282999bd2a0624e51ee0d803",
  "old_stem_body": {
    "sorted_unique_all_rows": true,
    "numeric_tokens": 0,
    "negations_in_stoplist": [
      "not",
      "no",
      "nor"
    ],
    "source": "outputs/stock_integrated_4h/prepare.py"
  },
  "coverage": {
    "AAPL": {
      "rows": 803,
      "no_news": 6,
      "news_without_target_paragraph": 0,
      "target_paragraph": 797
    },
    "AMZN": {
      "rows": 804,
      "no_news": 389,
      "news_without_target_paragraph": 3,
      "target_paragraph": 412
    }
  },
  "new_text": {
    "tokens": 7742049,
    "numeric_tokens": 668121,
    "negations": 12571,
    "repeated_occurrences": 6124781,
    "empty_news_windows": []
  },
  "LLM": {
    "records": 1972,
    "sha256": "c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7",
    "process_states": " 2333 Ts  \n 3030 Ts  \n",
    "continued": false
  },
  "predictive_fits": 0
}
```
