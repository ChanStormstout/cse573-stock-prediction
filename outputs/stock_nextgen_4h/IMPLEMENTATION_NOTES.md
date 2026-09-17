# Implementation notes and registered clarifications

These notes document engineering issues found before any P1--P3 outcome score
was computed. They do not change the target, model, prompt, input rows or later
evaluation labels.

## Forward calibration month wording

`PROTOCOL.md` says both that calibration starts with April and that method
selection uses March--August forward predictions. The implementation follows
the latter, already used throughout the project: March is predicted from
January--February rows, April from January--March, and so on. Every fit checks
that the latest training-label end precedes the earliest evaluation cutoff.
The April phrase is a wording error retained because the sealed protocol is
part of the running LLM manifest.

## Identical prompts and batch composition

The MLX ledger showed that Qwen token logits can differ when a byte-identical
prompt is evaluated in a different batch composition. The observed score and
log-probabilities remain unchanged in the private ledger and the sanitized
`LLM_SCORES.csv`. For the controlled P0→P1→P2→P3 sequence, a row whose entire
prompt input is byte-for-byte equal to the preceding variant carries forward
that preceding variant's normalized probability before calibration. This keeps
P0→P1 about content selection, P1→P2 about added context and P2→P3 about
deduplication. The rule was fixed before labels were loaded and uses no outcome
label. The input counts are 395 identical P0/P1 rows, 493 P1/P2 rows and 1,391
P2/P3 rows.

## Inventory correction

`inventory_v1` incorrectly reported only the already-conservative derived
`available_utc` comparison and allowed package filenames to appear as possible
event datasets. It is retained locally. `inventory_v2` separately reports the
raw `crawled_utc < published_utc` anomaly, restricts dataset discovery to the
raw-data area and is the inventory used for publication.

## Deduplication counts

P3 supplies the model with the representative report and aggregate report
counts. Independent-source counts are retained in the publication audit rather
than exposing private record identifiers in Git. A report can appear in
adjacent prediction windows, so all public dedup counts are explicitly called
window-cluster occurrences.
