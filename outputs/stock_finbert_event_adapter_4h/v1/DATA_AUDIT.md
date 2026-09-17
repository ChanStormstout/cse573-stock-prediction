# Data audit

The sealed extraction set passed the mechanical audit. It remains **model-provisional, not independent-human gold**.

## Counts

| Split | Articles | Positive articles | Facts |
|---|---:|---:|---:|
| train | 249 | 60 | 75 |
| development | 87 | 19 | 20 |
| check | 122 | 22 | 23 |

Training positives are AAPL 47 and AMZN 13; together they contain 75 facts.

## Fields actually present

`kind`, `action`, `old`, `new`, `unit`, and `evidence_ids`. The adapter receives no importance, sentiment, broker, disclosure-age label, or stock-return label.

## Mapping and grouping checks

- Revalidated all 458 source-body hashes and every stored title/body span: 0 failures.
- The sealed file has 93 links, but 0 have both endpoints inside the accepted 458-article panel (29 accepted articles are referenced at one endpoint).
- A separate conservative same-stock, within-14-day title/context scan found 1 high-similarity accepted pairs and 1 non-singleton accepted components: 0 cross-split components.
- None of those pairs crosses the fixed January--February forward-fold segments.
- Exact event-group IDs also have 0 cross-split groups.
- 57 accepted article-target pairs occur in the fixed 1,607-window corpus; 401 do not enter any fixed window.
- The duplicate check covers known conservative links; it is not proof that all semantic reposts were discovered.

## Consequence

The data are usable for the requested exploratory adapter comparison. They are too small and too provisionally labeled to support a formal extraction-quality claim, especially for AMZN.
