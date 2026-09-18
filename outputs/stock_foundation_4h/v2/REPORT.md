# FinBERT / Fin-ModernBERT matched pooling v2

## Plain-language result

We reran the encoder comparison after fixing a fairness problem in the old
ModernBERT run. The old run averaged the `[CLS]` and other special tokens into
the article vector; the canonical FinBERT path excludes special tokens. The
new run uses the same special-token-excluded pooling rule for the ModernBERT
articles, with the same title inputs, windows, PCA, metadata, LR selection and
no-news fallback.

The canonical FinBERT reproduction matched its saved probabilities to a
maximum absolute error of `1.72e-15`, so the ModernBERT comparison was allowed
to proceed. The new pooling changed the ModernBERT vectors substantially
(mean article-vector L2 difference from v1 `2.0673`), which means v1 and v2
are not interchangeable.

ModernBERT still did not become a stable improvement. It is higher than F2 in
some exposed periods and lower in others, with no training-period promotion
under the registered protocol. We do not select a stock-specific winner or
combine it after seeing development/later scores.

## Actual run

- 5,078 identical title/article keys, same order as canonical F2.
- Frozen `clapAI/Fin-ModernBERT`, pinned revision
  `31d3d96d5839a03dccd030bea40b77c2649a9a01`, FP32 on Apple MPS.
- 149,014,272 encoder parameters; zero trainable parameters and no gradients.
- 159 batches; 26.25 seconds; maximum length 256; no truncated title.
- Article vectors use only `attention_mask=1` and `special_tokens_mask=0`, then
  equal-average the articles in each original four-hour window.
- C is selected from earlier chronological forward folds and frozen before
  development/later evaluation. All periods are exposed exploratory
  historical backtests.

## Results

BA / Brier are shown separately; F2 is the canonical frozen FinBERT control.

| period | stock | F2 BA | F2 Brier | Modern v2 BA | Modern v2 Brier | Modern − F2 BA |
|---|---|---:|---:|---:|---:|---:|
| development | AAPL | 54.13% | 0.2803 | 46.25% | 0.2752 | −7.89pp |
| development | AMZN | 46.29% | 0.2977 | 54.17% | 0.2489 | +7.88pp |
| later | AAPL | 56.71% | 0.2856 | 54.52% | 0.2612 | −2.20pp |
| later | AMZN | 54.27% | 0.2617 | 54.97% | 0.2461 | +0.70pp |

Training-period forward OOF BA was AAPL `50.02%` for F2 versus `45.93%` for
Modern v2, and AMZN `48.14%` versus `51.48%`. This does not establish a
reliable cross-stock winner.

## Interpretation

**Observed:** the pooling repair changes the ModernBERT representation; the
canonical FinBERT path reproduces; Modern v2 has mixed period/stock results and
does not pass the registered promotion line.

**Not established:** that Fin-ModernBERT is ineffective in general, that a
different input (full body, target prefix, fine-tuning) would fail, or that
the historical pretraining overlap is temporally clean.

No fusion or extra ModernBERT tuning was run after this result. The next audit
stage is the independent TabPFN provenance/configuration check.
