# FinBERT / Fin-ModernBERT matched pooling v2

This run is a repair of the v1 encoder comparison, not a new architecture
search.  It keeps the original 1,607 AAPL/AMZN four-hour windows, title input,
cutoff, article order, article averaging, canonical J2 PCA/metadata, LR C
grid and past-only monthly selection.  It creates `v2` outputs and refuses to
overwrite v1.

The first stage trains the canonical FinBERT J2 path from the existing
special-token-excluded article embeddings and must reproduce the saved F2
probabilities.  If that parity check fails, the ModernBERT comparison is
blocked.  The second stage uses the same article texts and a shared helper
that mean-pools only tokens with `attention_mask=1` and
`special_tokens_mask=0`.  No encoder gradients are enabled.

The only changed component is the ModernBERT article embedding.  The pinned
local model is `clapAI/Fin-ModernBERT`, revision
`31d3d96d5839a03dccd030bea40b77c2649a9a01`, FP32, max length 256.  The
historical FinBERT/ModernBERT comparison remains an exploratory frozen-model
probe; pretraining overlap with this 2018 corpus cannot be excluded.

Selection uses March–August chronological forward folds and freezes the
August-trained C for September onward.  Development and later scores are
reported separately and never select a method or stock-specific winner.  No
new fine-tuning, prompt search or exposed-period tuning is allowed here.
