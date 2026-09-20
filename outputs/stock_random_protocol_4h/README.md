# Random-protocol four-hour study

**Current: core nested random baselines trained and independently verified; local
LLM analogy inference is running. Full proposed pipeline is not yet evaluated.**

- [Frozen protocol](PROTOCOL.md)
- [Analogy implementation detail](ANALOGY_IMPLEMENTATION.md)
- [Current handoff](../../docs/RANDOM_PROTOCOL_4H_HANDOFF.md)
- [Input audit](v1/INPUT_AUDIT.json)
- [Baseline execution](v1/BASELINE_EXECUTION.json)
- [Independent baseline verification](v1/BASELINE_VERIFICATION.json)
- [Analogy source verification](v1/ANALOGY_VERIFICATION.json)

## Fixed main comparisons

Paper-style keyword L1 LR; 36-feature R1 price-only; legacy-price+full-body
words; price+FinBERT; price+corrected FinModernBERT v2. All are fresh fits.
FinBERT remains the primary modern comparator; no test-picked stock-specific
encoder substitution. Existing probabilities, fitted transforms and supervised
reader gates are excluded. All historical periods remain exposed.

## Actual execution

Three seeds x ten outer folds x two stocks = 60 random blocks. Five methods
with three inner folds and three C values produce 3,000 actual estimator fits.
300 outer selected models and 24,105 probabilities replay within 1e-12; no
public predictive score has been released or used to redesign the mechanisms.
Additional grouped/chronological runs use independent directories and remain
separately labeled. LLM is frozen inference, not LLM finetuning.

## Commands

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/prepare.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/test_contracts.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/train.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/verify.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/diagnostic_train.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/verify_diagnostics.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/retrieval.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/verify_retrieval.py
work/stock-data/structured-env/bin/python3 outputs/stock_random_protocol_4h/infer.py
```

Do not launch a second inference concurrently. The already started local
continuation waits for inference completion, verifies output integrity, selects
finite analogy correction weights using inner data, then runs the frozen
54-pair fact reader pilot. It stops at semantic review; it does not promote
mechanical JSON validity to semantic gold or release stock scores automatically.

Weights, source passages, folds and sealed probabilities stay under
`work/stock-data/random_protocol_4h/v1`. No external ECNI lane is resumed.
