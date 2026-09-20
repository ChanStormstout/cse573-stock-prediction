# Ordered raw text and joint paragraph semantics — v1

This isolated finite study follows the owner's 2026-09-20 execution attachment.
All scores are exposed **exploratory random historical backtests**. No new
chronological claim is made. Existing FULL and text SVM are fixed comparators.

## Reproduce in the existing private environment

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/prepare.py
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/test_contracts.py
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/verify_inputs.py
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/train.py
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/verify.py
work/stock-data/finbert-env/bin/python3 outputs/stock_text_joint_4h/report.py
```

Preparation refuses any existing run directory. Training resumes only completed,
hash-matched blocks and refuses incomplete blocks. Never delete a failed block
in order to resume. A resume must use unchanged code, inputs and protocol.
The verifier prohibits estimator fits and never overwrites model weights.
Checkpoint reload is not independent retraining.

Raw articles, reconstructed token sequences, source maps, weights, calibration
pipelines, protected hashes and complete case evidence are private under
`work/stock-data/text_joint_4h/v1`. Public outputs contain aggregate audits,
selection records and numeric predictions, not redistributable original news.

## Frozen comparisons

- A1/A2/A3: raw ordered unigram LR; unigram+within-sentence bigram LR;
  calibrated SVM on the same unigram+bigram view. Three C candidates only.
- B1/B2: original FULL 16 price/binary chi-square-selected word features,
  plus training-only PCA16/uncompressed frozen FinBERT target-paragraph vectors.
  Scale semantic coordinates using training windows, then apply rho 0/.1/1.
  Zero rho reuses original FULL; missing evidence returns original FULL exactly.
- Reused old TF-IDF probabilities with matched PRICE fallback: no retraining.
- FULL shrinkage: select one of four positive shrink factors by inner Brier;
  preserve direction and exact no-news fallback.

LLM inference and continuation stay suspended at 1,972 records. No model-family,
feature, encoder or hyperparameter expansion based on exposed results is allowed.
