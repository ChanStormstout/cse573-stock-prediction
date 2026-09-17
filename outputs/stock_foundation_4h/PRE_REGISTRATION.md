# Finite foundation-model experiment — 2026-09-17

User explicitly authorized this new finite round after the earlier Phase-1 stop. That historical gate and failures are unchanged. All development/later periods are exposed exploratory backtests. Keep all 1607 four-hour AAPL/AMZN windows and their five-minute-before-start cutoffs. Do not select per-stock hindsight winners. No paid API, original articles or model binaries in Git.

## 1. Aggregation × regularization

Four fixed branches: article mean / near-republication group mean, each C=0.01 / 1. Use identical metadata, article-level training-only PCA16, scales, LR solver and original-news R1 fallback. Fit per stock on preceding data for March–August and finally January–August for development/later. No selection from development/later. Report representation contrast at each C and C contrast within each representation, including monthly differences and 1/5-day paired intervals. This isolates fixed-C representation effects, not economic causality.

## 2. Fin-ModernBERT

Pinned clapAI/Fin-ModernBERT revision 31d3d96d5839a03dccd030bea40b77c2649a9a01, Apache-2.0. Freeze encoder, FP32, eval mode, attention-masked mean pooling, max 256 tokens, same exact titles/articles/order as existing ProsusAI/finbert F2 (revision 4556d13015211d73dccd3fdd39d39232506f3e43). No target prefix or full-body extension in this matched substitution: adding those now would confound encoder comparison. Tokenizers differ, so record truncation. All PCA16 and scaling fit within past training articles/windows. Article averaging and old price columns identical to F2. C candidates .01/.1/1, chronological previous-month selection, March default .1; September onward freezes August-trained final classifier. Reproduce F2 with old embeddings through same new code. No encoder gradients. The model card reports financial text pretraining including FNSPID; historical contamination cannot be excluded. No claim of a temporally clean foundation model.

## 3. Chronos-2

Pinned amazon/chronos-2 revision 29ec3766d36d6f73f0696f85560a422f50e8498c, Apache-2.0; official chronos-forecasting 2.2.2. Frozen FP32 CPU inference initially. Forecast each stock/window independently, open and close as two related variates; no cross-window grouping or future-valued covariates. Last 512 scheduled regular-session five-minute bars with bar END <= cutoff. Retain missing scheduled bars as NaN, remove overnight non-trading time explicitly (trading-step axis). No actual future open in inputs. Forecast 49 steps maximum: locate target start and final bar via scheduled trading indices (usually steps 2/49; first session window steps 1/48). Use predicted median open at target start and median close at target end; quantile .1/.5/.9 spreads are not a joint distribution or direct direction probability.

Record raw median-difference direction BA/MCC separately. Fit a small training-only LR on three forecast features: median close-minus-open relative to last observed close, predicted target-open .9-.1 spread, predicted target-close .9-.1 spread. C=.01/.1/1 chosen only from earlier monthly forward scores. Also fit LR on the SAME 512-bar open/close history (relative log levels, train-only scaling, missing=mean) as matched information-length baseline. Report existing R1 but acknowledge different representations/lookbacks. A window with fewer than 48 observed historical bars, unavailable target indices, or invalid forecasts falls back to R1 and is counted; no selective dropping. Raw-direction results report forecast-covered subset and matching baselines explicitly, not fabricated Brier from medians. Classifier probability branch retains ALL evaluation windows.

## Selection and finite fusion

Train-only gate, stock-month equally weighted: >=3 forward months, >=2 months positive macro BA delta; macro BA improvement >=.01; neither stock loses >.01 BA; macro AND each stock Brier deterioration <=.002. Modern branch compared with matched F2; Chronos branch compared with same-context LR (R1 descriptive). At most one finite fusion per passing branch: .25/.5/.75 weights with existing F1, choose globally across both stocks using only earlier OOF scores, always include weight0 baseline; freeze final weight on train OOF. If no branch passes, no fusion, finetuning or broader grid. Seeds fixed573; deterministic frozen encoders and LR, no best-seed search. Modern inference budget2 hours; entire foundation inference<=8 device hours. Save completed outputs/failure reasons.

## Verification and reporting

Record model/inputs/code hashes, device/runtime, parameter counts, actual timings, classifier fits/reload parity, prediction alignment, future-bar rejection, cache mismatch rejection, complete keyed probabilities, monthly/coverage/constant predictions and 1/5-day paired intervals. New output/private run directories, refuse overwrite. Resumable inference must validate fingerprint and keys. Classifier decisions must not access development/later labels. Cases fixed by key hashes within changed-right/wrong/common-right/wrong; descriptive only. Report development/later separately and all unsuccessful branches. Update log/status, course report addendum and reproducible commands; repository refresh/check, commit/push/remote verification.

## Official sources checked before execution

- https://huggingface.co/clapAI/Fin-ModernBERT
- https://huggingface.co/amazon/chronos-2
- https://github.com/amazon-science/chronos-forecasting
