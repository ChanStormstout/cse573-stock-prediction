# Baseline roles and four-hour rationale

## Which methods are genuinely basic?

| Role | Method | What it tests |
|---|---|---|
| Sanity reference | Constant direction | Whether the model distinguishes the two classes; BA is exactly 50% if both occur |
| Basic learned baseline | Historical prices + logistic regression | How much is possible without news |
| Primary classical news baseline | Title TF-IDF + prices + logistic regression | Whether simple word features provide useful information |
| Stronger classical text control | Full-text word features + prices + logistic regression | Whether more text helps without a Transformer |
| Modern comparator | Frozen FinBERT title representations + prices | Whether semantic representation helps |
| Mechanism control | Article averaging + the same six reporting fields | Whether grouping contributes beyond metadata |

Baseline means a reference for a comparison; it does not necessarily imply a simple architecture. For this faculty narrative, however, the primary baseline is explicitly classical. FinBERT is never presented as the most basic model. Beating the classical title baseline alone does not prove that grouping works: the same-encoder/no-group comparison remains necessary.

SVM, random forest and naive Bayes are also conventional ML methods. They are not automatically more basic than logistic regression. Their scores should only enter this table when the task, sample set, split and evaluation protocol match. In particular, saved random-split SVM/RF scores cannot replace chronological controls.

No baseline was retrained for this slide revision. Measured cells are sourced from `outputs/stock_paper_methods_4h/v1/metrics.csv`. Full-text features remain non-neural word features; do not call them FinBERT embeddings.

## Why four hours?

Slide paragraph:

> We use four hours as an intraday compromise: more time for a possible news response than one hour, while retaining more prediction windows than a once-daily target. This is a task choice, not a demonstrated optimal horizon.

This is an explanation of scope, not a causal claim that news takes four hours to affect prices. It does not claim that daily prediction necessarily crosses overnight: the earlier daily study used its own session-based definition. More overlapping windows are not more independent market observations. Earlier one-hour/four-hour/daily comparisons in `outputs/stock_horizons/REPORT.md` did not establish a consistently superior horizon across stocks and periods.

The project adopted four hours after exploratory horizon work and the user's explicit scope decision. It must not be described as a universally optimal or prospectively chosen horizon.

## How to explain the figure

The blue FinBERT block is a frozen representation component. Teal blocks identify the representation changes under study: rule-based grouping, two-stage averaging and reporting metadata. The dashed connector passes group membership to averaging; it does not train FinBERT. The classifier learns on past labeled windows. News, reporting information and price features join before classification. With no admitted news, the saved price-only prediction is used.

The detailed figure's five-plus-one example explains aggregation weight only. The 50/50 split is not an upward probability or a judgment of equal economic importance.
