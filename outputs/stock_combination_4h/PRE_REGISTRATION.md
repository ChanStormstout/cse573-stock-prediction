# Finite combination validation and AMZN attribution

Registered before fitting the new J3 temperature. All existing outcomes are exposed historical replay.

## Four fixed systems
B0=existing F2; B1=F2 with positive temperature; B2=existing J3 (recent price + identical FinBERT representation); B3=J3 with its OWN past forward-OOF positive temperature. Reuse sealed classifier predictions, do not duplicate past classifier training. Fit per-stock temperatures only on original-news windows. Keep no-news R1 exactly unchanged. No intercept, reversed slope, probability threshold search or new encoder.

March has no calibration history and uses identity. April--August uses only earlier March--August OOF. September onward uses a frozen calibrator fit on March--August. Optimize the same bounded positive slope/log loss as the previous temperature control; minimum 30 fitting rows. Serialize and reload all calibrators. Report all four systems, BA/MCC/Brier/AUC, monthly and per-stock results. Preserve all 1607 windows (233 warmup rows have no OOF score).

Primary matched advancement comparison is B3 versus B1, isolating recent-price value after matched calibration. Secondary B3 versus B0 and B2 are descriptive. Gate: >=3 forward months; >=2 positive monthly macro BA differences; macro BA +1 percentage point; no stock loses >1 point BA; each stock AND macro Brier deterioration <=.002. No evaluation-period choice. This newly authorized finite combination is separate from the previous stopped grid; passing does not automatically launch more models.

## AMZN explanation, no retuning
Compare N1M against N0M, both with identical metadata. Report monthly results, news/no-news, current aggregation changed/unchanged, and paired direction transitions. Use the complete-period class denominators to add BA contributions by day. Show contribution concentration and 1/5-day paired block intervals; removing high-contribution days is a sensitivity calculation, never a replacement official score.

Using saved FINAL classifiers and transforms, execute four input/training combinations on development and later: article-trained/article input; article-trained/event input; event-trained/article input; event-trained/event input. Swap only the aggregation vector; retain model-specific training PCA/scaler, identical metadata and no-news R1. This distinguishes direct inference changes from retraining/selection effects, conditional on each intervention order; it is not economic causality. Check endpoints reproduce original N0M/N1M.

Fix descriptive case selection by hashed key within AMZN later changed-right, changed-wrong, unchanged-current-aggregation with changed decision, and no-news fallback. Inspect input metadata before writing explanations. These cases are explicitly post hoc and do not select the method.

## Deliverables and stopping
Write experiment report, course-report main draft and standalone offline historical-replay demo. No paper benchmark or independent extraction acceptance claims. Keep raw articles, models and intermediate vectors private. Update project status/log; repository checks, commit, push, remote HEAD verification.
