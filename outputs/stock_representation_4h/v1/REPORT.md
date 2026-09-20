# Representation diagnostics and aggregation order

**EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST.** All 1,607 original four-hour windows, three fixed seeds. No future generalization claim. Inner-only C selection. Grouped testing is a separate robustness protocol.

## Results: three-seed means and standard deviations

### preprocess

| stage      | method    | symbol   |   BA_mean |   BA_std |   MCC_mean |   MCC_std |   Brier_mean |   Brier_std |   predicted_up_mean |   predicted_up_std |
|:-----------|:----------|:---------|----------:|---------:|-----------:|----------:|-------------:|------------:|--------------------:|-------------------:|
| preprocess | CONTROL   | AAPL     |   0.69547 |  0.01916 |    0.39519 |   0.03848 |      0.20538 |     0.00134 |             0.58406 |            0.00543 |
| preprocess | CONTROL   | AMZN     |   0.61137 |  0.00490 |    0.22355 |   0.00918 |      0.23450 |     0.00324 |             0.54395 |            0.01689 |
| preprocess | FREQUENCY | AAPL     |   0.70125 |  0.00398 |    0.40580 |   0.00960 |      0.20700 |     0.00094 |             0.57493 |            0.01278 |
| preprocess | FREQUENCY | AMZN     |   0.60605 |  0.00542 |    0.21281 |   0.01008 |      0.23582 |     0.00263 |             0.54187 |            0.02022 |
| preprocess | FULL      | AAPL     |   0.67568 |  0.01207 |    0.35480 |   0.02441 |      0.22281 |     0.00363 |             0.58074 |            0.00072 |
| preprocess | FULL      | AMZN     |   0.60279 |  0.00809 |    0.20618 |   0.01608 |      0.25241 |     0.00221 |             0.54022 |            0.01619 |
| preprocess | NEGATION  | AAPL     |   0.69592 |  0.02035 |    0.39606 |   0.04062 |      0.20537 |     0.00134 |             0.58364 |            0.00829 |
| preprocess | NEGATION  | AMZN     |   0.61355 |  0.00459 |    0.22785 |   0.00853 |      0.23482 |     0.00278 |             0.54104 |            0.02092 |
| preprocess | NO_STEM   | AAPL     |   0.69087 |  0.01215 |    0.38533 |   0.02544 |      0.20816 |     0.00118 |             0.57908 |            0.00758 |
| preprocess | NO_STEM   | AMZN     |   0.60353 |  0.00285 |    0.20795 |   0.00455 |      0.23508 |     0.00176 |             0.54270 |            0.03098 |
| preprocess | NUMERIC   | AAPL     |   0.68690 |  0.00585 |    0.37826 |   0.01282 |      0.20606 |     0.00121 |             0.58655 |            0.00758 |
| preprocess | NUMERIC   | AMZN     |   0.61590 |  0.00547 |    0.23272 |   0.00998 |      0.23322 |     0.00299 |             0.54519 |            0.02163 |
| preprocess | PRICE     | AAPL     |   0.55835 |  0.00710 |    0.12163 |   0.01582 |      0.24679 |     0.00095 |             0.64384 |            0.02405 |
| preprocess | PRICE     | AMZN     |   0.52676 |  0.01063 |    0.05459 |   0.02156 |      0.25627 |     0.00250 |             0.60158 |            0.00800 |

### grouped

| stage   | method    | symbol   |   BA_mean |   BA_std |   MCC_mean |   MCC_std |   Brier_mean |   Brier_std |   predicted_up_mean |   predicted_up_std |
|:--------|:----------|:---------|----------:|---------:|-----------:|----------:|-------------:|------------:|--------------------:|-------------------:|
| grouped | FULL      | AAPL     |   0.52394 |  0.00859 |    0.04812 |   0.01719 |      0.27094 |     0.00310 |             0.56579 |            0.01184 |
| grouped | FULL      | AMZN     |   0.51740 |  0.01688 |    0.03481 |   0.03377 |      0.26504 |     0.01051 |             0.52488 |            0.01508 |
| grouped | GROUP_SVM | AAPL     |   0.49230 |  0.00381 |   -0.05054 |   0.02331 |      0.25067 |     0.00058 |             0.97717 |            0.00259 |
| grouped | GROUP_SVM | AMZN     |   0.51167 |  0.01854 |    0.02469 |   0.03901 |      0.25234 |     0.00183 |             0.61774 |            0.04889 |
| grouped | PRICE     | AAPL     |   0.55842 |  0.01672 |    0.12158 |   0.03508 |      0.24809 |     0.00224 |             0.64301 |            0.00471 |
| grouped | PRICE     | AMZN     |   0.51451 |  0.00515 |    0.02968 |   0.01050 |      0.25866 |     0.00231 |             0.60779 |            0.00471 |

### aggregation

| stage       | method        | symbol   |   BA_mean |   BA_std |   MCC_mean |   MCC_std |   Brier_mean |   Brier_std |   predicted_up_mean |   predicted_up_std |
|:------------|:--------------|:---------|----------:|---------:|-----------:|----------:|-------------:|------------:|--------------------:|-------------------:|
| aggregation | CONTROL       | AAPL     |   0.69547 |  0.01916 |    0.39519 |   0.03848 |      0.20538 |     0.00134 |             0.58406 |            0.00543 |
| aggregation | CONTROL       | AMZN     |   0.61137 |  0.00490 |    0.22355 |   0.00918 |      0.23450 |     0.00324 |             0.54395 |            0.01689 |
| aggregation | FULL          | AAPL     |   0.67568 |  0.01207 |    0.35480 |   0.02441 |      0.22281 |     0.00363 |             0.58074 |            0.00072 |
| aggregation | FULL          | AMZN     |   0.60279 |  0.00809 |    0.20618 |   0.01608 |      0.25241 |     0.00221 |             0.54022 |            0.01619 |
| aggregation | MAP_THEN_MEAN | AAPL     |   0.65216 |  0.00402 |    0.31532 |   0.00676 |      0.22082 |     0.00097 |             0.63678 |            0.00967 |
| aggregation | MAP_THEN_MEAN | AMZN     |   0.59271 |  0.00949 |    0.18571 |   0.01890 |      0.24049 |     0.00344 |             0.53151 |            0.01195 |
| aggregation | MEAN_LINEAR   | AAPL     |   0.65422 |  0.00589 |    0.31623 |   0.01264 |      0.21955 |     0.00151 |             0.61727 |            0.00829 |
| aggregation | MEAN_LINEAR   | AMZN     |   0.59411 |  0.00501 |    0.18916 |   0.00944 |      0.23899 |     0.00292 |             0.55100 |            0.01841 |
| aggregation | MEAN_THEN_MAP | AAPL     |   0.65087 |  0.00421 |    0.31273 |   0.00673 |      0.22085 |     0.00098 |             0.63719 |            0.01157 |
| aggregation | MEAN_THEN_MAP | AMZN     |   0.59062 |  0.00707 |    0.18152 |   0.01401 |      0.24081 |     0.00341 |             0.53192 |            0.00998 |
| aggregation | PRICE         | AAPL     |   0.55835 |  0.00710 |    0.12163 |   0.01582 |      0.24679 |     0.00095 |             0.64384 |            0.02405 |
| aggregation | PRICE         | AMZN     |   0.52676 |  0.01063 |    0.05459 |   0.02156 |      0.25627 |     0.00250 |             0.60158 |            0.00800 |
| aggregation | SET_BASE      | AAPL     |   0.64986 |  0.00589 |    0.31200 |   0.01294 |      0.22118 |     0.00076 |             0.64425 |            0.00829 |
| aggregation | SET_BASE      | AMZN     |   0.59506 |  0.01331 |    0.19051 |   0.02626 |      0.23943 |     0.00282 |             0.53566 |            0.01520 |

## Interpretation rules

- Preprocessing changes exactly one historical lexical operation per branch. Frequency can also change the 5,000-term vocabulary through term-frequency selection; that is part of this mechanism, not isolated coefficient-only reweighting.
- Grouped SVM has group-disjoint outer, inner and calibration splits. Historical ordinary SVM used stratified calibration. Thus cross-protocol score movement includes the necessary calibration isolation change. Components are association groups, not independently reviewed financial events.
- Aggregation uses a new matched TF-IDF + price LR reference (SET_BASE). It is not silently relabeled as historical FULL or SVM. All three semantic representations share that exact classifier/input design and C budget.
- MEAN_LINEAR keeps768 dimensions; the other two both use the same256-dimensional random mapping. MEAN_THEN_MAP vs MAP_THEN_MEAN is the strict aggregation-order comparison; linear vs nonlinear additionally changes feature geometry/dimension.
- The chunk pool already averages tokens within each chunk and selects target-company paragraphs. Mapping cannot recover missing passages, word-level relations or absent AMZN news.
- No-news windows return saved PRICE exactly; news with no qualifying paragraph use zero semantic block. No evidence-quality acceptance is implied.
- Paired intervals are descriptive after repeated exploration. Fixed thresholds, no outer-selected stock-specific winner or best seed.

## Actual training

```json
{
  "status": "COMPLETE_VERIFIED",
  "top_level_fit_calls": 5400,
  "SVM_internal_classifier_fits": 9000,
  "sigmoid_calibrators": 9000,
  "LR_fits": 2400,
  "fit_seconds": 1491.2011481826776,
  "device": "CPU",
  "threads": 2,
  "encoder_training": false,
  "encoder_inference": false,
  "LLM_continued": false,
  "counts": {
    "aggregation": 2400,
    "grouped": 600,
    "preprocess": 2400
  }
}
```

## Follow-up screen

```json
{
  "status": "DESCRIPTIVE_BUDGET_SCREEN_ONLY",
  "passes": false,
  "values": {
    "AAPL": {
      "SET_BASE": {
        "mean_delta_BA": 0.0022998917208062464,
        "positive_seeds": 2,
        "mean_delta_Brier": -0.0003569608305607208,
        "passes": true
      },
      "MEAN_THEN_MAP": {
        "mean_delta_BA": 0.0012889388639013937,
        "positive_seeds": 2,
        "mean_delta_Brier": -2.909480522583774e-05,
        "passes": true
      }
    },
    "AMZN": {
      "SET_BASE": {
        "mean_delta_BA": -0.0023460216759186117,
        "positive_seeds": 2,
        "mean_delta_Brier": 0.0010571679142500439,
        "passes": false
      },
      "MEAN_THEN_MAP": {
        "mean_delta_BA": 0.002089941845096477,
        "positive_seeds": 2,
        "mean_delta_Brier": -0.00032018974745162687,
        "passes": true
      }
    }
  },
  "decision": "Do not expand aggregation to price interactions; no automatic TabPFN or model grid"
}
```

See monthly_metrics.csv, coverage.csv, direction_changes.csv, paired_intervals.csv and selected_C files. All source text, vectors and weights remain private.
