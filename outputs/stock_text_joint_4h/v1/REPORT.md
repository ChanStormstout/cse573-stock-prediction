# Ordered text and joint FinBERT study

**COMPLETE / VERIFIED.** Read [plain-language findings](FINDINGS.md) first and [eight source-grounded case reviews](CASE_REVIEW.md) alongside the fixed case panel. The training-time execution record below preceded the final PASS; see FINAL_STATUS.json for the completed state.

**EXPLORATORY RANDOM HISTORICAL BACKTEST.** Same exposed 1,607 windows, three seeds and nested ten-fold protocol. No chronological scores are mixed here. FULL remains the fixed reference; SVM the strong classical comparator.

## Actual execution

{
  "status": "COMPLETE_PENDING_NO_FIT_VERIFICATION",
  "blocks": 60,
  "top_level_fit_calls": 4043,
  "actual_classifier_fits": 5243,
  "SVM_sigmoid_calibrators": 1800,
  "device": "CPU",
  "threads": 2,
  "encoder_fits": 0,
  "seconds": 3394.5511086658225,
  "wall_seconds": 3663.4196753749857,
  "LLM_continued": false
}

## Three-seed results

| method                   |   BA_mean_AAPL |   BA_mean_AMZN |   BA_std_AAPL |   BA_std_AMZN |   Brier_mean_AAPL |   Brier_mean_AMZN |   log_loss_mean_AAPL |   log_loss_mean_AMZN |
|:-------------------------|---------------:|---------------:|--------------:|--------------:|------------------:|------------------:|---------------------:|---------------------:|
| PAPER                    |         0.6725 |         0.5852 |        0.0101 |        0.0114 |            0.2272 |            0.2498 |               0.7166 |               0.7286 |
| PRICE                    |         0.5584 |         0.5268 |        0.0071 |        0.0106 |            0.2468 |            0.2563 |               0.6886 |               0.7084 |
| FULL                     |         0.6757 |         0.6028 |        0.0121 |        0.0081 |            0.2228 |            0.2524 |               0.7029 |               0.7581 |
| TFIDF_LR                 |         0.6494 |         0.6011 |        0.0120 |        0.0058 |            0.2210 |            0.2365 |               0.6339 |               0.6658 |
| TFIDF_SVM                |         0.6949 |         0.6012 |        0.0180 |        0.0049 |            0.2056 |            0.2353 |               0.6009 |               0.6634 |
| TFIDF_RF                 |         0.5986 |         0.5846 |        0.0071 |        0.0073 |            0.2332 |            0.2423 |               0.6590 |               0.6776 |
| TFIDF_LR_PRICE_FALLBACK  |         0.6491 |         0.6113 |        0.0127 |        0.0085 |            0.2210 |            0.2359 |               0.6339 |               0.6650 |
| TFIDF_SVM_PRICE_FALLBACK |         0.6955 |         0.6114 |        0.0192 |        0.0049 |            0.2054 |            0.2345 |               0.6003 |               0.6622 |
| TFIDF_RF_PRICE_FALLBACK  |         0.5991 |         0.5948 |        0.0068 |        0.0118 |            0.2331 |            0.2417 |               0.6588 |               0.6768 |
| A1                       |         0.6171 |         0.6106 |        0.0086 |        0.0050 |            0.2292 |            0.2389 |               0.6505 |               0.6711 |
| A2                       |         0.5921 |         0.6026 |        0.0168 |        0.0048 |            0.2346 |            0.2391 |               0.6616 |               0.6716 |
| A3                       |         0.6328 |         0.6116 |        0.0167 |        0.0108 |            0.2221 |            0.2341 |               0.6352 |               0.6613 |
| B1                       |         0.6759 |         0.5996 |        0.0132 |        0.0028 |            0.2239 |            0.2543 |               0.7050 |               0.7636 |
| B2                       |         0.6598 |         0.6040 |        0.0043 |        0.0122 |            0.2296 |            0.2569 |               0.7208 |               0.8266 |
| FULL_SHRINK              |         0.6757 |         0.6028 |        0.0121 |        0.0081 |            0.2133 |            0.2382 |               0.6177 |               0.6701 |
| TITLE_FINBERT_FUSION     |         0.6720 |         0.6041 |        0.0126 |        0.0091 |            0.2185 |            0.2454 |               0.6620 |               0.7117 |
| TITLE_MODERN_FUSION      |         0.6761 |         0.5990 |        0.0080 |        0.0103 |            0.2165 |            0.2433 |               0.6518 |               0.6969 |
| PARAGRAPH_FINBERT_FUSION |         0.6719 |         0.5975 |        0.0115 |        0.0084 |            0.2181 |            0.2447 |               0.6597 |               0.7024 |
| PARAGRAPH_MODERN_FUSION  |         0.6751 |         0.5978 |        0.0100 |        0.0097 |            0.2182 |            0.2414 |               0.6665 |               0.6862 |

A1: ordered raw unigrams + LR; A2: add within-sentence bigrams; A3: same text + calibrated SVM. A methods use exact PRICE fallback. B1/B2: original FULL features plus frozen target-company paragraph vectors, PCA16/uncompressed, in one LR. They are not complete-full-body encodings. No encoder was retrained.

## Interpretation boundaries

- A1 versus old TF-IDF changes order/frequency/numeric/negation retention together; it cannot identify one cause. A1/A2 and A2/A3 are closer component controls. A versus FULL changes both price handling and text/classifier choices.
- B1/B2 share inputs, gating and classifier; this is the matched PCA comparison. Older probability fusion has different metadata/scales/selection and is a historical system reference, not an isolated fusion-location ablation.
- FULL_SHRINK cannot change directions and supplies no new information. Its Brier/log-loss improvement is a confidence control.
- All transforms and SVM calibration text pipelines fit only allowed train rows. Selected FULL inner OOF is finite selection data, never learned stacking training.
- Day/group intervals are descriptive after repeated exploration, not fresh significance tests. Seeds/folds are not independent market samples.
- No per-stock outer-score winner is assembled. No attention, LoRA, RL, new data or LLM continuation.

## Coverage and correction scope

BA/MCC are undefined (blank) in single-class subgroups; TPR/TNR are reported only where their true class exists. Full-stock metrics contain both classes.

| symbol   | coverage                      | method      |        n |   errors |   changed |   repaired |   introduced |
|:---------|:------------------------------|:------------|---------:|---------:|----------:|-----------:|-------------:|
| AAPL     | no_news                       | A1          |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | A2          |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | A3          |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | B1          |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | B2          |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | FULL        |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | FULL_SHRINK |   6.0000 |   3.3333 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | no_news                       | TFIDF_SVM   |   6.0000 |   3.6667 |    2.3333 |     1.0000 |       1.3333 |
| AAPL     | target_paragraph              | A1          | 797.0000 | 292.0000 |  185.3333 |    73.0000 |     112.3333 |
| AAPL     | target_paragraph              | A2          | 797.0000 | 310.0000 |  228.0000 |    85.3333 |     142.6667 |
| AAPL     | target_paragraph              | A3          | 797.0000 | 283.6667 |  182.3333 |    75.6667 |     106.6667 |
| AAPL     | target_paragraph              | B1          | 797.0000 | 253.0000 |   25.0000 |    12.3333 |      12.6667 |
| AAPL     | target_paragraph              | B2          | 797.0000 | 266.6667 |   61.3333 |    23.6667 |      37.6667 |
| AAPL     | target_paragraph              | FULL        | 797.0000 | 252.6667 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | target_paragraph              | FULL_SHRINK | 797.0000 | 252.6667 |    0.0000 |     0.0000 |       0.0000 |
| AAPL     | target_paragraph              | TFIDF_SVM   | 797.0000 | 236.6667 |  143.3333 |    79.6667 |      63.6667 |
| AMZN     | news_without_target_paragraph | A1          |   3.0000 |   1.3333 |    1.0000 |     0.3333 |       0.6667 |
| AMZN     | news_without_target_paragraph | A2          |   3.0000 |   1.6667 |    1.3333 |     0.3333 |       1.0000 |
| AMZN     | news_without_target_paragraph | A3          |   3.0000 |   1.3333 |    1.0000 |     0.3333 |       0.6667 |
| AMZN     | news_without_target_paragraph | B1          |   3.0000 |   1.0000 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | news_without_target_paragraph | B2          |   3.0000 |   1.0000 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | news_without_target_paragraph | FULL        |   3.0000 |   1.0000 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | news_without_target_paragraph | FULL_SHRINK |   3.0000 |   1.0000 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | news_without_target_paragraph | TFIDF_SVM   |   3.0000 |   0.6667 |    0.3333 |     0.3333 |       0.0000 |
| AMZN     | no_news                       | A1          | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | A2          | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | A3          | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | B1          | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | B2          | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | FULL        | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | FULL_SHRINK | 389.0000 | 172.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | no_news                       | TFIDF_SVM   | 389.0000 | 176.0000 |  140.0000 |    68.3333 |      71.6667 |
| AMZN     | target_paragraph              | A1          | 412.0000 | 138.0000 |   84.0000 |    45.3333 |      38.6667 |
| AMZN     | target_paragraph              | A2          | 412.0000 | 144.3333 |   90.3333 |    45.3333 |      45.0000 |
| AMZN     | target_paragraph              | A3          | 412.0000 | 137.0000 |   71.6667 |    39.6667 |      32.0000 |
| AMZN     | target_paragraph              | B1          | 412.0000 | 147.3333 |   12.0000 |     4.6667 |       7.3333 |
| AMZN     | target_paragraph              | B2          | 412.0000 | 143.3333 |   48.0000 |    24.6667 |      23.3333 |
| AMZN     | target_paragraph              | FULL        | 412.0000 | 144.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | target_paragraph              | FULL_SHRINK | 412.0000 | 144.6667 |    0.0000 |     0.0000 |       0.0000 |
| AMZN     | target_paragraph              | TFIDF_SVM   | 412.0000 | 138.0000 |   66.0000 |    36.3333 |      29.6667 |

## Paired differences

| symbol   | reference   | method                   | resampling        |   delta_BA |     low |    high |   replicates |
|:---------|:------------|:-------------------------|:------------------|-----------:|--------:|--------:|-------------:|
| AAPL     | FULL        | A1                       | day1              |    -0.0585 | -0.0921 | -0.0281 |         1000 |
| AAPL     | FULL        | A1                       | day5              |    -0.0585 | -0.0902 | -0.0251 |         1000 |
| AAPL     | FULL        | A1                       | association_group |    -0.0585 | -0.0878 | -0.0307 |         1000 |
| AAPL     | FULL        | A2                       | day1              |    -0.0836 | -0.1198 | -0.0467 |         1000 |
| AAPL     | FULL        | A2                       | day5              |    -0.0836 | -0.1269 | -0.0474 |         1000 |
| AAPL     | FULL        | A2                       | association_group |    -0.0836 | -0.1199 | -0.0495 |         1000 |
| AAPL     | FULL        | A3                       | day1              |    -0.0429 | -0.0761 | -0.0118 |         1000 |
| AAPL     | FULL        | A3                       | day5              |    -0.0429 | -0.0763 | -0.0118 |         1000 |
| AAPL     | FULL        | A3                       | association_group |    -0.0429 | -0.0757 | -0.0171 |         1000 |
| AAPL     | FULL        | B1                       | day1              |     0.0002 | -0.0069 |  0.0072 |         1000 |
| AAPL     | FULL        | B1                       | day5              |     0.0002 | -0.0071 |  0.0072 |         1000 |
| AAPL     | FULL        | B1                       | association_group |     0.0002 | -0.0068 |  0.0071 |         1000 |
| AAPL     | FULL        | B2                       | day1              |    -0.0159 | -0.0295 | -0.0035 |         1000 |
| AAPL     | FULL        | B2                       | day5              |    -0.0159 | -0.0299 | -0.0026 |         1000 |
| AAPL     | FULL        | B2                       | association_group |    -0.0159 | -0.0281 |  0.0002 |         1000 |
| AAPL     | FULL        | FULL_SHRINK              | day1              |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AAPL     | FULL        | FULL_SHRINK              | day5              |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AAPL     | FULL        | FULL_SHRINK              | association_group |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AAPL     | FULL        | TFIDF_SVM_PRICE_FALLBACK | day1              |     0.0198 | -0.0048 |  0.0448 |         1000 |
| AAPL     | FULL        | TFIDF_SVM_PRICE_FALLBACK | day5              |     0.0198 | -0.0067 |  0.0477 |         1000 |
| AAPL     | FULL        | TFIDF_SVM_PRICE_FALLBACK | association_group |     0.0198 | -0.0044 |  0.0439 |         1000 |
| AAPL     | TFIDF_SVM   | A1                       | day1              |    -0.0778 | -0.1059 | -0.0463 |         1000 |
| AAPL     | TFIDF_SVM   | A1                       | day5              |    -0.0778 | -0.1119 | -0.0470 |         1000 |
| AAPL     | TFIDF_SVM   | A1                       | association_group |    -0.0778 | -0.1064 | -0.0497 |         1000 |
| AAPL     | TFIDF_SVM   | A2                       | day1              |    -0.1029 | -0.1377 | -0.0655 |         1000 |
| AAPL     | TFIDF_SVM   | A2                       | day5              |    -0.1029 | -0.1461 | -0.0657 |         1000 |
| AAPL     | TFIDF_SVM   | A2                       | association_group |    -0.1029 | -0.1376 | -0.0684 |         1000 |
| AAPL     | TFIDF_SVM   | A3                       | day1              |    -0.0622 | -0.0890 | -0.0354 |         1000 |
| AAPL     | TFIDF_SVM   | A3                       | day5              |    -0.0622 | -0.0934 | -0.0343 |         1000 |
| AAPL     | TFIDF_SVM   | A3                       | association_group |    -0.0622 | -0.0882 | -0.0387 |         1000 |
| AAPL     | TFIDF_SVM   | B1                       | day1              |    -0.0191 | -0.0456 |  0.0044 |         1000 |
| AAPL     | TFIDF_SVM   | B1                       | day5              |    -0.0191 | -0.0459 |  0.0049 |         1000 |
| AAPL     | TFIDF_SVM   | B1                       | association_group |    -0.0191 | -0.0416 |  0.0044 |         1000 |
| AAPL     | TFIDF_SVM   | B2                       | day1              |    -0.0351 | -0.0604 | -0.0123 |         1000 |
| AAPL     | TFIDF_SVM   | B2                       | day5              |    -0.0351 | -0.0594 | -0.0134 |         1000 |
| AAPL     | TFIDF_SVM   | B2                       | association_group |    -0.0351 | -0.0554 | -0.0095 |         1000 |
| AAPL     | TFIDF_SVM   | FULL_SHRINK              | day1              |    -0.0193 | -0.0453 |  0.0054 |         1000 |
| AAPL     | TFIDF_SVM   | FULL_SHRINK              | day5              |    -0.0193 | -0.0475 |  0.0068 |         1000 |
| AAPL     | TFIDF_SVM   | FULL_SHRINK              | association_group |    -0.0193 | -0.0434 |  0.0058 |         1000 |
| AAPL     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | day1              |     0.0005 | -0.0028 |  0.0036 |         1000 |
| AAPL     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | day5              |     0.0005 | -0.0027 |  0.0034 |         1000 |
| AAPL     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | association_group |     0.0005 | -0.0029 |  0.0038 |         1000 |
| AMZN     | FULL        | A1                       | day1              |     0.0078 | -0.0113 |  0.0277 |         1000 |
| AMZN     | FULL        | A1                       | day5              |     0.0078 | -0.0101 |  0.0264 |         1000 |
| AMZN     | FULL        | A1                       | association_group |     0.0078 | -0.0111 |  0.0273 |         1000 |
| AMZN     | FULL        | A2                       | day1              |    -0.0002 | -0.0204 |  0.0215 |         1000 |
| AMZN     | FULL        | A2                       | day5              |    -0.0002 | -0.0181 |  0.0179 |         1000 |
| AMZN     | FULL        | A2                       | association_group |    -0.0002 | -0.0212 |  0.0202 |         1000 |
| AMZN     | FULL        | A3                       | day1              |     0.0088 | -0.0075 |  0.0250 |         1000 |
| AMZN     | FULL        | A3                       | day5              |     0.0088 | -0.0048 |  0.0231 |         1000 |
| AMZN     | FULL        | A3                       | association_group |     0.0088 | -0.0072 |  0.0247 |         1000 |
| AMZN     | FULL        | B1                       | day1              |    -0.0032 | -0.0091 |  0.0019 |         1000 |
| AMZN     | FULL        | B1                       | day5              |    -0.0032 | -0.0089 |  0.0022 |         1000 |
| AMZN     | FULL        | B1                       | association_group |    -0.0032 | -0.0087 |  0.0027 |         1000 |
| AMZN     | FULL        | B2                       | day1              |     0.0013 | -0.0123 |  0.0142 |         1000 |
| AMZN     | FULL        | B2                       | day5              |     0.0013 | -0.0111 |  0.0132 |         1000 |
| AMZN     | FULL        | B2                       | association_group |     0.0013 | -0.0120 |  0.0159 |         1000 |
| AMZN     | FULL        | FULL_SHRINK              | day1              |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AMZN     | FULL        | FULL_SHRINK              | day5              |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AMZN     | FULL        | FULL_SHRINK              | association_group |     0.0000 |  0.0000 |  0.0000 |         1000 |
| AMZN     | FULL        | TFIDF_SVM_PRICE_FALLBACK | day1              |     0.0086 | -0.0081 |  0.0241 |         1000 |
| AMZN     | FULL        | TFIDF_SVM_PRICE_FALLBACK | day5              |     0.0086 | -0.0058 |  0.0234 |         1000 |
| AMZN     | FULL        | TFIDF_SVM_PRICE_FALLBACK | association_group |     0.0086 | -0.0088 |  0.0241 |         1000 |
| AMZN     | TFIDF_SVM   | A1                       | day1              |     0.0095 | -0.0205 |  0.0412 |         1000 |
| AMZN     | TFIDF_SVM   | A1                       | day5              |     0.0095 | -0.0269 |  0.0409 |         1000 |
| AMZN     | TFIDF_SVM   | A1                       | association_group |     0.0095 | -0.0195 |  0.0429 |         1000 |
| AMZN     | TFIDF_SVM   | A2                       | day1              |     0.0015 | -0.0275 |  0.0330 |         1000 |
| AMZN     | TFIDF_SVM   | A2                       | day5              |     0.0015 | -0.0343 |  0.0334 |         1000 |
| AMZN     | TFIDF_SVM   | A2                       | association_group |     0.0015 | -0.0259 |  0.0358 |         1000 |
| AMZN     | TFIDF_SVM   | A3                       | day1              |     0.0104 | -0.0172 |  0.0399 |         1000 |
| AMZN     | TFIDF_SVM   | A3                       | day5              |     0.0104 | -0.0201 |  0.0394 |         1000 |
| AMZN     | TFIDF_SVM   | A3                       | association_group |     0.0104 | -0.0155 |  0.0402 |         1000 |
| AMZN     | TFIDF_SVM   | B1                       | day1              |    -0.0016 | -0.0308 |  0.0327 |         1000 |
| AMZN     | TFIDF_SVM   | B1                       | day5              |    -0.0016 | -0.0326 |  0.0283 |         1000 |
| AMZN     | TFIDF_SVM   | B1                       | association_group |    -0.0016 | -0.0330 |  0.0328 |         1000 |
| AMZN     | TFIDF_SVM   | B2                       | day1              |     0.0029 | -0.0278 |  0.0365 |         1000 |
| AMZN     | TFIDF_SVM   | B2                       | day5              |     0.0029 | -0.0301 |  0.0326 |         1000 |
| AMZN     | TFIDF_SVM   | B2                       | association_group |     0.0029 | -0.0267 |  0.0353 |         1000 |
| AMZN     | TFIDF_SVM   | FULL_SHRINK              | day1              |     0.0016 | -0.0289 |  0.0356 |         1000 |
| AMZN     | TFIDF_SVM   | FULL_SHRINK              | day5              |     0.0016 | -0.0301 |  0.0329 |         1000 |
| AMZN     | TFIDF_SVM   | FULL_SHRINK              | association_group |     0.0016 | -0.0295 |  0.0365 |         1000 |
| AMZN     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | day1              |     0.0102 | -0.0163 |  0.0380 |         1000 |
| AMZN     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | day5              |     0.0102 | -0.0172 |  0.0360 |         1000 |
| AMZN     | TFIDF_SVM   | TFIDF_SVM_PRICE_FALLBACK | association_group |     0.0102 | -0.0146 |  0.0373 |         1000 |

See metrics.csv, monthly_metrics.csv, selection_records.csv, direction_changes.csv, oracle_scope_diagnostic.csv and CASE_NOTES.md. The oracle scope is explicitly undeployable and never used for selection.
