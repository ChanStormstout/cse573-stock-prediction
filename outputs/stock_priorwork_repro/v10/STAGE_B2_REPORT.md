# V10 Stage B2 frozen NEWS+PRICE report

Status: **PASS**. All 18 frozen branches and 108 monthly models were executed and independently replayed. No B2 result selected or tuned a model.

## Four-hour joint results

| method        | stock   | phase       |   n |       BA |    Brier |
|:--------------|:--------|:------------|----:|---------:|---------:|
| PAPER_2G_L1LR | AAPL    | development | 126 | 0.47997  | 0.262699 |
| PAPER_2G_L1LR | AAPL    | later       | 178 | 0.508147 | 0.255695 |
| TFIDF_LR      | AAPL    | development | 126 | 0.515467 | 0.255788 |
| TFIDF_LR      | AAPL    | later       | 178 | 0.501389 | 0.259288 |
| TFIDF_RF      | AAPL    | development | 126 | 0.511663 | 0.262369 |
| TFIDF_RF      | AAPL    | later       | 178 | 0.492611 | 0.264703 |
| PAPER_2G_L1LR | AMZN    | development | 126 | 0.478664 | 0.301801 |
| PAPER_2G_L1LR | AMZN    | later       | 179 | 0.507384 | 0.28204  |
| TFIDF_LR      | AMZN    | development | 126 | 0.543599 | 0.26325  |
| TFIDF_LR      | AMZN    | later       | 179 | 0.434043 | 0.273651 |
| TFIDF_RF      | AMZN    | development | 126 | 0.521336 | 0.259938 |
| TFIDF_RF      | AMZN    | later       | 179 | 0.497309 | 0.262992 |

## Daily joint results

| horizon_window     | method         | stock   | phase       |   n |       BA |    Brier |
|:-------------------|:---------------|:--------|:------------|----:|---------:|---------:|
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AAPL    | development |  42 | 0.5      | 0.25     |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AAPL    | later       |  62 | 0.5      | 0.25     |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AAPL    | development |  42 | 0.527273 | 0.263889 |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AAPL    | later       |  62 | 0.39027  | 0.31581  |
| 1d:DNEWS_24H       | TFIDF_KNN      | AAPL    | development |  42 | 0.565909 | 0.292381 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AAPL    | later       |  62 | 0.48973  | 0.34129  |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AAPL    | development |  42 | 0.479545 | 0.383914 |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AAPL    | later       |  62 | 0.502703 | 0.35116  |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AAPL    | development |  42 | 0.584091 | 0.247781 |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AAPL    | later       |  62 | 0.47027  | 0.257588 |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AAPL    | development |  42 | 0.379545 | 0.257584 |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AAPL    | later       |  62 | 0.57027  | 0.248351 |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AMZN    | development |  42 | 0.479167 | 0.302596 |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AMZN    | later       |  62 | 0.436259 | 0.320449 |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AMZN    | development |  42 | 0.388889 | 0.279649 |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AMZN    | later       |  62 | 0.483804 | 0.253542 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AMZN    | development |  42 | 0.576389 | 0.282857 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AMZN    | later       |  62 | 0.503135 | 0.34     |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AMZN    | development |  42 | 0.486111 | 0.299773 |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AMZN    | later       |  62 | 0.434169 | 0.347132 |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AMZN    | development |  42 | 0.5      | 0.25     |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AMZN    | later       |  62 | 0.5      | 0.25     |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AMZN    | development |  42 | 0.555556 | 0.266663 |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AMZN    | later       |  62 | 0.416928 | 0.269609 |

## Four-hour system-level increments

| method        | stock   | phase       |   DELTA_BA_VS_R1 |   DELTA_BA_VS_NEWS_ONLY |
|:--------------|:--------|:------------|-----------------:|------------------------:|
| PAPER_2G_L1LR | AAPL    | development |      -0.0821501  |              0.0207911  |
| PAPER_2G_L1LR | AAPL    | later       |      -0.030504   |             -0.00978906 |
| PAPER_2G_L1LR | AMZN    | development |      -0.0324675  |             -0.00278293 |
| PAPER_2G_L1LR | AMZN    | later       |      -0.00112641 |             -0.0097622  |
| TFIDF_LR      | AAPL    | development |      -0.0466531  |              0.0154665  |
| TFIDF_LR      | AAPL    | later       |      -0.0372616  |             -0.00309461 |
| TFIDF_LR      | AMZN    | development |       0.0324675  |             -0.00463822 |
| TFIDF_LR      | AMZN    | later       |      -0.0744681  |             -0.0775344  |
| TFIDF_RF      | AAPL    | development |      -0.0504564  |             -0.0352434  |
| TFIDF_RF      | AAPL    | later       |      -0.0460402  |              0.0462296  |
| TFIDF_RF      | AMZN    | development |       0.0102041  |             -0.00371058 |
| TFIDF_RF      | AMZN    | later       |      -0.0112015  |             -0.0212766  |

## Daily system-level increments

| horizon_window     | method         | stock   | phase       |   DELTA_BA_VS_DPRICE |   DELTA_BA_VS_NEWS_ONLY |
|:-------------------|:---------------|:--------|:------------|---------------------:|------------------------:|
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AAPL    | development |           0.120455   |               0         |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AAPL    | later       |          -0.0702703  |               0         |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AMZN    | development |          -0.0416667  |               0.0138889 |
| 1d:DNEWS_24H       | PAPER_2G_L1LR  | AMZN    | later       |           0          |               0         |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AAPL    | development |           0.147727   |               0         |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AAPL    | later       |          -0.18       |               0         |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AMZN    | development |          -0.131944   |              -0.0138889 |
| 1d:DNEWS_24H       | TFIDF_ADABOOST | AMZN    | later       |           0.0475444  |              -0.0130617 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AAPL    | development |           0.186364   |              -0.0681818 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AAPL    | later       |          -0.0805405  |               0.0135135 |
| 1d:DNEWS_24H       | TFIDF_KNN      | AMZN    | development |           0.0555556  |               0.229167  |
| 1d:DNEWS_24H       | TFIDF_KNN      | AMZN    | later       |           0.0668757  |               0.0172414 |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AAPL    | development |           0.1        |               0         |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AAPL    | later       |          -0.0675676  |              -0.0140541 |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AMZN    | development |          -0.0347222  |              -0.0347222 |
| 1d:DNEWS_OVERNIGHT | PAPER_1G_L1LR  | AMZN    | later       |          -0.00208986 |              -0.0282132 |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AAPL    | development |           0.204545   |               0         |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AAPL    | later       |          -0.1        |               0         |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AMZN    | development |          -0.0208333  |               0         |
| 1d:DNEWS_OVERNIGHT | PAPER_2G_L1LR  | AMZN    | later       |           0.0637409  |              -0.0109718 |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AAPL    | development |           0          |              -0.120455  |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AAPL    | later       |           0          |               0.119459  |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AMZN    | development |           0.0347222  |               0         |
| 1d:DNEWS_OVERNIGHT | TFIDF_LR       | AMZN    | later       |          -0.0193312  |              -0.0668757 |

These are system-level comparisons when classifier families differ; they are not pure causal attribution to news. Development/later are historical evaluations under their registered labels, not untouched tests.
