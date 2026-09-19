# Prior-work clean reproduction and one-day study

**Verification: PASS.** Exploratory historical study; existing controls preserved.

## Four-hour BA

| method          |   ('AAPL', 'development') |   ('AAPL', 'later') |   ('AMZN', 'development') |   ('AMZN', 'later') |
|:----------------|--------------------------:|--------------------:|--------------------------:|--------------------:|
| PAPER_1G_L1LR   |                    0.5125 |              0.543  |                    0.5525 |              0.4816 |
| PAPER_1G_LINSVM |                    0.5557 |              0.5979 |                    0.528  |              0.5065 |
| PAPER_2G_L1LR   |                    0.4575 |              0.5183 |                    0.5    |              0.5078 |
| TFIDF_ADABOOST  |                    0.5045 |              0.51   |                    0.4994 |              0.5092 |
| TFIDF_KNN       |                    0.5672 |              0.5482 |                    0.5499 |              0.4833 |
| TFIDF_LINSVM    |                    0.5    |              0.4993 |                    0.5835 |              0.5074 |
| TFIDF_LR        |                    0.5    |              0.5    |                    0.5    |              0.4934 |
| TFIDF_RF        |                    0.5342 |              0.5192 |                    0.5563 |              0.5083 |

## One-day BA

| method          |   ('AAPL', 'development') |   ('AAPL', 'later') |   ('AMZN', 'development') |   ('AMZN', 'later') |
|:----------------|--------------------------:|--------------------:|--------------------------:|--------------------:|
| PAPER_1G_L1LR   |                    0.5    |              0.4879 |                    0.5    |              0.4909 |
| PAPER_1G_LINSVM |                    0.4609 |              0.6311 |                    0.5149 |              0.4764 |
| PAPER_2G_L1LR   |                    0.5    |              0.5    |                    0.5    |              0.5106 |
| TFIDF_ADABOOST  |                    0.4626 |              0.5209 |                    0.4569 |              0.4806 |
| TFIDF_KNN       |                    0.5063 |              0.4882 |                    0.5319 |              0.4871 |
| TFIDF_LINSVM    |                    0.4754 |              0.5124 |                    0.5387 |              0.53   |
| TFIDF_LR        |                    0.5    |              0.5    |                    0.5089 |              0.5317 |
| TFIDF_RF        |                    0.4513 |              0.5344 |                    0.447  |              0.5397 |

## Joint news+price (development/later combined)

| stock   | horizon            | method              | phase             |   n |   accuracy |       ba |         mcc |   precision |   recall |       f1 |   up_recall |   down_recall |   pred_up |   true_up | constant   |      auc |      brier |
|:--------|:-------------------|:--------------------|:------------------|----:|-----------:|---------:|------------:|------------:|---------:|---------:|------------:|--------------:|----------:|----------:|:-----------|---------:|-----------:|
| AAPL    | 4h                 | JOINT_TFIDF_KNN     | development_later | 304 |   0.490132 | 0.498308 | -0.00361548 |    0.475728 | 0.675862 | 0.558405 |    0.675862 |     0.320755  |  0.677632 |  0.476974 | False      | 0.50848  |   0.322207 |
| AMZN    | 4h                 | JOINT_TFIDF_KNN     | development_later | 305 |   0.47541  | 0.49182  | -0.0192251  |    0.463519 | 0.755245 | 0.574468 |    0.755245 |     0.228395  |  0.763934 |  0.468852 | False      | 0.484482 |   0.297833 |
| AAPL    | 4h                 | JOINT_PAPER_1G_L1LR | development_later | 304 |   0.509868 | 0.52082  |  0.0472302  |    0.491071 | 0.758621 | 0.596206 |    0.758621 |     0.283019  |  0.736842 |  0.476974 | False      | 0.527695 |   0.260511 |
| AMZN    | 4h                 | JOINT_PAPER_1G_L1LR | development_later | 305 |   0.508197 | 0.523094 |  0.0523989  |    0.484444 | 0.762238 | 0.592391 |    0.762238 |     0.283951  |  0.737705 |  0.468852 | False      | 0.498683 |   0.25357  |
| AAPL    | 4h                 | JOINT_PAPER_2G_L1LR | development_later | 304 |   0.480263 | 0.499198 | -0.00281772 |    0.476534 | 0.910345 | 0.625592 |    0.910345 |     0.0880503 |  0.911184 |  0.476974 | False      | 0.5054   |   0.260496 |
| AMZN    | 4h                 | JOINT_PAPER_2G_L1LR | development_later | 305 |   0.481967 | 0.501273 |  0.00323865 |    0.469636 | 0.811189 | 0.594872 |    0.811189 |     0.191358  |  0.809836 |  0.468852 | False      | 0.485518 |   0.254867 |
| AAPL    | 1d:DNEWS_OVERNIGHT | JOINT_TFIDF_KNN     | development_later | 104 |   0.490385 | 0.492844 | -0.0141928  |    0.425926 | 0.511111 | 0.464646 |    0.511111 |     0.474576  |  0.519231 |  0.432692 | False      | 0.488889 |   0.305    |
| AMZN    | 1d:DNEWS_OVERNIGHT | JOINT_TFIDF_KNN     | development_later | 104 |   0.480769 | 0.496454 | -0.00746979 |    0.449275 | 0.659574 | 0.534483 |    0.659574 |     0.333333  |  0.663462 |  0.451923 | False      | 0.496081 |   0.314287 |
| AAPL    | 1d:DNEWS_OVERNIGHT | JOINT_TFIDF_LINSVM  | development_later | 104 |   0.432692 | 0.441996 | -0.116337   |    0.383333 | 0.511111 | 0.438095 |    0.511111 |     0.372881  |  0.576923 |  0.432692 | False      | 0.439171 | nan        |
| AMZN    | 1d:DNEWS_OVERNIGHT | JOINT_TFIDF_LINSVM  | development_later | 104 |   0.461538 | 0.469578 | -0.0614897  |    0.42623  | 0.553191 | 0.481481 |    0.553191 |     0.385965  |  0.586538 |  0.451923 | False      | 0.463792 | nan        |
| AAPL    | 1d:DNEWS_OVERNIGHT | JOINT_PAPER_1G_L1LR | development_later | 104 |   0.451923 | 0.429944 | -0.145917   |    0.333333 | 0.266667 | 0.296296 |    0.266667 |     0.59322   |  0.346154 |  0.432692 | False      | 0.422599 |   0.251273 |
| AMZN    | 1d:DNEWS_OVERNIGHT | JOINT_PAPER_1G_L1LR | development_later | 104 |   0.451923 | 0.5      |  0          |    0.451923 | 1        | 0.622517 |    1        |     0         |  1        |  0.451923 | True       | 0.474244 |   0.251018 |

## Random diagnostic

| label                                                | stock   | method   |   n |   accuracy |       ba |        mcc |   precision |   recall |       f1 |   up_recall |   down_recall |   pred_up |   true_up | constant   |      auc |    brier |
|:-----------------------------------------------------|:--------|:---------|----:|-----------:|---------:|-----------:|------------:|---------:|---------:|------------:|--------------:|----------:|----------:|:-----------|---------:|---------:|
| RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE | AAPL    | TFIDF_LR | 161 |   0.540373 | 0.5      | 0          |    0.540373 | 1        | 0.701613 |    1        |     0         |  1        |  0.540373 | True       | 0.720099 | 0.247235 |
| RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE | AAPL    | TFIDF_RF | 161 |   0.68323  | 0.671559 | 0.360567   |    0.669811 | 0.816092 | 0.735751 |    0.816092 |     0.527027  |  0.658385 |  0.540373 | False      | 0.743709 | 0.217579 |
| RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE | AMZN    | TFIDF_LR | 161 |   0.515528 | 0.500386 | 0.00348474 |    0.515723 | 0.987952 | 0.677686 |    0.987952 |     0.0128205 |  0.987578 |  0.515528 | False      | 0.519308 | 0.249249 |
| RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE | AMZN    | TFIDF_RF | 161 |   0.565217 | 0.559391 | 0.12829    |    0.558559 | 0.746988 | 0.639175 |    0.746988 |     0.371795  |  0.689441 |  0.515528 | False      | 0.590825 | 0.243648 |

XGBoost and Word2Vec are recorded unavailable in the existing environment; neither was installed.
