# Dataset audit

| horizon   | stock   | phase       |   n |   news_coverage |   no_news |   mean_articles |   dates |
|:----------|:--------|:------------|----:|----------------:|----------:|----------------:|--------:|
| 4h        | AAPL    | development | 126 |        0.984127 |         2 |         8.1129  |      42 |
| 4h        | AAPL    | later       | 178 |        1        |         0 |        13.8483  |      60 |
| 4h        | AAPL    | oof         | 499 |        0.991984 |         4 |        12.3636  |     167 |
| 4h        | AMZN    | development | 126 |        0.579365 |        53 |         1.79452 |      42 |
| 4h        | AMZN    | later       | 179 |        0.469274 |        95 |         1.70238 |      60 |
| 4h        | AMZN    | oof         | 499 |        0.517034 |       241 |         1.77132 |     167 |
| 1d        | AAPL    | development |  84 |        1        |         0 |        34.7738  |      42 |
| 1d        | AAPL    | later       | 124 |        1        |         0 |        52.8387  |      62 |
| 1d        | AAPL    | oof         | 328 |        1        |         0 |        46.4085  |     164 |
| 1d        | AMZN    | development |  84 |        0.904762 |         8 |         4.76316 |      42 |
| 1d        | AMZN    | later       | 124 |        0.870968 |        16 |         3.9537  |      62 |
| 1d        | AMZN    | oof         | 328 |        0.844512 |        51 |         4.00361 |     164 |

AAPL/AMZN coverage asymmetry, two stocks, a short span, overlapping 4h windows, repeated articles, provider availability timestamps and exposed later periods limit generalization. No hard rows were removed.
