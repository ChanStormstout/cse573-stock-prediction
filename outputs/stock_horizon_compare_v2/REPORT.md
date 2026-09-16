# Horizon comparison

Exploratory historical comparison only. The one-hour target remains the original project task; all three horizons have now been inspected, so none is an untouched target-selection test.

## Metrics

| symbol   | horizon   | method      | split          |      C |   news_coverage |   n |   balanced_accuracy |     mcc |   brier |   predicted_up_fraction |
|:---------|:----------|:------------|:---------------|-------:|----------------:|----:|--------------------:|--------:|--------:|------------------------:|
| AAPL     | 1h        | price       | development    | 0.0100 |          0.9921 | 252 |              0.5120 |  0.0275 |  0.2517 |                  0.7460 |
| AAPL     | 1h        | price       | exposed_replay | 0.0100 |          1.0000 | 364 |              0.4423 | -0.1336 |  0.2653 |                  0.7555 |
| AAPL     | 1h        | price_title | development    | 0.1000 |          0.9921 | 252 |              0.4956 | -0.0106 |  0.2546 |                  0.7778 |
| AAPL     | 1h        | price_title | exposed_replay | 0.1000 |          1.0000 | 364 |              0.4620 | -0.0885 |  0.2747 |                  0.7582 |
| AMZN     | 1h        | price       | development    | 1.0000 |          0.6667 | 252 |              0.4887 | -0.0227 |  0.2536 |                  0.5357 |
| AMZN     | 1h        | price       | exposed_replay | 1.0000 |          0.5342 | 365 |              0.5066 |  0.0134 |  0.2580 |                  0.6000 |
| AMZN     | 1h        | price_title | development    | 1.0000 |          0.6667 | 252 |              0.5158 |  0.0315 |  0.2523 |                  0.4921 |
| AMZN     | 1h        | price_title | exposed_replay | 1.0000 |          0.5342 | 365 |              0.5011 |  0.0022 |  0.2631 |                  0.6000 |
| AAPL     | 4h        | price       | development    | 0.1000 |          0.9841 | 126 |              0.5132 |  0.0385 |  0.2573 |                  0.8651 |
| AAPL     | 4h        | price       | exposed_replay | 0.1000 |          1.0000 | 178 |              0.5292 |  0.0726 |  0.2665 |                  0.7978 |
| AAPL     | 4h        | price_title | development    | 0.1000 |          0.9841 | 126 |              0.5132 |  0.0385 |  0.2595 |                  0.8651 |
| AAPL     | 4h        | price_title | exposed_replay | 0.1000 |          1.0000 | 178 |              0.5189 |  0.0520 |  0.2693 |                  0.8427 |
| AMZN     | 4h        | price       | development    | 0.1000 |          0.5794 | 126 |              0.5009 |  0.0019 |  0.2620 |                  0.6111 |
| AMZN     | 4h        | price       | exposed_replay | 0.1000 |          0.4693 | 179 |              0.4620 | -0.0792 |  0.2677 |                  0.6425 |
| AMZN     | 4h        | price_title | development    | 1.0000 |          0.5794 | 126 |              0.4777 | -0.0447 |  0.2771 |                  0.6190 |
| AMZN     | 4h        | price_title | exposed_replay | 1.0000 |          0.4693 | 179 |              0.4839 | -0.0339 |  0.2775 |                  0.6536 |
| AAPL     | 1d        | price       | development    | 0.1000 |          1.0000 |  42 |              0.4386 | -0.1494 |  0.2865 |                  0.7857 |
| AAPL     | 1d        | price       | exposed_replay | 0.1000 |          1.0000 |  61 |              0.5427 |  0.1553 |  0.2882 |                  0.9180 |
| AAPL     | 1d        | price_title | development    | 0.1000 |          1.0000 |  42 |              0.4591 | -0.1168 |  0.2868 |                  0.8571 |
| AAPL     | 1d        | price_title | exposed_replay | 0.1000 |          1.0000 |  61 |              0.5276 |  0.1111 |  0.2937 |                  0.9344 |
| AMZN     | 1d        | price       | development    | 0.0100 |          1.0000 |  42 |              0.4663 | -0.0693 |  0.2581 |                  0.6667 |
| AMZN     | 1d        | price       | exposed_replay | 0.0100 |          1.0000 |  61 |              0.4941 | -0.0123 |  0.2507 |                  0.3607 |
| AMZN     | 1d        | price_title | development    | 0.1000 |          1.0000 |  42 |              0.4351 | -0.1316 |  0.2793 |                  0.6429 |
| AMZN     | 1d        | price_title | exposed_replay | 0.1000 |          1.0000 |  61 |              0.5419 |  0.0844 |  0.2588 |                  0.4426 |

## Interpretation rule

A horizon is only a promising follow-up if the price+title model exceeds its matched price baseline across both stocks and both reported time ranges without worse Brier score. A single highest score is not sufficient to replace the original target.