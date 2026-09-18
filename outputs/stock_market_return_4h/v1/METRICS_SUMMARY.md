# Market state and continuous-return auxiliary experiment — v1

## Plain-language result

This run tested whether adding the continuous four-hour return as an auxiliary training signal improves the unchanged four-hour direction task. It used the original price features, plus a secondary variant that appends the already-frozen F1 probability. The model and parameter were chosen from March–August chronological forward OOF only; development (September–October) and later (November onward) were scored separately and were not used for selection.

The selected auxiliary objective did not pass the registered promotion line. For the raw-price branch the chosen C2 lambda was 0.0, so the OOF selection preferred the classification-only loss. For the price+F1 branch it selected lambda 0.5, but the later-period AMZN result remained below 50%. This is evidence against claiming a stable improvement, not evidence that the entire project is invalid.

## External market-data audit

The fixed SPY/QQQ one-minute Alpaca probe stopped with `STOP_AUTH_OR_DATA_AUDIT`: credentials_present=False, status_counts={'AUTH_REQUIRED_NO_CREDENTIALS': 8}. No daily or synthetic ETF substitute was used.

## Selected parameters

| input       |   selected_parameter |
|:------------|---------------------:|
| price/C0    |                 0.1  |
| price/C1    |                 1    |
| price/C2    |                 0    |
| price_F1/C0 |                 0.1  |
| price_F1/C1 |                 0.01 |
| price_F1/C2 |                 0.5  |

## Results by period

### OOF selected model (March–August)

| symbol   | method             |   n |     BA |     MCC |   Brier |   pred_up |   return_MAE |   return_corr |
|:---------|:-------------------|----:|-------:|--------:|--------:|----------:|-------------:|--------------:|
| AAPL     | C0 price           | 383 | 0.5387 |  0.0761 |  0.2611 |    0.5170 |     nan      |      nan      |
| AAPL     | C0 price+F1        | 383 | 0.5539 |  0.1061 |  0.2598 |    0.5222 |     nan      |      nan      |
| AAPL     | C1 return ridge    | 383 | 0.5176 |  0.0356 |  0.2584 |    0.3838 |       0.0064 |        0.0275 |
| AAPL     | C1 return ridge+F1 | 383 | 0.5173 |  0.0353 |  0.2583 |    0.3708 |       0.0065 |        0.0371 |
| AAPL     | C2 joint           | 383 | 0.5259 |  0.0511 |  0.2854 |    0.5274 |       0.0056 |       -0.0274 |
| AAPL     | C2 joint+F1        | 383 | 0.5380 |  0.0749 |  0.2835 |    0.5352 |       0.0062 |        0.0543 |
| AAPL     | F0                 | 383 | 0.5002 |  0.0004 |  0.2650 |    0.5352 |     nan      |      nan      |
| AAPL     | F1                 | 383 | 0.5030 |  0.0061 |  0.2593 |    0.6214 |     nan      |      nan      |
| AAPL     | F2                 | 383 | 0.5002 |  0.0004 |  0.2602 |    0.5352 |     nan      |      nan      |
| AAPL     | F6                 | 383 | 0.5534 |  0.1105 |  0.2452 |    0.6554 |     nan      |      nan      |
| AAPL     | R1                 | 383 | 0.5268 |  0.0529 |  0.2577 |    0.4648 |     nan      |      nan      |
| AMZN     | C0 price           | 382 | 0.5249 |  0.0520 |  0.2603 |    0.6466 |     nan      |      nan      |
| AMZN     | C0 price+F1        | 382 | 0.5144 |  0.0323 |  0.2646 |    0.7304 |     nan      |      nan      |
| AMZN     | C1 return ridge    | 382 | 0.5061 |  0.0124 |  0.2615 |    0.4031 |       0.0081 |       -0.1594 |
| AMZN     | C1 return ridge+F1 | 382 | 0.5058 |  0.0117 |  0.2655 |    0.4084 |       0.0085 |       -0.1381 |
| AMZN     | C2 joint           | 382 | 0.5262 |  0.0541 |  0.2793 |    0.6257 |       0.0070 |        0.0241 |
| AMZN     | C2 joint+F1        | 382 | 0.5310 |  0.0661 |  0.2885 |    0.6754 |       0.0079 |       -0.0926 |
| AMZN     | F0                 | 382 | 0.5187 |  0.0394 |  0.2578 |    0.6623 |     nan      |      nan      |
| AMZN     | F1                 | 382 | 0.5327 |  0.0653 |  0.2606 |    0.4817 |     nan      |      nan      |
| AMZN     | F2                 | 382 | 0.4814 | -0.0384 |  0.2749 |    0.6283 |     nan      |      nan      |
| AMZN     | F6                 | 382 | 0.5514 |  0.1104 |  0.2520 |    0.6859 |     nan      |      nan      |
| AMZN     | R1                 | 382 | 0.5218 |  0.0457 |  0.2562 |    0.6545 |     nan      |      nan      |

### Development (September–October)

| symbol   | method             |   n |     BA |     MCC |   Brier |   pred_up |   return_MAE |   return_corr |
|:---------|:-------------------|----:|-------:|--------:|--------:|----------:|-------------:|--------------:|
| AAPL     | C0 price           | 126 | 0.5510 |  0.1155 |  0.2478 |    0.7381 |     nan      |      nan      |
| AAPL     | C0 price+F1        | 126 | 0.5669 |  0.1518 |  0.2482 |    0.7381 |     nan      |      nan      |
| AAPL     | C1 return ridge    | 126 | 0.5680 |  0.1421 |  0.2417 |    0.6508 |       0.0071 |        0.2090 |
| AAPL     | C1 return ridge+F1 | 126 | 0.5925 |  0.1946 |  0.2422 |    0.6587 |       0.0071 |        0.2120 |
| AAPL     | C2 joint           | 126 | 0.5238 |  0.0518 |  0.2503 |    0.6984 |       0.0071 |       -0.1989 |
| AAPL     | C2 joint+F1        | 126 | 0.5484 |  0.1060 |  0.2508 |    0.7063 |       0.0074 |        0.0451 |
| AAPL     | F0                 | 126 | 0.5046 |  0.0130 |  0.2601 |    0.8571 |     nan      |      nan      |
| AAPL     | F1                 | 126 | 0.5794 |  0.1955 |  0.2500 |    0.7937 |     nan      |      nan      |
| AAPL     | F2                 | 126 | 0.5413 |  0.1067 |  0.2803 |    0.8175 |     nan      |      nan      |
| AAPL     | F6                 | 126 | 0.5035 |  0.0154 |  0.2573 |    0.9444 |     nan      |      nan      |
| AAPL     | R1                 | 126 | 0.5621 |  0.1489 |  0.2441 |    0.7778 |     nan      |      nan      |
| AMZN     | C0 price           | 126 | 0.5315 |  0.0639 |  0.2750 |    0.6349 |     nan      |      nan      |
| AMZN     | C0 price+F1        | 126 | 0.5343 |  0.0686 |  0.2739 |    0.6111 |     nan      |      nan      |
| AMZN     | C1 return ridge    | 126 | 0.5195 |  0.0382 |  0.2562 |    0.5476 |       0.0117 |        0.0880 |
| AMZN     | C1 return ridge+F1 | 126 | 0.5288 |  0.0561 |  0.2579 |    0.5159 |       0.0116 |        0.0732 |
| AMZN     | C2 joint           | 126 | 0.5306 |  0.0608 |  0.2971 |    0.5952 |       0.0108 |       -0.0053 |
| AMZN     | C2 joint+F1        | 126 | 0.4935 | -0.0128 |  0.2978 |    0.5794 |       0.0116 |        0.0783 |
| AMZN     | F0                 | 126 | 0.4833 | -0.0329 |  0.2710 |    0.5714 |     nan      |      nan      |
| AMZN     | F1                 | 126 | 0.5566 |  0.1118 |  0.2481 |    0.4206 |     nan      |      nan      |
| AMZN     | F2                 | 126 | 0.4629 | -0.0728 |  0.2977 |    0.5556 |     nan      |      nan      |
| AMZN     | F6                 | 126 | 0.5250 |  0.0510 |  0.2652 |    0.6429 |     nan      |      nan      |
| AMZN     | R1                 | 126 | 0.5111 |  0.0223 |  0.2719 |    0.6190 |     nan      |      nan      |

### Later exposed period (November onward)

| symbol   | method             |   n |     BA |     MCC |   Brier |   pred_up |   return_MAE |   return_corr |
|:---------|:-------------------|----:|-------:|--------:|--------:|----------:|-------------:|--------------:|
| AAPL     | C0 price           | 178 | 0.5264 |  0.0563 |  0.2709 |    0.6742 |     nan      |      nan      |
| AAPL     | C0 price+F1        | 178 | 0.5152 |  0.0323 |  0.2689 |    0.6742 |     nan      |      nan      |
| AAPL     | C1 return ridge    | 178 | 0.5254 |  0.0525 |  0.2621 |    0.6292 |       0.0106 |        0.0626 |
| AAPL     | C1 return ridge+F1 | 178 | 0.5027 |  0.0055 |  0.2614 |    0.6180 |       0.0106 |        0.0554 |
| AAPL     | C2 joint           | 178 | 0.5147 |  0.0307 |  0.2786 |    0.6517 |       0.0099 |       -0.0994 |
| AAPL     | C2 joint+F1        | 178 | 0.5204 |  0.0430 |  0.2757 |    0.6573 |       0.0105 |        0.0449 |
| AAPL     | F0                 | 178 | 0.5189 |  0.0520 |  0.2694 |    0.8427 |     nan      |      nan      |
| AAPL     | F1                 | 178 | 0.5174 |  0.0418 |  0.2587 |    0.7753 |     nan      |      nan      |
| AAPL     | F2                 | 178 | 0.5671 |  0.1523 |  0.2856 |    0.7360 |     nan      |      nan      |
| AAPL     | F6                 | 178 | 0.4939 | -0.0137 |  0.2529 |    0.7303 |     nan      |      nan      |
| AAPL     | R1                 | 178 | 0.5387 |  0.0860 |  0.2558 |    0.7191 |     nan      |      nan      |
| AMZN     | C0 price           | 179 | 0.5021 |  0.0043 |  0.2740 |    0.6257 |     nan      |      nan      |
| AMZN     | C0 price+F1        | 179 | 0.5138 |  0.0284 |  0.2710 |    0.6145 |     nan      |      nan      |
| AMZN     | C1 return ridge    | 179 | 0.5528 |  0.1086 |  0.2539 |    0.6201 |       0.0141 |        0.1097 |
| AMZN     | C1 return ridge+F1 | 179 | 0.5544 |  0.1104 |  0.2556 |    0.5866 |       0.0141 |        0.0696 |
| AMZN     | C2 joint           | 179 | 0.5026 |  0.0054 |  0.2885 |    0.6145 |       0.0139 |       -0.1902 |
| AMZN     | C2 joint+F1        | 179 | 0.4973 | -0.0055 |  0.2849 |    0.6089 |       0.0140 |        0.0892 |
| AMZN     | F0                 | 179 | 0.4979 | -0.0043 |  0.2677 |    0.5978 |     nan      |      nan      |
| AMZN     | F1                 | 179 | 0.5436 |  0.0872 |  0.2583 |    0.4693 |     nan      |      nan      |
| AMZN     | F2                 | 179 | 0.5427 |  0.0869 |  0.2617 |    0.5978 |     nan      |      nan      |
| AMZN     | F6                 | 179 | 0.5116 |  0.0244 |  0.2545 |    0.6592 |     nan      |      nan      |
| AMZN     | R1                 | 179 | 0.5085 |  0.0174 |  0.2722 |    0.6089 |     nan      |      nan      |

## Promotion check

The registered engineering line was June–August mean BA gain ≥1 percentage point for both stocks, no stock loss >1 point, and mean Brier worsening ≤0.002. It is a screening rule, not a significance test.

| variant   | method   |   choice |   base_JunAug_mean_BA |   method_JunAug_mean_BA |   delta_BA |   AAPL_delta_BA |   AMZN_delta_BA |   delta_Brier | passes_engineering_gate   |
|:----------|:---------|---------:|----------------------:|------------------------:|-----------:|----------------:|----------------:|--------------:|:--------------------------|
| price     | C1       |   1.0000 |                0.5764 |                  0.5455 |    -0.0309 |         -0.0231 |         -0.0388 |        0.0090 | False                     |
| price     | C2       |   0.0000 |                0.5764 |                  0.5772 |     0.0007 |          0.0076 |         -0.0061 |        0.0008 | False                     |
| price_F1  | C1       |   0.0100 |                0.5784 |                  0.5217 |    -0.0567 |         -0.0657 |         -0.0477 |        0.0098 | False                     |
| price_F1  | C2       |   0.5000 |                0.5784 |                  0.5750 |    -0.0034 |          0.0083 |         -0.0151 |        0.0020 | False                     |

