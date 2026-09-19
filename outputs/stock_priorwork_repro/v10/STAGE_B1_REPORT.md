# V10 Stage B1 report

Status: **PASS**. DPRICE is complete and NEWS+PRICE was not fitted.

## DPRICE aggregate BA

| stock   | phase       |       BA |    Brier |
|:--------|:------------|---------:|---------:|
| AAPL    | OOF         | 0.472101 | 0.264788 |
| AAPL    | development | 0.379545 | 0.25759  |
| AAPL    | later       | 0.57027  | 0.24835  |
| AMZN    | OOF         | 0.446721 | 0.253791 |
| AMZN    | development | 0.520833 | 0.251163 |
| AMZN    | later       | 0.436259 | 0.256261 |

## Frozen text methods

{
  "24h": [
    "TFIDF_ADABOOST",
    "PAPER_2G_L1LR",
    "TFIDF_KNN"
  ],
  "4h": [
    "TFIDF_LR",
    "PAPER_2G_L1LR",
    "TFIDF_RF"
  ],
  "overnight": [
    "PAPER_1G_L1LR",
    "TFIDF_LR",
    "PAPER_2G_L1LR"
  ]
}
