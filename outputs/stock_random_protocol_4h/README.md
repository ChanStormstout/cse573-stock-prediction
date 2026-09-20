# Random-protocol four-hour study

Status: prepared; baseline training next. [Frozen protocol](PROTOCOL.md).

Main comparison: paper-style keyword LR, R1 price-only, full-body words+price,
FinBERT+price, corrected FinModernBERT+price. Subsequent mechanisms use the same
splits and remain separately identifiable. Existing model predictions are not
reused as new holdout scores. All data are exposed historical course data.

Commands (repository root):

```sh
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/prepare.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/test_contracts.py
work/stock-data/finbert-env/bin/python3 outputs/stock_random_protocol_4h/train.py
```

Weights, raw text, fold assignments and sealed outer probabilities stay under
`work/stock-data/random_protocol_4h/v1`. Source hashes and compact execution
evidence are public. No external ECNI entity/reader lane is resumed.
