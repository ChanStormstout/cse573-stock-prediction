# FNSPID Amazon historical-news candidates

This lane measures whether FNSPID can fill Amazon news-context gaps without
using outcome labels. It treats date-only records as available at the next XNYS
regular open and keeps direct, multi-company and indirect relations separate.

Run with the environment that provides pandas, DuckDB, PyArrow and
exchange-calendars:

```bash
PYTHONPATH=/path/to/work/stock-data/deps \
/path/to/work/stock-data/finbert-env/bin/python3 \
  outputs/stock_fnspid_amzn_4h/build_candidates.py \
  --source-repo /path/to/source/repository \
  --output outputs/stock_fnspid_amzn_4h/v1 \
  --private-output /path/to/work/stock-data/fnspid_amzn_4h/v1

PYTHONPATH=/path/to/work/stock-data/deps \
/path/to/work/stock-data/finbert-env/bin/python3 \
  outputs/stock_fnspid_amzn_4h/verify.py \
  --source-repo /path/to/source/repository \
  --output outputs/stock_fnspid_amzn_4h/v1 \
  --private-output /path/to/work/stock-data/fnspid_amzn_4h/v1
```

`extract_review_cards.py` is a separate outcome-free pass over the local raw
CSV files. It writes 120 private blinded cards and a public hash/count audit;
the scan can take several minutes because the public FNSPID distribution is
large.

Private candidate mappings are never committed. Public outputs contain hashes,
counts and time-safe window mappings only.
