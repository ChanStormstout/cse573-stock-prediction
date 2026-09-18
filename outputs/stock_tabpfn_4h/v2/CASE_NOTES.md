# TabPFN v2 case notes

Cases are fixed by the common window key before reading the labels: changed
directions between v1 n=1 and v2 n=8, common-right, and common-wrong windows.
The full keyed probabilities are in `comparison_predictions.csv`.

- A changed direction shows the ensemble configuration matters; it does not
  show which configuration is causally correct.
- Common errors remain evidence that changing `n_estimators` cannot repair all
  stock/news information failures.
- No case was selected from development or later scores, and no per-stock
  winner was promoted.
