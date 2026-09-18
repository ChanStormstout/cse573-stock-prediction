# Implementation plan

1. `prepare.py` loads the fixed `R1`/`F1_new` predictions and the ten-state
   vector from the current `paper_methods_4h/v1/inputs.pkl`. It checks keys,
   labels, cutoffs, both experts' chronological fit records, and parity between
   the issued R1 column and its canonical private `P_own_lr` probability file
   without writing a predictive result. It creates the registered advantage
   target only after those checks.
2. `run_gate.py` requires an explicit `--approve-gate-run` flag, builds each
   chronological fold, fits G0--G3 with the fixed weighted ridge protocol,
   saves all evidence and metrics, and freezes the August controller for the
   exposed periods.
3. `verify.py` first performs a real-input preflight without predictive
   execution. It checks advantage arithmetic, both expert provenances, exact
   R1 fallback and nested fallbacks, fold-local median/mean/scale transforms,
   direction-disagreement headroom semantics, finite weights, and G3-to-G2
   nesting. After a separately approved result run it independently reconstructs
   prediction mapping, no-news/exact-R1 behavior, metrics, monthly metrics,
   advancement, chronology, preprocessing and coefficient structure without
   calling the runner's metric or advancement helpers.
4. `report.py` renders the saved CSV/JSON artifacts into a human-readable
   report after an approved run.

Implementation and the real-input preflight have executed. G0--G3 predictive
replay has **not** executed: there is no oracle, metrics, advancement or
prediction result directory. The required later command is:

```bash
work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/run_gate.py \
  --approve-gate-run --output outputs/stock_specific_gate_4h/v1
```
