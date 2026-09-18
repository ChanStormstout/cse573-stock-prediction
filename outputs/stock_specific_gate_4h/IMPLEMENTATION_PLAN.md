# Implementation plan

1. `prepare.py` loads the fixed `R1`/`F1_new` predictions and the ten-state
   vector from the current `paper_methods_4h/v1/inputs.pkl`.  It checks keys,
   labels, cutoffs, and expert provenance without writing a result.
2. `run_gate.py` requires an explicit `--approve-gate-run` flag, builds each
   chronological fold, fits G0--G3 with the fixed weighted ridge protocol,
   saves all evidence and metrics, and freezes the August controller for the
   exposed periods.
3. `verify.py` reconstructs metrics and checks time safety, no-news equality,
   finite weights, advantage arithmetic, and the G3-to-G2 nesting regression.
4. `report.py` renders the saved CSV/JSON artifacts into a human-readable
   report after an approved run.

No command in this implementation plan has been executed.  The required later
command is:

```bash
work/stock-data/finbert-env/bin/python3 outputs/stock_specific_gate_4h/run_gate.py \
  --approve-gate-run --output outputs/stock_specific_gate_4h/v1
```
