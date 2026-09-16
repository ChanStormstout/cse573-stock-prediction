# Execution-only batching amendment

Sequential forecasting takes roughly3.3 seconds per price window, before larger
news inputs. Before reading any LLM outcome scores, benchmark native MLX batches
of4 against the first8 already saved sequential price forecasts. Compare raw
texts, parsed probabilities/directions and validation, not target correctness.
If any parsed probability/direction/validity differs or batching fails, keep the
original sequential run. If all eight agree, stop and preserve partial serial
run and execute ALL three comparisons afresh in a separate batched directory.
No prompt/model/budget/selection changes; do not splice a preferred prediction.
Small parity checks cannot prove bitwise equality on all later inputs. Batch
floating-point effects and exact-text differences are reported. Primary full
results then belong to the batched engine. Batch elapsed time is shared across
rows for cost aggregation, not measured per-request latency. The initial serial
and benchmark costs are additional and retained. Original run.py snapshot stays
in its run directory. No model training occurs.
