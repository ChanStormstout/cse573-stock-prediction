# Output consistency diagnostic — declared during inference, before LLM scoring

At the first42 price-only outputs,11 forecasts returned p_up=.5 and DOWN,
violating the registered tie rule UP. Two raw examples were read without
joining their target labels. All three synthetic smoke outputs had been .5 UP.
The frozen prompt, generation and primary invalid-fallback scoring remain unchanged.

To avoid confusing forecast quality with this output convention, ALSO report
native declared UP/DOWN accuracy/BA/MCC on every window. A malformed/missing
direction still falls back to UP and is flagged. Its Brier is not computed from
hard labels; raw numeric probabilities are reported separately if valid in [0,1],
with .5 for missing/invalid numeric probability. Direction and raw probability
may disagree. Thus this is not one coherent probabilistic forecast system and
cannot be silently promoted to the primary method. No prompt repair, rerun or
hyperparameter selection is authorized by this diagnostic. No outcome-based
choice between the primary and diagnostic results.

This is an interface diagnostic added after observing unlabeled generated
outputs, not the original registered primary comparison. The full article pack,
window coverage and frozen model remain the same. Counts of ties, raw numeric
validity, native direction validity and contradictory outputs must be shown.
