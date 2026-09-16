# One targeted second residual version: zero news intercept

Registered after v3 diagnosis and before this variant's fitting. All historical periods already exposed. This is an additional exploratory test, not a newly independent test.

Observed in v3 AAPL: past price OOF mean .4991 vs actual positive fraction .5864; later price mean .5866 vs actual .4888. News intercepts body .5943 and semantic .4175, and98.3% of later semantic corrections were positive. The free news intercept can absorb calibration/class-prior error instead of news-specific information; this is a plausible failure mechanism, not proof the intercept explains every error.

Change exactly one mechanism: fix news intercept to zero in both residual and matched joint controls. Joint-control price intercept remains free. Body100/PCA16, C grid, monthly boundaries, offset rows, clipping, router, calibration, thresholds, budget gate, frozen evaluation all unchanged. No feature dimension search, new encoder or more gate candidates. Recompute the entire OOF/selection chain for this variant, not merely subtract an intercept from old test predictions. End-to-end model weights must be trained and saved.

One such extra version only. Report both free and zero variants, even if the new one is worse. No subsequent search prompted by these scores. As before, baseline is retained if the predeclared development gate fails. Online version may run with the same fixed online protocol, separately labeled.
