# Stage 5 claim-only audit protocol

This stage audits interpretation, not predictive performance. Before running
the audit, the protocol is fixed as follows:

1. Read only the saved public calibration manifests and current-facing reports.
2. Record every saved slope, count negative coefficients, and separate the
   early unconstrained Platt files from later constrained files.
3. Produce a machine-readable table of safe and unsafe claims for ISSUE-012
   through ISSUE-021.
4. Do not fit, retrain, run inference, alter predictions, search parameters,
   or choose a method from development/later scores.

The result is an interpretation correction and verification artifact. It does
not create a new model, new score, or independent generalization claim.
