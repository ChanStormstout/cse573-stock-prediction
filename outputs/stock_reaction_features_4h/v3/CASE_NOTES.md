# Reaction v3 case notes

Cases were fixed before reading the June--August gate outcome.  They are
diagnostics, not extra training labels.

1. **Target association:** retain a ticker/legal-name match, but record that a
   multi-company article can mention the target only in a list or comparison.
   The target-pair key remains auditable and the evidence tier is preserved.
2. **Completed-bar cutoff:** a pre-article feature may use the 15:40 bar for an
   article available at 15:44, but never a bar ending after 15:44.  The verifier
   changes all future closes after the cutoff and requires identical context.
3. **Maturity:** a 240-minute reaction label is used for training only after its
   end time is before the evaluation article's availability.  A later article
   cannot train an earlier score.
4. **AR1 gate failure:** 240-minute AR1 improves June--August mean BA for both
   stocks and satisfies the Brier and positive-month checks, but its macro AUC
   delta is negative (`-0.00570`).  The full gate therefore fails; no W0--W3
   prediction is produced.  The 60-minute candidate also fails stock and macro
   conditions.
5. **AR2 unavailable:** no target-context FinBERT binary was available.  The
   run records `NOT_RUN_MODEL_UNAVAILABLE` and does not substitute cached title
   vectors.

The cases separate extraction/association quality, article reaction prediction,
and four-hour direction.  A correct association is not evidence that its
reaction is predictable, and a reaction-model score is not evidence of causal
stock movement.
