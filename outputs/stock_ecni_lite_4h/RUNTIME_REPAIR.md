# Pre-result exact-fallback repair

The first isolated execution stopped during the first method before any public
prediction or metric file was written. The mathematical round trip
`sigmoid(logit(base))` can differ from `base` by floating-point rounding, so the
bit-for-bit fallback assertion stopped the run.

The bounded repair explicitly copies the baseline probability on rows where
all eligible evidence blocks are inactive. This enforces the preregistered
fallback and does not alter active-row features, fitting, selection or scoring.
The incomplete temporary directories are preserved outside the repository.

After the completed run, the first fit-free verification pass incorrectly
counted warm-up rows, whose baseline probability is intentionally missing, as
fallback mismatches because `NaN != NaN`. It nevertheless replayed all saved
evaluated probabilities with maximum error zero. The verifier-only repair
restricted the fallback contract to the 1,374 evaluated rows. No model was
refit and no prediction or metric changed. The initial verifier output is
preserved as `VERIFICATION_pre_scope_fix.json`; the corrected verifier passed.
