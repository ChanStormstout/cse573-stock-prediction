# Reaction v2 case notes

1. **Time boundary repair:** every valid context row passed the completed-bar
   check; perturbing only future bars left the recomputed context unchanged.
2. **Article-level signal:** the 240-minute AR1 probe passed the predeclared
   outer monthly gate.  The 30, 60 and 120-minute probes did not.
3. **Window-level failure:** with the fixed F1 W0 protocol, the promoted
   240-minute W1 changed AAPL from 48.70% to 51.98% but changed AMZN from
   60.63% to 44.43%.  W3's strict two-article fallback recovered AMZN to
   55.95%, still below W0, and did not improve AAPL.
4. **Interpretation:** an article reaction can be predictable without being a
   reliable residual correction for the four-hour stock direction.  This is a
   downstream failure, not evidence that the time-safe reaction labels are
   invalid.
5. **AR2:** the target-context FinBERT binary was unavailable, so AR2 is
   explicitly `NOT_RUN_MODEL_BINARY_UNAVAILABLE`; existing article vectors
   were not substituted.
