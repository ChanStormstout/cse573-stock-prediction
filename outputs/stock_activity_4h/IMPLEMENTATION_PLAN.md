# Activity v1 implementation plan

1. Audit raw `activity` without inspecting four-hour outcomes.
2. Build completed-bar 15/60-minute current and history-only relative fields in
   `work/stock-data/activity_4h/`; keep row-level evidence private.
3. Reproduce canonical R1 as A0 and require probability error at most `1e-12`.
4. Run the independent preflight verifier before any A1 execution.
5. Only a later explicit `APPROVE_ACTIVITY_RUN` may write v1 candidate results.

The runner contains a post-run verifier/report path, but they are deliberately
not invoked at this checkpoint.
