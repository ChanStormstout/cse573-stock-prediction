# Reaction v4 data audit

- All v4 reaction labels use same-session bars beginning no earlier than article availability.
- Pre-article return and realized-volatility values are `NaN` when their contiguous completed-bar context is unavailable; each has a separate `valid` flag.
- `AAPL` occurrences expanded as *American Association for Physician Leadership* are rejected only when no independent Apple evidence is present. The deterministic private review-card count is 5; this narrow rule does not resolve all entity ambiguity.
- Candidate-pair count: 89953; canonical group count: 85397; Jan--August feasibility gate: PASS.
