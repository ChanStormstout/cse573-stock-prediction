# FNSPID AMZN four-hour handoff

## Current state

The outcome-blind candidate-coverage stage is complete and verified in
`outputs/stock_fnspid_amzn_4h/v3`.

- 804 canonical AMZN windows mapped.
- 389 original no-news windows retained.
- Date-only FNSPID rows become available at the next XNYS regular open.
- Direct, multi-company and indirect relations remain separate.
- 120 private blinded review cards are ready; no labels or independent review
  have been supplied.
- No stock outcome was used and no predictive model was fitted.

## Main finding

Direct-only FNSPID candidates cover 176/389 original gaps in the 24-hour view
and 314/389 in the three-session view. Direct plus multi-company candidates
cover more, but their density is high enough that they remain blocked pending
semantic review.

## Next permitted work

Independent entity-relation review of the frozen 120 cards. A predictive
experiment requires a new protocol after review and must keep exact price
fallback for windows without validated context.

## Entry points

- [Protocol](../outputs/stock_fnspid_amzn_4h/PRE_REGISTRATION.md)
- [Report](../outputs/stock_fnspid_amzn_4h/v3/REPORT.md)
- [Coverage summary](../outputs/stock_fnspid_amzn_4h/v3/COVERAGE_SUMMARY.json)
- [Verification](../outputs/stock_fnspid_amzn_4h/v3/VERIFICATION.json)

