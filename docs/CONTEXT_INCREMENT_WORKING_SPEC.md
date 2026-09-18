# Context Increment + Fact Change Working Specification

Starting SHA: `27f7b477459c28e48aa3d3321ba874ebff8a36b1`. This lane preserves all
historical experiments and does not authorize real Mmeta/M1, news-direction, joint,
Activity v2, router, thread/newness, or market-state candidate scoring.

| # | Requirement | Code/artifact | Test/status |
|---|---|---|---|
| 1 | Publish permitted existing Activity evidence byte-identically | allowlist/inventory | pending inventory |
| 2 | Bounded SPY/QQQ access probe, distinguish credential vs provider failures | `market_intake.py` | pending |
| 3 | Audit every raw-news metadata field without outcomes | `audit_news_schema.py`, `audit_v1/` | pending |
| 4 | Build deterministic 64-pair, outcome-blind reading pilot | `build_news_pairs.py`, `run_pair_reader.py` | pending |
| 5 | Freeze M0/Mmeta/M1 and no-real-score boundary | `MARKET_PREREGISTRATION.md` | pending |
| 6 | Reproduce R1 in new private audit path | `baseline_adapter.py`, `audit_v1/` | pending |
| 7 | Synthetic separate-process train/save/verify/report and faults | `tests/test_synthetic_e2e.py` | pending |

Invariant: `thread`, `ord_in_thread`, and highlights have unresolved financial-event
semantics unless this lane proves otherwise. `SPY` and `QQQ` must be consolidated
5-minute raw-adjusted data or this market lane remains blocked. Postrun verification
must not fit, rebuild input, or overwrite a model.

## Final pre-predictive review issues

| ID | Severity | Finding | Required repair/status |
|---|---|---|---|
| CONTEXT-001 | BLOCKING | Existing 64-pair selection is AAPL-biased | rebuild 8×2×4 balanced manifest |
| CONTEXT-002 | HIGH | Numeric difference stratum overstated action semantics | neutral strata only |
| CONTEXT-003 | BLOCKING | Comparator labeled numeric difference as supported change | remove semantic labels |
| CONTEXT-004 | HIGH | Legacy title flags drove target association | reaction-v4 target contract required |
| CONTEXT-005 | BLOCKING | Market runner/verifier/reporter are stubs | implement before real approval |
| CONTEXT-006 | BLOCKING | R1 audit is probability-only | independent 1,374-row refit required |
| CONTEXT-007 | HIGH | Synthetic fixture bypasses production paths | full fixture/corruption suite required |
| CONTEXT-008 | HIGH | Qwen availability was not hash-audited | 13/13 exact files now verified |
| CONTEXT-009 | MEDIUM | Initial checklist was stale | statuses updated in this section |
