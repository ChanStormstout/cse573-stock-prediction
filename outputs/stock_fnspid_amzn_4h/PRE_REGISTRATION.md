# FNSPID Amazon candidate-coverage lane (v1)

## Purpose

Test whether the existing FNSPID corpus can supply additional Amazon-specific
historical-news candidates to the canonical 804 AMZN four-hour windows. This is
an **outcome-blind coverage and quality experiment**, not a stock-prediction
experiment.

## Frozen inputs

- Amazon issuer: SEC CIK `1018724`, resolved ticker `AMZN`.
- Relinked FNSPID edges and complete metadata from ECNI E1R.
- Canonical four-hour window keys and cutoffs from the existing course task.
- XNYS sessions generated with the installed `exchange_calendars` data and
  checked exactly against the accepted repository schedule over their overlap.

No label, return, saved prediction, BA, MCC or Brier value may enter this lane.

## Conservative availability rule

The relevant 2018--2019 FNSPID rows are date-only. A record dated on calendar
day `d` becomes available at the **regular XNYS open of the first session whose
session date is strictly later than `d`**. It cannot be used at an earlier
cutoff. This rule deliberately prevents same-day intraday use.

## Relation strata

Candidates remain separated as:

1. `DIRECT_TARGET_HIGH_CONFIDENCE`;
2. `MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE`;
3. `INDIRECT_OR_COMPETITOR` (diagnostic only).

These are mechanical relinking claims, not human gold labels. Direct and
multi-company rows remain provisional until independent semantic review.

## Deduplication

Rows use a stable article group: nonempty URL hash, otherwise normalized-title
hash, otherwise near-title fingerprint, otherwise record hash. If a group has
multiple relation claims, the public group classification is conservative:
multi-company, then direct, then indirect. Repeated rows do not become multiple
independent events.

## Fixed coverage views

- previous 24 clock hours;
- current and previous two XNYS sessions, always subject to availability time.

For each view report direct-only, multi-company and diagnostic-indirect counts,
plus the number of canonical no-course-news windows that gain at least one
candidate. Coverage is not predictive improvement.

## Frozen review sample

Select 120 unique article groups by deterministic hash, balancing direct versus
multi-company, year, source, matched field and body availability where feasible.
The public manifest contains hashes and strata only. Private source text may be
attached later for blinded review. No stock outcome is used for sampling.

## Stop rule

Do not train a direction model from this lane until an independent review has
established acceptable target-relation precision and the downstream experiment
has a separately frozen protocol. Failure to add credible candidate coverage is
a valid stopping result.

