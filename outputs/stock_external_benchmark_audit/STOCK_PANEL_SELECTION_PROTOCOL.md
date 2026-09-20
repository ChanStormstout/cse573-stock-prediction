# Outcome-blind stock-panel protocol

Use only training-period metadata. Start from securities with stable ticker/issuer identity, valid daily prices, common calendar coverage and a sector mapping. Within each GICS sector, count days with at least one eligible pre-cutoff news record. Split coverage into within-sector terciles using training years only, then select two stocks per tercile by: longest common coverage, fewest missing prices, stable identity, then lexicographic permanent identifier. Returns and predictability are prohibited selection inputs.

Target 66 stocks: 11 sectors × 6. If infeasible, retain the largest deterministic balanced panel with at least five sectors and four stocks per sector. Publish every exclusion and reason.

Within each sector/coverage stratum, deterministically hash the permanent issuer ID and assign approximately 2/3 companies to training, 1/6 to development-unseen, and 1/6 to locked-unseen. An unseen company supplies no downstream direction labels to training; it is not claimed unseen to pretrained encoders.

No named stock is provisionally eligible yet because FNSPID's full per-stock timestamp/text coverage table has not been acquired. The current eligible universe is therefore `PENDING_METADATA_AUDIT`, not an invented list.
