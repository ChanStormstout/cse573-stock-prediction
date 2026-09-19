# V8 canonical-lineage correction

This addendum is committed before V8 predictions. The canonical F1 lineage is:
`stock_baseline.build()` creates the target-stock candidate universe using its
title association, filters and deduplication; `stock_horizons.dataset()` writes
those selected keys as `article_keys`; `stock_integrated_4h.prepare()` assigns
`news_record_keys = article_keys`, reads raw full article bodies and creates
`stem_body`.

Therefore:

`CANONICAL_F1_ASSOCIATION = legacy target-title candidate association + canonical filters/dedup`.

`CANONICAL_F1_TEXT = full raw article bodies of associated articles transformed by canonical prepare.py cleaning/stemming`.

Association and text representation are separate. V7 remains a valid historical
stop under its earlier protocol, but its execution blocker is superseded by
this confirmed lineage. V8 changes neither the method matrix nor grids: it only
uses the matched association/full-text reconstruction and V7's fixed DPRICE
formula, evaluates both daily windows separately, and preserves V6/V7.
