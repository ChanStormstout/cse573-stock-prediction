# Diagram Brief
## User Goal
One standalone editable slide matching the supplied earlier branching framework, adapted to current proposed full-text method. Audience: faculty. No Google Slides mutation.
## Source Inventory
|id|source|type|role|priority|notes|
|R|user screenshot 8d612b28|image|style/layout|high|blue/teal, upper encoder/lower grouping, shared combine|
|M|current fulltext_figure_v1 brief and discussion|method|content|highest|company passages plus separate original titles, extra detail retained|
## Requirement Traceability
|id|requirement|source evidence|must/should/may|planned visual encoding|
|1|reference branching shape|R|must|encoder above grouping, pool above report features|
|2|full bodies used|M|must|prepare-text node selects company passages and retains titles|
|3|separate title/body summaries|M|must|pooling node names each separately|
|4|no invented dimensionality|M|must|remove old 16/6/16 and PCA dimension|
|5|report metadata retained|M|must|grouped records feed counts/sources/ages/coverage|
|6|proposal status|discussion|must|title says proposed; no results|
## Semantic Model
Prepared records retain original title and source/time metadata. Encode titles and company passages separately with frozen FinBERT. Group only body passages, with compatible numeric/action/period cues and unmatched details retained. Group IDs govern body pooling only. Average title vectors separately. Combine the two text summaries with reporting and historical-price features in logistic regression. No attention/LLM or verified event understanding is claimed.
## Style Contract
Use reference blue/teal; bold Helvetica Neue; orthogonal gray arrows; dashed group membership. Six columns instead of reference five because company-passage preparation needs an explicit stage. No icons: the supplied reference has none.
## Open Assumptions
Exact grouping thresholds and reduction dimensions await preregistration. The figure abstracts train-only scaling/compression; fixed dimensions are deliberately not invented. Group outputs retain report provenance for counts and ages.
