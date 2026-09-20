# Source-grounded review of the fixed case panel

The panel was specified before new fits: seed 573, stable-hash selection from
repaired/harmed/both-wrong/both-right categories, plus fixed input-type
supplements. The eight reviews below use the first prespecified entry for each
listed category, not a handpicked confidence extreme. Full source sentences
and article keys are in the private `case_evidence.json`; public descriptions
are paraphrases. These are assistant reviews, not independent extraction gold.

| Stock / method / category | Canonical row ID | FULL → new P(up); actual | What the source contains | What the case supports |
|---|---|---|---|---|
| AAPL / A3 / harmed | 2bd11b8dcffcc1d5ab8e7192 | .4050 → .5086; down | A sector report covers several electronics companies; another item combines a fund's Apple holding reduction with short-interest information for another company. | Restored numeric and repeated text includes non-Apple facts and holdings reports. The new model crosses the threshold incorrectly. This alone does not establish which token caused it. |
| AAPL / A3 / repaired | 0d88105c65087db3546f7946 | .3042 → .5047; up | Supplier shares falling alongside iPhone production concerns, and a separate Apple analyst downgrade with an unchanged price target. | News tone and the next four-hour direction can differ. Correcting this window does not show that a positive-sentiment rule was learned. |
| AAPL / B2 / harmed | 7cc4d1072d0c01ad679316de | .4400 → .9009; down | An investment manager's new Apple position and an analyst reiterating a buy rating. | A semantic branch can turn a modest base probability into a confidently wrong one. A reiterated rating is not automatically a new upgrade. |
| AAPL / B1 / repaired | f4e4d66e4fbb890d889e8779 | .5224 → .3648; down | An analyst reiteration with favorable product-positioning language, plus reports about possible headphones and a laptop update. | A correct downward change coexists with favorable source language; market direction cannot be read directly from adjective polarity. |
| AMZN / A3 / harmed | 371495a701fe5f2e208380dc | .6140 → .4788; up | Broad analyst roundups cover many firms and include interest-rate commentary; one opening item concerns Apple rather than Amazon. | Full raw text contains mixed objects and macro background. Adding local bigrams does not explicitly resolve which action belongs to Amazon. |
| AMZN / A3 / repaired | 605f7494c387b23c6a44f828 | .1400 → .5785; up | A sector-price roundup alongside a report that Amazon is developing a chip for AI tasks. | A direct company development is present and this prediction was repaired; one example cannot demonstrate repeatable event usefulness. |
| AMZN / B2 / harmed | 727b14e198f57e0a3f38ce0f | .5255 → .3866; up | A combined holdings item discusses one manager reducing Boeing and another increasing Amazon. | Both actions and company objects appear in the same headline/body. Target-name presence is weaker than validated event-object extraction. |
| AMZN / B1 / repaired | e915cfe8fa27a83ed78ea3ab | .3854 → .7851; up | The same broad analyst-roundup article family as another selected September 27 case. | A shared article can occur in distinct price windows. Article identity and subsequent window labels must not be treated as independent event examples. |

## What remains unknown

No deletion, word-weight intervention or semantic-object ablation was performed
here. Therefore the table does not attribute each failure to one causal feature.
It distinguishes observed source content and prediction flips from hypotheses
about model behavior. Negation, numeric, multi-company, near-repost and no-news
supplements are enumerated in `CASE_PANEL_COVERAGE.json` and `case_index.csv`.
Every no-news case has exact PRICE fallback; every B case without target evidence
has exact FULL fallback. Missing categories, if any, are recorded with zero
available examples rather than filled with invented cases.
