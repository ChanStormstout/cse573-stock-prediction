# Simplified full-text method slide

Standalone one-slide PowerPoint delivered locally as `Slide6_FullText_Method.pptx`. Google Slides was not opened or modified in this task. This supersedes the visual presentation of fulltext_figure_v1, not any experimental result.

Three numbered columns establish left-to-right reading. An illustrative pair of Apple reports carries the explanation: blue passages repeat a target-price cut, a teal passage adds guidance. Group repeated content while preserving the additional passage, encode with frozen FinBERT, average within/across groups, and combine body/title summaries, reporting metadata and completed historical prices in logistic regression. The figure is a proposed extension, not a tested result. Missing-body/news behavior remains in concise speaker notes.

## Design

Scientific-figure-making was consulted for readable Helvetica-style text, restrained semantic color and vector export. Matplotlib was not used because this is an editable architecture illustration rather than a data plot. Icons come from Lucide (ISC license), rather than copied Flaticon assets. Document/layers/chart/scan icons represent article, group, price and text encoder roles. The shapes, labels and arrows are native editable objects; icons are embedded vector SVGs.

Two native renders were inspected: first review shortened jargon (body/title representation to summary), shortened the conclusion, and clarified the column titles. Final 1280 x 720 render has no observed clipping/overlap; the three-column reading order and icon/color roles remain clear. PPTX structural, layout/font-policy, one-slide count and artifact-tool import checks passed. These are presentation checks only.

## Rebuild

`build_slide.mjs` uses the installed artifact-tool runtime and workstation paths; adjust those paths on another machine. Preserve finalizer output/receipt versioning. Native binary and preview stay local; the source, icons and license are committed. Asset sources: https://github.com/lucide-icons/lucide/tree/main/icons and https://lucide.dev. No raw course/news text or predictive data is included; example wording is authored.
