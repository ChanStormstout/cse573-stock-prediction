# Slide 6: company-focused full text (proposal)

Updated the existing Google Slides slide 6 in place on 2026-09-21. The live deck contains 16 slides. Text comparison of the before/after exports confirms that only slide 6 changed (whitespace normalized). Google Drive showed Saved to Drive. Other slides and author-entered results were not edited or independently validated.

## Main idea

Read company-relevant body passages as well as titles. Retain context, negation, numbers and evidence IDs. Group repeated passage content conservatively, retain unmatched details, and separate conflicting updates. Frozen FinBERT encodes the text. Average passage vectors within groups, then across groups; keep a separate title summary. Join those summaries with reporting/coverage metadata and completed historical prices in a regularized logistic-regression classifier.

The example is illustrative: report B repeats a target-price cut but contains additional guidance. Group the repeated passage, retain the extra passage. Dataset novelty is not market-first disclosure; a similarity group is not a validated financial event.

## Status boundary

This is a proposed extension. No new experiment, model training, predictive score or quality acceptance was produced. Previous headline-method results cannot be assigned to this full-text design. Missing reliable body retains titles; absent admitted news uses the price-only fallback. Precise extraction/grouping thresholds and dimensions remain to be preregistered before experiments.

## Editable deliverables

- `fulltext_method.drawio`: authoritative vector figure.
- `fulltext_method.svg`: portable vector export.
- `fulltext_method.json`: diagram geometry and text for native slide generation.
- `Slide6_FullText_Proposal_v2.pptx`: local one-slide PowerPoint, editable shapes/text/arrows.
- `slide6_preview.png` and `fulltext_method.pdf`: local previews.
- Live deck: https://docs.google.com/presentation/d/1_xlkQI09kQmm2wSjHOUtYJDAflbOQPsv1dGqrKzwT5w/edit

Binary exports and the private before/after whole-deck backups are not committed. No source news is redistributed.

## Build and review

Run `make_figure.py`, then `export_drawio.py` from repository root. The custom exporter supports this figure's explicit rectangle/text/polyline primitives; it is not a general draw.io renderer. PDF font export uses installed macOS Arial. SVG/PPTX follow the deck's Helvetica Neue. `build_slide.mjs` uses the installed artifact-tool runtime and private `work/presentation_meeting/fulltext_v1/before_edit.pptx` reference. Workstation paths must be adapted elsewhere; preserve versioned finalizer receipts rather than overwriting them.

Draw.io strict/static checks pass. One-slide PowerPoint package, geometry/font-policy and artifact-tool reimport checks pass. Rendered slide and saved live Google Slides were visually inspected. See `defect-log.md`. No model verifier was run.
