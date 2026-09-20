# CSE 573 faculty meeting — 21 September 2026

## Presentation

16 English slides: 13 main slides for a 10–12 minute discussion and three appendix slides. The deck uses the supplied Simple Light Mode template. The native tables and method diagrams remain editable.

The final local file is `CSE573_News_Context_Faculty_Meeting_v4.pptx`. Presentation binaries remain outside Git under the existing repository policy. [Presenter notes](PRESENTER_NOTES.md) include slide-specific sources and explanations. [Method proposal](METHOD_PROPOSAL.md) records the design boundary.

## Narrative

1. Can news add predictive information beyond recent prices?
2. Establish the price + FinBERT reference in plain language.
3. Show historical gains and their lack of stability across companies/periods.
4. Distinguish a repeated report from a changed company fact.
5. Link company, actor, action, values, time and source evidence.
6. Propose a bounded adjustment to the unchanged reference forecast.
7. Test history metadata, current facts and relative fact changes separately.

The course connection is classical ML plus modern language representations and entity/relation mining. It is grounded in the supplied CSE 573 group-project deck and the assigned news-prediction paper, not assumptions about the instructor's private preferences. Internal method codes are confined to source notes where needed. The supplied Module1 IntroductoryLecture PDF was identified as CSE 464 software-quality material and was not used as CSE 573 requirements.

## Evidence boundary

- Reported scores are copied from saved canonical historical results at source checkpoint `652b1999407a2e248064d61e8758cd68050dcf89`.
- All existing evaluation periods are exposed exploratory historical backtests.
- The context-adjustment pipeline is proposed and untested. It is not a demonstrated improvement over price + FinBERT.
- The target-price example is explicitly illustrative. Its values are not dataset observations.
- No new training, inference, labels, acquisition, or experimental protocol changes occurred.
- No fabricated scores or blanket claim of leakage in the instructor's paper is presented.
- Independent entity review remains pending. This presentation does not authorize a paused research lane.

## Build

`build_slides.mjs` is the exact local authoring source, using the installed Presentations artifact-tool runtime and the requested template. Its absolute paths identify this workstation's dependencies; rebuilding elsewhere requires replacing those paths. Outputs and finalizer receipts use versioned names and should not overwrite an existing delivered deck. The template asset is not redistributed. Validation receipts and rendered previews stay under `work/presentation_meeting/build/`.
