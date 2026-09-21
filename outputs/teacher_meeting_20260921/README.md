# Current delivery: grouped-news method, v10

The current local presentation is `CSE573_Grouped_News_Faculty_Meeting_v10.pptx`: 18 editable English slides (15 main, 3 appendix), using the requested Simple Light Mode template. It replaces the proposed factual-change correction as the main story with the **implemented frozen FinBERT + near-report grouping + six reporting metadata fields + historical prices + small classifier**.

- [Current presenter notes and sources](PRESENTER_NOTES_v10.md)
- [Plain-language method explanation](METHOD_EXPLANATION_v10.md)
- [Delivery manifest](DELIVERY_MANIFEST_v10.json)
- [Versioned authoring source](build_slides_v10.mjs)

Slides 4–9 explain a six-report example, grouping guards, two-level averaging, metadata, and the final predictor. Slide 10 explains actual training. Slides 11–13 show matched historical comparisons and attribution limits. FinBERT and balanced accuracy are defined. Color distinguishes existing components from the changed representation; it does not imply unsupported hierarchy.

All results are saved chronological, exposed exploratory backtests. No new training or independent predictive verification was performed. Illustrative weights are labeled as examples, not measured forecasts. Grouping is heuristic title similarity, not validated financial-event identity. Existing v9 and earlier presentations remain unchanged. Presentation binaries remain local under repository policy.

---

# Historical v9 documentation (superseded presentation narrative)

# CSE 573 faculty meeting — 21 September 2026

## Presentation

18 English slides: 15 main slides for a 10–12 minute discussion and three appendix slides. The deck uses the supplied Simple Light Mode template. The native tables and method diagrams remain editable.

The final local file is `CSE573_News_Context_Faculty_Meeting_v9.pptx`. Presentation binaries remain outside Git under the existing repository policy. [Presenter notes](PRESENTER_NOTES.md) include slide-specific sources and explanations. [Method proposal](METHOD_PROPOSAL.md) records the design boundary.

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

## Readability revision

The updated deck retains the light template, with navy headings, teal emphasis on proposed information increments, pale grouping backgrounds and stronger typographic hierarchy. Diagrams distinguish the reference path from the proposed correction. Numeric evidence and method status are unchanged. Earlier PowerPoint versions remain local.

## Logic-led visual revision

Removed arbitrary one-sided backgrounds from parallel text comparisons. Three completed-experiment observations now precede a separate full-width hypothesis, explicitly labeled as a question rather than a proven cause. Color inside diagrams distinguishes existing components and proposed changes. Short definitions accompany financial language features, BA and extraction fact F1. A new evidence page records matched provisional extraction results and the observed N0415 maintain-rating/raise-target case. Extraction metrics are never presented as direction accuracy. The stock-results page keeps all four cells and gives later-period relative changes with an explicit mixed-development qualification. No scores were invented or hidden; no prediction experiment was run.

## Evaluation target page

Slide 14 states the user's aspiration of BA at least 60% for both companies. It is explicitly not a performance forecast or measured result. Actual new-pipeline results are NOT RUN. This does not change a protocol, promotion gate or run authorization.
