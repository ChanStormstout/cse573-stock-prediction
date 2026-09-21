# Defect Log
## Pass 0 - Initial Plan Review
|issue|reference evidence|planned fix|
|old method dimensions|16/6/16 in screenshot|omit because proposal dimensions not fixed|
|old title-only input|Available titles|add full bodies and company-passage preparation|
|group identity ambiguity|fact checks|use compatible fact cues, not verified truth|
## Screenshot Review

## Screenshot Evidence
|pass|screenshot path|capture type|full canvas visible|crop/viewport notes|
## Initial semantic audit placeholder (superseded)
Completed in the final audits below.
## Initial red-team placeholder (superseded)
Completed in the final audits below.
## Remaining Gaps
Reference layout intentionally expands by one preparation stage.

Static preflight: zero FAIL, two reviewed warnings. The 840 px horizontal gap at source-row height contains the two vertically split processing stages, so forcing uniform same-y spacing would destroy the reference topology. Typography warning applies to diagram modules only: the larger slide heading is composed outside the draw.io canvas. Module/annotation weights and scale intentionally follow the supplied reference.

## Cycle 1 — canvas-only pass1.png
Nine-zone inventory (observations, not invented defects):
1. TL source fill matches reference blue.
2. TL source text is inside bounds.
3. TL source arrow reaches preparation.
4. TL preparation company label touches/crosses border: P1, reduce label size.
5. T encoder sits above grouping as required.
6. T encoder arrow is forward.
7. T frozen status visible.
8. T encoder second line is close to edge: P2, preserve while checking native export.
9. TR body pooling first line is dense: P2, shorten average to mean.
10. TR separate title average is stated.
11. TR group-ID route enters below vector route.
12. TR pooling output reaches combine.
13. L preparation fans into both branches.
14. L no hidden title deletion implied.
15. L preparation stage adds a column versus reference, intentionally.
16. C group title exceeds its boundary: P1, shorten to Group passages.
17. C similarity line touches boundary: P1, shorten to Text + fact cues.
18. C extra-details line fits.
19. C dashed membership line does not cross encoder-vector arrow.
20. C group-ID text is outside routing path.
21. R report label is close to side edges: P2, shorten separators and reduce 1 px.
22. R three modalities enter combine at distinct ports.
23. R classifier text fits.
24. R model-output arrow points down.
25. BL legend is legible and left aligned.
26. BL solid/dashed definitions are explicit.
27. B price node is distinct from metadata.
28. B price route enters combine, not the language encoder.
29. BR probability output is four-hour and has no invented value.
30. BR fallback caption stays on canvas.
Patches: preparation font 29→26; grouping labels shortened; pooled-text labels shortened; reporting text uses compact separators. No semantic edge changes.
|1|work/presentation_meeting/reference_fulltext_v4/pass1.png|canvas-only|yes|1600 px whole-diagram render|

## Cycle 2 — canvas-only pass2.png
1. TL source labels fit; source/preparation edge clear.
2. TL company passage label now fits: prior P1 fixed.
3. T encoder two lines remain close but inside; check native font wrapping.
4. T encoder→pool arrow unobstructed.
5. TR body two-level mean label fits.
6. TR title averaging is explicitly independent.
7. L fork topology matches reference upper/lower split.
8. L preparation label reads as processing, not raw data.
9. C group title fits: prior P1 fixed.
10. C fact-cue label fits: prior P1 fixed.
11. C dashed membership routes only into body pooling.
12. R report/count/coverage labels now have visible side margins.
13. R merge ports are distinct and classifier output points down.
14. BL legend reflects both colors and both line styles.
15. B completed-price node avoids volume/tick assumptions.
16. BR probability output and missing-news fallback are visible.
P2 refinement for native export: shorten within/across phrase, reduce source and encoder fonts 30→28 to reserve native text insets. No scientific change.
|2|work/presentation_meeting/reference_fulltext_v4/pass2.png|canvas-only|yes|1600 px whole-diagram render|

## Cycle 3 — canvas-only pass3.png at slide width
1. TL source readable at1196px; no clipping.
2. T encoder text readable; shorter wording keeps fit.
3. TR body/title averaging remains separate.
4. L prepare module is aligned with input and reference branch split.
5. C group membership dash remains visible at reduced scale.
6. R merge/classifier arrows remain readable.
7. BL legend readable but secondary in hierarchy.
8. B past-price arrow avoids metadata block.
9. BR output and fallback are visible.
Native-slide rendering then revealed extra line wraps in source time, target-company selection, encoder description, extra detail, and reporting ages. These are P2 readability defects without clipping. Shortened to Source / time, Target passages, Frozen encoder, Keep extra text, and two compact metadata lines. Semantics unchanged; FinBERT definition stays in concise notes and the existing deck glossary.
|3|work/presentation_meeting/reference_fulltext_v4/pass3.png|canvas-only|yes|1196 px whole-diagram render|
|4|work/presentation_meeting/reference_fulltext_v4/render/slide-1.png|canvas-only|yes|native slide, before wrap repairs|

## Final Screenshot Review — native slide
All five wrapping defects from the prior native slide are fixed. Complete nine-zone check: TL source fits; T frozen encoder fits; TR body/title pooling has three intended lines; L target passages and titles fit; C grouping/cues/extra text fit; R reporting/merge/classifier fit; BL legend legible; B price node/caption readable; BR probability arrow and label clear. No P0/P1 remaining.
|5|work/presentation_meeting/reference_fulltext_v4/render_final/slide-1.png|canvas-only|yes|1280 x720 final slide|

## Requirement And Semantic Audit
|check|observed screenshot|expected from reference|actual|status|
|branch shape|final slide|encoder above grouping|same, with prep stage inserted|PASS|
|full text|final slide|adapt old input|bodies plus target passages retained|PASS|
|title preservation|final slide|new content requirement|original titles and separate average|PASS|
|membership|final slide|dashed control|body group IDs into pooling|PASS|
|report information|final slide|parallel metadata|count/source/age/coverage pathway|PASS|
|prices|final slide|direct join|completed price bars join separately|PASS|
|dimensions|final slide|old values not transferable|no unsupported fixed dimensions|PASS|
|output|final slide|up probability|four-hour up probability|PASS|

## Red-Team Visual Audit
These are checked observations, not a claim that thirty defects were found. Actual defects and repairs are recorded in the cycle history above.
|zone|check|status|
|---|---|---|
|TL|Input lists both title and body|PASS|
|TL|Time/source provenance preserved|PASS|
|TL|Input to preparation points right|PASS|
|T|Prepared original titles retained|PASS|
|T|Company-target passage preparation explicit|PASS|
|T|Frozen encoder feeds both text summaries|PASS|
|TR|Body two-level pooling stated|PASS|
|TR|Separate title average stated|PASS|
|TR|No inherited PCA16 claim|PASS|
|L|Upper encoding branch source visible|PASS|
|L|Lower grouping branch source visible|PASS|
|L|No accidental filtering of no-news windows implied|PASS|
|C|Body group IDs use dashed line|PASS|
|C|Membership route avoids data-vector line|PASS|
|C|Extra unmatched text retained|PASS|
|C|Fact cues do not claim truth verification|PASS|
|R|Grouped records feed report statistics|PASS|
|R|Coverage separate from sentiment|PASS|
|R|Text/report/price join once|PASS|
|R|Classifier output points downward|PASS|
|BL|Legend explains color roles|PASS|
|BL|Legend explains line styles|PASS|
|BL|No decorative visual elements|PASS|
|B|Price node says completed bars|PASS|
|B|No assumed volume field|PASS|
|B|Price connector enters join|PASS|
|BR|Output is four-hour probability|PASS|
|BR|No fake probability number|PASS|
|BR|No-news fallback caption legible|PASS|
|Whole|Native wrapping repaired without omissions|PASS|

## Self-score
Internal design assessment, not external review: semantic accuracy9/10, reference-layout fidelity8/10 (added preparation stage), readability9/10 (six columns require smaller labels), typography9/10, editability/export9/10 (custom renderer, not diagrams.net native screenshot). Total44/50. No dimension below6.

|Dimension|Score|
|---|---|
|Semantic accuracy|9/10|
|Reference layout fidelity|8/10|
|Readability|9/10|
|Typography|9/10|
|Editability and export|9/10|
|TOTAL|44/50|

## Final Remaining Gaps
The diagram adds one necessary preparation stage relative to the reference. Border colors and font are measured/visual approximations; fill and ink colors sampled from supplied screenshot. No exact pixel-match claim. Diagram PDF uses Arial while native PPTX/SVG use Helvetica Neue. Rendered readability checked in both; native PowerPoint application itself was not launched.
