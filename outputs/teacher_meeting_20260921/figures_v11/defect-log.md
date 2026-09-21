# Figure review record

No predictive experiment was run. These checks concern figure geometry and rendering only.

## Static preflight

Both final draw.io files pass strict structural validation with zero errors or warnings. Final visual-quality scans contain zero FAIL items. Reviewed warnings: the overview has deliberately uneven horizontal gaps because branches occupy different vertical bands. The example's title A and note height estimates are conservative: both visibly fit. Its hierarchy/density warnings were reviewed; this teaching example intentionally uses five large meaningful boxes rather than decorative content.

A pre-render membership-arrow/text collision was corrected before the first preview.

## Three export-review cycles

| Canvas zone | Cycle 1: SVG export | Cycle 2: revised SVG | Cycle 3: PDF at 1,050 px and slide-native figure |
|---|---|---|---|
| Top left | Title and encoder hierarchy clear; encoder subtitle too wide (P1) | Shortened subtitle fits | Readable; frozen status visible |
| Top | Pool label too wide (P1) | Shorter three-line wording fits | Native slide wraps to four lines inside box; no clipping |
| Top right | Trained classifier/output relationship clear | Same, no defect | Readable with explicit arrow |
| Left | Input source/arrival label exceeded box (P1) | Three short lines fit | No clipping |
| Center | Grouping subtitle exceeded box (P1); group-ID arrow clear | Shortened labels fit; no crossing | Solid data vs dashed membership visible |
| Right | Join and classifier fit | No defect | Prediction path readable |
| Bottom left | Legend and fallback readable | No defect | Readable; long fallback line remains inside canvas |
| Bottom | Price branch fit; example count annotation clear | No defect | Readable; no label/arrow collision |
| Bottom right | Example product text too close to box edge (P2) | Shortened to Apple product launch | Example weights and probability disclaimer clear |

All cycle-1 P1 issues were corrected before slide delivery. Cycle 2 had no P0/P1 issues. Cycle 3 confirmed geometry at reduced size and in native slide rendering. No missing required component or connector was found. Self-review assessment: semantics 5/5, readability 4/5, editability 5/5; this is not external peer review.

## Export limitations

SVG/PNG and PowerPoint use Helvetica Neue. PDF uses embedded Arial for portable Unicode arrows; the geometry and content are identical but glyph metrics differ slightly. PowerPoint itself was not opened; artifact-tool re-import, native-vector render, and package/layout checks passed. The overview abstracts scaling and training detail into accompanying slide notes. The grouping remains heuristic, not independently validated event identity.
