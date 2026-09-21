# Reference-style full-text method figure

The owner requested the earlier branching figure style, replacing the rejected alternative layouts. This version keeps the upper frozen encoder, lower grouping branch, separate text/report/price summaries and right-hand join/classifier/output. It adds explicit target-company passage preparation, keeps original titles and pools their vectors separately. Group IDs govern body pooling only. Old fixed16/6/16 dimensions are not transferred to the untested extension.

## Deliverables

- `method.drawio`: editable authoritative diagram.
- `method.svg`: vector preview.
- `method.pdf`: local PDF export.
- `Slide6_FullText_Reference_Final.pptx`: local one-slide PowerPoint with editable shapes/text/arrows.
- `preview.png`: local native-slide preview.

Google Slides was not opened or changed. This is a proposed full-text design, with no new experiment or score. Earlier artifacts are preserved.

## Reproduction and review

Run `make_figure.py` and `export_drawio.py` from repository root. `build_slide.mjs` uses the installed artifact-tool runtime and its finalizer; adapt workstation paths elsewhere and preserve versioned receipts. Custom primitive exporter generates PDF/SVG/geometry from draw.io; the PPTX uses the same geometry. No raster model pipeline or external icon asset is embedded. Source screenshot is layout/style authority only and is not redistributed.

Three canvas-only vector review cycles and two native-slide render checks are recorded in `defect-log.md`. Strict draw.io structure passes; static quality has zero FAIL with two reviewed warnings (branch spacing and diagram-only type hierarchy). Package, layout/font policy and first-party PPTX reimport pass. Self-score44/50 is internal design assessment, not independent review. Remaining approximations are font rendering and the added preparation column, as recorded. Raw project data and full source slide exports are not included.
