# Layout Grid
## Canvas
- width: 2040
- height: 850
- scale assumption: diagram scaled uniformly to 1196 px on 1280 x 720 slide
- margin: 24 px diagram, 42 px slide
## Grid Lines
|name|x|y|purpose|
|source|24|270|raw news|
|prepare|334|270|new text preparation|
|split|674|100/415|encoder vs body grouping|
|summary|1024|85/415/670|text/report/price|
|join|1444|275|combine|
|predict|1744|280/620|classifier and output|
## Region Boxes
All exact major boxes are listed in visual-spec Shapes. Source/prepare have 70 px gutter; encoder/group 80 px gutter to summaries; join/model 70 px gutter.
## Repeated Components
Ten rounded modules, two semantic fills; consistent 2 px strokes and centered labels. One dashed group-membership route. No decorative matrices or symbols.
## Drawing Order
1. background
2. connectors
3. module fills and strokes
4. text
5. legend and brief fallback caption
