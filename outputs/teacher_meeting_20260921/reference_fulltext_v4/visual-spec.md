# Visual Spec
## Source
- Reference image: user clipboard 8d612b28, 2646 x 1288.
- Target drawio: method.drawio
- Canvas: 2040 x 850, diagram only; slide adds heading.
- Font policy: Helvetica Neue consistent with preceding deck.
## Global Style
- Background: #FFFFFF.
- Primary font: Helvetica Neue, bold 30-34 px modules, 26 px annotations; slide title 34 px.
- Stroke style: 2 px boxes, 2.5 px arrows; subtle rounded corners 8 px; no shadows.
- Arrow style: filled block heads, orthogonal; dashed group IDs (8/6).
- Color palette: sampled dominant blue #EEF3F8, teal #E9F4F1, ink #1E314B, teal ink #2E6C72. Border #9CB5C6 and arrows #607181 approximate anti-aliased screenshot strokes.
## Regions
|id|bbox x,y,w,h|role|visual notes|
|inputs|24,270,580,165|raw news + preparation|new explicit selection stage|
|split|674,100,270,475|encoder/group branches|matches reference hierarchy|
|summaries|1024,85,340,685|pool/report/prices|stacked like reference|
|prediction|1444,275,550,445|combine/classify/output|single fan-in|
## Text Blocks
|id|bbox x,y,w,h|text|font|alignment|priority|
|legend|24,640,900,110|Blue standard; teal proposed; solid data; dashed membership|26|left|secondary|
|footer|24,804,1970,38|Missing body/title fallback and no-news price fallback|26|left|secondary|
## Shapes
|id|bbox x,y,w,h|type|fill|stroke|notes|
|source|24,270,240,165|rounded rect|blue|blue border|raw inputs|
|prepare|334,270,270,165|rounded rect|teal|teal border|target passages plus original titles|
|encoder|674,100,270,140|rounded rect|blue|blue border|frozen|
|groups|674,415,270,160|rounded rect|teal|teal border|body passages only|
|pool|1024,85,340,175|rounded rect|teal|teal border|separate body/title summary|
|report|1024,415,340,150|rounded rect|teal|teal border|metadata|
|price|1024,670,340,100|rounded rect|blue|blue border|past complete bars|
|combine|1444,275,230,225|rounded rect|blue|blue border|join inputs|
|model|1744,280,250,170|rounded rect|blue|blue border|trained LR|
|out|1744,620,250,100|rounded rect|blue|blue border|4h probability|
## Connectors
|id|from|to|route|arrowheads|label|notes|
|raw|source|prepare|straight|target|none|news records|
|encode|prepare|encoder|up/right|target|none|titles and company passages|
|group|prepare|groups|down/right|target|none|body passages and provenance|
|vectors|encoder|pool|straight|target|none|separate text vectors|
|ids|groups|pool|right/up/right|target|group IDs|dashed; body only|
|report|groups|report|straight|target|none|records and memberships|
|text|pool|combine|right/down/right|target|none|two summaries|
|meta|report|combine|right/up/right|target|none|report features|
|prices|price|combine|right/up/right|target|none|price features|
|predict|combine|model|straight|target|none|joined features|
|prob|model|out|down|target|none|probability|
## Semantic Relations And Flow
Directed acyclic data path. Preparation fans out to encoder and passage grouping. Pooling gets vectors and group assignments; grouping feeds report statistics. Text/report/price streams join once.
## Icons And Images
None in reference or output. All semantic primitives remain editable.
