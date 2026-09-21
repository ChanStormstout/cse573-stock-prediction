# Grouped-news architecture: semantic brief

Audience: course instructor; figures must explain inputs, information organization, and prediction without unexplained internal method codes.

Content authority: `outputs/stock_paper_methods_4h/run.py`, its v1 report, and `outputs/stock_integrated_4h/run.py`. Style authority: the existing Simple Light Mode slides. The editable draw.io sources are authoritative; SVG/PDF/PNG and PowerPoint primitives are derived from their geometry and labels.

| Requirement | Evidence | Encoding |
|---|---|---|
| Frozen title encoder | N1M reuses FinBERT title vectors | Blue encoder labeled frozen |
| Rule-based grouping | Token Jaccard and action/number/quarter guards | Teal grouping module, no learned claim |
| Two-stage average | Mean within near-report groups, then across groups | Teal averaging module and separate worked example |
| Membership controls aggregation | Groups select which title vectors are averaged | Dashed group-ID dependency |
| Six metadata fields | Five log count/age fields and one repeat flag | Separate teal metadata branch |
| Historical prices | Six return/range pairs plus four summary/time fields | Separate 16-feature price branch |
| Prediction | Logistic regression; no-news R1 fallback | Trained classifier and explicit fallback note |

Every connector represents a data or membership dependency. No edge may cross a label or unrelated box. Metadata retains source/arrival records from grouped input; grouping does not invent source or first-disclosure information. Numeric dimensions are from code. The illustration has five similar target-price titles and one product title; this is not a measured case or forecast.

Excluded: full-body event extraction, generated JSON, attention, graph inference, trained encoder, and claims of optimal event weighting. No raw course text or model binary is included.

The overall figure abstracts feature scaling and fitting details into training notes. PCA and scaling fit past training data only. The separate module figure explains equal group weights, not equal economic importance.
