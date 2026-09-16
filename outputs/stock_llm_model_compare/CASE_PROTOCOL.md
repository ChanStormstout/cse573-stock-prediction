# Fixed review selection

Select before reading the new performance table: compare old frozen→9B original,
9B original→revised prompt, revised prompt→staged. For each stock in the exposed
April panel, select one case from each exact-fact outcome (fixed, regressed,
both wrong, both correct), using smallest SHA256(case ID + comparison name).
Absent cells are reported, never filled with a handpicked success. At most24
comparison slots; repeated article IDs are read once with all model outputs.

Additionally select up to5 staged rule-rejection cases by SHA256(ID), and up to5
positive-reference articles lost at the gate. Show raw source and predictions
locally, publish paraphrased findings only. This panel diagnoses components and
cannot estimate the prevalence or causal importance of each error type.
