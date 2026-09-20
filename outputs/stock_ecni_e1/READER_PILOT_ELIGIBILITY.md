# Reader-pilot eligibility

## Decision

A reader pilot is not yet authorized. The proposed 600 train/development plus 300 locked human-review examples is not supported by qualified evidence at E1.

The sampled Nasdaq subset contains substantial bodies, so the categories `repetition`, `numeric update`, `new event`, `new period`, `action change`, `denial/correction`, `different actor`, `different target company`, `hypothetical/opinion`, and `insufficient evidence` are textually plausible. However, target association has not passed quality review, EDT access/terms are unresolved, and the SEC pilot could not be selected without a frozen panel.

Recommended next evidence gate: acquire a metadata-complete index and freeze the panel; then construct 120 outcome-blind candidate pairs across sectors and time, double-label 60, and estimate category prevalence and agreement. Expand toward 600/300 only when at least 30 independently reviewed examples exist for each category used in acceptance metrics. These counts are evidence-planning values, not a model-performance threshold.
