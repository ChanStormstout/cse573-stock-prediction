# Human reader gold protocol

Annotators receive target-company identity, cutoff-safe predecessor/current evidence, timestamps, and sentence IDs. They do not receive prices, returns, direction labels, model predictions, or later evidence.

For each pair label: target identity; actor and object; event/action; assertion status (asserted, denied, corrected, hypothetical/opinion, unknown); event period; numeric values/units and direction of change; evidence spans; predecessor comparability; novelty relation; and whether abstention is required. Every non-unknown fact needs a cited evidence span. Missing or conflicting evidence must be `unknown`, not guessed.

Near-duplicate/event families are isolated across train, development, and locked review. Two annotators label each locked item independently; disagreements are adjudicated by a third review with both rationales hidden until adjudication. Report per-field agreement, evidence support, coverage, and abstention correctness. Future acceptance thresholds are frozen before reader training: at least 90% target identity and evidence-span support, at least 85% actor/object and event/action, and at least 90% correct abstention on insufficient-evidence controls, each with counts and confidence intervals. Independent human review must not be claimed until performed.
