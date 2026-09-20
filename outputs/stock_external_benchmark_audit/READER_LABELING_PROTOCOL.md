# Reader labeling protocol

The reader receives target company, current evidence with sentence IDs, and optionally an earlier compatible candidate. It returns the ECNI statement schema, evidence IDs, grounded numbers and `unknown`/abstention. It never receives future returns and never generates causal market explanations.

Use a frozen open 3B--4B model if available, with LoRA/QLoRA only after a zero/few-shot baseline. FinBERT is the established compact financial encoder/control; Fin-ModernBERT is the long-context financial encoder with an explicit FNSPID overlap warning; Qwen is the structured evidence reader, not the direction predictor.

Maintain two event-family-isolated sets: train/development reader labels and locked human reader evaluation. Initial planning values are about 600 train/development examples and 300 locked examples, subject to a class-support audit. Stratify repetition, numeric update, new period, denial/correction, different actor, different company, hypothetical, opinion and insufficient evidence. Near-duplicate families cannot cross sets.

Quality metrics cover target/object, event/action, assertion status, numeric grounding, evidence selection, abstention, coverage and family-level confidence intervals. Independent human adjudication is required before reader-derived predictive features are authorized.
