# V2: fixed binary-choice scoring, registered before outcome scoring

Motivation: by452 price-only generative outputs,433 numeric answers were exactly
.5 and19 were malformed. p=.5/DOWN contradictions were frequent. No LLM outcome
scores were joined to guide this interface change. Original frozen JSON protocol
and all outputs are preserved; do not call the interrupted three-way JSON run
complete or silently replace its scores.

V2 keeps all1,607 inputs/609 evaluation windows and the price/news/joint source
packs, model revision, nonthinking mode, seed and baseline training unchanged.
Remove the illustrative JSON containing .5 and all requests to verbalize a
probability/rationale. Ask for exactly UP or DOWN. Require each answer to be one
native tokenizer token; stop if not. At the answer position read model log
probabilities for both tokens and normalize within these two allowed choices:
p_up=exp(logP(UP)-logsumexp(logP(UP),logP(DOWN))). Direction=p_up>=.5.
This is constrained-choice token preference, not calibrated market probability.
Record original unconstrained top token and total probability mass on both
choices; small choice mass is a diagnostic, not grounds to drop a window.

Same native batch4 engine; max1 answer token, no sampling/LoRA. Full three-way
comparison over609 windows. No future outcomes enter prompt, selection, scoring
or batching. All predictions stay; engine failures stop the run. The JSON
version is a partial interface diagnostic only, not compared on an unequal
subset as a full winner/loser. Code/weights/inputs/output logprobs are sealed.

No rationale is generated for V2. Cases examine supplied evidence, probability
changes across information variants and actual outcomes; do not invent a model
reason or claim a faithful explanation. Report BA/MCC/Brier/months plus paired
1-/5-day descriptive blocks against matched LR and old title baseline. Same
fixed joint-vs-LR case selection. No examples, fine-tuning or increased history.

Execution: finish the underway JSON price group if practical, otherwise preserve
its exact partial count; stop its expensive remaining generation once V2 synthetic
smoke validates the scoring interface. This is an explicit budget/quality
amendment after unlabeled outputs, not a pre-registered success claim. Publish
which branches did/did not finish and all incurred inference costs.

Synthetic smoke_v1 caught a numerical issue: native low-precision logprob
normalization yielded a total UP+DOWN mass above1 (about1.02–1.04), although
conditional binary probabilities were bounded. Before any real V2 forecast,
re-normalize the entire vocabulary logprob vector in float32, then select UP/DOWN.
This common normalizer preserves their relative preference while making the
choice-mass diagnostic meaningful. Keep both smoke runs; no market-label tuning.
