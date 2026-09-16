# Targeted second SFT variant: separate event-presence decision loss

Registered after the first expanded run's epoch-1 development predictions were
all empty (0/20 facts), before inspecting any expanded student's check predictions.
The fixed three-epoch first run must complete and remain available.

Diagnosis: despite weighting event-containing articles to 2/3 total article loss,
mean completion-token CE gives the single empty/nonempty branching decision much
less weight in long positive JSON than in the five-token empty completion. On the
249 training examples, summed coefficients on that one-token decision are 4.1482
for positive articles versus 16.6 for negative articles (4.002 times larger).
This arithmetic identifies an objective imbalance, not proof it explains all
errors. Repeated punctuation/key tokens and autoregressive generation can also
matter; inspect development generation and actual training examples.

Second and final targeted variant in this round: same model, data, seed, LoRA,
learning rate, three epochs, completion mean CE and positive weighting. Add one
next-token CE at the first token where compact empty/nonempty event JSON differs.
Use independent class-balanced weights on this auxiliary term: N/(2*n_positive)
and N/(2*n_negative), each class totaling N/2 per epoch. In the Qwen tokenizer,
the branch is answer token index 2. This does not supply labels at inference.
It cannot fix wrong facts, absent evidence or limited data by itself.

Development checkpoint selection remains 2*TP-FP-FN, then exact set, then earlier
epoch. Compare the original expanded SFT, this variant and frozen model. Do not
select or modify on check results. No further loss/grid expansion this round.
This is a targeted engineering experiment, not a paper novelty claim.
