# Faculty meeting notes

Suggested length: 10–12 minutes, with slides 14–16 held for questions.

## Slide 1: News context for stock prediction

Opening: We study whether news adds useful information to a four-hour stock-direction forecast. The next method is a proposal, not a claimed experimental improvement. Present the mechanism and ask for feedback on a finite evaluation.
Source checkpoint: 652b1999407a2e248064d61e8758cd68050dcf89.

## Slide 2: The prediction task

Explain the cutoff in ordinary language: the model must make its decision before the four-hour interval begins. Windows overlap and are not independent observations. Four hours is the fixed current task, not a horizon selected for this deck. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_foundation_4h/v1/REPORT.md

## Slide 3: The course contribution is information mining

Course basis: the supplied CSE 573 group-project slides, pages 3, 7 and 20, require classical ML plus modern AI and a Semantic Web or Web Mining connection. The assigned Alostad and Davulcu paper asks whether news selected by breaking Twitter activity improves direction prediction. Our proposed extension instead asks whether relations between past and current reports reveal useful fact changes. We do not reproduce the Twitter mechanism. Reference: https://journals.sagepub.com/doi/10.3233/WEB-170349. This is evidence for the project's framing, not a claim about the instructor's private preferences.

## Slide 4: The reference model combines news meaning and prices

F2 uses ProsusAI/FinBERT title vectors, equal article averaging, PCA fitted on training articles, price features and news metadata in logistic regression. Special-token-excluded pooling is the canonical implementation. With no admitted news, the stored F2 system returns R1. The encoder is frozen; the downstream classifier really is trained. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_paper_methods_4h/run.py and https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_foundation_4h/v2/REPORT.md. FinBERT basis: https://arxiv.org/abs/1908.10063

## Slide 5: FinBERT gains vary across companies and periods

All numbers are historical saved values, not invented or newly trained for this presentation. F0 is price plus title word features; F1 is price plus full-body word features selected from the same admitted article lineage; F2 is the canonical frozen FinBERT system. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_finbert_event_adapter_4h/v1/REPORT.md. The later period was exposed and used during the broader exploration. Do not call these independent test generalization estimates.

## Slide 6: What our experiments changed in our thinking

These are distinct findings, not equivalent comparisons. Modern v2 used matched token pooling. Event full-fact F1 is a provisional extraction metric, not stock BA and not independent human gold. V10 is a separate classical experiment; the statement is its descriptive all-cell finding. The last panel is a hypothesis. Sources: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_foundation_4h/v2/REPORT.md; https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_finbert_event_adapter_4h/v1/REPORT.md; https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_priorwork_repro/v10/CLASSICAL_LANE_FINAL_SUMMARY.md.

## Slide 7: The same negative wording can mean different things

Illustrative example only. These dollar amounts are invented to explain the method and are not reported observations from the dataset. The distinction depends on the same target, analyst, event, and time context. A new publisher or dissimilar wording alone is not evidence of a new financial fact. A price response is not assumed.

## Slide 8: A news report becomes a traceable set of facts

Proposed representation and illustrative example, not an observed dataset card. A current and prior event are comparable only when target company, actor, event type, units and relevant period agree. Store the two source sentence identifiers, publication and availability times. A missing old value remains unknown. Company mentions alone do not create direct event relations. The graph is used first for retrieval, joins and evidence checking; a graph neural network is not required. The extracted numeric change is computed by code. Changes become classifier inputs; the reader does not assume a stock response.

## Slide 9: Proposed method: add evidence of what changed

Proposed architecture, not yet evaluated as this end-to-end configuration. Preserve the canonical F2 probability exactly as a fixed offset. In a separate branch, retrieve prior target-specific articles available by the cutoff, compare supported facts and make a compact context vector. The correction is active only when company identity and evidence are accepted. A graph is a provenance structure at first, not a claim to have trained a GNN. Prior work on inter-news relations motivates the question: https://arxiv.org/abs/2410.10614. This is not a reproduction of FININ.

## Slide 10: How extracted information changes the forecast

Proposed details: z has at most 12 predefined features; no unconstrained news-specific intercept. One candidate correction is alpha*tanh(w^T z), alpha in {0,0.1,0.25,0.5} log-odds, with L2 C in {0.01,0.1}. Alpha=0 is the unchanged base. These are proposed finite settings to preregister, not selected values or current authorization to run. Shared effects reduce AMZN's sparse-event problem; no per-stock later-period winner selection. This differs from prior F1 absolute-event corrections by using F2 as the offset and relative news facts; neither component is guaranteed to add predictive information.

## Slide 11: Training follows the order information becomes available

Chronological proposal: inside each outer fold fit or adapt the reader only on earlier annotations, index only earlier articles, create base probabilities and event features without fitting on each row's label, and train the correction on those OOF rows. All preprocessing is fold-local. Purge training labels whose end time overlaps the next evaluation cutoff. Keep the same two-stock architecture. Modern pretraining on historical text remains a retrospective caveat even if downstream time order is correct. Existing development and later periods remain exposed, not new holdouts.

## Slide 12: The experiment must isolate the value of context

Proposed matched ablations only; none is executed during deck creation. Keep canonical windows, F2 offset, data cutoffs, shared predictor capacity and selection budget fixed. K1 checks extra history/availability. K2 uses current-article facts without pair relations. K3 adds verified pair changes with matched count/age features. Report monthly BA/MCC/Brier, 1- and 5-day block paired intervals, coverage and changed/repaired/introduced errors. Require improvement on each stock across multiple training months with a Brier guardrail before selecting a candidate. Do not choose the method using exposed later-period results.

## Slide 13: Monday discussion and the next decision

The review pack is a prerequisite, not a completed semantic validation. No independent entity review has passed yet. The broader 66-company external-data lane remains separate from the AAPL/AMZN course task and has no predictive result. This meeting can agree on a narrow contribution and data scope. Ask whether to prioritize a careful negative result plus clear mechanism test over a broad new model search. Sources: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_ecni_e1rv2_gpt_review/GPT_REVIEW_README.md and https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/docs/CURRENT_STATUS.md.

## Slide 14: Appendix: price + FinBERT across periods

Saved canonical F2 values, no retraining. Brier is mean squared probability error; lower is better. A constant 0.5 forecast has Brier 0.25. BA=50% is a chance benchmark, not a proof of statistical significance. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_finbert_event_adapter_4h/v1/REPORT.md.

## Slide 15: Appendix: what the evaluation supports

Use precise, respectful language in the meeting. Distinguish confirmed implementation leakage, repeated development-set exposure, different evaluation protocols and uncertain pretraining overlap. Source: https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/docs/METHOD_VALIDITY_AUDIT.md; https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/docs/EXPERIMENT_CORRECTIONS.md; https://github.com/ChanStormstout/cse573-stock-prediction/blob/652b1999407a2e248064d61e8758cd68050dcf89/outputs/stock_priorwork_repro/v10/CLASSICAL_LANE_FINAL_SUMMARY.md. We have not conducted an independent paper-wide leakage audit for this deck.

## Slide 16: Appendix: sources and method status

References: https://arxiv.org/abs/1908.10063; https://arxiv.org/abs/2410.10614. Project source checkpoint: 652b1999407a2e248064d61e8758cd68050dcf89. Exact evidence links appear in notes on each result slide. No paper's benchmark gains are transferred to this project. No model training, data acquisition, entity labeling or reader calls took place in making these slides.
