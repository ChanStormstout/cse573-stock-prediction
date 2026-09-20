# Proposed news-context correction for price + FinBERT

Status: design for faculty discussion only. No predictive run is authorized or claimed by this document.

## Why this design

Averaging article representations compresses financial language but does not explicitly preserve who changed what relative to a previous report. Previous event extraction improved provisional fact accuracy without a selected direction gain. Therefore relative change is a hypothesis to test, not an established solution. Missing relevant news and broader market information remain separate limitations.

## Information flow

1. Preserve the canonical price + FinBERT forecast, including its existing price fallback.
2. Retrieve only articles available before the prediction cutoff. Match target company, actor and event type; retain availability and publication time separately.
3. Compare current and prior supported facts. Output repeat, changed fact, explicit correction/denial, background or unknown, with paired evidence sentence identifiers.
4. Compute numeric changes in code only when entity, actor, units and period match. Dataset first appearance is not market first disclosure.
5. Store evidence relations for joins and retrieval. A graph neural network is not required.
6. Construct at most 12 fixed features for relations, age and accepted evidence. Train a shared, strongly regularized correction on chronological out-of-sample reference probabilities.
7. Return the exact reference probability when evidence is missing or rejected.

## Proposed finite correction

`logit(p_final) = logit(p_reference) + g * alpha * tanh(w^T z)`.

`g` is an evidence-quality gate, not a direction selector. Proposed candidate alpha values are 0, 0.1, 0.25 and 0.5 log-odds; proposed L2 C values are 0.01 and 0.1. Alpha zero retains the reference. No unrestricted event intercept and no exposed-period per-stock winner selection. These settings require a future preregistration before execution.

## Chronological training and evaluation

Reader annotations, retrieval indexes, transforms and correction training must use only each fold's past. Keep near-duplicate event groups separate between training and evaluation; retain uncertain labels. Purge labels that have not matured by the next prediction cutoff. Use nested chronological selection for the proposed correction. Modern encoder pretraining remains a retrospective limitation on 2018 data.

Matched ablations: unchanged reference; reference plus history metadata; plus current-article facts; plus relative fact changes. Match available information and capacity where testing a component. Report both stocks, monthly BA/MCC/Brier, coverage, repaired and introduced errors, and 1-/5-day block paired uncertainty. Repeated exploration limits statistical claims. New untouched data are needed to confirm generalization.

## What would count as learning

- Better relation extraction with no direction gain: reading improved, predictive increment unverified.
- A gain only from coverage/age: attribute it to history metadata.
- A gain only in one exposed period: retain as exploratory, not a stable winner.
- A repeatable gain on both stocks with sound probabilities and later fresh confirmation: evidence for the proposed mechanism.

The existing 515-card entity review remains pending and is a prerequisite for any dependent external-data lane; this design does not bypass it.
