# Diagnostic addendum — 2026-09-16, before computation

The implementation audit found one promised descriptive comparison missing from the latest report: an old single branch selected exclusively using past OOF results. This is a post-exposure diagnostic, not a new independent test or training experiment.

Complete the missing row without fitting anything. For each stock use the saved March–August 2018 OOF predictions of price/title/body/semantic; retain branches whose mean monthly Brier is no more than title Brier + 0.002; choose highest mean monthly BA, then lower Brier, then fixed complexity order price/title/body/semantic. Freeze this choice before reading its September-onward score. Preserve each branch's own output in no-news windows (no implicit price replacement). Report the chosen branch and all OOF candidates, both stocks and both later periods. Do not change the deployed selected system or the original selection records.

Additionally decompose the saved AMZN full-body linear model on all 24 selected cases into intercept, price and text logit contributions; show the five largest absolute nonzero text contributions. Exact coefficient contributions explain the model's calculation, not market causation or semantic understanding. No text perturbations or new performance candidates are needed. Verify reconstructed probabilities against the saved predictions.
