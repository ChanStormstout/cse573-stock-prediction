# Protocol implementation amendment

This amendment was saved on 2026-09-17 before the official full-context run was started and before any adapter extraction metric or downstream four-hour result existed.

The first implementation protected the complete candidate sentence but supplied only adjacent sentences as context. That did not fully implement the requested title-plus-target-company-passage input. The partial run contained training losses only. Its 15 completed forward fits were not used to select a threshold, adapter, event feature, residual, or stock model and remain in the private work directory with an `aborted_adjacent_context` name.

The official run restarts all A0/A1/A2 folds and seeds with identical inputs:

- first sequence: `TARGET`, title, complete candidate sentence, and evidence sentence ID;
- second sequence: every other numbered complete sentence from the extracted target-company passages;
- only the second sequence may be shortened by the tokenizer at the fixed 512-token limit;
- no label, split, fold, seed, threshold, loss, learning rate, epoch limit, or selection rule changes.

This correction implements the original requested information scope. It was not motivated by a favorable extraction or stock-prediction score.

## Frozen-layer cache implementation note

Before any official A1/A2 fold was accepted for selection, the partially frozen variants were changed to cache the deterministic output of embeddings plus the frozen lower ten BERT layers. Those modules remain in eval mode; only the registered top two layers and task heads use training mode. Full-float caching gave exact equality. To avoid macOS swap, the frozen cache is stored as float16 and restored to float32 before the trainable layers. A fixed 255-unit probe (uniform coverage plus the longest 128 units, token lengths 134--512) gave maximum probability differences `4.8071e-5` for A1 and `3.3498e-5` for A2, below the registered cache tolerance `5e-5`. Official checkpoint reload remains a separate full-float inference check at `1e-6`.

One A1 fold finished during the implementation check. It was preserved privately in `adapter_runs_pre_top_cache_20260917` but removed from the official progress set before any extraction threshold, adapter score, check result, or stock result was computed. Official A1 and A2 folds restart from zero with the cache implementation. The nine already completed A0 folds are unchanged because A0 trains only cached classification heads and never backpropagates through encoder layers.

## Partial-sharing clarification

Before any downstream residual result was computed, C2 was fixed to the user's requested **shared event effects plus a small stock-specific bias**. Concretely, it uses the same 16 event coefficients for both stocks and adds one strongly shrunk AMZN event-window offset (the indicator is scaled by 0.25 before the common L2 penalty). The earlier phrase “stock-specific event interactions” in `PRE_REGISTRATION.md` was imprecise and would have implied another 16 AMZN coefficients, which is not justified by 13 positive AMZN training articles. No development or later prediction was read when this implementation detail was fixed.
