# GPT-assisted annotation and local Qwen extraction experiments

Main forecast task remains **four-hour AAPL/AMZN direction**. This directory
implements the upstream annotation and extraction component. Extraction metrics
are not stock-direction BA. Raw news, GPT outputs, labels, and adapters stay under
`work/stock-data/annotation/`, outside Git. See REPORT.md for completed results.

## Actual workflow

1. Seal `PROTOCOL.md`; content-only candidates from Jan–Apr 2018, excluding the
   registered tiny pilot. Save exact source spans, grouping and input hashes.
2. Split 300 Jan–Feb training / 60 March development / 100 April checking pairs.
   Use two separate GPT 6 High chats with identical source packets. Each packet
   has 20 or fewer IDs. Multiple packets can share one browser request. Save the
   visible JSONL response, then strictly import it; do not replace missing replies
   with heuristic labels.
3. Compare A/B fields and evidence; source-read disagreements and fixed 10% audits.
   Preserve `uncertain` and `conflict` separately from no-event examples.
4. The original March panel had zero positives. The content-only amendment adds
   38 March candidates. A separately reported April challenge adds 39 candidates.
   See DEVELOPMENT_AMENDMENT.md; these supplements do not alter training labels.
5. Conservative post-annotation grouping drops repeated broker/action/value facts
   within seven publication days, retaining earliest available observations.
   Broker dictionary/grouping is incomplete; labels are **model-provisional**.
6. Seal training data; fit MLX QLoRA with completion-only loss and fixed 3 epochs.
   Select checkpoint on March facts. Optional second loss variant is registered
   in GATE_LOSS_PROTOCOL.md. No April-based student model selection.
7. Compare existing rules, frozen Qwen, and selected adapters on identical inputs.
   Report original April and enriched challenge separately, plus per-stock data.
   Stop downstream promotion if extraction or quality prerequisites fail.

## Commands (with owner-provided local data)

```sh
python3 outputs/stock_llm_annotation/prepare.py --out work/stock-data/annotation/v1
python3 outputs/stock_llm_annotation/labels.py packets --data work/stock-data/annotation/v1 --out work/stock-data/annotation/v1/packets
python3 outputs/stock_llm_annotation/verify.py --data work/stock-data/annotation/v1
```

Browser replies are imported with `bulk_import.py` (explicit packet IDs, pass,
conversation URL and immutable output directory). `labels.py compare` accepts the
two validated label directories. `seal.py` refuses missing labels, stale
adjudications, duplicate IDs and unresolved audits. No command creates fake GPT
or human annotations. Actual snapshot directories are versioned v1/v2/v3.

```sh
work/stock-data/structured-env/bin/python outputs/stock_llm_4h/train_adapter.py --data work/stock-data/annotation/student_v2 --out work/stock-data/annotation/runs/qlora_v2 --expanded-contract --balance-events --seed 573 --max-length 2048
# Second fixed variant adds --event-gate-loss and uses a different --out.
work/stock-data/structured-env/bin/python outputs/stock_llm_4h/infer.py --data work/stock-data/annotation/student_v3 --out NEW_FROZEN_RUN --split check --expanded-contract
# Adapted inference additionally uses --adapter RUN/selected.
work/stock-data/finbert-env/bin/python outputs/stock_llm_4h/rule_baseline.py --data work/stock-data/annotation/student_v3 --out NEW_RULE_RUN
python3 outputs/stock_llm_annotation/evaluate.py --data work/stock-data/annotation/student_v3 --rule RULE_RUN --frozen FROZEN_RUN --tuned TUNED_RUN --out NEW_COMPARISON
```

`NEW_*` must not already exist. The public repository does not include licensed
source data or adapters, so a clone alone cannot reproduce training. Model
revision and environment are recorded in the run manifests; the existing local
MLX environment is used, not a paid API or Sol job.

## Checks and limitations

```sh
python3 -m unittest discover -s outputs/stock_llm_annotation -p 'test*.py'
work/stock-data/structured-env/bin/python outputs/stock_llm_annotation/check_mlx_loss.py
```

Literal matching checks syntax/numbers/spans, not full semantic correctness.
Teacher agreement is not independent human review. Excluding uncertain inputs
reduces coverage and makes the accepted subset easier; report exclusions. Article
selection is enriched and selected-input recall does not measure full-document
recall. Unknown/multiple brokers may leave undetected duplicates. All historical
market periods have been exposed; no new generalization claim is supported.
The earliest downstream extractor boundary is **2018-05-01**; preserve earlier
forecast windows as the unchanged baseline, and train corrections only on
eligible past-only OOF base probabilities. Do not label a failed or unrun
four-hour branch as a new model result.
