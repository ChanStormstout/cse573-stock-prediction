# Case notes

Cases and outcome-selection rules were fixed in `case_selection.json` before adapter and downstream results were revealed. Labels remain model-provisional.

## Extraction cases

| Cohort | ID | Gold type/action | Frozen Qwen | QLoRA Qwen | Selected FinBERT | Exact? |
|---|---|---|---|---|---|---:|
| qwen_historical_current_confusion | N0400 | no-event | rating|initiate, target_price|raise | no-event | no-event | yes |
| maintain_rating_and_raise_target | N0415 | rating|maintain, target_price|raise | no-event | target_price|raise | rating|maintain, target_price|raise | yes |
| multi_company_target_object | N0417 | target_price|raise | no-event | no-event | target_price|raise | yes |

These rows test extraction only. A correct fact does not imply that the next four-hour direction is predictable.

## Window cases

- Duplicate cohort `AMZN|2018-04-27 15:30:00+00:00` contains 1 multi-report clusters (2 reports). The window feature builder counts each cluster once and keeps report count separately.
- No-event fallback `AAPL|2018-04-24 13:30:00+00:00`: D1=0.614803422991, D2=0.614803422991, D3=0.614803422991, D4=0.614803422991; exact equality is verified.

## Prediction-error cases

- `extraction_correct_prediction_wrong`: no window met the pre-fixed rule; no replacement case was cherry-picked.
- `extraction_error_prediction_wrong`: no window met the pre-fixed rule; no replacement case was cherry-picked.
