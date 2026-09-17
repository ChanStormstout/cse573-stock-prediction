# Case audit

Cases use fixed categories and deterministic key hashing. The input fields were recorded before the outcome field was appended. Raw article text and titles remain outside Git.

Selected paragraph variant: **P0**. Independent extraction review: **not performed**.

| Stock | Split | Selection slots | News | Target passages | R0 | R1 | P0 | Selected paragraph | F0 | F6 | Actual | Error notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AAPL | validation | baseline_right_new_wrong | 1 | 12 | 0.393 | 0.446 | 0.636 | 0.636 | 0.420 | 0.541 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | validation | both_right | 1 | 12 | 0.498 | 0.615 | 0.697 | 0.697 | 0.603 | 0.656 | 1 | none recorded |
| AAPL | validation | P0_P1_direction_changed | 1 | 12 | 0.536 | 0.620 | 0.445 | 0.445 | 0.564 | 0.533 | 0 | final_prediction_error |
| AAPL | validation | P0_P3_direction_changed | 1 | 12 | 0.764 | 0.639 | 0.538 | 0.538 | 0.732 | 0.588 | 1 | none recorded |
| AAPL | validation | news | 1 | 2 | 0.514 | 0.484 | 0.554 | 0.554 | 0.546 | 0.519 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | validation | recent_direction_changed | 1 | 12 | 0.564 | 0.466 | 0.606 | 0.606 | 0.534 | 0.536 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | validation | P0_P2_direction_changed | 1 | 8 | 0.583 | 0.523 | 0.526 | 0.526 | 0.578 | 0.525 | 1 | none recorded |
| AAPL | validation | baseline_wrong_new_right | 1 | 12 | 0.588 | 0.506 | 0.457 | 0.457 | 0.575 | 0.481 | 0 | none recorded |
| AAPL | validation | both_wrong, no_news | 0 | 0 | 0.509 | 0.493 | 0.493 | 0.493 | 0.493 | 0.493 | 1 | llm_prediction_error, final_prediction_error |
| AAPL | test | baseline_wrong_new_right | 1 | 12 | 0.539 | 0.368 | 0.493 | 0.493 | 0.521 | 0.431 | 0 | none recorded |
| AAPL | test | both_right | 1 | 12 | 0.636 | 0.562 | 0.506 | 0.506 | 0.624 | 0.534 | 1 | none recorded |
| AAPL | test | P0_P2_direction_changed | 1 | 11 | 0.693 | 0.628 | 0.558 | 0.558 | 0.685 | 0.593 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | test | P0_P1_direction_changed | 1 | 12 | 0.518 | 0.497 | 0.457 | 0.457 | 0.542 | 0.477 | 0 | none recorded |
| AAPL | test | P0_P3_direction_changed | 1 | 12 | 0.526 | 0.509 | 0.598 | 0.598 | 0.551 | 0.554 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | test | baseline_right_new_wrong, recent_direction_changed | 1 | 12 | 0.356 | 0.501 | 0.554 | 0.554 | 0.409 | 0.528 | 0 | llm_prediction_error, final_prediction_error |
| AAPL | test | news | 1 | 12 | 0.682 | 0.483 | 0.390 | 0.390 | 0.696 | 0.437 | 0 | none recorded |
| AAPL | test | both_wrong | 1 | 12 | 0.669 | 0.568 | 0.582 | 0.582 | 0.662 | 0.575 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | both_wrong | 0 | 0 | 0.544 | 0.584 | 0.584 | 0.584 | 0.584 | 0.584 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | baseline_right_new_wrong | 1 | 2 | 0.395 | 0.377 | 0.625 | 0.625 | 0.398 | 0.501 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | news | 1 | 6 | 0.599 | 0.606 | 0.576 | 0.576 | 0.723 | 0.591 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | recent_direction_changed | 1 | 4 | 0.482 | 0.668 | 0.571 | 0.571 | 0.387 | 0.620 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | P0_P1_direction_changed | 1 | 4 | 0.702 | 0.594 | 0.499 | 0.499 | 0.574 | 0.546 | 0 | final_prediction_error |
| AMZN | validation | P0_P2_direction_changed | 1 | 2 | 0.601 | 0.543 | 0.564 | 0.564 | 0.712 | 0.553 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | no_news | 0 | 0 | 0.451 | 0.499 | 0.499 | 0.499 | 0.499 | 0.499 | 1 | llm_prediction_error, final_prediction_error |
| AMZN | validation | both_right | 1 | 2 | 0.589 | 0.788 | 0.541 | 0.541 | 0.576 | 0.664 | 1 | none recorded |
| AMZN | validation | P0_P3_direction_changed | 1 | 2 | 0.703 | 0.842 | 0.600 | 0.600 | 0.664 | 0.721 | 0 | llm_prediction_error, final_prediction_error |
| AMZN | validation | baseline_wrong_new_right | 1 | 2 | 0.470 | 0.316 | 0.576 | 0.576 | 0.603 | 0.446 | 0 | llm_prediction_error |
| AMZN | test | P0_P3_direction_changed | 1 | 2 | 0.481 | 0.466 | 0.607 | 0.607 | 0.457 | 0.537 | 1 | none recorded |
| AMZN | test | baseline_wrong_new_right | 1 | 2 | 0.340 | 0.398 | 0.609 | 0.609 | 0.369 | 0.504 | 1 | none recorded |
| AMZN | test | news | 1 | 4 | 0.768 | 0.710 | 0.585 | 0.585 | 0.790 | 0.647 | 1 | none recorded |
| AMZN | test | both_wrong | 1 | 2 | 0.326 | 0.371 | 0.573 | 0.573 | 0.295 | 0.472 | 1 | final_prediction_error |
| AMZN | test | recent_direction_changed | 1 | 8 | 0.586 | 0.471 | 0.522 | 0.522 | 0.552 | 0.497 | 0 | llm_prediction_error |
| AMZN | test | P0_P2_direction_changed | 1 | 2 | 0.458 | 0.297 | 0.548 | 0.548 | 0.472 | 0.423 | 1 | final_prediction_error |
| AMZN | test | both_right | 1 | 6 | 0.423 | 0.430 | 0.564 | 0.564 | 0.448 | 0.497 | 0 | llm_prediction_error |
| AMZN | test | no_news | 0 | 0 | 0.674 | 0.722 | 0.722 | 0.722 | 0.722 | 0.722 | 1 | none recorded |
| AMZN | test | P0_P1_direction_changed | 1 | 2 | 0.505 | 0.563 | 0.499 | 0.499 | 0.497 | 0.531 | 1 | llm_prediction_error |
| AMZN | test | baseline_right_new_wrong | 1 | 4 | 0.442 | 0.336 | 0.466 | 0.466 | 0.500 | 0.401 | 1 | llm_prediction_error, final_prediction_error |

`input_error` means news existed but the target-paragraph selector returned no eligible passage. `selection_change` only says P0 and the selected paragraph variant crossed 0.5 differently; it is not a causal explanation. The unavailable market-factor branch is recorded separately from model error.

The following fixed slots had no eligible window and therefore no case was substituted: AAPL/validation/paragraph_direction_changed; AAPL/test/paragraph_direction_changed; AAPL/test/no_news; AMZN/validation/paragraph_direction_changed; AMZN/test/paragraph_direction_changed.
