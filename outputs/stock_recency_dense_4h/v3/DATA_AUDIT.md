# Data audit

- Official input: `work/stock-data/paper_methods_4h/v1/inputs.pkl`, 1,607 unique AAPL/AMZN four-hour windows.
- Labels are the sign of the stored four-hour `target_return`; the independent verification reproduced every label.
- Evaluation rows in this run: 765 forward-OOF (March–August), 252 development (September–October), and 357 later (November onward).
- The equal R1/F1/F2 prediction columns reproduce the saved official reference probabilities with maximum absolute error 0.
- Recency age uses the XNYS schedule and only labels matured before the evaluation boundary. Dense windows use raw regular-session five-minute bars, exactly 48 bars, without interpolation or after-hours data.
- Article association is limited to article IDs already accepted in official windows. Raw news and cached features remain private.
- The raw seventh bar column is documented separately in `ACTIVITY_AUDIT.md` and is not used.
- All development and later values are exposed historical backtests.
