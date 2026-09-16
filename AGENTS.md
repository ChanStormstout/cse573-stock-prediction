# Project maintenance

- Main task is AAPL/AMZN FOUR-HOUR direction. Preserve one-hour history separately.
- Read docs/CURRENT_STATUS.md and outputs/PROJECT_LOG.md before experiments.
- Preserve historical runs; use new run directories. All existing evaluation periods are exposed exploratory backtests.
- After project changes, update the log and review entrypoints, run relevant checks, commit the task's changes and push to the configured GitHub remote as requested by the owner. Do not commit unrelated user changes.
- Keep raw course/news data, environments, cached features and model binaries out of Git. Add result files only through scripts/repository_results.txt after inspection.
- Run python3 scripts/refresh_repository.py and python3 scripts/check_repository.py before committing. Verify remote HEAD after push.
- Do not silently change repository visibility, expand third-party redistribution rights, or claim independent extraction review has passed.
