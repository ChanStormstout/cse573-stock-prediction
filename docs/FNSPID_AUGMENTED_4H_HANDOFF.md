# FNSPID-augmented chronological four-hour replay handoff

## Completed run

- Protocol: `outputs/stock_fnspid_augmented_4h/PRE_REGISTRATION.md`
- Report: `outputs/stock_fnspid_augmented_4h/v1/REPORT.md`
- Verification: `outputs/stock_fnspid_augmented_4h/v1/VERIFICATION.json`
- Public predictions and metrics: `outputs/stock_fnspid_augmented_4h/v1/`
- Private text, combined vectors and 228 models:
  `work/stock-data/fnspid_augmented_4h/v1/`

The run keeps all 1,607 canonical windows. It adds 1,322 deduplicated direct
FNSPID groups under a next-open availability rule and three-session lookback.
Independent verification is PASS; maximum model replay error is
`7.771561172376096e-16`.

## Main result

FNSPID markedly raises AMZN news coverage, including later from 84/179 to
160/179 windows. Development scores improve for several methods, with augmented
FinModernBERT reaching 62.06% BA for AMZN. The same method falls to 43.96% in
the later period. No augmented method improves both stocks across forward OOF,
development and later periods.

## Next bounded question

Keep the augmented corpus as the default for later news experiments. The next
useful comparison is a preregistered, past-only relevance/novelty gate versus
unfiltered augmentation, with exact price fallback when evidence is absent.
Do not select a stock-specific gate from the exposed development/later scores.
LLM, attention and analogy branches were not rerun in v1.
