# Entity relink audit

A deterministic locked set of 600 edge candidates was frozen at `655967663cba199b940cf4251a07e09c22c682bbdd5fab4a1f8eb118383cfb6a`. It spans the complete-census output and is stored privately without article bodies.

- Relation counts: `{'INDIRECT_OR_COMPETITOR': 123, 'DIRECT_TARGET_HIGH_CONFIDENCE': 285, 'MULTI_COMPANY_DIRECT_HIGH_CONFIDENCE': 192}`
- High-confidence candidates: `477/600`; indirect/competitor controls: `123/600`.
- Rule counts: `{'UNIQUE_DISTINCTIVE_SEC_ISSUER_ALIAS': 590, 'EXPLICIT_EXCHANGE_OR_DOLLAR_TICKER': 10}`
- Native-tag agreement: `167/600`; disagreement: `433/600`.
- Every sampled candidate has an explicit distinctive SEC alias or explicit exchange/dollar-ticker rule; only the two registered high-confidence relation classes enter panel coverage.

This is an **algorithmic evidence-rule support audit**, not independent human gold and not measured precision. Native FNSPID association quality remains the original E1 result.
