# Entity semantic labeling guide

Review only the target company, title, evidence context, alias, field, publisher, native tag, and date shown on each blinded card. Do not consult stock prices or outcomes. Assign exactly one label:

- `DIRECT_TARGET`: target company is itself a substantive actor, object, or subject.
- `MULTI_COMPANY_DIRECT`: target is substantive and the report directly covers multiple companies.
- `INDIRECT_COMPETITOR_OR_COUNTERPARTY`: target appears only through another entity's event or comparison.
- `VENUE_OR_MARKET_NAME_ONLY`: alias denotes an exchange, index, listing venue, or market description.
- `PUBLISHER_OR_SOURCE_NAME_ONLY`: alias occurs only as a publisher/source identity.
- `INCIDENTAL_MENTION`: target is named without a substantive target event or statement.
- `NO_TARGET_EVIDENCE`: displayed evidence does not support the target relation.
- `UNCERTAIN`: available context is insufficient to decide.

For NDAQ, “shares trade on Nasdaq,” “Nasdaq-listed,” or a Nasdaq index reference is venue/market use, not NASDAQ, Inc. corporate evidence. Do not accept the relinker's prior class as gold. Record reviewer identity independently. The frozen gate is ≥0.90 selected-panel point precision, Wilson lower bound ≥0.85, no company below 0.60 with ≥5 reviews, and NDAQ ≥0.90.
