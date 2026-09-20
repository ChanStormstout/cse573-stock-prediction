# Semantic relation labels

Choose exactly one label for each review card.

- `DIRECT_TARGET`: The target company is itself a substantive actor, object, or subject.
- `MULTI_COMPANY_DIRECT`: The target is substantive and the report directly covers multiple companies.
- `INDIRECT_COMPETITOR_OR_COUNTERPARTY`: The target appears only through another entity's event or comparison.
- `VENUE_OR_MARKET_NAME_ONLY`: The matched alias denotes an exchange, index, listing venue, or market description.
- `PUBLISHER_OR_SOURCE_NAME_ONLY`: The matched alias occurs only as a publisher or source identity.
- `INCIDENTAL_MENTION`: The target is named without a substantive target event or statement.
- `NO_TARGET_EVIDENCE`: The available source evidence does not support the target relation.
- `UNCERTAIN`: The available source evidence is insufficient to decide reliably.
