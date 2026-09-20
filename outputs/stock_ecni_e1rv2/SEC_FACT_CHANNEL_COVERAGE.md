# SEC fact-channel coverage

Training-period counts use SEC submission metadata only and no outcomes.

- Filer classes: `{'DOMESTIC_REGISTRANT': 54, 'FOREIGN_PRIVATE_ISSUER': 12}`
- Form totals: `{'8-K': 2903, '10-Q': 655, '10-K': 237, '6-K': 1343, '20-F': 55}`
- Companies with zero matching frozen-source forms: **0/66**
- Counts by filer class: `{'DOMESTIC_REGISTRANT': {'8-K': 2903, '10-Q': 655, '10-K': 237, '6-K': 0, '20-F': 0}, 'FOREIGN_PRIVATE_ISSUER': {'8-K': 0, '10-Q': 0, '10-K': 0, '6-K': 1343, '20-F': 55}}`
- Per-company total-form range: **19–238**; max/min ratio **12.53**.
- Mechanical severe-inequality warning: **True**, defined as any zero-coverage company or a max/min ratio of at least 5.

Foreign-private issuers are evaluated with 6-K and 20-F in addition to the domestic 8-K/10-Q/10-K channel. A zero is explicit missingness in the frozen SEC source and is not interpreted as absence of factual disclosure. Unequal form volume is retained as a future missingness/coverage variable; facts are never fabricated.
