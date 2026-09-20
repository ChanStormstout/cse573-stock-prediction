# SEC factual channel pilot

The bounded pilot selected one company per represented historical industry by deterministic CIK hash. It retained **66** 8-K/10-Q/10-K filing metadata objects from 2018–2021. Each object stores acceptance time separately from report period; **38/66** have `period_end != availability_time` calendar date, demonstrating the point-in-time distinction. The row-level ledger remains private and no returns were accessed.
