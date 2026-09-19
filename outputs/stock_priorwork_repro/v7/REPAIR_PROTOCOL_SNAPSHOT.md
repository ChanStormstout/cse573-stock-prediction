# V7 repair addendum

This addendum is committed before v7 prediction execution. v6 four-hour
predictions and results will not be changed. v6 daily results remain historical
preliminary evidence; v7 supersedes v6 only for the one-day task. No method,
grid, news window, feature definition, or stock-specific selection is added in
response to v6 outcomes. Both `DNEWS_OVERNIGHT` and `DNEWS_24H` remain
preregistered and are evaluated separately.

Daily article eligibility, target association, full-text representation and
ordering must be reconstructed from the accepted canonical four-hour source and
must reproduce all 1,607 canonical article-key memberships and normalized full
text before daily fitting. Failure to establish that contract stops v7.

For completed prior session j, define `r_j = log(close_j / open_j)`. For target
session t, DPRICE is exactly: `DRET_1=r_(t-1)`;
`DRET_2=r_(t-2)+r_(t-1)`; `DRET_5=sum(r_(t-5)..r_(t-1))`;
`RANGE_1=(high_(t-1)-low_(t-1))/open_(t-1)`;
`RV_5=sqrt(sum(r_j^2, j=t-5..t-1))`; `MEAN_5=mean(r_j, j=t-5..t-1)`;
and `HISTORY_AGE_HOURS=cutoff-latest completed prior regular-session end`.
Five complete prior sessions are required. Target-day OHLC/activity and the
target-day overnight gap are excluded from predictive inputs.

The repair adds independent canonical/news/time/chronology/preprocessing/metric
verification for v6, and independent session, daily label, article membership,
full-text, DPRICE, preprocessing, freeze, metric and no-row-deletion checks
for v7. The old v6 11-check verifier is a shallow contract check, not full
independent validation.
