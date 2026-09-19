# V9 full-grid compliance repair

V6/V8 retained the registered method matrix and grids but did not issue every
candidate on each completed OOF month. Their values are preserved as
`HISTORICAL_FIXED_DEFAULT_IMPLEMENTATION`, not full-grid-selected results.
No earlier development/later value changes V9 methods, grids, text, news
windows or stock choices.

V9 scores every registered candidate after each March--August OOF month using
only past training rows. March issues the fixed defaults. April--August issue a
parameter selected from all candidate evidence from strictly earlier OOF
months; September onward freezes the March--August full-grid winner while
training rows may expand chronologically. LR/SVM C=[.01,.1,1]; RF has 4,
AdaBoost 4 and KNN 6 configurations, for 29 text candidates. DPRICE has its
three C candidates. The registered no-news fallback and global weakest-stock
method ranking remain unchanged. V9 serializes issued models privately and
requires independent lineage, grid, chronology, perturbation, metric and
reload verification before interpretation.
