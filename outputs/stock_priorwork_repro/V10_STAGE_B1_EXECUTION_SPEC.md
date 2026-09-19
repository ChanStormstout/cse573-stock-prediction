# V10 Stage B1 execution specification

This code-only checkpoint restates the frozen V10 addendum. Stage B1 runs only
daily DPRICE with seven prior-session price features, `C in {0.01, 0.1, 1.0}`
and chronological March--August selection followed by September--February C
freezing. It separately ranks the eight V9 issued NEWS-only methods using only
March--August monthly OOF predictions, then freezes three methods per horizon
for a later B2. It does not fit NEWS+PRICE models.
