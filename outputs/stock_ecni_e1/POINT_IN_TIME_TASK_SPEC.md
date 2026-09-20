# Point-in-time daily task specification

## Status

Frozen as a task contract, conditional on a later defensible stock panel.

- Cutoff: regular-market open minus five minutes for the target XNYS session.
- Target: same-session open-to-close binary direction. E1 does not construct, inspect, or summarize these labels.
- Population: retain every eligible stock-session, including no-news days.
- `EXACT_INTRADAY_USABLE`: eligible only when its recorded availability is no later than the cutoff.
- `DATE_ONLY_CONSERVATIVE`: eligible at the next applicable regular-market open **after** the recorded calendar date; never assigned to a same-date pre-open cutoff.
- `AMBIGUOUS` and `INVALID`: excluded from primary text input.
- SEC evidence: available at the recorded acceptance timestamp, with amendments retained as separate versions.
- Duplicate groups remain evidence about dissemination and are not collapsed in E1.

## Ledger query contract

`available(company_id, cutoff)` returns only ledger objects with the same permanent company ID, a usable time class, and an effective availability time `<= cutoff`. The implementation must resolve date-only records under the policy above before comparison. No future article, filing amendment, event label, price, or return may affect membership.

## Future market/sector provenance plan

Use a frozen, point-in-time daily source for SPY and the eleven Select Sector SPDR proxies (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY), subject to availability and inception checks. Features may use only completed prior sessions' returns/volatility. No market feature is downloaded, tuned, or scored in E1.
