# Aborted during joint follow-up

The frozen news-only matrix completed but the daily joint follow-up stopped
because the implementation assumed the four-hour R1 field set for DPRICE.
Artifacts are preserved and are not interpreted. v5 only corrects the daily
price-column lookup for its preregistered prior-session fields; it does not
change the model, grid, source data or selection.
