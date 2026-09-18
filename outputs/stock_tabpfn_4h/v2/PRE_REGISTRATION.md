# TabPFN v2.5 provenance and corrected configuration probe

This is a separate audit family. It keeps the original AAPL/AMZN four-hour
windows, past-only chronological splits, OLD+RECENT price inputs, and the
previous cross-stock price input. The existing `stock_goal60_4h/v1` artifacts
are preserved and are used only as a matched historical control.

Before running, the local bundled TabPFN runtime and checkpoint are audited.
The checkpoint is the public default classifier path from Prior Labs, and the
runtime metadata is retained. The local file is not labelled synthetic-only:
the package metadata says this default classifier is fine-tuned on real data.

The corrected probe uses the same pinned local checkpoint and
`n_estimators=8`, the library default ensemble size. The old v1 probe used
`n_estimators=1`; its predictions remain the n=1 control. CPU inference,
random state 573, one preprocessing job, no telemetry, and no tuning are
fixed before evaluation. Own-price and cross-stock branches are run with the
same inputs and split schedule. No development/later score selects a branch.

This experiment does not claim that TabPFN is synthetic-only, does not
reproduce the TabPFN pretraining paper, and does not treat a larger ensemble as
new independent data. If the checkpoint or runtime fingerprint changes, the
run stops rather than resuming.
