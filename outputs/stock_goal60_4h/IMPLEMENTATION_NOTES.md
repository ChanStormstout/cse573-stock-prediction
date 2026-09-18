# Implementation clarifications

- The registered June–August gate chooses at most two mechanisms for a final frozen September-onward blend. Any March–August blend scores are necessarily **post-selection descriptions**, because the architecture gate used those months. They are not presented as an independent outer validation of that blend. Blend weights remain selected using strictly earlier issued probabilities. No retrospectively gated blend is described as a fully nested outer system.
- History reaction features use prefix sums over the exchange's scheduled five-minute grid. Missing any required bar gives unknown. Start/end returns can span nights and include that price gap; a separate overnight fraction is recorded. Realized volatility sums intraday bar returns and is not described as continuous overnight volatility.
- Existing A1 has three independently fitted January–February checkpoints. Each is encoded and reduced separately; incompatible representation coordinates are not averaged. Only resulting classifier probabilities are averaged.
- Preliminary local runs were archived before improving complete upstream cache fingerprints and a Python syntax compatibility change. Final records use the revised source and are replayed. They are not additional model searches and their scores are not used for selection.
- Frozen-model inference, TabPFN conditioning, classifier fitting, and new bilinear gradient optimization are reported separately. No new encoder fine-tuning occurs here.

- Verification also found float32 neural outputs rounding the float64 fallback by at most 2.98e-8. No direction changed. All final system probabilities use float64 before fallback assignment and round-trip CSV parsing; preliminary records are retained privately and excluded.

- Matched-input review found the 8+8 title control initially inherited recent-price R1 while its F2 comparator used OLD price columns. That preliminary contrast and blend were excluded and archived. The final 8+8 control uses exactly F2_new price columns and metadata; all affected source-fingerprint runs were replayed. This is a matching repair, not a new parameter search.
