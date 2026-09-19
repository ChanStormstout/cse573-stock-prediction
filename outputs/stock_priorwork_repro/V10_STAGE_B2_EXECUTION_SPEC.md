# V10 Stage B2 frozen execution specification

Stage B2 executes only the 18 branches already frozen in
`v10/STAGE_B2_FROZEN_CONFIGURATION.json`. Each branch fits one direct
NEWS+PRICE classifier for September 2018 through the final valid February 2019
rows, using expanding chronological training data and the branch's frozen
method family and parameter. No B2 result may select a method or parameter.

The text representation is the verified canonical full-body `stem_body`.
Count or TF-IDF preprocessing and text-only chi-square selection are fitted on
each month's training rows. The price block is separately standardized on the
same training rows and then sparse-concatenated with the selected text block.
Four-hour branches use the exact 16 canonical R1 features; daily branches use
the exact seven verified DPRICE features.

No-news rows retain an all-zero text block and an active price block. There is
no post-prediction prior, price-only, or other fallback override. Exactly 108
monthly joint models are expected, with zero candidate-grid fits. All final
bundles are written once, hashed, reloaded, and independently refit by a
separate verifier. Existing Stage A/B1, V6--V9, F0/F1/F2, Market Context,
relation-reader, NEWS-only models, and DPRICE models are immutable.

Development/later results are descriptive exposed historical evaluations and
must not identify a winner. A passing final audit freezes the classical lane;
no B2.1 or further exposed classical score search is authorized.
