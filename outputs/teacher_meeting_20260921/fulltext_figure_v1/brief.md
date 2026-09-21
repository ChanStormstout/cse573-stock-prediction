# Slide 6: company-focused full-text extension
Audience: CSE573 faculty. Proposed architecture, not a completed or scored experiment.
Content authority: user discussion of company-paragraph selection, title preservation, passage-level grouping, frozen FinBERT, metadata and price inputs. Existing slide is style only.
Must show target-company selection, context/negation/numeric preservation, repeated-block grouping with unmatched details retained, separate titles, metadata/coverage, prices, classifier, 4h output, missing-body/news fallback.
No invented feature dimensions or results; no claim of verified event identity or market-first disclosure. Fixed pooling first; no attention or LLM claimed.
The example is illustrative, not a historical case.
Input is already cutoff-qualified news. Unknown or unreliable body selection falls back to titles. Feature scaling/compression/classifier fit on past training only. Sentence IDs retained for inspection.
Semantic traceability: input->selection->grouping->encoding/group pooling->prediction; titles bypass body processing into encoding; metadata derived from raw reports plus selection/group records; prices go directly to classifier.
