# v2 pooling case notes

This run is a representation-control audit rather than a new event extractor.
The fixed comparison cases are the windows where Modern v2 changes the
direction relative to canonical F2, plus unchanged common-right and
common-wrong windows. They are stored in `modern_predictions.csv` and
`finbert_match_predictions.csv` by the same window key.

- A changed direction is evidence that the encoder representation matters;
  it is not evidence that the new direction is causally correct.
- A common wrong direction is not repaired by switching encoders; these cases
  motivate the stop on further encoder-only search.
- A common right direction is preserved by both representations and is not a
  reason to claim generalization.

Independent human event-quality review is not part of this run and remains
unavailable.
