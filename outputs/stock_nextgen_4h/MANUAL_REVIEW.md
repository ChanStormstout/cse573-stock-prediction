# Paragraph engineering review gate

After `input_quality.py` passes its mechanical checks, inspect every fixed card
in the private `QUALITY_CARDS.md` before LLM inference. Do not reveal outcome
labels while doing this review.

Check and record:

1. selected units are complete target-company sentences or paragraphs;
2. no source-truncated tail is accepted as a complete unit;
3. promotional/navigation text is excluded;
4. a missing target paragraph does not trigger an unrelated introduction.

Write `manual_review.json` inside the versioned paragraph directory with this
schema:

```json
{
  "status": "PASS_ENGINEERING_GATE or FAIL",
  "reviewer": "reviewer identity or role",
  "independent_review": false,
  "labels_hidden": true,
  "cases_reviewed": 16,
  "checked": ["concrete checks performed"],
  "remaining_limitations": ["known limitations"],
  "input_version": "versioned directory name",
  "quality_sha256": "SHA-256 of quality.json"
}
```

`PASS_ENGINEERING_GATE` means only that the limited input-engineering gate
passed. It is not independent semantic validation. `llm_choice.py` verifies the
review status and hashes this file into its immutable run manifest. It refuses
to start after a failed or missing review.
