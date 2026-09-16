# Pre-real-data schema amendment

The synthetic smoke showed a schema-interface defect: staged gate literally
returned `events/uncertain` from the slash-separated schema placeholder.
The original protocol prohibited smoke-based prompt tuning. This is an explicit
exception/amendment before any real new-model outputs: replace composite enum
placeholders with concrete valid JSON examples plus separately listed enums in
the revised and staged prompts. Do not modify the original prompt comparison.

The original code snapshot and all smoke output remain in smoke_v2. No financial
development or check label prompted this change. The revised prompts are frozen
after this correction; this does not establish extraction performance. Smoke_v1
failed before model loading because sandboxed Metal access was unavailable.
