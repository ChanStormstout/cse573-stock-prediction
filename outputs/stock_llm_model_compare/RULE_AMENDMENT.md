# Development-only bounded rule correction

Registered while staged inference was still in March development, before its
April outputs were inspected. Case D0014 supplied an exact, correct model quote
with a target lowered **by $2 to $168**. v1 rejects two USD amounts unless they
are explicitly from/to or to/from. That conservative rule loses the new target.

One v2 rule is added: for exactly two explicit USD amounts in the grammatical
pattern `by AMOUNT to NEW_TARGET`, retain NEW_TARGET and leave old=null. AMOUNT
is a change magnitude, not an old target. Do not reconstruct an implied old value.
Keep v1 inference/prompts and all other rejections. Replay the saved stage outputs
without further model calls; report v1 and v2 separately on both panels.

This is a single development-informed parser refinement, not an April-tuned
model, additional prompt search or new independent result. No other fixes enter
this v2. Unit checks include refusal to guess currency and refusal to merge
three numeric amounts.
