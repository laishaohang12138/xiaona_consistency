# Next Actions

## Priority Order

### P0

1. Start using the handoff pack in every new chat window.
2. Start using `review_packet.json` as the default GPT review input.
3. Keep `winner_bank` bootstrap deferred until review-only angle, clothing, lighting, and 3D topology invariance are mature enough for industrial LoRA screening.
4. Continue clean-lane review-only replay and input-manifest completion before any curated winner promotion.

### P1

1. Improve `three_quarter` manual review prompts further.
2. Keep refining review-only invariance diagnostics around:
   - angle noise from Nano Banana prompt/view mismatch
   - clothing and OUTER-style occlusion
   - lighting / exposure drift
   - face/body 3D topology consistency
3. Keep refining face drift diagnostics around:
   - age impression drift
   - face shape drift
   - expression drift
   - over-smoothing drift
4. Strengthen cross-batch winner-bank review language only after the invariance gates above are stable.

### P2

1. Continue lane-specific evidence for `three_quarter` and `side`.
2. Keep heavy models shortlist-only unless existing explicit metrics fail to explain the decision.
3. Do not reopen Optuna until frozen-benchmark governance is actually ready.

## Immediate Working Loop

For a new shot batch:

1. run `preflight_batch`
2. run `shot_review` only when preflight is clean or explicitly forced
3. inspect `gpt_review_packet.json` / `review_packet.json`
4. compare `top1`, `top2`, and `top3` for diagnostic review
5. do not promote into winner bank until the review-only invariance gates are mature
6. continue manifest completion and clean-lane replay

## Acceptance Checks

The current workflow is in a good state when:

- GPT can use `review_packet.json` without needing raw debug spelunking
- human reviewers can understand the batch summary and top candidate differences quickly
- winner-bank promotion remains explicitly deferred by policy
- review-only angle / clothing / lighting / topology invariance can be replayed cleanly before curated-bank promotion

## Do Not Do

- do not let machine top1 auto-enter the training bank
- do not start `winner_bank` bootstrap until review-only invariance and 3D topology consistency are mature
- do not add more absolute anchors casually
- do not re-open Optuna because of temporary ranking discomfort
- do not push heavy models into the full main path unless explicit metrics stop being useful
