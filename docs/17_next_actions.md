# Next Actions

## Priority Order

### P0

1. Start using the handoff pack in every new chat window.
2. Start using `review_packet.json` as the default GPT review input.
3. Promote the first human-approved winner into `outputs/winner_bank.json`.
4. Re-run `shot_review` after that promotion to activate real cross-batch drift evidence.

### P1

1. Improve `three_quarter` manual review prompts further.
2. Keep refining face drift diagnostics around:
   - age impression drift
   - face shape drift
   - expression drift
   - over-smoothing drift
3. Strengthen cross-batch winner-bank review language so GPT can explain drift without reading debug-only fields.

### P2

1. Continue lane-specific evidence for `three_quarter` and `side`.
2. Keep heavy models shortlist-only unless existing explicit metrics fail to explain the decision.
3. Do not reopen Optuna until frozen-benchmark governance is actually ready.

## Immediate Working Loop

For a new shot batch:

1. run `shot_review`
2. inspect `review_packet.json`
3. compare `top1`, `top2`, and `top3`
4. decide the winner with GPT plus human review
5. promote only the human-approved winner
6. re-run review when winner-bank drift needs to be checked

## Acceptance Checks

The current workflow is in a good state when:

- GPT can use `review_packet.json` without needing raw debug spelunking
- human reviewers can understand the batch summary and top candidate differences quickly
- winner-bank promotion is manual and explicit
- cross-batch drift becomes visible immediately after curated-bank promotion

## Do Not Do

- do not let machine top1 auto-enter the training bank
- do not add more absolute anchors casually
- do not re-open Optuna because of temporary ranking discomfort
- do not push heavy models into the full main path unless explicit metrics stop being useful
