# Next Actions

## Priority Order

### P0

1. Start using the handoff pack in every new chat window.
2. Generate `input_manifest_completion_plan.json` before rerunning clean front / three_quarter replay.
3. Fill `prompt_id` from the real prompt file, prompt slot, or batch family.
4. Fill `anchor_source` only after confirming the actual truth anchors used.
5. Fill `seed_unavailable_reason` when Nano Banana did not expose per-image seeds.
6. Keep `winner_bank` unfrozen; use it only as mutable GPT-plus-human review memory.
7. Do not use mutable winner-bank entries for final admission, identity truth, body truth, or parameter fitting.
8. Keep `segformer_body_truth_fusion` as the default heavy chain for `body_gold_threequarter_review`, `body_gold_side90_shadow`, and `body_gold_back180_shadow`.
9. Treat current angle governance as metadata-blocked, not geometry-failed.
10. Treat current clothing governance as OUTER-replay-blocked, not simple-outfit-failed.
11. Treat current lighting governance as warning-heavy, not lighting-high failed.
12. Use `prepare_lighting_replay_pack` to keep lighting replay inputs separate from clean-lane review inputs.
13. Use `prepare_outer_replay_pack` to keep outerwear replay inputs separate from clean-lane review inputs.
14. Use `prepare_topology_replay_pack` to keep side/back topology replay inputs separate from clean-lane and OUTER replay inputs.
15. Use `prepare_replay_collection_plan` to convert manifest, lighting, OUTER, and side/back topology gaps into the next collection queue.
16. Keep Optuna and all parameter fitting disabled until project optimization is complete.
17. Interpret `116-1` body truth through pose/gait-aware metrics before calling body drift.
18. Read side/back `PASS` rows through `same_truth_projection_uncertainty`; they are priority-review evidence, not new truth anchors.
19. Treat final training-set admission as out of scope; route evidence packets to the external training decision flow.
20. Treat final image-set membership as out of scope; route evidence packets to the external dataset-curation flow.
21. Use body topology partition fields before calling multi-pose lower-body drift.

### P1

1. Improve `three_quarter` manual review prompts further.
2. Keep refining review-only invariance diagnostics around angle noise, clothing occlusion, and lighting drift.
3. Keep refining face drift diagnostics around age impression, face shape, expression, and over-smoothing.
4. Treat body 3D topology as already stabilized for `three_quarter` review, and move the next validation effort to `side` and `back` with the same truth-fusion chain.
5. Use `pose_gait_body_truth` and `optimization_focus.gait_invariance` to review `gait_tolerant_topology_margin_review` rows before calling body drift.
6. Use `same_truth_projection_confidence` and `same_truth_projection_uncertainty` to separate derived side/back support from hard truth.
7. Strengthen mutable winner-bank review language without freezing it as release truth.
8. Prepare a controlled lighting replay pack so the next round measures light change directly instead of inferring it from current front batches.
9. Keep the lighting replay pack small and lane-pure before changing any lighting gate threshold.
10. Prepare a controlled OUTER replay pack so clothing and occlusion are measured directly instead of inferred from simple outfits.
11. Use `head_topology_*` fields to review local face drift before treating a high global topology score as stable identity.
12. Use `body_topology_*` partition fields to separate torso/shoulder/waist structural drift from gait-phase and lower-limb pose projection.

### P2

1. Continue lane-specific evidence for `three_quarter` and `side`.
2. Do not expand beyond truth fusion and current face canonical until lighting / clothing invariance is cleaner.
3. Do not reopen Optuna until project optimization and frozen-benchmark governance are actually ready.

## Immediate Working Loop

For a new shot batch:

1. run `preflight_batch`
2. run `shot_review`; for `three_quarter` / `side` / `back` BODY_GOLD profiles the runtime should now auto-pick truth fusion unless you explicitly override it
3. run `prepare_manifest_completion_plan` for split manifests
4. inspect `review_handoff_packet.json` first
5. inspect `optimization_focus` to keep clothing replay, gait margin review, and topology validation separate
6. inspect `pose_gait_body_truth` before treating body deltas as identity drift
7. inspect `same_truth_projection_uncertainty` before trusting side/back priority rows
8. inspect body topology partition weak parts before treating lower-body asymmetry as drift
9. inspect `gpt_review_packet.json` / `review_packet.json` only when candidate-level detail is needed
10. compare `top1`, `top2`, and `top3` for diagnostic review
11. record a human-confirmed winner only as mutable winner-bank memory, never as frozen truth
12. continue manifest completion and clean-lane replay
13. do not spend another iteration rescuing `three_quarter` body topology unless a new replay shows regression
14. after manifest completion, prioritize lighting replay before reopening any new algorithm branch
15. run `prepare_lighting_replay_pack` before collecting the next lighting replay batch
16. run `prepare_outer_replay_pack` before collecting the next OUTER replay batch
17. run `prepare_topology_replay_pack` before collecting side/back topology replay images
18. run `prepare_replay_collection_plan` and follow `immediate_operator_queue` before expanding the replay backlog

## Acceptance Checks

The current workflow is in a good state when:

- GPT can use `review_packet.json` without needing raw debug spelunking
- GPT can start from `review_handoff_packet.json` without needing raw debug spelunking
- human reviewers can understand the batch summary and top candidate differences quickly
- body-truth reviewers can see pose/gait read counts before opening candidate-level JSON
- optimization review can separate OUTER evidence gaps, gait/topology margin rows, and side/back topology validation
- winner-bank entries remain mutable and clearly separated from frozen truth
- review-only angle / clothing / lighting / topology invariance can be replayed cleanly before winner-bank freezing or external admission handoff
- `replay_collection_plan.json` can tell the operator which metadata, lighting, OUTER, and side/back topology evidence to collect next
- side/back review rows expose same-truth projection confidence and uncertainty without introducing any new truth anchor
- GPT/human review can see the weakest head-topology partition for each face-canonical row without opening raw debug caches
- GPT/human review can see the weakest body-topology partition and pose-explained delta for each body-canonical row without opening raw HMR2 caches
- machine ranking is visibly labeled as review-priority evidence, not final image-set membership

## Do Not Do

- do not let machine top1 auto-enter the training bank
- do not ask this repository to make final training-set admission decisions
- do not ask this repository to make final image-set membership decisions
- do not freeze `winner_bank` until review-only invariance and pose/gait-aware body consistency are mature
- do not add more absolute anchors casually
- do not re-open Optuna or any parameter fitting because of temporary ranking discomfort
- do not push heavy models into the full main path unless explicit metrics stop being useful
