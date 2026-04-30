# Current State

Updated for the current review-only governance state.

## What Is Working

### Workflow and UX

- Task-oriented workflow entry is live in `check_consistency.py`.
- `review_handoff_packet.json` is the first file to send into GPT or human review.
- `review_packet.json` and `gpt_review_packet.json` remain detail files for candidate-level analysis.
- `review_invariance_status.json` tracks angle, clothing, lighting, and topology maturity.
- `input_manifest_completion_plan.json` tracks which input-manifest fields still need operator confirmation.
- `prepare_lighting_replay_pack` now scaffolds a controlled lighting replay workspace under `input_replay/lighting/`.
- `prepare_outer_replay_pack` now scaffolds a controlled OUTER replay workspace under `input_replay/outer/`.
- `prepare_topology_replay_pack` now scaffolds controlled side/back topology replay workspaces under `input_replay/topology/`.
- `replay_collection_plan.json` turns manifest, lighting, OUTER, and side/back topology gaps into an operator queue.
- Replay run commands now use isolated `outputs/replay/...` artifact directories so controlled variants do not overwrite main `outputs/`.
- `promote_winner` supports shortlist-based manual promotion.
- `winner_bank_status` exposes current curated-bank readiness.

### Batch Review Stack

Current shot-batch review stack includes:

- absolute face anchor evidence
- cloth-free body identity evidence
- 3D and world3d cohesion evidence
- head topology partition evidence for local face-structure drift review
- body topology partition evidence for torso, shoulder/neck, waist/pelvis, leg axis, lower-body volume, and gait-phase review
- shortlist-only heavy parser review
- pairwise compare cards for top candidates
- training-admission advice as evidence only
- `three_quarter` / `side` / `back` BODY_GOLD review lanes now prefer `segformer_body_truth_fusion` when the operator does not override the heavy provider
- the latest three-quarter truth-fusion replay reached `body_canonical_coverage=1.0` and removed batch-wide `BODY_TOPOLOGY_SUPPORT_WEAK`
- `review_run_index.json` now points `three_quarter_clean_snapshot` to the latest truth-fusion snapshot instead of the older baseline snapshot

### Master Consistency

Each reviewed item now carries a `master_consistency_card` with:

- `face_master_alignment`
- `body_master_alignment`
- `world3d_master_alignment`
- `hybrid_master_alignment`
- `lane_validity`
- `face_drift`
- `manual_focus`
- `manual_review_prompts`

This is especially useful for `three_quarter`, `side`, and `back` review where a simple score is not enough.

## What Is Not Release-Grade Yet

- review-only invariance is currently `NOT_READY`.
- lighting invariance is now a `WARN`, not a hard `FAIL`: current issue is widespread lighting warning noise, not severe lighting-high evidence.
- input split manifests still need `prompt_id` and `anchor_source`.
- OUTER is a governed inactive asset and has not yet completed review-only occlusion replay.
- `side/back` are still review or shadow lanes, not main training release lanes.
- side scoring still shows partial fallback to `profile_like` on some surfaces.
- non-front face signal remains weak in many side batches because the absolute face master is frontal.
- winner bank is not frozen; mutable entries may support human review, but cannot define truth, training admission, or fitting data.
- parameter fitting remains disabled until project optimization is complete.

## Current Body Topology Read

For the latest `three_quarter` replay:

- old boundary-plus-light-geometry baseline stayed at `body_topology_support_mean=0.6607`
- truth fusion with canonical body truth raised this to `0.7586`
- `body_truth_support_mean` rose from `0.6828` to `0.7616`
- the `topology_consistency` gate is now `PASS`
- combined front + three_quarter `body_topology_top3_mean` is now `0.7361`
- lane-aware read is `front=0.6945`, `three_quarter=0.7777`
- the remaining main blockers are now manifest metadata, OUTER occlusion replay, face angle / face identity, and lighting replay, not canonical body topology

## Current Pose/Gait Body Truth Read

- `A-Core_01_0deg_MASTER.png` is the only face truth.
- `Task-63987060-116-1.png` is the only body truth.
- Body truth is read as pose/gait-aware absolute truth, not as a rigid flat-template overlay.
- Prefer `body_pose_independent_truth_alignment`, `body_gait_tolerant_topology_similarity`, and `body_core_measurement_similarity` when gait or stance differs.
- Use `body_topology_partition_mean_similarity`, `body_topology_weakest_part_similarity`, and `body_pose_explained_delta_score` to separate regional body topology drift from pose/gait projection.
- Use `body_pose_sensitive_measurement_similarity` and `body_pose_measurement_gap` to explain which deltas are likely pose/gait expression rather than body drift.
- `review_status_board.json` and `review_handoff_packet.json` now expose `pose_gait_body_truth` read counts, metric means, and review examples before opening candidate-level JSON.
- `gait_tolerant_topology_margin_review` separates high core body-truth support with marginal gait-tolerant topology from true body-drift risk.
- `optimization_focus` now summarizes clothing, gait, and topology next actions in both the status board and handoff packet.
- Mutable winner-bank entries can support review memory only; they cannot override `116-1`.

## Current Governance Read

- `angle_invariance` is now blocked only by incomplete manifest intent fields
- front angle center distance is `32.3708`, but the lane-aware front tolerance is now `35.0`, so it is no longer treated as a geometry failure
- three-quarter angle center distance is `8.6891`, well inside the current `18.0` tolerance
- `clothing_invariance` is now blocked by missing controlled OUTER replay evidence, not by simple-outfit evidence failure
- intrinsic simple-outfit evidence is currently strong enough on both front and three-quarter lanes:
  - front: clothfree cohesion `0.9686`, body-under-clothes continuity `0.6497`
  - three_quarter: clothfree cohesion `0.9760`, body-under-clothes continuity `0.6095`
- a dedicated OUTER replay scaffold now exists:
  - `input_replay/outer/front/<family>/<prompt_leaf>`
  - `input_replay/outer/three_quarter/<family>/<prompt_leaf>`
  - every prompt leaf already contains `input_manifest.json` and `_input_manifest_metadata_template.json`
  - current replay image count is still `0`, so the clothing gate is prepared for collection but not yet validated from controlled OUTER replay
- `lighting_invariance` remains a `WARN` because `SKIN_LIGHTING_RISK_WARN` is still very frequent, especially in front batches, but there is no `SKIN_LIGHTING_RISK_HIGH` evidence in the current clean snapshots
- a dedicated lighting replay scaffold now exists:
  - `input_replay/lighting/front/{neutral_base,bright_exposure,dim_exposure,warm_cast,cool_cast}`
  - `input_replay/lighting/three_quarter/{neutral_base,bright_exposure,dim_exposure,warm_cast,cool_cast}`
  - every variant directory already contains `input_manifest.json` and `_input_manifest_metadata_template.json`
  - current replay image count is still `0`, so the lighting gate is prepared for collection but not yet validated from controlled replay
- `replay_collection_plan.json` now prioritizes the next collection wave:
  - clean-lane manifest completion first
  - lighting variants by lane warning pressure
  - OUTER starter-wave prompt leaves before the full prompt backlog
  - controlled `input_replay/topology/` side/back variants before truth-fusion topology validation
- a dedicated topology replay scaffold now exists:
  - `input_replay/topology/side/{side_left_profile,side_right_profile}`
  - `input_replay/topology/back/{back180_neutral,back180_subtle_gait_shift}`
  - every variant directory already contains `input_manifest.json` and `_input_manifest_metadata_template.json`
  - current topology replay image count is still `0`, so side/back topology validation is prepared for collection but not yet measured

## Current Known Interpretation

For the recent side batch:

- routing is not the main failure
- the batch is correctly recognized as side-oriented
- the real problem is front-core evaluation pressure on `face` and `upper`
- batch advice correctly says to reroute to a matching lane profile instead of treating it as `BODY_GOLD.front_core`

## Current Evidence Output Shape

Use `review_handoff_packet.json` first.

Then add detail files only when needed:

1. `review_status_board.json`
2. `review_invariance_status.json`
3. `replay_collection_plan.json`
4. `input_manifest_completion_plan.json`
5. `gpt_review_packet.json`
6. `review_packet.json`

For body-truth triage, start with `pose_gait_body_truth` in the handoff/status board; open `gpt_review_packet.json` only when a non-consistent read needs candidate-level inspection.
For optimization triage, start with `optimization_focus`; it keeps clothing replay, gait/topology margin review, and side/back topology validation separate.
For head-topology triage, use `face_canonical_summary.head_topology_*` in `gpt_review_packet.json` to identify local jaw, contour, center-axis, or lateral-balance drift when global face topology remains high.
For body-topology triage, use `canonical_truth_summary.body_topology_*` to identify whether the weakest region is trunk/shoulder/waist structure or lower-limb gait projection.

## Current Governance State

- machine output remains evidence only
- no auto-promotion into winner bank
- no final training-set admission inside this project
- no final image-set membership decision inside this project
- `project_scope` is now `screening_and_evidence_only`; final training decisions belong to the external training decision flow and final image-set construction belongs to the external dataset-curation flow
- `seal_training_admission` is disabled by default and can only record an already-external decision as an audit ledger with an explicit environment override
- winner bank is mutable review memory and is not frozen release truth
- mutable winner-bank entries must not feed parameter fitting, final admission, or final image-set membership
- front top candidates may be reviewed and manually recorded, but remain mutable until freeze governance is explicitly reopened
- Nano Banana batches may use `seed_unavailable_reason` instead of a fabricated seed when the generator does not expose one.
