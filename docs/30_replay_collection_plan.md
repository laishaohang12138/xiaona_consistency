# Replay Collection Plan

## Purpose

`replay_collection_plan.json` turns current review-only gaps into an operator queue.

It is not a scoring change, fitting pass, winner-bank freeze, final image-set decision, or final training-set admission step. It only tells the operator what evidence to collect next before the machine screening layer can judge invariance more cleanly.

## Command

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow prepare_replay_collection_plan
```

This refreshes:

- `outputs/review_run_index.json`
- `outputs/lighting_replay_pack.json`
- `outputs/outer_replay_pack.json`
- `outputs/topology_replay_pack.json`
- `outputs/review_invariance_status.json`
- `outputs/input_manifest_completion_plan.json`
- `outputs/replay_collection_plan.json`

`prepare_review_handoff` and `refresh_review_status_board` also refresh the plan.

## Read Order

Start with:

1. `overall_status`
2. `summary`
3. `immediate_operator_queue`
4. `collection_queue`

Use `all_tasks` only when you need the full backlog.

## Output Isolation

Replay run commands include `--artifacts-dir outputs/replay/...`.

This keeps controlled replay evidence from overwriting the main `outputs/` review state. For example, a topology variant writes to:

```powershell
outputs/replay/topology/side/side_left_profile/
```

## What It Covers

- Clean-lane manifest metadata gaps for front and three-quarter replay.
- Controlled lighting replay collection under `input_replay/lighting/`.
- Governed OUTER occlusion replay collection under `input_replay/outer/`.
- Controlled side and back topology replay collection under `input_replay/topology/`, then validation using profile-default `segformer_body_truth_fusion`.

## Ground Rules

- Face truth remains `A-Core_01_0deg_MASTER.png`.
- Body truth remains `Task-63987060-116-1.png`.
- Body truth must be read through pose/gait-aware metrics.
- Winner bank remains mutable review memory and is not frozen.
- Parameter fitting remains disabled until project optimization is complete.
- Final image-set construction belongs to the external dataset-curation flow, not this repository.
- Final training-set admission belongs to the external training decision flow, not this repository.
