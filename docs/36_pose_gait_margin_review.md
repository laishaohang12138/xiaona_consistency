# Pose/Gait Margin Review

This note documents `outputs/pose_gait_margin_review_sheet.json`.

## Scope

- This is review routing only.
- It does not decide final image-set membership.
- It does not decide final training-set admission.
- It does not freeze `winner_bank`.
- It does not fit pose/gait parameters.
- `Task-63987060-116-1.png` remains the only body truth.

## Purpose

The sheet separates apparent body deltas into review categories before any body drift call is made.

It starts from `outputs/consistency_confidence_matrix.json` and extracts rows with:

- `gait_tolerant_topology_margin_review`
- `manual_review_required`
- pose/gait-related evidence gaps

## Categories

- `manual_body_truth_review`: unresolved body-truth rows that need direct manual review
- `structural_margin_review`: body-truth and pose/gait signals are both weak enough to require structure-first review
- `pose_lane_projection_review`: yaw, stance, or lane projection is the likely confound
- `lighting_confounded_pose_review`: lighting risk can affect perceived continuity
- `clothing_or_occlusion_confounded_review`: garment or visible-body evidence can hide body truth
- `gait_tolerant_review`: current body evidence is mostly consistent, but gait/stance needs confirmation

## Review Use

Review P0 rows first.

For each row, resolve to one of:

- `pose_or_gait_explains_delta`
- `clothing_or_occlusion_explains_delta`
- `lighting_confounds_read`
- `structural_body_review_needed`
- `unexplained_body_drift_risk`
- `insufficient_evidence`

The resolution is manual review evidence only. It is not dataset membership, winner truth, or training admission.

## Workflow

Run:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow prepare_pose_gait_margin_review
```

The workflow refreshes supporting status files, updates `consistency_confidence_matrix.json`, writes `pose_gait_margin_review_sheet.json`, and refreshes the handoff packet.
