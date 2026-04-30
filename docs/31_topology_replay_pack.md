# Topology Replay Pack

## Purpose

`topology_replay_pack.json` prepares controlled side/back replay inputs for 3D topology consistency and pose/gait-aware body truth review.

It exists because front and three-quarter topology are already stable enough for review, while side/back still need lane-pure validation with the same truth-fusion chain.

## Command

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow prepare_topology_replay_pack
```

This creates:

- `input_replay/topology/side/side_left_profile/`
- `input_replay/topology/side/side_right_profile/`
- `input_replay/topology/back/back180_neutral/`
- `input_replay/topology/back/back180_subtle_gait_shift/`

Each directory contains:

- `input_manifest.json`
- `_input_manifest_metadata_template.json`

## Review Rules

- Use the profile default heavy provider; side/back BODY_GOLD profiles already prefer `segformer_body_truth_fusion`.
- Keep full body and feet visible whenever possible.
- Change side/back view or mild gait/stance only.
- Do not intentionally change identity, body structure, outfit class, lighting, or framing.
- Read `Task-63987060-116-1.png` through pose/gait-aware metrics.
- Read side/back review-only `PASS` rows through `same_truth_projection_uncertainty`; a derived projection is not a new truth anchor.

## Non-Goals

- Do not freeze winner bank from this replay.
- Do not use topology replay for parameter fitting.
- Do not use topology replay as final training-set admission or final image-set membership.
- Do not call body drift from gait-sensitive rows without manual structure review.

## Same-Truth Projection

Side/back topology replay should produce `same_truth_projection_confidence` and `same_truth_projection_uncertainty` in `review_only_breakdown_v2`.

Use those fields to distinguish two cases:

- high body topology with acceptable projection uncertainty: priority review evidence
- weak or uncertain projection: keep as manual review, even if one raw metric looks good
