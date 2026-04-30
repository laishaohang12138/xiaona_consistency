# Body Topology Partition Evidence

This note documents the body topology partition layer used for pose/gait-aware review.

## Scope

- This is screening evidence only.
- It does not decide final training-set admission.
- It does not decide final image-set membership.
- It does not introduce a new body truth anchor.
- `Task-63987060-116-1.png` remains the only body truth.
- Gait and stance are treated as projection factors before body drift is called.

## Metric Shape

`body_canonical_hmr2` now emits `body_topology_partition_v1` from existing HMR2/SMPL canonical measurements:

- `body_topology_partition_mean_similarity`
- `body_topology_weakest_part`
- `body_topology_weakest_part_similarity`
- `body_topology_torso_core_similarity`
- `body_topology_shoulder_neck_frame_similarity`
- `body_topology_waist_pelvis_similarity`
- `body_topology_leg_axis_similarity`
- `body_topology_lower_body_volume_similarity`
- `body_topology_gait_phase_similarity`
- `body_pose_explained_delta_score`

The partition layer is a re-read of existing canonical measurements. It is not a fitted parameter set and does not use winner-bank entries.

## Interpretation

`torso_core`, `shoulder_neck_frame`, and `waist_pelvis` are the most structural body-shape reads.

`leg_axis` and `lower_body_volume` can carry real topology drift, but they are also affected by stance and camera projection.

`gait_phase` is reported separately so lower-limb asymmetry can be reviewed as gait/pose evidence instead of being folded directly into body-shape truth.

`body_pose_explained_delta_score` rises when core body measurements hold while lower-limb or pose-sensitive measurements explain the remaining delta.

## Review Use

Start with:

1. `body_pose_independent_truth_alignment`
2. `body_gait_tolerant_topology_similarity`
3. `body_topology_partition_mean_similarity`
4. `body_topology_weakest_part_similarity`
5. `body_pose_explained_delta_score`

If the weakest part is `leg_axis` or `lower_body_volume` and `body_pose_explained_delta_score` is high, review gait/stance before calling body drift.

If the weakest part is `torso_core`, `shoulder_neck_frame`, or `waist_pelvis`, treat it as a stronger structural review priority.

## Output Surfaces

The fields are exposed in:

- heavy evidence metric specs
- `canonical_truth_summary`
- `review_only_breakdown_v2`
- `pose_gait_body_truth` metric means and examples
- topology review prompts in `gpt_review_packet.json`
