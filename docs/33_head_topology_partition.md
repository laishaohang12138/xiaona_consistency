# Head Topology Partition Evidence

## Purpose

`head_topology_partition_v1` splits the face canonical topology read into regional head-structure evidence.

It is review-only evidence. It does not create a new truth anchor, does not freeze the winner bank, does not participate in final training-set admission or final image-set membership, and does not permit parameter fitting.

Truth policy remains unchanged:

- face truth: `A-Core_01_0deg_MASTER.png`
- body truth: `Task-63987060-116-1.png`
- side/back derived views use same-truth projection, not new anchors

## Runtime Source

The fields are produced by `face_pose_canonical_3ddfa` or its `face_pose_canonical_bridge` fallback after canonical landmarks are available.

The provider still runs in `shadow_only` mode.

## Fields

The compact GPT packet now exposes:

- `head_topology_mean_similarity`
- `head_topology_weakest_part`
- `head_topology_weakest_part_similarity`
- `head_topology_upper_face_similarity`
- `head_topology_mid_face_similarity`
- `head_topology_lower_face_similarity`
- `head_topology_contour_similarity`
- `head_topology_center_axis_similarity`
- `head_topology_lateral_balance_similarity`

The full review packet also keeps `head_topology_partition`.

## Interpretation

Use the global `canonical_face_topology_similarity` as the broad continuity read.

Use the partition fields to explain local drift:

- `upper_face_similarity`: forehead, brow, eye-zone structure
- `mid_face_similarity`: nose bridge and cheek-plane structure
- `lower_face_similarity`: mouth, chin, and jaw relationship
- `contour_similarity`: outer face and jawline continuity
- `center_axis_similarity`: nose-mouth-chin axis stability
- `lateral_balance_similarity`: left/right canonical balance

The weakest partition should become a manual review focus before calling the image identity-stable.

## Current Role

This improves multi-angle head topology review by making local drift visible when the overall topology score is high.

It is especially useful for:

- three-quarter fake-face detection
- lower-face and jaw narrowing drift
- nose-mouth-chin axis changes
- contour and far-side cheek collapse
- side-view same-truth projection review

## Non-Goals

- Do not treat a high partition score as training admission.
- Do not use side/profile support anchors as truth.
- Do not fit thresholds from this until the project optimization phase is complete.
- Do not call final LoRA dataset acceptance inside this repository.
