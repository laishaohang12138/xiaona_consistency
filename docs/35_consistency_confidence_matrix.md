# Consistency Confidence Matrix

This note documents `outputs/consistency_confidence_matrix.json`.

## Scope

- This is screening and review evidence only.
- It does not decide final training-set admission.
- It does not decide final image-set membership.
- It does not freeze `winner_bank`.
- It does not fit parameters.
- Face truth remains `A-Core_01_0deg_MASTER.png`.
- Body truth remains `Task-63987060-116-1.png`, interpreted through pose/gait-aware evidence.

## Purpose

The matrix converts the existing review packet, invariance gates, manifest state, and replay plan into one per-image confidence view.

It is meant to answer:

- which candidate rows need high-priority manual review
- which confidence axis is currently weakest
- whether apparent body drift is pose/gait explainable
- whether missing metadata or replay evidence is limiting confidence
- whether top ranking is stable enough to trust as review priority

It is not meant to answer:

- which images enter the final dataset
- which images enter final LoRA training
- which candidate becomes frozen identity truth
- what thresholds should be fitted

## Output Shape

The batch section reports:

- `evidence_confidence_band_counts`
- `review_priority_counts`
- `pose_gait_read_counts`
- `top_unresolved_evidence_gaps`
- `weakest_axes`
- `axis_summary`
- `ranking_stability`
- `global_blockers`

The candidate rows report:

- `consistency_signal_score`
- `evidence_confidence_score`
- `evidence_confidence_band`
- `review_priority`
- `axes.face_identity`
- `axes.head_topology`
- `axes.body_truth`
- `axes.body_topology_partition`
- `axes.pose_gait_explanation`
- `axes.clothing_independence`
- `axes.lighting_robustness`
- `axes.lane_pose_trace`
- `axes.metadata_traceability`
- `unresolved_evidence_gaps`

## Interpretation

`consistency_signal_score` summarizes visual and structural consistency evidence that is already available.

`evidence_confidence_score` also includes traceability and evidence-maturity factors, so missing `prompt_id`, missing `anchor_source`, empty replay packs, or missing topology partition fields can lower confidence without implying image failure.

`review_priority` means review routing only. It is not membership or admission.

`ranking_stability` labels close machine-ranking gaps so top candidates are compared manually instead of treated as stable winners.

## Current Known Limits

If current packets were generated before head/body topology partition fields were added, the matrix will show:

- `BODY_TOPOLOGY_PARTITION_NOT_IN_CURRENT_PACKET`
- `HEAD_TOPOLOGY_PARTITION_NOT_IN_CURRENT_PACKET`

That means the packet needs to be refreshed with current evidence, not that the candidate failed topology.

If manifest fields are incomplete, the matrix will show `PROMPT_METADATA_TRACEABILITY_INCOMPLETE`.

If OUTER, lighting, side, or back replay directories are empty, this remains an evidence gap rather than a final consistency verdict.

## Workflow

Run:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow refresh_consistency_confidence_matrix
```

The workflow refreshes the supporting status files, writes `outputs/consistency_confidence_matrix.json`, updates `review_status_board.json`, and updates `review_handoff_packet.json`.
