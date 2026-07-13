# Handoff Index

This repository uses a small handoff pack so a new chat window can recover context quickly.

Read in this order:

1. [15_project_memory.md](./15_project_memory.md)
2. [16_current_state.md](./16_current_state.md)
3. [17_next_actions.md](./17_next_actions.md)
4. [29_pose_gait_body_truth_policy.md](./29_pose_gait_body_truth_policy.md) when reviewing pose/gait-aware body truth.
5. [30_replay_collection_plan.md](./30_replay_collection_plan.md) when preparing the next lighting / OUTER / side-back topology collection wave.
6. [31_topology_replay_pack.md](./31_topology_replay_pack.md) when preparing controlled side/back topology replay.
7. [32_same_truth_projection_uncertainty.md](./32_same_truth_projection_uncertainty.md) when reviewing side/back same-truth projection confidence.
8. [33_head_topology_partition.md](./33_head_topology_partition.md) when reviewing local face/head topology drift under high global topology scores.
9. [34_body_topology_partition.md](./34_body_topology_partition.md) when reviewing pose/gait-aware body topology partition drift.
10. [25_clothing_invariant_surface_bridge.md](./25_clothing_invariant_surface_bridge.md) when reviewing clothing-invariant surface evidence.
11. [26_densepose_wsl_deployment.md](./26_densepose_wsl_deployment.md) when preparing official DensePose deployment.
12. [37_truth_fusion_gpu_stack.md](./37_truth_fusion_gpu_stack.md) when running maximum-evidence GPU truth-fusion review.

Use this pack when:

- opening a new chat window
- handing work from one reviewer to another
- recovering context after a long pause

Fast rules:

- This project provides screening, explainable evidence, risk routing, and ranking only.
- Final training-set admission is outside this repository and belongs to the external training decision flow.
- Final image-set construction is outside this repository and belongs to the external dataset-curation flow.
- Absolute anchors stay frozen unless there is an explicit governance decision.
- Winner bank is mutable review memory for now, not frozen release truth.
- Optuna and parameter fitting stay frozen until project optimization and frozen benchmark governance are truly ready.

Recommended new-window prompt:

`Read docs/14_handoff_index.md, docs/15_project_memory.md, docs/16_current_state.md, and docs/17_next_actions.md first, then continue from the current state.`

Operational shortcut:

- Run `python check_consistency.py --workflow shot_review --profile <profile>` for a new shot batch.
- Add `--heavy-provider segformer_body_truth_fusion --device-policy cuda` for maximum-evidence GPU review.
- Run `python check_consistency.py --workflow prepare_topology_replay_pack` before side/back topology replay collection.
- Run `python check_consistency.py --workflow prepare_replay_collection_plan` before the next controlled replay collection.
- Read `outputs/review_packet.json` for GPT-assisted review.
- Use `python check_consistency.py --workflow promote_winner --interactive` only after GPT plus human confirm the winner.
