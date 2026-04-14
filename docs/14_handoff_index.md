# Handoff Index

This repository uses a small handoff pack so a new chat window can recover context quickly.

Read in this order:

1. [15_project_memory.md](./15_project_memory.md)
2. [16_current_state.md](./16_current_state.md)
3. [17_next_actions.md](./17_next_actions.md)
4. [25_clothing_invariant_surface_bridge.md](./25_clothing_invariant_surface_bridge.md) when reviewing clothing-invariant surface evidence.
5. [26_densepose_wsl_deployment.md](./26_densepose_wsl_deployment.md) when preparing official DensePose deployment.

Use this pack when:

- opening a new chat window
- handing work from one reviewer to another
- recovering context after a long pause

Fast rules:

- This project provides explainable evidence and ranking only.
- Final training admission is always decided by custom GPT plus human review.
- Absolute anchors stay frozen unless there is an explicit governance decision.
- Optuna stays frozen until anchor coverage and benchmark freeze are truly ready.

Recommended new-window prompt:

`Read docs/14_handoff_index.md, docs/15_project_memory.md, docs/16_current_state.md, and docs/17_next_actions.md first, then continue from the current state.`

Operational shortcut:

- Run `python check_consistency.py --workflow shot_review --profile <profile>` for a new shot batch.
- Read `outputs/review_packet.json` for GPT-assisted review.
- Use `python check_consistency.py --workflow promote_winner --interactive` only after GPT plus human confirm the winner.
