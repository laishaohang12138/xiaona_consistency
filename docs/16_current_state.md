# Current State

Updated for the current repository state after commits:

- `c36022b` Add workflow review packet and winner promotion UX
- `9eb393c` Refine review guidance and master consistency evidence

## What Is Working

### Workflow and UX

- Task-oriented workflow entry is live in `check_consistency.py`.
- `review_packet.json` is the main review protocol for GPT plus human review.
- `promote_winner` supports shortlist-based manual promotion.
- `winner_bank_status` exposes current curated-bank readiness.

### Batch Review Stack

Current shot-batch review stack includes:

- absolute face anchor evidence
- cloth-free body identity evidence
- 3D and world3d cohesion evidence
- shortlist-only heavy parser review
- pairwise compare cards for top candidates
- training-admission advice as evidence only

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

- `side/back` are still review or shadow lanes, not main training release lanes.
- side scoring still shows partial fallback to `profile_like` on some surfaces.
- non-front face signal remains weak in many side batches because the absolute face master is frontal.
- winner-bank drift checks are ready, but only become meaningful after at least one human-approved winner is promoted into `outputs/winner_bank.json`.

## Current Known Interpretation

For the recent side batch:

- routing is not the main failure
- the batch is correctly recognized as side-oriented
- the real problem is front-core evaluation pressure on `face` and `upper`
- batch advice correctly says to reroute to a matching lane profile instead of treating it as `BODY_GOLD.front_core`

## Current Evidence Output Shape

Use `review_packet.json` first.

Read order inside the packet:

1. `batch_summary`
2. `ranked_review_packet`
3. `pairwise_compare_cards`
4. `winner_bank_status`
5. `items`
6. `debug`

## Current Governance State

- machine output remains evidence only
- no auto-promotion into winner bank
- no auto-admission into training set
- curated winner bank may still be empty depending on the current review cycle
- winner-bank bootstrap is currently deferred until review-only angle, clothing, lighting, and 3D topology invariance are mature enough for industrial LoRA screening
- front top candidates may be reviewed for diagnostics, but should not be promoted into winner bank under the current policy
