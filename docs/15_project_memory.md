# Project Memory

## Core Role

This repository is a screening and evidence system for XiaoNa LoRA candidate review.

It does:

- explainable QA metrics
- batch-relative ranking
- shortlist evidence for GPT plus human review
- mutable winner-bank review memory and cross-batch drift notes
- route Nano Banana 2 draw batches into review buckets

It does not:

- decide final training-set admission
- seal images into a final training set
- auto-admit images into the final training set
- replace custom GPT plus human review
- redefine XiaoNa identity through support anchors
- freeze winner-bank entries as identity or body truth

## Frozen Principles

1. Machine role is `advisory_only` or `evidence_only`.
2. Review decision owner is `custom_gpt_plus_human`; final training-set admission belongs to the external training decision flow.
3. Absolute face identity is defined only by `A-Core_01_0deg_MASTER.png`.
4. Absolute body identity is defined only by `Task-63987060-116-1.png`.
5. Support anchors are assist-only and must never become identity masters by drift.
6. Body truth review is pose/gait-aware: separate body structure from stance, gait, and local pose expression.
7. Winner bank is not frozen in the current phase; it is mutable human-review memory only.
8. Optuna and parameter fitting are frozen until project optimization and frozen benchmark governance are ready.
9. Input stays simple. The system adapts to a flat shot batch instead of forcing complex folder structure.
10. Review outputs must be readable by humans and stable for GPT programmatic analysis.
11. Mutable winner-bank entries must not feed final admission or parameter fitting.
12. This repository does not participate in final training-set admission; it only screens, ranks, explains, and packages evidence.

## Current Review Model

Primary review artifacts:

- `outputs/qa_report.json`
- `outputs/ranked_candidates.json`
- `outputs/review_packet.json`
- `outputs/winner_bank_candidate.json`
- `outputs/winner_bank_report.json`

Primary user-facing workflow entry:

- `shot_review`
- `inspect_review_packet`
- `refresh_review_status_board`
- `prepare_front_bootstrap_review`
- `winner_bank_status`

## Screening Governance Memory

- `BODY_GOLD.front_core` is the main identity lane.
- `BRIDGE.simple_outfit` is a bridge review lane, not a new identity definition lane.
- `side/back` are still shadow or review-oriented lanes.
- `three_quarter` can be used as stronger advisory evidence than `strict side`, but still needs conservative review.
- `winner_bank` can be updated as mutable human-review memory, but it is not frozen release truth.

## Review Discipline

For every shot batch:

1. human first-pass removes obvious bad generations
2. machine produces metrics and shortlist evidence
3. GPT plus human use top candidates for diagnostic review
4. optional winner-bank recording remains manual and mutable
5. no winner-bank entry can become final admission or fitting truth by itself

Before winner-bank freezing, external admission handoff, or parameter fitting is reopened:

1. angle noise from Nano Banana prompt/view mismatch must be replay-stable
2. clothing / OUTER-style occlusion must not create false identity confidence
3. lighting and exposure drift must not be confused with identity drift
4. face/body 3D topology consistency must be stable across clean-lane replay
5. pose/gait-sensitive body differences must be separated from unexplained body-structure drift

For every future new winner:

1. compare against absolute masters
2. compare against current batch
3. compare against mutable winner-bank memory
4. keep the record mutable until freeze governance is explicitly reopened
5. route evidence to the external training decision flow if needed
