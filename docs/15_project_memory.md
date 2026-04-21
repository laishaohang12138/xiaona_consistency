# Project Memory

## Core Role

This repository is an evidence system for XiaoNa LoRA training governance.

It does:

- explainable QA metrics
- batch-relative ranking
- shortlist evidence for GPT plus human review
- cross-batch drift evidence through winner bank governance after the bootstrap gate is reopened

It does not:

- auto-admit images into the final training set
- replace custom GPT plus human review
- redefine XiaoNa identity through support anchors

## Frozen Principles

1. Machine role is `advisory_only` or `evidence_only`.
2. Final decision owner is always `custom_gpt_plus_human`.
3. Absolute face identity is defined only by the front face master.
4. Absolute body identity is defined only by the front full-body master.
5. Support anchors are assist-only and must never become identity masters by drift.
6. Optuna is frozen until anchor coverage and frozen benchmark governance are ready.
7. Input stays simple. The system adapts to a flat shot batch instead of forcing complex folder structure.
8. Review outputs must be readable by humans and stable for GPT programmatic analysis.
9. Winner bank bootstrap is deferred until review-only angle, clothing, lighting, and 3D topology invariance are mature.

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

## Training Governance Memory

- `BODY_GOLD.front_core` is the main identity lane.
- `BRIDGE.simple_outfit` is a training-admission bridge lane, not a new identity definition lane.
- `side/back` are still shadow or review-oriented lanes.
- `three_quarter` can be used as stronger advisory evidence than `strict side`, but still needs conservative review.

## Review Discipline

For every shot batch:

1. human first-pass removes obvious bad generations
2. machine produces metrics and shortlist evidence
3. GPT plus human use top candidates for diagnostic review
4. winner-bank promotion stays deferred until review-only invariance gates are mature

Before winner-bank bootstrap is reopened:

1. angle noise from Nano Banana prompt/view mismatch must be replay-stable
2. clothing / OUTER-style occlusion must not create false identity confidence
3. lighting and exposure drift must not be confused with identity drift
4. face/body 3D topology consistency must be stable across clean-lane replay

For every future new winner:

1. compare against absolute masters
2. compare against current batch
3. compare against curated winner bank
4. only then decide training admission
