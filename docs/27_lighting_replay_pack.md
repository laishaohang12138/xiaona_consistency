# Lighting Replay Pack

## Purpose

This pack exists to test lighting invariance under controlled replay conditions.

It is not a training pack.
It is not a winner-bank pack.
It must stay separate from `input_split/front` and `input_split/three_quarter`.

## Root

The workflow creates:

- `input_replay/lighting/front/*`
- `input_replay/lighting/three_quarter/*`
- `outputs/lighting_replay_pack.json`

## Variant Buckets

Each lane gets the same five lighting buckets:

1. `neutral_base`
2. `bright_exposure`
3. `dim_exposure`
4. `warm_cast`
5. `cool_cast`

These buckets are intentionally simple.
The goal is to measure whether the system still recognizes the same person and body truth when only lighting changes.

## Collection Rules

1. Keep the same lane inside one lane folder.
2. Change lighting only.
3. Do not intentionally change identity, body structure, outfit class, framing, or pose family at the same time.
4. Do not mix replay images back into `input_split/`.
5. Fill `prompt_id` and `anchor_source` only after the actual batch provenance is confirmed.

## Suggested Size

Per lane and per lighting bucket:

1. minimum: `6`
2. target: `8`
3. upper bound: `12`

This keeps the replay pack small enough to maintain, but large enough to show whether lighting warnings are systematic.

## Operator Loop

1. Run `prepare_lighting_replay_pack`.
2. Put images into `input_replay/lighting/<lane>/<variant>/`.
3. Run `prepare_lighting_replay_pack` again to refresh `input_manifest.json` and `_input_manifest_metadata_template.json`.
4. Fill `prompt_id`, `seed` or `seed_unavailable_reason`, and `anchor_source`.
5. Run `preflight_batch` and `shot_review` on each variant directory separately.
6. Compare lighting replay outputs before changing lighting thresholds or governance gates.

## Why This Matters

If lighting invariance is judged only from mixed production batches, angle noise, outfit noise, and identity noise get mixed together.

A controlled lighting replay pack isolates the question:

"Did lighting change, or did the identity/body truth actually drift?"
