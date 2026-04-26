# OUTER Replay Pack

## Purpose

This pack is for review-only clothing and occlusion replay.

Use it to answer one question:

- after outerwear changes, is this still the same Xiaona face and the same body truth?

It is not for training admission.
It is not for winner-bank bootstrap.

## Root Layout

The replay root is:

- `input_replay/outer/`

The structure is:

- `input_replay/outer/front/<family>/<prompt_leaf>/`
- `input_replay/outer/three_quarter/<family>/<prompt_leaf>/`

Each prompt leaf already contains:

- `input_manifest.json`
- `_input_manifest_metadata_template.json`

## What One Prompt Leaf Means

One prompt leaf means one OUTER prompt only.

Examples:

- one blazer front baseline prompt
- one trench three-quarter tracking prompt
- one rigid coat front signature prompt

Do not mix two prompt leaves together.

## Collection Rules

- Keep one prompt leaf for one outerwear prompt only.
- Keep front and three_quarter separate.
- Change outerwear or occlusion only.
- Do not intentionally change identity, body structure, lighting, or framing.
- Do not move these images back into `input_split/` clean lanes.

## Suggested Pack Size

For each prompt leaf:

- minimum: 4
- target: 6
- maximum: 8

This is enough for controlled review-only replay.

## Operator Loop

1. Put images into the correct prompt leaf.
2. Run `prepare_outer_replay_pack` again.
3. Fill `seed` or `seed_unavailable_reason`.
4. Fill `anchor_source` only after the actual batch provenance is confirmed.
5. Run `preflight_batch` and `shot_review` on that prompt leaf when you want controlled OUTER evidence.

## Why This Matters

Without a dedicated OUTER replay pack, clothing warnings are mixed with angle noise, lighting noise, and generation drift.

With this pack, clothing or occlusion becomes its own controlled test.
