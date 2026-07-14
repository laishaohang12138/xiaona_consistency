# Runtime Operator Manual

## Purpose
- This manual is for daily human-in-the-loop review work.
- It is written for an operator who does not need to read code.
- The goal is to run a batch review, inspect the evidence, and record a human winner.

## What This System Does
- It checks candidate images against the frozen XiaoNa truth anchors.
- It outputs evidence, rankings, and review packets.
- It does not decide final training-set admission; that belongs to the external training decision flow.
- It does not decide final image-set membership; that belongs to the external dataset-curation flow.

## Frozen Truth Rules
1. `A-Core_01_0deg_MASTER.png` is the absolute face truth.
2. `Task-63987060-116-1.png` is the absolute body truth.
3. `Task-63987060-97-1.png` is only an upper-body support anchor.
4. QA is `evidence_only`.
5. Final training admission and final image-set membership are out of scope for this repository.

## Daily Workflows
1. `shot_review`
   - Run one batch review.
   - Generate `qa_report.json`, `ranked_candidates.json`, `review_packet.json`, and standalone Shadow evidence files.
   - Use this when a new Nano Banana 2 batch is ready.
2. `inspect_review_packet`
   - Read the latest review summary without opening raw JSON.
   - Use this when you want the shortest human-readable summary.
3. `promote_winner`
   - Write one human-confirmed winner into the winner bank.
   - Use this only after manual review is finished.
4. `winner_bank_status`
   - Check winner bank status and drift across batches.
   - Use this when you want to know whether the recent winners are starting to drift.
5. `prepare_replay_collection_plan`
   - Build the next metadata, lighting, OUTER, and side/back topology collection queue.
   - Use this before a controlled replay collection round.
6. `prepare_topology_replay_pack`
   - Create controlled side/back topology replay directories and manifest templates.
   - Use this before collecting side/back validation images.
7. `run_identity_repeatability_shadow`
   - Re-execute the face measurement chain under the preregistered three-domain protocol.
   - Requires explicit image paths and confirmation; it never scans all of `input` automatically.
8. `run_body_repeatability_shadow`
   - Re-execute HMR2 under the separate body three-domain protocol.
   - Reuses every baseline/trial reconstruction for body core and native topology, without extra HMR2 executions.
   - Reports core components and topology coordinate-axis quantiles separately and never changes review results.

## Recommended Daily Sequence
1. Run `shot_review`.
2. Run `inspect_review_packet`.
3. Compare top candidates manually.
4. Confirm one winner.
5. Run `promote_winner`.
6. Periodically run `winner_bank_status`.
7. Before side/back topology replay, run `prepare_topology_replay_pack`.
8. Before controlled replay, run `prepare_replay_collection_plan`.

## Start Command
Run the interactive entry:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --interactive
```

Then choose one workflow from the menu.

Heavy repeatability workflows require explicit confirmation because each selected image runs one
baseline plus 13 serialized model executions. CUDA is the default. NVIDIA/WHEA risk blocks execution
unless the operator selects CPU or explicitly acknowledges the recorded hardware risk. The body
workflow stops when the baseline is unavailable or a trial fails, preserves the resumable checkpoint,
and applies the preregistered cooldown after each real HMR2 execution.

Example body command:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow run_body_repeatability_shadow --repeatability-image .\input\selected.png --repeatability-confirm --device-policy cuda --require-gpu
```

Body outputs are written under `outputs/body_repeatability_runs/<run_id>/`. Reusing the same run ID
resumes only when the full source, protocol, adapter, provider and implementation contract matches.
Protocol v0.2 preserves each available topology trial's full 20670-coordinate residual in that trial's
`result.json`. `run_summary.json` reports fixed signed and absolute quantiles separately for x, y, and z;
it does not emit a vertex norm, coordinate-axis aggregate, topology score, stability label, or threshold.
Read `axis_availability` before interpreting the run: a missing core or topology axis changes the status to
`COMPLETE_WITH_UNAVAILABLE_MEASUREMENTS` even when all 13 trial executions themselves completed.

## Workflow Details

### 1. shot_review
Choose this when you want to review a fresh batch.

The interactive flow will ask for:
- runtime profile
- heavy evidence provider
- optional benchmark preset in advanced cases

Expected outputs:
- `outputs/qa_report.json`
- `outputs/ranked_candidates.json`
- `outputs/review_packet.json`
- `outputs/winner_bank_candidate.json`
- `outputs/winner_bank_report.json`
- `outputs/identity_evidence_shadow.json`
- `outputs/body_evidence_shadow.json`

How to use the result:
- Read `review_packet.json` first.
- Use `ranked_candidates.json` as a shortlist reference.
- Use `qa_report.json` only when you need detailed evidence per image.
- Use the two `*_evidence_shadow.json` files only for mathematical-chain diagnosis. They do not alter ranking, review routing, the winner bank, or external dataset decisions.
- In `body_evidence_shadow.json`, `body_topology.measurement` is usable only when readiness is `READY` and provider comparison is `MATCH`. It contains a 20670-coordinate signed zero-pose SMPL vertex delta, not a score.
- Signature-only body artifacts remain `BLOCKED`. Provider v4 regenerates body artifacts with all 6890 canonical vertices; do not manually relabel legacy signatures as native topology.
- Topology repeatability is preregistered in body protocol v0.2 but remains unexecuted until this explicit workflow is run. It shares the same 14 HMR2 reconstructions as body core.

### 2. inspect_review_packet
Choose this when you do not want to read raw JSON.

This workflow prints a concise summary of:
- run status
- engine status
- truth anchors
- active heavy provider
- active shadow view classifier
- batch blockers
- top candidate
- screening/routing advice

Use this workflow after every `shot_review`.

### 3. promote_winner
Choose this only after you manually confirm the winner.

The workflow will ask you to select:
- winner image, or
- winner rank

Then it writes the promoted record into the winner bank.

Important:
- winner promotion is not the same thing as final training-set admission
- the winner bank is mutable review memory, not an auto-ingest training list or final image-set decision

### 4. winner_bank_status
Choose this when you want to see:
- how many winners are recorded
- whether the recent winners show drift
- whether the bank is still stable enough to use as a review reference

### 4b. external admission audit
`seal_training_admission` is disabled by default in the screening project.

Use it only as an audit ledger after an external training-decision flow has already made the decision. To record that external decision, set:

```powershell
$env:XIAONA_ALLOW_EXTERNAL_ADMISSION_AUDIT="1"
```

Important:
- this does not make a local admission decision
- this does not decide final image-set membership
- the manifest is an external audit ledger, not a local training-set builder
- the environment override is an explicit assertion that the external decision is already complete
- local release gates, preflight results, and evidence completeness are recorded as advisory audit context; they neither authorize nor veto the external decision
- every locally emitted compatibility field such as `training_admission_allowed` or `eligible_for_training_seal` remains `false`

### 5. prepare_replay_collection_plan
Choose this when you are preparing a controlled replay round instead of reviewing a fresh mixed batch.

The workflow refreshes `outputs/replay_collection_plan.json` and gives an `immediate_operator_queue`.

Use it to plan:
- which front / three-quarter manifest fields must be completed first
- which lighting variants need images next
- which OUTER prompt leaves belong in the starter wave
- whether side/back lane topology is ready to run with `segformer_body_truth_fusion`
- which isolated `outputs/replay/...` artifact directory each replay command should use

Important:
- this plan produces screening evidence only
- it does not freeze winner bank
- it does not authorize parameter fitting
- it does not make final training-set admission decisions
- it does not make final image-set membership decisions

### 6. prepare_topology_replay_pack
Choose this when you want to collect side/back topology evidence separately from clean-lane, lighting, and OUTER replay.

The workflow creates:
- `input_replay/topology/side/side_left_profile`
- `input_replay/topology/side/side_right_profile`
- `input_replay/topology/back/back180_neutral`
- `input_replay/topology/back/back180_subtle_gait_shift`

Important:
- side/back BODY_GOLD profiles already prefer `segformer_body_truth_fusion`
- do not override to a weaker provider unless debugging
- do not treat gait-sensitive body deltas as drift without topology review
- this is still screening evidence only

## How To Read The Main Signals

### Preflight Stages
- `metadata_only` checks whether the batch has usable prompt intent and provenance before visual runtime initialization.
- A metadata-only result never claims observed lane purity, lane mismatch, or geometry failure; those fields remain deferred.
- `visual` runs the face/pose router only after the metadata gate passes.
- `VISUAL_PREFLIGHT_RUNTIME_UNAVAILABLE` means the visual environment failed and remains a hard stop.

### Run Status
- `ok`
  - The pipeline completed normally.
- `engine_fatal`
  - The core vision engines were unavailable.
  - Do not trust the batch as a usable review result.
  - Fix the environment first, then rerun the batch.

### Engine Status
Engine status tells you whether the core face and pose engines are usable.

Current policy:
- classic OpenCV fallback is disabled by default
- engine failure should stop trust in the batch, not silently downgrade evidence

If engine status is fatal:
1. stop review
2. do not pick a winner from this run
3. rerun only after the engine is healthy

### Truth Anchors
Always verify that the review packet still points to:
- face truth: `A-Core_01_0deg_MASTER.png`
- body truth: `Task-63987060-116-1.png`
- upper support: `Task-63987060-97-1.png`

If these are wrong, the batch evidence is being judged against the wrong reference system.

### Heavy Evidence
Heavy evidence is the heavier model-based evidence layer.

Default industrial mode:
- `segformer_body_fusion`

This combines:
- garment parsing evidence
- body-measure evidence

What to look at:
- provider name
- confidence
- coverage
- cache status

Interpretation:
- high coverage + stable confidence = stronger evidence support
- low availability or empty evidence = the batch still relies mostly on light signals

### ViewClf
`ViewClf` is the shadow view classifier status.

What it means:
- it is a second lane judgment signal
- it does not replace the primary lane router
- it is used for comparison, audit, and later benchmark validation

Use it like this:
- if the primary route and shadow route agree, lane confidence is stronger
- if they disagree often, inspect the batch more carefully
- do not manually override the batch only because `ViewClf` disagrees

## What To Review Manually

### For Front
Focus on:
- is this clearly XiaoNa
- is the face still `A-Core_01`
- is the full body still close to `116-1`
- any obvious pollution in legs, feet, waist, or shoulder line

### For Three-Quarter
Focus on:
- age drift
- face shape drift
- whether XiaoNa still looks like the same person under slight turn

### For Side / Back
Focus on:
- silhouette stability
- body thickness and geometry
- leg line and foot quality
- whether the image is only useful as observation or support material

## Recommended Reading Order
When one batch is finished:

1. read `review_packet` summary
2. check `run_status`
3. check truth anchors
4. check `heavy_evidence_summary`
5. check `ViewClf`
6. inspect top 3 candidates manually
7. decide whether the batch is:
   - usable for human shortlist review
   - observation only
   - reroute to another lane/profile

## What Not To Do
- Do not treat QA PASS as automatic main-training admission.
- Do not promote a winner before human review.
- Do not use a batch with `engine_fatal` as evidence.
- Do not let support anchors redefine XiaoNa.
- Do not use benchmark mode as a daily review replacement.

## Common Situations

### Situation A: The batch finishes normally
What to do:
1. open `inspect_review_packet`
2. compare top candidates manually
3. promote one winner if the batch is good enough

### Situation B: `engine_fatal`
What to do:
1. stop the review
2. fix the face/pose environment
3. rerun `shot_review`

### Situation C: Heavy evidence is weak or unavailable
What to do:
1. keep using the batch as a light-evidence review batch
2. be more conservative with garment or geometry judgments
3. rerun later if the heavy provider environment was unstable

### Situation D: Batch gate says reroute
What to do:
1. do not force the batch into the current lane
2. treat the reroute as process guidance
3. rerun or review under a more suitable profile

## When To Use Benchmark
Do not use benchmark in the daily winner workflow.

Use benchmark only when:
- you changed rules
- you changed a provider
- you changed thresholds
- you want to compare heavy providers
- you want to validate shadow view classifier behavior on frozen labels

## Minimal Command Cheat Sheet

Interactive daily use:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --interactive
```

Direct QA run:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --mode qa
```

Inspect latest review packet:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow inspect_review_packet
```

Show winner bank status:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow winner_bank_status
```

## Final Rule
- QA is the evidence officer.
- Human review is the final judge.
- If evidence is weak, choose conservatively.
- Underfill is safer than pollution.
