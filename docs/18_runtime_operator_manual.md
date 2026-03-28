# Runtime Operator Manual

## Purpose
- This manual is for daily human-in-the-loop review work.
- It is written for an operator who does not need to read code.
- The goal is to run a batch review, inspect the evidence, and record a human winner.

## What This System Does
- It checks candidate images against the frozen XiaoNa truth anchors.
- It outputs evidence, rankings, and review packets.
- It does not decide the final main-training admission by itself.

## Frozen Truth Rules
1. `A-Core_01_0deg_MASTER.png` is the absolute face truth.
2. `Task-63987060-116-1.png` is the absolute body truth.
3. `Task-63987060-97-1.png` is only an upper-body support anchor.
4. QA is `evidence_only`.
5. Final sealing belongs to human review and the downstream decision flow.

## Daily Workflows
1. `shot_review`
   - Run one batch review.
   - Generate `qa_report.json`, `ranked_candidates.json`, and `review_packet.json`.
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

## Recommended Daily Sequence
1. Run `shot_review`.
2. Run `inspect_review_packet`.
3. Compare top candidates manually.
4. Confirm one winner.
5. Run `promote_winner`.
6. Periodically run `winner_bank_status`.

## Start Command
Run the interactive entry:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --interactive
```

Then choose one workflow from the menu.

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

How to use the result:
- Read `review_packet.json` first.
- Use `ranked_candidates.json` as a shortlist reference.
- Use `qa_report.json` only when you need detailed evidence per image.

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
- admission advice

Use this workflow after every `shot_review`.

### 3. promote_winner
Choose this only after you manually confirm the winner.

The workflow will ask you to select:
- winner image, or
- winner rank

Then it writes the promoted record into the winner bank.

Important:
- winner promotion is not the same thing as main training admission
- the winner bank is a human-approved reference memory, not an auto-ingest training list

### 4. winner_bank_status
Choose this when you want to see:
- how many winners are recorded
- whether the recent winners show drift
- whether the bank is still stable enough to use as a review reference

## How To Read The Main Signals

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
