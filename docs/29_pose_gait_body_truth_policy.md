# Pose/Gait-Aware Body Truth Policy

## Purpose

This policy records the current governance correction:

- `A-Core_01_0deg_MASTER.png` is the only face truth.
- `Task-63987060-116-1.png` is the only body truth.
- `winner_bank` is not frozen at the current stage.
- Parameter fitting stays disabled until project optimization is complete.

The body truth is absolute, but it must be read as a living body under pose and
gait variation, not as a rigid flat-template overlay.

## Winner Bank State

Current state:

```text
winner_bank = mutable human-review memory
```

Allowed now:

- manually record or update human-confirmed candidates
- use winner-bank entries for cross-batch review notes
- inspect drift as advisory evidence

Not allowed now:

- freeze `winner_bank` as release truth
- use `winner_bank` as a new identity or body master
- use `winner_bank` as training-admission authority
- feed `winner_bank` entries into Optuna or other parameter fitting

## Parameter Fitting Hold

Do not run parameter fitting while the project is still optimizing:

- no Optuna fitting
- no threshold fitting from candidate-review data
- no fitting from mutable winner-bank entries

Allowed work during this phase:

- evidence-schema refinement
- controlled replay collection
- manual review language refinement
- provider availability and coverage validation

## Absolute Truth Anchors

Face:

```text
A-Core_01_0deg_MASTER.png
```

Body:

```text
Task-63987060-116-1.png
```

Support anchors and winner-bank entries may help interpretation, but they must
not redefine XiaoNa.

## Pose/Gait Interpretation

Body consistency must separate three questions:

1. Does the body structure still match `116-1` after reducing pose noise?
2. Is the gait or stance variation plausible for the `116-1` body truth?
3. Is any remaining delta unexplained body drift?

Preferred review signals:

- `body_pose_independent_truth_alignment`
- `body_gait_tolerant_topology_similarity`
- `body_core_measurement_similarity`
- `body_pose_sensitive_measurement_similarity`
- `body_pose_measurement_gap`

Interpretation:

- high pose-independent truth + noisy pose-sensitive measurement means gait or
  stance may explain the local asymmetry
- weak pose-independent truth + weak topology means possible true body drift
- high pose-independent truth + high pose-sensitive measurement + small
  measurement gap, with gait-tolerant topology only slightly below the pass band,
  is `gait_tolerant_topology_margin_review`: review it as a topology/gait margin,
  not as automatic body drift
- high exact pose similarity is helpful, but it is not required if the structural
  body truth is still supported

## Review Rule

When pose/gait and body structure disagree, review order is:

1. face truth against `A-Core_01`
2. body structure truth against `116-1`
3. pose/gait compatibility
4. clothing, lighting, and local rendering risks
5. mutable winner-bank drift notes

The final decision remains GPT-plus-human. Machine output is evidence only.
