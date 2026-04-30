# FACE_LOCK Prompt Asset

## Import Status

- `v0.1.1-nb2-topology-16x1-copyready` is landed under `prompts/face_lock/`.
- The current state is `landed_review_only`.
- This pack is not an active runtime formula and does not create new truth anchors.
- Machine-readable governance lives in `manifest.yaml`.

## Current Contents

- Imported source files live under `assembled_shortlist/`.
- The pack contains 16 complete copy-ready prompts: `FL-01~FL-16`.
- Each prompt already includes:
  - reference priority rule
  - identity hard-lock
  - 3D face topology lock
  - head-neck-body continuity
  - hair / ear / accessory lock
  - skin / texture / lighting lock
  - wardrobe, background, camera, pose, and quality discipline
  - shot-specific plugin
  - negative core
  - negative finish

## Slot Families

- `FL-01~FL-02`: canonical front baselines.
- `FL-03~FL-06`: 10-22 degree yaw identity carry.
- `FL-07~FL-08`: micro pitch stability.
- `FL-09~FL-10`: head-neck-body attachment hold.
- `FL-11~FL-12`: front light microrelief.
- `FL-13~FL-14`: clavicle, hairline, and ear-root readability.
- `FL-15~FL-16`: 28-30 degree contour threshold.

## Governance Note

- `A-Core_01_0deg_MASTER.png` remains the only face truth.
- `Task-63987060-116-1.png` remains the only body truth.
- FACE_LOCK reinforces face topology and attachment consistency only.
- FACE_LOCK outputs are screening evidence, not final training-set admission.
- Do not freeze winner bank or fit parameters from this pack under the current project policy.
