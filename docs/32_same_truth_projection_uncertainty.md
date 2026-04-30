# Same-Truth Projection Uncertainty

## Purpose

The project has one face truth and one body truth:

- `A-Core_01_0deg_MASTER.png`
- `Task-63987060-116-1.png`

Side, back, and strong three-quarter views must not create new truth anchors. They must read the same truths through canonical, pose-normalized, and topology-aware projection evidence.

## Runtime Evidence

`review_only_breakdown_v2` now exposes a same-truth projection node:

- `same_truth_projection_mode`
- `same_truth_projection_policy`
- `same_truth_projection_confidence`
- `same_truth_projection_uncertainty`
- `same_truth_projection_reliability`
- `face_projection_confidence`
- `body_projection_confidence`
- `projection_uncertainty_reasons`

The policy is always `same_truth_projection_not_new_anchor`.

## Lane Interpretation

- `front`: direct truth surface; face and body truth both stay close to the original anchor views.
- `three_quarter`: canonical pose projection; face topology and body topology absorb moderate yaw.
- `side`: same-truth side projection; body topology, world3d, and core measurements dominate.
- `back`: same-truth back projection body-only; face evidence is withheld and uncertainty starts higher.

## Review Rule

High projection confidence can promote a candidate into priority review, but high uncertainty must keep the result conservative.

For side/back, a review-only `PASS` is still a priority-review signal only. It is not final training admission, not winner-bank freeze evidence, and not a new truth source.

## Why This Exists

A single front face truth and a single body truth cannot literally observe every side/back surface. The system therefore compares derived canonical structure:

- face topology and pose-normalized identity from the face truth
- body core topology, HMR2 shape, and core measurements from the body truth
- world3d and lane-membership evidence as support

This solves the practical LoRA screening problem while preserving the strict truth policy.
