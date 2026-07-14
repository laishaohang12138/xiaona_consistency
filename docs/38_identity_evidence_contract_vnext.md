# Identity Evidence Contract vNext - Shadow Specification v0.1

The normative mathematical architecture, identifiability boundaries, and
future promotion conditions are defined in
[`39_underlying_mathematical_model.md`](./39_underlying_mathematical_model.md).

## Purpose

This contract adds traceable native identity measurements without changing the
current review score, review status, ranking, review route, or Winner Bank.

The contract separates four different concepts:

1. native identity residual;
2. observation eligibility and scope;
3. measurement provenance;
4. calibration state.

No v0.1 field is a probability, admission decision, or fitted confidence bound.

## Governance Invariants

- Face truth is `A-Core_01_0deg_MASTER.png` with `ABSOLUTE_FROZEN` authority.
- Body truth is `Task-63987060-116-1.png` with `ABSOLUTE_FROZEN` authority.
- Both truth assets are immutable and pinned by SHA-256.
- Winner Bank is mutable review memory and has no truth authority.
- Current candidate batches must not fit parameters, thresholds, weights, ranks,
  nuisance bases, covariance matrices, or calibration intervals.
- Every v0.1 record uses `decision_influence: NONE`.

Truth integrity is checked before provider construction and GPU engine
initialization. A missing or mismatched truth asset stops runtime creation.

## Physical Isolation

The existing pipeline finishes these operations before Shadow evidence is built:

1. current scoring;
2. `review_only_score_v2` routing;
3. pairwise ranking cards;
4. Winner Bank governance;
5. QA report and review packet writes.

Shadow evidence is then written separately to:

```text
outputs/identity_evidence_shadow.json
```

It is not consumed by the current score, confidence matrix, review packet,
review route, Winner Bank, training admission, or image-set decision path.

## Observation Eligibility

Eligibility is declared per axis, never once for the whole image:

```text
MEASURABLE
CONDITIONAL
PRIOR_DEPENDENT
UNOBSERVABLE
UNAVAILABLE
UNASSESSED
```

`CHAIN_VALID` means required artifacts were available and parsed. It does not
mean the detector is stable. Stability requires a separate perturbation study.

The initial lane mapping is a Shadow scope declaration, not a calibrated fact:

- front face identity: `MEASURABLE`;
- three-quarter face identity: `CONDITIONAL`;
- side face identity: `PRIOR_DEPENDENT`;
- back face identity: `UNOBSERVABLE`.

The same eligibility is declared independently for `face_shape`. Front is
measurable, three-quarter is conditional, side is model-prior-dependent, and
back is unobservable. An available side residual is therefore not promoted to
front-equivalent evidence.

## Native Face Identity Measurement

The current identity vector is an InsightFace runtime embedding carried inside
the face canonical artifact. It is not an embedding re-extracted from a
canonicalized 3D face image.

The v0.1 residual is:

```text
e_unit = e / max(||e||_2, epsilon)
r_face = acos(clip(a_unit dot x_unit, -1, 1))
```

The unit is radians and lower means more consistent. The residual remains
`SHADOW_UNCALIBRATED`. Missing model or alignment identifiers are emitted as
contract gaps instead of being silently inferred.

The legacy `canonical_identity_vector` field remains a compatibility alias.
New artifacts also declare `runtime_face_embedding_raw` and
`runtime_face_embedding_unit`.

## Native Face Projection Shape Measurement

`face_shape` compares corresponding canonical 2D landmarks with weighted,
Huber-IRLS Procrustes alignment. Each iteration removes weighted translation,
weighted RMS scale, and the best proper `SO(2)` rotation. Reflection is never
allowed:

```text
X_hat = (X - weighted_center(X)) / weighted_rms(X)
R_star = argmin over R in SO(2) sum_i w_i ||A_hat_i - X_hat_i R||^2
r_shape = sqrt(2 * weighted_mean(huber(||A_hat_i - X_hat_i R_star||)))
```

The unit is normalized shape distance and lower means more consistent. This is
canonical **projection geometry**, not absolute 3D facial truth. Its validity
therefore depends on landmark correspondence, visibility, and the upstream
canonicalization model, especially outside front view.

Point count alone is not a correspondence contract. When both artifacts carry
`landmark_schema_id`, the IDs must match; a mismatch withholds the shape
residual. A missing or one-sided ID remains observable only in Shadow mode and
emits an explicit correspondence gap. Promotion requires a frozen point-order
schema for both truth and candidate artifacts.

When both artifacts provide per-landmark visibility, the pair weight is their
elementwise minimum. One-sided visibility is retained with an explicit gap;
missing visibility falls back to uniform weights and is also reported as a gap.
The Huber delta `0.05`, iteration limit `20`, and convergence tolerance
`0.000000001` are fixed Shadow engineering defaults, not fitted parameters and
not calibrated thresholds.

Low-, middle-, and high-y quantiles plus lateral and center-axis bands are
diagnostics inside one `face_geometry` evidence block. They are intentionally
coordinate bands, not anatomical upper/middle/lower labels, because the current
artifact contract does not yet freeze axis orientation. They are not
independent evidence and cannot be counted as multiple votes. The whole axis
remains `SHADOW_UNCALIBRATED` with `decision_influence: NONE`.

## Raw Observation Metadata

Raw observation values must include their measurement contract:

```text
value
unit
algorithm_id
input_region
image_scale
provider_version
```

Provider-estimated visibility must be named as an estimate. A numeric quality
value is not converted into a universal reliability score in v0.1.

## Evidence Lineage

The lineage graph stores only direct derivation edges:

```text
OBSERVED_FROM
TRANSFORMED_FROM
DERIVED_FROM
```

Shared-upstream relationships are computed by graph traversal and are not
persisted as pairwise edges. The graph must remain acyclic. Multiple diagnostics
derived from one landmark set remain in one evidence family and cannot become
independent votes merely by using different formulas.

## Provider Measurement Contract

An artifact name or provider label alone does not prove that reference and
candidate measurements use the same instrument. Each face axis therefore
builds a provider contract containing the model, model bundle SHA-256,
provider version, execution backend, preprocessing/alignment contract, source
field, dimension or landmark count, and geometry schema where applicable.

Two hashes have deliberately different meanings:

- `observed_contract_sha256` fingerprints every field currently observed, even
  when the contract is incomplete;
- `comparable_contract_sha256` exists only when every required field is
  resolved.

Missing fields produce `PARTIAL_MATCH`, never a comparability claim. Conflicts
between known critical fields produce `MISMATCH` and withhold the corresponding
native residual. Asset path, candidate name, rank, and Winner Bank state are not
part of the provider fingerprint.

New direct 3DDFA artifacts record the selected landmark order, native
224-pixel model-crop coordinate convention, exporter SHA-256, and a composite
SHA-256 over required model assets. New runtime embeddings record the
InsightFace package version and a composite SHA-256 over the expected
`buffalo_l` ONNX files. Legacy artifacts remain valid inputs but normally stay
`PARTIAL` until regenerated; they are not silently upgraded to complete.
CPU/CUDA or ONNX Execution Provider changes are contract changes, so a backend
mismatch is withheld instead of being interpreted as identity drift.

## Repeatability Contract

Repeatability is never represented by one generic score. Every native
measurement carries three separate domains:

```text
numerical_repeatability
preprocessing_repeatability
admissible_perturbation_stability
```

The first checks identical re-execution, the second checks codec/resampling
changes, and the third checks small preregistered crop and gamma perturbations.
Results retain native residual units and report only trial count, min, median,
max, spread, perturbation family, and detector-chain transitions. No
stable/unstable threshold or combined score exists in v0.1.

Multi-source runs also emit `cross_source_descriptors`. They describe the
distribution of per-source medians, maxima, and spreads, plus each exact
preregistered trial across sources. Detector-chain diagnostics are aggregated
only as distributions of source-level medians, so a source with more available
trials cannot silently dominate the batch. These are descriptive Shadow
statistics: they do not pool axes, fit a covariance model, estimate a
population confidence interval, or create a stable/unstable label.

Detector-chain transitions are decomposed without a fitted threshold. Each
trial records normalized face-box IoU and center/scale deltas, raw normalized
InsightFace five-point displacement, five-point similarity-shape residual after
removing translation/scale/rotation, and the 3DDFA canonical pose delta when
available. This separates input framing motion from landmark-shape jitter and
prevents an alignment change from being silently described as identity drift.
The diagnostics remain part of the same upstream measurement chain and are not
independent votes.

The probe list is preregistered in
`configs/identity_repeatability_protocol.yaml`. Execution is disabled by
default, serialized to at most one GPU job, and requires the explicit
`run_identity_repeatability_shadow` workflow plus a confirmation flag. Until
trials are actually run, all three domains report `NOT_MEASURED`; determinism
must not be inferred from missing data.

### Repeatability Execution Workflow

The workflow never scans `input` automatically. Every source image must be
named explicitly. One source creates one byte-identical baseline re-execution
and 13 preregistered trials: three identical-input runs, four codec/resampling
runs, and six local crop/gamma perturbations.

```powershell
.\.venv\Scripts\python.exe check_consistency.py `
  --workflow run_identity_repeatability_shadow `
  --repeatability-image input\candidate.png `
  --repeatability-run-id candidate_face_contract_v0_1 `
  --repeatability-confirm
```

CUDA is the workflow default and is fail-fast: both the Torch CUDA path and
ONNX Runtime CUDA execution provider must be available. CPU execution requires
the explicit `--device-policy cpu` override and becomes a different run
contract. The workflow initializes neither body truth fusion nor surface
occlusion providers.

On Windows, a CUDA repeatability run also checks NVIDIA-linked WHEA event 17
records since the current boot. Any observed NVIDIA PCIe corrected hardware
error, or a failed risk probe, blocks model initialization. The only bypass is
the explicit `--allow-gpu-hardware-risk` acknowledgement; both the risk snapshot
and override state are frozen into the run contract. Selecting
`--device-policy cpu` bypasses GPU execution rather than bypassing the evidence.

Each baseline and trial writes its image and result atomically under
`outputs/identity_repeatability_runs/<run_id>/items/`. Repeating the same
command skips terminal work units when the protocol, source hashes, selected
axes, adapter contract, model bundles, and execution backend still match.
Failed work units remain untouched unless `--repeatability-retry-failed` is
present. A baseline failure pauses all trials for that source, avoiding 13
uninformative heavy calls. Contract drift under the same run ID is a hard
resume error rather than an append.

Completed materialized images are deterministic derivatives of the frozen
source hash and protocol, so their SHA-256 and byte size are recorded and the
files are removed after successful measurement. Failed work units retain their
materialized image for diagnosis. A successful 3DDFA direct export likewise
removes its redundant input copy only after the export artifact is found. This
keeps resume and forensic evidence while preventing full-resolution storage
from multiplying with every trial.

The primary analysis pair is `run_manifest.json` plus `run_summary.json`. Raw
`result.json` files are needed only when diagnosing a failed trial, a detector
chain transition, or a withheld provider comparison. These files remain
Shadow evidence: they do not alter QA status, rank, review routing, Winner Bank,
training admission, or final image-set membership.

## Explicitly Deferred

The following are research assets and have no v0.1 decision role:

- nuisance basis construction or subtraction;
- SVD rank selection or ridge parameter selection;
- identity leakage gates;
- repeatability threshold fitting or combined stability scores;
- hard-negative thresholds;
- weighted IRLS Procrustes promotion into any decision path;
- body counterfactual rendering;
- covariance or Mahalanobis fusion;
- total uncertainty or confidence intervals;
- checkpoint convergence probabilities.

Promotion requires a frozen independent benchmark, frozen provider contracts,
an explicit parameter-fitting policy change, replay validation, and one-axis-at-
a-time impact review.
