# Body Canonical Artifact Bridge

## Purpose

This bridge exists to let the QA system consume `116-1` body truth and candidate body shape evidence before direct HMR2 inference is wired into the runtime.

It does **not** create new truth anchors.

- `A-Core_01` remains the only face truth.
- `116-1` remains the only body truth.
- The artifact is only a derived canonical representation.

## Runtime target

The heavy provider name is:

```text
body_canonical_hmr2
```

The provider expects:

1. Master truth artifact:

```text
outputs/master_truth/body_master_shape_only.json
```

2. Candidate artifact for each image, either:

```text
input/<image>.png.body_canonical.json
input/<image_stem>.body_canonical.json
```

If either artifact is missing, the provider returns structured `unavailable` evidence instead of faking a score.

## Artifact schema

The JSON schema is:

```json
{
  "schema_version": "body_canonical_artifact_v1",
  "provider_name": "hmr2",
  "provider_family": "body_canonical",
  "provider_version": "hmr2_export_v1",
  "model_id": "4d_humans_hmr2",
  "source_path": "D:/xiaona_consistency/input/v2(0).png",
  "source_role": "candidate",
  "shape_beta": [0.0, 0.1, 0.2],
  "pose_vector": [0.0, 0.1, 0.2],
  "canonical_measurements": {
    "waist_to_shoulder": 0.52,
    "hip_to_waist": 1.31
  },
  "measurement_scales": {
    "waist_to_shoulder": 0.08,
    "hip_to_waist": 0.10
  },
  "fit_confidence": 0.91,
  "coverage": 0.88,
  "notes": "",
  "conversion_meta": {}
}
```

### Required fields

- `schema_version`
- `source_role`
- `shape_beta`

### Recommended fields

- `pose_vector`
- `fit_confidence`
- `coverage`
- `canonical_measurements`
- `measurement_scales`

## Converter script

Use:

```powershell
.\.venv\Scripts\python.exe .\export_body_canonical_artifact.py --help
```

Supported raw input formats:

- `.json`
- `.npz`
- `.npy`
- `.pkl`

The converter accepts explicit dotted paths when the raw export field names do not match the default heuristics.

## Example: export 116-1 as master truth

```powershell
.\.venv\Scripts\python.exe .\export_body_canonical_artifact.py `
  --input D:\hmr2_exports\116-1_result.pkl `
  --output D:\xiaona_consistency\outputs\master_truth\body_master_shape_only.json `
  --source-image D:\xiaona_consistency\anchors\full\Task-63987060-116-1.png `
  --source-role master_truth `
  --beta-key pred_smpl_params.betas `
  --global-orient-key pred_smpl_params.global_orient `
  --measurement waist_to_shoulder=0.52 `
  --measurement hip_to_waist=1.31 `
  --measurement-scale waist_to_shoulder=0.08 `
  --measurement-scale hip_to_waist=0.10
```

## Example: export one candidate image

```powershell
.\.venv\Scripts\python.exe .\export_body_canonical_artifact.py `
  --input D:\hmr2_exports\v2_0_result.pkl `
  --output D:\xiaona_consistency\input\v2(0).png.body_canonical.json `
  --source-image D:\xiaona_consistency\input\v2(0).png `
  --source-role candidate `
  --beta-key pred_smpl_params.betas `
  --global-orient-key pred_smpl_params.global_orient `
  --confidence-key score
```

## Example: batch export candidate sidecars

When HMR2 exports are already in one directory, use the batch bridge:

```powershell
.\.venv\Scripts\python.exe .\batch_export_body_canonical_artifacts.py `
  --candidate-export-dir D:\hmr2_exports\batch_001 `
  --image-dir D:\xiaona_consistency\input `
  --output-mode adjacent `
  --beta-key pred_smpl_params.betas `
  --global-orient-key pred_smpl_params.global_orient `
  --confidence-key score
```

Matching rule:

- export file stem and image stem are normalized before matching
- suffixes like `_result` are ignored
- punctuation differences such as `v2_0_result.pkl` vs `v2(0).png` are tolerated

If a candidate export cannot be matched to exactly one image, the batch script reports it in `skipped`.

### Recommended workflow

1. Convert `116-1` into `outputs/master_truth/body_master_shape_only.json`
2. Batch-export candidate sidecars into `input/*.body_canonical.json`
3. Run QA with:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --mode qa --profile body_gold_fullbody --heavy-provider segformer_body_truth_fusion
```

## Current scoring behavior

The scaffold provider currently emits:

- `body_shape_truth_alignment`
- `body_shape_beta_similarity`
- `canonical_measurement_similarity`
- `body_pose_delta_similarity`
- `body_mesh_fit_confidence`

This is enough to:

- validate the provider contract
- benchmark canonical body truth evidence
- keep the QA chain stable before direct HMR2 integration

## Next step

After the artifact bridge is stable on frozen benchmarks, replace the external sidecar dependency with direct HMR2 inference inside:

```text
core/qa_heavy_body_canonical.py
```
