# Face Pose Canonical Artifact Bridge

## Purpose

This bridge lets the QA system consume `A-Core_01` face truth and candidate face normalization evidence before direct 3DDFA-V3-style inference is wired into the runtime.

It does **not** create new truth anchors.

- `A-Core_01` remains the only face truth.
- The artifact is only a derived canonical representation.
- The provider is shadow-only and does not overwrite the current face score.

## Runtime target

The shadow provider name is:

```text
face_pose_canonical_bridge
```

The provider expects:

1. Master truth artifact:

```text
outputs/master_truth/face_master_canonical.json
```

2. Candidate artifact for each image, either:

```text
input/<image>.png.face_pose_canonical.json
input/<image_stem>.face_pose_canonical.json
input/<image>.png.face_canonical.json
input/<image_stem>.face_canonical.json
```

If either artifact is missing, the provider returns structured `unavailable` evidence instead of faking a score.

## Artifact schema

The JSON schema is:

```json
{
  "schema_version": "face_pose_canonical_artifact_v1",
  "provider_name": "3ddfa_v3",
  "provider_family": "face_canonical_shadow",
  "provider_version": "3ddfa_v3_export_v1",
  "source_path": "D:/xiaona_consistency/input/v2(0).png",
  "source_role": "candidate",
  "canonical_landmarks": [0.1, 0.2, 0.3],
  "canonical_identity_vector": [0.1, 0.2, 0.3],
  "pose_euler_deg": {
    "yaw": 4.2,
    "pitch": -1.3,
    "roll": 0.8
  },
  "visible_face_coverage": 0.88,
  "frontalization_quality": 0.81,
  "pose_fit_confidence": 0.90,
  "notes": ""
}
```

### Required fields

- `schema_version`
- `source_role`

### Recommended fields

- `canonical_landmarks`
- `canonical_identity_vector`
- `pose_euler_deg`
- `visible_face_coverage`
- `frontalization_quality`
- `pose_fit_confidence`

## Current scoring behavior

The scaffold provider currently emits:

- `face_pose_normalization_confidence`
- `canonical_face_landmark_similarity`
- `canonical_face_identity_similarity`
- `pose_delta_similarity`
- `pose_delta_deg`
- `visible_face_coverage`
- `frontalization_quality`
- `pose_fit_confidence`

This is enough to:

- validate the provider contract
- support side/3Q face normalization diagnostics
- keep the QA chain stable before direct 3DDFA-V3 integration

## Recommended workflow

1. Export `A-Core_01` into `outputs/master_truth/face_master_canonical.json`
2. Export candidate sidecars into `input/*.face_pose_canonical.json`
3. Run QA normally:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --mode qa --profile body_gold_fullbody
```

Because the provider is shadow-only, it is enabled by default and only adds evidence. It does not change the current face gating.

## Converter scripts

Single artifact:

```powershell
.\.venv\Scripts\python.exe .\export_face_pose_canonical_artifact.py `
  --input D:\face_exports\A-Core_01_result.pkl `
  --output D:\xiaona_consistency\outputs\master_truth\face_master_canonical.json `
  --source-image D:\xiaona_consistency\anchors\face\front\A-Core_01_0deg_MASTER.png `
  --source-role master_truth `
  --landmarks-key landmarks_2d `
  --identity-key identity_vector `
  --pose-key pose_euler_deg
```

Batch candidate sidecars:

```powershell
.\.venv\Scripts\python.exe .\batch_export_face_pose_canonical_artifacts.py `
  --candidate-export-dir D:\face_exports\batch_001 `
  --image-dir D:\xiaona_consistency\input `
  --output-mode adjacent `
  --landmarks-key landmarks_2d `
  --identity-key identity_vector `
  --pose-key pose_euler_deg
```
