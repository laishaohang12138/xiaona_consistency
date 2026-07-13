# Clothing-Invariant Surface Bridge

## Purpose

This bridge lets the QA system consume external DensePose / SAM2 / custom
surface-occlusion evidence without making those models hard runtime
dependencies.

It does not create a new XiaoNa truth source.

- `A-Core_01_0deg_MASTER` remains the only face truth.
- `Task-63987060-116-1` remains the only body truth.
- Surface occlusion evidence only explains how much clothing blocks or distorts
  visible body evidence.

## Provider

Runtime provider name:

```text
clothing_surface_occlusion_bridge
```

The provider is also included as a fourth component inside:

```text
segformer_body_truth_fusion
```

If no sidecar exists, it returns structured unavailable evidence and the review
chain falls back to parser + body topology.

When `XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT` is set, the bridge can create a
missing sidecar before returning evidence. The maximum-evidence GPU review path
sets:

```text
XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT=densepose,sam2
XIAONA_SURFACE_OCCLUSION_DEVICE=cuda
```

Use `sam2` when DensePose WSL is not ready, and `off` for sidecar-only review.

## Candidate Sidecar Files

For `input/example.png`, any of these files will be consumed:

```text
input/example.png.surface_occlusion.json
input/example.surface_occlusion.json
input/example.png.densepose.json
input/example.densepose.json
input/example.png.sam2.json
input/example.sam2.json
```

## Artifact Schema

```json
{
  "schema_version": "clothing_surface_occlusion_artifact_v1",
  "provider_name": "external_surface_occlusion",
  "provider_family": "clothing_invariant_surface",
  "provider_version": "external_surface_occlusion_v1",
  "model_id": "densepose_or_sam2",
  "source_path": "D:/xiaona_consistency/input/example.png",
  "source_role": "candidate",
  "track_id": "",
  "metrics": {
    "visible_body_surface_alignment": 0.82,
    "garment_occlusion_index": 0.31,
    "garment_boundary_risk": 0.12,
    "visible_body_ratio": 0.55,
    "visible_face_ratio": 0.10,
    "visible_arm_ratio": 0.12,
    "visible_leg_ratio": 0.18,
    "clothing_surface_confidence": 0.91
  },
  "conversion_meta": {
    "raw_input": "",
    "notes": ""
  }
}
```

## Converter

Use the generic converter when an external model has already produced a JSON
export:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_artifact.py `
  --input D:\surface_exports\example.json `
  --output D:\xiaona_consistency\input\example.png.surface_occlusion.json `
  --source-image D:\xiaona_consistency\input\example.png `
  --model-id densepose_custom_v1
```

For the official WSL DensePose deployment, use the dedicated exporter:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_densepose.py `
  --image D:\xiaona_consistency\input\example.png `
  --output D:\xiaona_consistency\input\example.png.densepose.json
```

If you already know the scalar metrics, you can write the sidecar directly:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_artifact.py `
  --output D:\xiaona_consistency\input\example.png.surface_occlusion.json `
  --source-image D:\xiaona_consistency\input\example.png `
  --visible-body-surface-alignment 0.82 `
  --garment-occlusion-index 0.31 `
  --garment-boundary-risk 0.12 `
  --visible-body-ratio 0.55 `
  --confidence 0.91
```

## SAM2 Exporter

First deploy the checkpoint into the project-local path:

```powershell
.\.venv\Scripts\python.exe .\deploy_surface_occlusion_sam2.py
```

Default local deployment path:

```text
D:\xiaona_consistency\external\models\SAM2\sam2.1_hiera_tiny.pt
```

The local runtime can also export SAM2 silhouette sidecars:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_sam2.py `
  --image D:\xiaona_consistency\input\example.png `
  --output D:\xiaona_consistency\input\example.png.surface_occlusion.json `
  --device cuda
```

Batch mode:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_sam2.py `
  --input-dir D:\xiaona_consistency\input `
  --output-dir D:\xiaona_consistency\input `
  --device cuda
```

Default model:

```text
facebook/sam2.1-hiera-tiny
```

The exporter now prefers the locally deployed checkpoint above. Use
`--prefer-hf` only when you explicitly want Hugging Face cache/download
instead of the project-local deployment.

SAM2 is used only as silhouette / occlusion evidence. It does not classify
clothing and does not identify XiaoNa. Clothing semantics still come from
Human Parsing / Segformer; identity truth still comes from face/body canonical
truth.

On Windows, SAM2 may warn that the optional `_C` extension is unavailable and
skip some post-processing. This is allowed for the sidecar workflow as long as
the exporter returns a valid artifact, but the warning should be recorded when
auditing model quality.

## Runtime

Run the existing truth-fusion provider:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --mode qa --profile body_gold_threequarter_review --heavy-provider segformer_body_truth_fusion
```

The resulting review-only score will expose:

- `clothing_invariant_score`
- `clothing_invariant_confidence`
- `garment_occlusion_index`
- `garment_boundary_risk`
- `visible_body_surface_alignment`
- `occlusion_adjusted_truth_score`

## Governance Rule

High clothing-invariant score means:

```text
The visible evidence still supports the two absolute truths under clothing variation.
```

It does not mean:

```text
The image is approved for training admission or final image-set membership.
```

Training admission and final image-set construction remain external decisions. This repository only reports clothing-invariant evidence.
