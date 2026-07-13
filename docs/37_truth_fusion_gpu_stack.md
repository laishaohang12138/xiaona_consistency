# Truth Fusion GPU Stack

## Scope

This project can run the maximum-evidence review stack for batch screening:

- `segformer_body_truth_fusion`
- HMR2 / 4D-Humans body canonical truth bridge
- DensePose / SAM2 clothing surface occlusion sidecars
- 3DDFA-V3 face canonical shadow bridge
- InsightFace `buffalo_l` identity embedding through ONNXRuntime CUDA

The stack remains review evidence only. It does not decide final image-set
membership or training admission.

## Default Review Mode

Interactive `shot_review` now defaults to:

```text
heavy_provider = segformer_body_truth_fusion
device_policy = cuda
surface_occlusion_auto = densepose,sam2
```

Equivalent explicit command:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py `
  --workflow shot_review `
  --profile body_gold_fullbody `
  --heavy-provider segformer_body_truth_fusion `
  --device-policy cuda `
  --surface-occlusion-auto densepose,sam2
```

Use `--require-gpu` when CPU fallback should be treated as a hard failure:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py `
  --workflow shot_review `
  --heavy-provider segformer_body_truth_fusion `
  --device-policy cuda `
  --require-gpu
```

To raise pose/gait and body-canonical evidence coverage beyond the default
shortlist, run full-group Top-N heavy review:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py `
  --workflow shot_review `
  --profile body_gold_fullbody `
  --heavy-provider segformer_body_truth_fusion `
  --heavy-candidate-mode full_group `
  --heavy-max-candidates 20 `
  --device-policy cuda `
  --surface-occlusion-auto densepose,sam2 `
  --require-gpu
```

Use `--heavy-max-candidates 0` only when the whole group should receive heavy
evidence. For large Nano Banana batches, Top20 is the preferred daily reliability
mode before any full-batch expensive replay.

## Environment Variables

The CLI sets these defaults for truth-fusion GPU review:

```text
XIAONA_INSIGHTFACE_DEVICE=cuda
XIAONA_HUMAN_PARSING_DEVICE=cuda
XIAONA_SEGFORMER_DEVICE=cuda
XIAONA_HMR2_DEVICE=cuda
XIAONA_3DDFA_V3_DEVICE=cuda
XIAONA_SURFACE_OCCLUSION_DEVICE=cuda
XIAONA_SAM2_DEVICE=cuda
XIAONA_DENSEPOSE_DEVICE=cuda
XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT=densepose,sam2
```

## Surface Occlusion

`clothing_surface_occlusion_bridge` still consumes existing sidecars first.
When `XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT` is enabled and no sidecar exists,
it tries exporters in order.

For maximum evidence:

```text
densepose,sam2
```

For local Windows-only review when DensePose WSL is not ready:

```text
sam2
```

For sidecar-only review:

```text
off
```

## Readiness Check

Run:

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py --workflow setup_external_models
```

The status output reports:

- 3DDFA repo and assets
- 4D-Humans repo and SMPL file
- Torch CUDA availability
- ONNXRuntime CUDAExecutionProvider availability
- SAM2 local checkpoint
- DensePose WSL bootstrap files

## Interpretation

GPU readiness raises evidence confidence and coverage. It does not add new
truth anchors:

- face truth remains `A-Core_01_0deg_MASTER.png`
- body truth remains `Task-63987060-116-1.png`
- pose and gait remain projection factors before body drift is called
