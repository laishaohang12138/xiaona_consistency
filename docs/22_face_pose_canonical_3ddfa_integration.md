# 3DDFA-V3 Face Canonical Direct Integration

## Purpose

This provider lets the project try direct `3DDFA-V3` inference for face canonical shadow evidence before falling back to the existing artifact bridge.

- Truth anchor does not change.
- `A-Core_01` is still the only face truth.
- This provider is still `shadow-only`.
- If direct inference is unavailable, the runtime falls back to `face_pose_canonical_bridge`.

## Provider name

```text
face_pose_canonical_3ddfa
```

## Default behavior

The runtime now requests:

```text
face_canonical: face_pose_canonical_3ddfa
```

If `3DDFA-V3` is not available locally, the provider does not break QA. It returns direct-export guidance and continues through the existing bridge path.

## Recommended local layout

Clone the official repository into:

```text
external/3DDFA-V3
```

Expected default entrypoint:

```text
external/3DDFA-V3/demo_lite_export.py
```

The project now prefers a local renderer-free lite exporter instead of the official `demo.py`.

- It still uses the official `3DDFA-V3` weights and detectors.
- It avoids the CPU renderer build step.
- It writes project-readable JSON directly.
- If `demo_lite_export.py` is absent, the provider falls back to `demo.py`.

## Optional environment variables

If the repo is not stored under the default path, set:

```powershell
$env:XIAONA_3DDFA_V3_REPO="D:\models\3DDFA-V3"
```

If a different Python interpreter is required:

```powershell
$env:XIAONA_3DDFA_V3_PYTHON="D:\venvs\3ddfa\python.exe"
```

If the direct CLI needs extra arguments:

```powershell
$env:XIAONA_3DDFA_V3_EXTRA_ARGS="--onnx --mode cpu"
```

If you need a fully custom command, use:

```powershell
$env:XIAONA_3DDFA_V3_CMD_TEMPLATE="{python} {entrypoint} --inputpath {input_dir} --savepath {output_dir}"
```

Supported placeholders:

- `{python}`
- `{entrypoint}`
- `{input_dir}`
- `{output_dir}`
- `{image_path}`
- `{repo}`

## Runtime behavior

When direct inference works:

1. The provider exports canonical data for `A-Core_01` if `outputs/master_truth/face_master_canonical.json` is missing.
2. The provider exports candidate canonical data for the current image if no cached sidecar exists.
3. The provider then reuses the existing bridge scoring logic.

This keeps scoring stable while still moving the project toward direct `3DDFA-V3` integration.

## Local prerequisites

The repo alone is not enough. The local clone also needs these model assets under:

```text
external/3DDFA-V3/assets
```

Required files:

- `face_model.npy`
- `large_base_net.pth`
- `net_recon.pth`
- `retinaface_resnet50_2020-07-20_old_torch.pth`
- `similarity_Lm3D_all.mat`

## Fallback behavior

If direct inference cannot run:

- existing `face_pose_canonical_bridge` sidecars are still used
- if no sidecars exist, the provider returns structured `unavailable` evidence
- the QA run does not crash

## Output interpretation

This provider adds or reuses:

- `face_pose_normalization_confidence`
- `canonical_face_landmark_similarity`
- `canonical_face_identity_similarity`
- `pose_delta_similarity`
- `visible_face_coverage`
- `frontalization_quality`
- `pose_fit_confidence`

Identity similarity is still backed by the project's current face embedding chain. `3DDFA-V3` is used here to improve face canonical geometry and normalization, not to redefine identity truth.
