# DensePose WSL Deployment

## Why WSL

DensePose is maintained as part of the official Detectron2 project.
The official Detectron2 installation path is source build oriented and the
current Windows runtime in this repository does not have the compiler toolchain
needed for that build.

Current local facts:

- Windows `.venv` has PyTorch CUDA and can run SAM2.
- Windows runtime does not have `cl`.
- Windows runtime does not have `nvcc`.
- WSL Ubuntu is installed, but the distro runtime still needs to be verified
  before running non-interactive bootstrap commands.

Because of that, the recommended deployment path is:

```text
WSL Ubuntu -> official Detectron2 + DensePose -> export DensePose JSON -> convert to .densepose.json sidecar -> consume from clothing_surface_occlusion_bridge
```

## Prepare Bootstrap

Write the bootstrap script and manifest into the repository:

```powershell
.\.venv\Scripts\python.exe .\deploy_surface_occlusion_densepose_wsl.py
```

Files created:

```text
external\models\DensePose\bootstrap_densepose_wsl.sh
external\models\DensePose\bootstrap_densepose_wsl.json
```

## Attempt Automatic Bootstrap

If WSL is fully usable, run:

```powershell
.\.venv\Scripts\python.exe .\deploy_surface_occlusion_densepose_wsl.py --run
```

The bootstrap script will:

1. install Linux build/runtime packages
2. clone the official Detectron2 repository
3. create a WSL Python venv
4. install `torch` and `torchvision`
5. install official Detectron2 from source
6. install official DensePose from `detectron2/projects/DensePose`
7. download a baseline DensePose checkpoint

## DensePose Output Role

DensePose is not a new identity truth source.

It should only provide surface evidence such as:

- `visible_body_surface_alignment`
- `visible_body_ratio`
- `visible_arm_ratio`
- `visible_leg_ratio`
- `clothing_surface_confidence`

Those values are then consumed by:

```text
core/qa_heavy_surface_occlusion.py
```

## Sidecar Conversion

Once DensePose produces a dump or an external JSON summary, convert it into a
project sidecar.

Direct image-to-sidecar path from Windows:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_densepose.py `
  --image D:\xiaona_consistency\input\example.png `
  --output D:\xiaona_consistency\input\example.png.densepose.json
```

Batch path:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_densepose.py `
  --input-dir D:\xiaona_consistency\input `
  --output-dir D:\xiaona_consistency\input
```

If you already have an official DensePose dump `.pkl`, convert it directly:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_densepose.py `
  --dump-file D:\xiaona_consistency\outputs\densepose_smoke.pkl `
  --source-image D:\xiaona_consistency\input\example.png `
  --output D:\xiaona_consistency\input\example.png.densepose.json
```

For precomputed external JSON summaries, the generic converter is still valid:

```powershell
.\.venv\Scripts\python.exe .\export_surface_occlusion_artifact.py `
  --input D:\densepose_exports\example.json `
  --output D:\xiaona_consistency\input\example.png.densepose.json `
  --source-image D:\xiaona_consistency\input\example.png `
  --model-id densepose_detectron2 `
  --provider-name densepose_surface_occlusion
```

The bridge will automatically consume:

```text
example.png.densepose.json
```

## Governance Boundary

DensePose helps answer:

```text
How much real body surface is still visible under clothing variation?
```

It does not answer:

```text
Who defines XiaoNa?
```

Face truth remains `A-Core_01_0deg_MASTER`.
Body truth remains `Task-63987060-116-1`.
