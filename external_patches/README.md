# External Patch Bundle

This directory preserves local changes made inside nested external repositories
without vendoring their full contents into the main repository.

## Why This Exists

- `external/3DDFA-V3` and `external/4D-Humans` are separate Git repositories.
- They also contain model assets that are too large for normal GitHub pushes.
- The main repository therefore tracks:
  - patch files for tracked source changes
  - helper scripts added locally
  - instructions for rebuilding the same local setup

## Included Bundles

### `3DDFA-V3`

- Patch file:
  - `external_patches/3DDFA-V3/local_modifications.patch`
- Added helper script copy:
  - `external_patches/3DDFA-V3/demo_lite_export.py`

Apply from the external repo root:

```powershell
git -C external/3DDFA-V3 apply ..\\..\\external_patches\\3DDFA-V3\\local_modifications.patch
Copy-Item external_patches\\3DDFA-V3\\demo_lite_export.py external\\3DDFA-V3\\demo_lite_export.py
```

Required external assets are not stored here:

- `assets/face_model.npy`
- `assets/large_base_net.pth`
- `assets/net_recon.pth`
- `assets/retinaface_resnet50_2020-07-20_old_torch.pth`
- `assets/similarity_Lm3D_all.mat`

### `4D-Humans`

- Patch file:
  - `external_patches/4D-Humans/local_modifications.patch`
- Added helper script copy:
  - `external_patches/4D-Humans/demo_xiaona_export.py`

Apply from the external repo root:

```powershell
git -C external/4D-Humans apply ..\\..\\external_patches\\4D-Humans\\local_modifications.patch
Copy-Item external_patches\\4D-Humans\\demo_xiaona_export.py external\\4D-Humans\\demo_xiaona_export.py
```

Required external assets are not stored here:

- `data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- downloaded 4D-Humans checkpoints and cache files

## Expected Workflow

1. Clone the external repositories into `external/`.
2. Download the required upstream assets separately.
3. Apply the local patch file.
4. Copy the helper export script into the external repo.
5. Run the project as usual.
