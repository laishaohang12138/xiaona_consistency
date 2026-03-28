# HMR2 Body Canonical Direct Integration

## Purpose

This provider upgrades `body_canonical_hmr2` from a pure artifact reader into a direct bridge:

- it can try external `HMR2 / 4D-Humans` export first
- if direct export is not configured or fails, it still accepts the existing artifact workflow
- it does not change truth governance

`116-1` is still the only body truth anchor.

## Provider name

```text
body_canonical_hmr2
```

## Runtime behavior

The provider now has two valid input modes:

1. Existing artifact mode
   - `outputs/master_truth/body_master_shape_only.json`
   - `input/*.body_canonical.json`

2. Direct bridge mode
   - run an external HMR2 command
   - collect a raw `.pkl/.json/.npz/.npy`
   - convert it into the same artifact schema internally
   - reuse the current scoring path

## Important constraint

Unlike the face-side `3DDFA-V3` integration, this provider does **not** assume a stable official single-image export CLI.

The recommended way is to provide a command template that writes raw exports into a known output directory.

## Default local integration

The repository now includes a minimal local entrypoint:

```text
export_body_canonical_direct_hmr2.py
```

If the following are present, the provider can try local direct export without any command template:

- `external/4D-Humans/`
- `export_body_canonical_direct_hmr2.py`
- `data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
  or `external/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

GPU is used automatically when `torch.cuda.is_available()` is true.

## Optional environment overrides

```powershell
$env:XIAONA_HMR2_REPO="D:\models\4D-Humans"
$env:XIAONA_HMR2_PYTHON="D:\venvs\hmr2\python.exe"
$env:XIAONA_HMR2_ENTRYPOINT="D:\xiaona_consistency\export_body_canonical_direct_hmr2.py"
$env:XIAONA_HMR2_DEVICE="cuda"
```

If you need a custom export wrapper, you may still provide a full template:

```powershell
$env:XIAONA_HMR2_CMD_TEMPLATE="{python} {entrypoint} --image_path {image_path} --output_dir {output_dir} --device {device}"
```

Supported placeholders:

- `{python}`
- `{entrypoint}`
- `{input_dir}`
- `{output_dir}`
- `{image_path}`
- `{repo}`
- `{device}`

## Output expectation

The command template must write at least one raw export file under:

```text
outputs/heavy_evidence_cache/body_canonical_hmr2/direct_runs/<hash>/output
```

Supported formats:

- `.pkl`
- `.json`
- `.npz`
- `.npy`

The raw export should contain fields that can be mapped to:

- `shape_beta`
- `pose_vector` or `global_orient + body_pose`
- optional `fit_confidence`
- optional `coverage`
- optional `canonical_measurements`

## Fallback behavior

If direct export is not configured or blocked:

- the provider continues reading existing artifacts
- if artifacts are also missing, it returns structured `unavailable` evidence
- QA does not crash

## Recommended rollout

1. Keep the current artifact workflow as the stable baseline
2. Add `XIAONA_HMR2_CMD_TEMPLATE`
3. Validate one `116-1` master export
4. Validate a small candidate batch
5. Only then use `segformer_body_truth_fusion` for wider review
