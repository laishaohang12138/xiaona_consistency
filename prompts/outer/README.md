# OUTER Prompt Asset

## Import Status
- `v0.3.0-nb2-8x3` raw shortlist import has been fully landed into the repo.
- The current state is a governed inactive asset.
- This folder is not yet the active OUTER runtime pack.
- Machine-readable governance lives in `manifest.yaml`.

## Current Contents
- Imported source files live under `assembled_shortlist_v0_3_0_nb2_8x3/`.
- The landed shortlist now covers `OT-A01~H03` across eight outerwear families.
- Family coverage:
  - `OT-A01~A03` blazer
  - `OT-B01~B03` collarless jacket
  - `OT-C01~C03` overshirt
  - `OT-D01~D03` cardigan
  - `OT-E01~E03` trench
  - `OT-F01~F03` rigid coat
  - `OT-G01~G03` short jacket
  - `OT-H01~H03` long panel coat

## Governance Note
- Do not bind OUTER as active runtime scope until the full runtime formula, negative prompts, and release gates are assembled.
- Preserve original source filenames during import; normalize naming only after the full pack is present.
- OUTER may be used for review-only clothing / occlusion invariance replay while `runtime_active=false`.
- OUTER must not be used for training admission or winner-bank bootstrap under the current policy.
