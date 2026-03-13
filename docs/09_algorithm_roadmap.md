# Algorithm Roadmap

## Confirmed Route
1. Engineering decoupling first
2. High-order algorithms added later through interfaces
3. First algorithm upgrade: Human Parsing
4. Side/back BODY GOLD work enters through shadow lanes before quota promotion

## Current Runtime Reality
- `core/` is now the main implementation surface; `check_consistency.py` is a CLI wrapper
- Human Parsing is already the default upstream region source for constitution and skin
- Legacy segmentation remains as fallback, not as the preferred default
- BODY GOLD front core remains conservative; `side_90` and `back_180` are governance targets, not fully unlocked runtime lanes

## Why Decouple First
- Rules, thresholds, prompt doctrine, and runtime heuristics must stay patchable without replacing the whole pipeline
- Rollback safety matters more than squeezing new models into the main loop
- Side/back expansion requires profile, anchor, quota, and report governance before heavier model upgrades

## Provider Interfaces To Introduce
- `SubjectMaskProvider`
- `SkinRegionProvider`
- `HandFootProvider`
- `BodyMeasureProvider`

## Integration Order
1. Human Parsing stays as the default upstream provider
2. Tone-anchor semantics are separated from identity-anchor semantics
3. Heel / foot-index support is added to lower-body QA
4. Hand Landmarker is added for hand-end anomaly checks
5. BodyMeasureProvider is added for sagittal / posterior geometry checks
6. YOLO is used for ROI and clutter cleanup only
7. SAM is used for mask refinement only
8. SMPL-X / Anthropometry runs on shortlist only

## Explicit Non-Route
- Do not replace the pipeline wholesale with YOLO + SAM
- Do not make high-order algorithms the first step
- Do not treat side/back shadow lanes as released BODY GOLD quota before anchor coverage and report evidence are stable
- Do not move dynamic batch reasoning into Custom GPT knowledge
