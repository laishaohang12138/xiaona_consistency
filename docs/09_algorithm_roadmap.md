# Algorithm Roadmap

## Confirmed Route
1. Engineering decoupling first
2. High-order algorithms added later through interfaces
3. First algorithm upgrade: Human Parsing

## Why Decouple First
- Current logic is concentrated in one large script
- Rules, thresholds, prompt doctrine, and runtime heuristics are mixed together
- Directly adding large models now would increase coupling and reduce rollback safety

## Provider Interfaces To Introduce
- `SubjectMaskProvider`
- `SkinRegionProvider`
- `HandFootProvider`
- `BodyMeasureProvider`

## Integration Order
1. Legacy providers stay as default
2. Human Parsing replaces the upstream region source for constitution and skin
3. Heel / foot-index support is added to lower-body QA
4. Hand Landmarker is added for hand-end anomaly checks
5. YOLO is used for ROI and clutter cleanup only
6. SAM is used for mask refinement only
7. SMPL-X / Anthropometry runs on shortlist only

## Explicit Non-Route
- Do not replace the pipeline wholesale with YOLO + SAM
- Do not make high-order algorithms the first step
- Do not move dynamic batch reasoning into Custom GPT knowledge

