# Algorithm Policy

## Confirmed Route
1. Engineering decoupling first
2. High-order algorithms later
3. First algorithm upgrade = Human Parsing

## Provider Strategy
- keep legacy logic as default provider first
- add new algorithms through stable interfaces

## Integration Order
- Human Parsing
- heel / foot-index support
- Hand Landmarker
- YOLO for ROI and clutter cleanup only
- SAM for mask refinement only
- SMPL-X shortlist only

## Explicit Non-Route
- do not replace the whole system with YOLO + SAM first

