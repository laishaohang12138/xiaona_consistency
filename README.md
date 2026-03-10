# xiaona_consistency

XiaoNa LoRA training engineering repository.

## Current Scope
- QA pipeline in [check_consistency.py](./check_consistency.py)
- Frozen project rules in `docs/`
- Machine-readable project config in `configs/`
- Versioned prompt assets in `prompts/`
- Custom GPT knowledge export in `kb_export/`
- Anchor assets in `anchors/`

## Confirmed Route
1. Engineering decoupling first
2. Human Parsing as the first high-order algorithm
3. YOLO and SAM only as later support layers
4. Custom GPT knowledge ingests frozen rule documents only

## Repository Notes
- `input/`, `outputs/`, and `calib_pass/` are excluded from Git as local/dynamic data
- `.venv/` and IDE files are excluded from Git

