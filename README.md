# xiaona_consistency

XiaoNa LoRA training engineering repository.

## Current Scope
- Entrypoint CLI in [check_consistency.py](./check_consistency.py)
- Main QA implementation in `core/`
- Frozen project rules in `docs/`
- Machine-readable project config in `configs/`
- Versioned prompt assets in `prompts/`
  `body_gold/` is active and `bridge/` is now landed as a scoped prompt asset pack
- Custom GPT knowledge export in `kb_export/`
- Anchor assets in `anchors/`
- Offline benchmark replay support for threshold tuning
- Offline Optuna tuning support on top of benchmark replay
- Preset-driven Optuna modes in `configs/optuna_mode_presets.json`

## Runtime Status
- The QA pipeline is modularized into `core/qa_runtime.py`, `core/qa_pipeline.py`, `core/qa_scoring.py`, `core/qa_consistency.py`, `core/qa_features.py`, and `core/providers.py`
- Human Parsing is the default upstream provider for `subject_mask` and `skin_region`, with legacy fallback retained
- BODY GOLD currently runs as a conservative front-core lane, while side/back work is staged through shadow profile lanes
- BRIDGE `v0.3.1-nb2` prompt assets are landed in `prompts/bridge/`, but QA/runtime governance is still BODY GOLD centric
- `outputs/qa_report.json` now writes `report_meta` plus `items`, including provider policy, anchor snapshot, layer quota snapshot, and threshold hash
- Benchmark replay can re-score a saved `qa_report.json` under threshold overrides without rerunning vision models
- Optuna search can optimize replay metrics from `configs/optuna_search_space.template.json` without touching the main QA path
- `configs/optuna_guard.json` keeps Optuna locked until frozen benchmark labels and anchor coverage are ready
- `configs/optuna_mode_presets.json` provides user-facing review / front / 3q / side / full-release fit presets

## Confirmed Route
1. Engineering decoupling first
2. Human Parsing as the first high-order algorithm
3. Hand/foot and body-measure providers before heavier 3D escalation
4. YOLO and SAM only as later support layers
5. Custom GPT knowledge ingests frozen rule documents only

## Repository Notes
- `input/`, `outputs/`, and `calib_pass/` are excluded from Git as local/dynamic data
- `.venv/` and IDE files are excluded from Git
