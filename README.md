# xiaona_consistency

XiaoNa LoRA candidate screening and consistency-evidence repository.

## Current Scope
- Entrypoint CLI in [check_consistency.py](./check_consistency.py)
- Main QA implementation in `core/`
- Frozen project rules in `docs/`
- Machine-readable project config in `configs/`
- Versioned prompt assets in `prompts/`
  `body_gold/` is active; `bridge/`, `neckline/`, `outer/`, and `face_lock/` are landed as scoped prompt asset packs with separate governance
- Custom GPT knowledge export in `kb_export/`
- Anchor assets in `anchors/`
- Offline benchmark replay support for threshold tuning
- Offline Optuna tuning support on top of benchmark replay
- Preset-driven Optuna modes in `configs/optuna_mode_presets.json`

## Runtime Status
- The QA pipeline is modularized into `core/qa_runtime.py`, `core/qa_pipeline.py`, `core/qa_scoring.py`, `core/qa_consistency.py`, `core/qa_features.py`, and `core/providers.py`
- Human Parsing is the default upstream provider for `subject_mask` and `skin_region`, with legacy fallback retained
- BODY GOLD currently runs as a conservative front-core lane, while side/back work is staged through shadow profile lanes
- BRIDGE `v0.3.1-nb2` and NECKLINE `v0.1.1-nb-clean` prompt assets are landed in `prompts/bridge/` and `prompts/neckline/`, but QA/runtime governance is still BODY GOLD centric
- FACE_LOCK `v0.1.1-nb2-topology-16x1-copyready` is landed in `prompts/face_lock/` as review-only structured face-layer prompt evidence
- `outputs/qa_report.json` now writes `report_meta` plus `items`, including provider policy, anchor snapshot, layer quota snapshot, and threshold hash
- Benchmark replay can re-score a saved `qa_report.json` under threshold overrides without rerunning vision models
- Optuna search can optimize replay metrics from `configs/optuna_search_space.template.json` without touching the main QA path
- `configs/optuna_guard.json` keeps Optuna locked until project optimization, frozen benchmark labels, and anchor coverage are ready
- `configs/optuna_mode_presets.json` provides user-facing review / front / 3q / side / full-release fit presets
- Winner bank is currently mutable review memory, not frozen release truth or a fitting source
- Body truth is pose/gait-aware: `Task-63987060-116-1.png` remains the only body truth while pose-sensitive deltas are interpreted separately
- Side/back evidence now carries same-truth projection confidence and uncertainty; derived projections do not create new truth anchors
- Final training-set admission and final image-set construction are outside this repository; outputs are screening, review-priority ranking, risk routing, and evidence packets only
- `prepare_replay_collection_plan` turns manifest, lighting, OUTER, and side/back topology gaps into the next controlled replay collection queue
- `prepare_topology_replay_pack` scaffolds controlled side/back topology replay directories under `input_replay/topology/`

## Confirmed Route
1. Engineering decoupling first
2. Human Parsing as the first high-order algorithm
3. Hand/foot and body-measure providers before heavier 3D escalation
4. YOLO and SAM only as later support layers
5. Custom GPT knowledge ingests frozen rule documents only

## Repository Notes
- `input/`, `outputs/`, and `calib_pass/` are excluded from Git as local/dynamic data
- `.venv/` and IDE files are excluded from Git

## Handoff Pack
- Use [docs/14_handoff_index.md](./docs/14_handoff_index.md) when opening a new chat window.
- The recommended read order is:
  `14_handoff_index.md -> 15_project_memory.md -> 16_current_state.md -> 17_next_actions.md`
- This pack exists so workflow context does not depend on a single long chat window.
