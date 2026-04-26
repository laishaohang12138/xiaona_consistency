# XiaoNa Training Project Charter

## Project Identity
- Project name: XiaoNa LoRA Training Engineering
- Single scope: this repository and its linked anchor, prompt, QA, and batch review assets
- Primary objective: build a high-purity, reusable, iteratively upgradable screening and evidence system for XiaoNa candidate batches

## Version Roadmap
1. XiaoNa v1.0 Core
   - Strong identity consistency
   - Stable body architecture
   - Production-safe training assets
2. XiaoNa v2.0
   - Better spatial thickness
   - Better dressed-body stability
   - Better 3D consistency
3. XiaoNa v3.0
   - More human liveliness
   - Controlled micro-variation
   - Better "alive person" feeling

## Current Priority
- Freeze and stabilize v1.0 Core first
- Do not trade identity or body architecture for style, beauty, or novelty

## Working Scope
- Training asset planning
- BODY GOLD / BRIDGE / NECKLINE / OUTER / FACE LOCK progression
- Anchor usage rules
- Prompt architecture maintenance
- QA review and final arbitration
- Candidate screening, risk routing, and evidence packaging
- Batch retrospectives
- Patch decisions
- Version naming and release discipline

## Non-Goals
- General image chatting
- Cross-project creative work
- Style-first prompt experimentation
- Large unsupervised rule rewrites
- Final training-set admission inside this repository

## Tool Roles
- ChatGPT / Codex
  - Architecture, QA policy, prompt structure, patch strategy, version design
- Gemini
  - Candidate image generation workshop
- Nano Banana Pro / Nano Banana 2
  - High-consistency candidate generation and comparison
- ComfyUI / local environment
  - Batch generation, training, QA, regression checking

## Frozen Decision
- Route order is fixed for now:
  1. Engineering decoupling
  2. Static rules and prompt assets frozen into files
  3. High-order algorithms added through provider interfaces
  4. First algorithm upgrade is Human Parsing

## Repository Layers
- `docs/`: frozen project rules and operating doctrine
- `configs/`: machine-readable project configuration
- `prompts/`: versioned prompt assets
- `kb_export/`: static documents prepared for Custom GPT knowledge ingestion

## Source of Truth Policy
- Repository files are the primary source of truth
- Custom GPT knowledge is a published read-only mirror of frozen rules
- Dynamic batch data is not a source of truth
- `winner_bank` is mutable review memory in the current phase, not frozen truth
- final training-set admission belongs to the external training decision flow, not this repository
- Parameter fitting is disabled until project optimization is complete
- `A-Core_01_0deg_MASTER.png` is the only face truth
- `Task-63987060-116-1.png` is the only body truth, interpreted with pose/gait-aware consistency
