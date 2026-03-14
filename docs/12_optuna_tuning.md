# Optuna Tuning

## Purpose
- Run offline parameter search on top of `benchmark_report()`
- Tune thresholds and fusion weights without rerunning face / pose / parsing models
- Keep search isolated from the main QA pipeline

## Install
- Project venv:
  - `.\.venv\Scripts\pip.exe install optuna`

## CLI
- Basic run:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json`
- Save study summary and best override:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json --optuna-output outputs/optuna_study_result.json --optuna-best-override-out outputs/optuna_best_override.json`
- Persist study to sqlite:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json --optuna-storage-path outputs/optuna_study.db`
- Override the guard file explicitly:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.frozen.json --optuna-search-space configs/optuna_search_space.front_core.json --optuna-guard-path configs/optuna_guard.json`

## Guard
- `Optuna` is now guarded by `configs/optuna_guard.json`
- Default state is locked: candidate-review benchmark files must not feed parameter fitting
- Even after unlocking, the run is still blocked unless all of these are true:
  - label file sets `dataset_role=benchmark_frozen`
  - label file sets `optuna_ready=true`
  - anchor registry covers the required view buckets declared in the guard file
- This keeps daily shot-selection runs and pre-anchor-fill review data out of the tuning source by default

## Unlock Sequence
1. Finish the missing anchor lanes and verify `configs/anchor_registry.yaml`
2. Create a separate frozen label set, not the daily candidate-review file
3. Set frozen labels to `dataset_role=benchmark_frozen` and `optuna_ready=true`
4. Flip `optuna_locked` to `false` in `configs/optuna_guard.json`
5. Run lane-specific Optuna presets instead of mixing front / 3q / side shadow into one study

## Recommended Presets
- Front mainline:
  - `configs/optuna_search_space.front_core.json`
- Three-quarter review lane:
  - `configs/optuna_search_space.three_quarter_review.json`
- Side-90 shadow lane:
  - `configs/optuna_search_space.side90_shadow.json`

## Search Space Shape
```json
{
  "schema_version": "qa_optuna_search_space_v1",
  "objective": {
    "metric_path": "metrics.release_safety_score",
    "direction": "maximize"
  },
  "study": {
    "name": "body_gold_release_safety",
    "sampler": "tpe",
    "seed": 42,
    "n_trials": 40
  },
  "fixed_override": {},
  "parameters": [
    {
      "name": "overall_pass",
      "path": "profile_thresholds.overall_pass",
      "type": "float",
      "low": 0.68,
      "high": 0.82,
      "step": 0.01
    }
  ],
  "constraints": [
    {
      "type": "order",
      "lower_path": "profile_thresholds.overall_warn",
      "upper_path": "profile_thresholds.overall_pass",
      "min_gap": 0.02
    }
  ]
}
```

## Supported Parameter Types
- `float`
- `int`
- `categorical`

## Supported Constraints
- `order`
  - Enforces `lower_path <= upper_path - min_gap`

## Supported Objective Paths
- Any numeric path inside the benchmark result JSON
- Common choices:
  - `metrics.release_safety_score`
  - `metrics.macro_f1`
  - `metrics.pass_precision`
  - `agreement_metrics.view_lane_accuracy`
  - `group_metrics.view_lane.front.metrics.release_safety_score`
  - `group_metrics.view_lane.three_quarter.metrics.release_safety_score`
  - `group_metrics.task_profile.body_gold_side90_shadow.metrics.release_safety_score`

## Notes
- `fixed_override` is merged before trial values
- CLI `--threshold-override-file/json` is merged after `fixed_override`, so it can pin part of the search
- `--optuna-storage-path` uses `load_if_exists=True`, so trial counts in the result are cumulative for that sqlite study
- Candidate-review labels still work in `--mode benchmark`; the guard only blocks `--mode optuna`
- Search paths reuse the existing threshold override schema, for example:
  - `profile_thresholds.overall_pass`
  - `profile_weights.face`
  - `consistency.warn_threshold.constitution_soft`
  - `consistency.score_fusion.overall.confidence_floor`
- Current implementation is single-objective only
- Mixed-profile benchmark sets are supported, but top-level `profile_thresholds` / `profile_weights` apply to every item profile inside replay
