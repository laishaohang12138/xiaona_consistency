# Optuna Tuning

## Purpose
- Run offline parameter search on top of `benchmark_report()`
- Tune thresholds and fusion weights without rerunning face / pose / parsing models
- Keep search isolated from the main QA pipeline

## Install
- Project venv:
  - `.\.venv\Scripts\pip.exe install optuna`

## CLI
- List available presets:
  - `python check_consistency.py --optuna-list-presets`
- Generate the replay report with the preset's recommended runtime profile first:
  - `python check_consistency.py --profile body_gold_threequarter_review`
- Basic run:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json`
- Run with a preset:
  - `python check_consistency.py --mode optuna --optuna-preset front_core_fit --benchmark-labels configs/benchmark_labels.frozen.json --optuna-output outputs/optuna_front_result.json --optuna-best-override-out outputs/optuna_front_best.json`
- Save study summary and best override:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json --optuna-output outputs/optuna_study_result.json --optuna-best-override-out outputs/optuna_best_override.json`
- Persist study to sqlite:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.local.json --optuna-search-space configs/optuna_search_space.template.json --optuna-storage-path outputs/optuna_study.db`
- Override the guard file explicitly:
  - `python check_consistency.py --mode optuna --benchmark-labels configs/benchmark_labels.frozen.json --optuna-search-space configs/optuna_search_space.front_core.json --optuna-guard-path configs/optuna_guard.json`

## Guard
- `Optuna` is now guarded by `configs/optuna_guard.json`
- User-facing presets are defined in `configs/optuna_mode_presets.json`
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
- `review_only`
  - Candidate-review only, blocks fitting
- `front_core_fit`
  - Front mainline fit
  - Uses `configs/optuna_search_space.front_core.json`
- `three_quarter_fit`
  - Three-quarter review fit
  - Uses `configs/optuna_search_space.three_quarter_review.json`
- `side90_shadow_fit`
  - Side-90 shadow fit
  - Uses `configs/optuna_search_space.side90_shadow.json`
- `back180_shadow_fit`
  - Back-180 shadow fit
  - Uses `configs/optuna_search_space.back180_shadow.json`
- `full_release_fit`
  - Full cross-lane release fit
  - Uses `configs/optuna_search_space.template.json`

## Input Collection Rules
- `review_only`
  - Include daily candidate-review batches, mixed candidate lanes, and temporary shortlist observations
  - Exclude parameter fitting entirely
- `front_core_fit`
  - Include frozen front-only full-body benchmark images, front false-warn / false-pass regressions, and production-like front compositions
  - Exclude `three_quarter`, `side_90`, `back_180`, cropped headshots, and daily candidate-review batches
- `three_quarter_fit`
  - First generate the source report with `--profile body_gold_threequarter_review`
  - Include frozen `three_quarter` benchmark images with stable upper/full-body evidence and manually sealed soft-review regressions
  - Exclude front-only release images, side shadow candidates, and unfrozen exploratory samples
- `side90_shadow_fit`
  - Include frozen `side_90` full-body benchmark images with feet in frame and manually confirmed side-shadow regressions
  - Exclude side headshots, mixed front/3q release images, and unfrozen side candidates
- `back180_shadow_fit`
  - Include frozen `back_180` full-body benchmark images with complete back contour, full lower-body evidence, and feet in frame
  - Exclude mixed front/3q/side samples, cropped back views, and unfrozen back candidates
- `full_release_fit`
  - Include a balanced frozen benchmark covering `front`, `three_quarter`, `side_90`, and `back_180`
  - Exclude candidate-review data and any lane that is not yet anchor-frozen

## Single-Lane Vs Global Fit
- Single-lane fit can affect the whole system if you tune shared parameters and then promote them globally
- Safe rule:
  - Lane-specific `best_override` should stay lane-local first
  - Only promote a lane-specific override into a broader release config after cross-lane benchmark validation
- `front_core_fit`
  - This is closest to the main release lane, so its fitted thresholds are the most likely to influence final release behavior
- `three_quarter_fit`
  - It is now isolated through `body_gold_threequarter_review`, so it no longer has to mutate the front mainline directly
  - Even so, treat it as review-lane optimization first, not as an automatic global promotion, because it still tunes shared consistency/fusion surfaces
- `side90_shadow_fit` / `back180_shadow_fit`
  - These are safer to keep isolated because they correspond to dedicated shadow profiles
  - Their fitted threshold overrides should remain scoped to `body_gold_side90_shadow` / `body_gold_back180_shadow` first
- `full_release_fit`
  - This is the only preset intended to optimize a final cross-lane merged release surface

## Back-180 Handling
- `back_180` should not be treated as a face-driven lane
- Preferred fitting target:
  - `body_gold_back180_shadow`
- Preferred features to tune:
  - upper/full thresholds
  - overall thresholds
  - depth-lite thresholds
  - `profile_like` upper/full geometric fusion weights
- Do not overfit back_180 on face similarity:
  - the profile already sets `face=0.00`
  - back fitting should focus on body contour, posture, framing, leg geometry, and back-view consistency evidence

## What Goes Into `input/`
- `Optuna` itself does not read `input/`; it replays a saved `outputs/qa_report.json`
- The practical workflow is:
  1. Put the candidate image set for one lane into `input/`
  2. Run QA once with the preset's recommended runtime profile to produce `outputs/qa_report.json`
  3. Export or maintain the matching benchmark label file
  4. Run `benchmark` or `optuna` on that saved report
- Because of that, the real fitting source is `qa_report.json + benchmark_labels.frozen.json`, not the live `input/` directory

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
  - `group_metrics.task_profile.body_gold_threequarter_review.metrics.release_safety_score`
  - `group_metrics.task_profile.body_gold_side90_shadow.metrics.release_safety_score`
  - `group_metrics.task_profile.body_gold_back180_shadow.metrics.release_safety_score`

## Notes
- Preset mode is safer than manually mixing `--optuna-search-space` and `--optuna-guard-path`
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
