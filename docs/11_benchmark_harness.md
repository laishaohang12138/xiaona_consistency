# Benchmark Harness

## Purpose
- Provide a low-cost offline replay path for threshold tuning
- Reuse `outputs/qa_report.json` instead of rerunning image models for every sweep
- Prepare a stable objective surface for later Optuna integration

## What It Replays
- Module thresholds from `task_profiles`
- Profile weights for `face / upper / full`
- Consistency gates from `configs/consistency_thresholds.yaml`
- BODY CONSTITUTION scoring ranges and weights
- DEPTH 3D LITE scoring ranges and weights
- Face-quality warning flags derived from report debug fields

## What It Does Not Replay
- Raw face embedding extraction
- Raw Human Parsing / MediaPipe inference
- Any missing debug field that was not written into the source report

## CLI
- Export label template:
  - `python check_consistency.py --mode benchmark --benchmark-template-out configs/benchmark_labels.local.json`
- Run benchmark:
  - `python check_consistency.py --mode benchmark --benchmark-labels configs/benchmark_labels.local.json`
- Run benchmark on a specific report with an in-memory override:
  - `python check_consistency.py --mode benchmark --benchmark-report outputs/qa_report.json --benchmark-labels configs/benchmark_labels.local.json --threshold-override-file configs/threshold_override.local.json`

## Label File Shape
```json
{
  "schema_version": "qa_benchmark_labels_v1",
  "items": {
    "example.png": {
      "expected_status": "WARN",
      "weight": 1.0,
      "notes": "human sealed as backup only"
    }
  }
}
```

## Output Metrics
- `exact_accuracy`
- `macro_f1`
- `pass_precision`
- `pass_recall`
- `false_pass_rate`
- `release_safety_score`

## Usage Rule
- Benchmark labels should come from frozen human arbitration, not temporary mood judgments
- Prefer a benchmark set that covers:
  - front core
  - three-quarter soft review
  - side/back shadow lanes
  - known false-pass and false-warn regressions
