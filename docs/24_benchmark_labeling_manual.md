# Benchmark Labeling Manual

## Purpose

- This document is for operators who do not want to edit JSON directly.
- Use it when you need to fill benchmark labels for a QA batch.
- Recommended workflow: export JSON template -> export CSV -> edit in Excel/WPS -> import back to JSON.

## Files

Current v2 batch template:

- [benchmark_labels_v2_current.template.json](/D:/xiaona_consistency/outputs/benchmark_labels_v2_current.template.json)

Helper scripts:

- [export_benchmark_labels_csv.py](/D:/xiaona_consistency/export_benchmark_labels_csv.py)
- [import_benchmark_labels_csv.py](/D:/xiaona_consistency/import_benchmark_labels_csv.py)
- [autofill_benchmark_labels.py](/D:/xiaona_consistency/autofill_benchmark_labels.py)

## Optional: Auto-Prefill First

If you do not want to start from a blank template, generate an automatic draft first:

```powershell
.\.venv\Scripts\python.exe .\autofill_benchmark_labels.py `
  --template outputs\benchmark_labels_v2_current.template.json `
  --report outputs\qa_report.json `
  --output outputs\benchmark_labels_v2_current.auto.json
```

Important:

- this is only a machine draft
- it is not frozen benchmark truth
- you still need to review and correct it

Recommended workflow:

1. generate `auto.json`
2. export it to CSV
3. edit only the wrong rows in Excel/WPS
4. import the CSV back to JSON

## Step 1. Export CSV

```powershell
.\.venv\Scripts\python.exe .\export_benchmark_labels_csv.py `
  --input outputs\benchmark_labels_v2_current.auto.json `
  --output outputs\benchmark_labels_v2_current.edit.csv
```

After this, open:

- [benchmark_labels_v2_current.edit.csv](/D:/xiaona_consistency/outputs/benchmark_labels_v2_current.edit.csv)

Use Excel or WPS to edit it.

## Step 2. Only Fill These Columns

You only need to touch:

- `expected_status`
- `expected_task_profile`
- `expected_view_lane`
- `expected_view_lane_detail`
- `must_have_reasons`
- `must_not_have_reasons`
- `notes`

For the current v2 batch, the simplest starting point is:

- `expected_view_lane`: `side_90`
- `expected_task_profile`: leave blank at first, or fill the lane profile you want to validate
- `expected_status`: fill `WARN` or `FAIL`
- `notes`: write your human reason in plain Chinese

## Step 3. Recommended Minimal Labeling Rule

For this batch, do not try to over-label.

Use this rule:

1. If the image still has review value, fill `WARN`
2. If the image is clearly not usable, fill `FAIL`
3. Keep `expected_view_lane=side_90`
4. Put the main human reason into `notes`

You can leave these empty in the first pass:

- `expected_view_lane_detail`
- `must_have_reasons`
- `must_not_have_reasons`

## Step 4. Import CSV Back Into JSON

```powershell
.\.venv\Scripts\python.exe .\import_benchmark_labels_csv.py `
  --template outputs\benchmark_labels_v2_current.auto.json `
  --csv outputs\benchmark_labels_v2_current.edit.csv `
  --output outputs\benchmark_labels_v2_current.filled.json
```

This creates:

- [benchmark_labels_v2_current.filled.json](/D:/xiaona_consistency/outputs/benchmark_labels_v2_current.filled.json)

## Step 5. Run Benchmark Replay

```powershell
.\.venv\Scripts\python.exe .\check_consistency.py `
  --mode benchmark `
  --benchmark-report outputs\qa_report.json `
  --benchmark-labels outputs\benchmark_labels_v2_current.filled.json `
  --benchmark-output outputs\benchmark_result_v2_current.json
```

## Practical Advice

- Do not edit the JSON manually.
- Do not try to judge everything at once.
- First finish `expected_status` and `notes`.
- After the first pass is complete, run replay once.
- Only then decide whether you need to refine lane detail or reason constraints.
