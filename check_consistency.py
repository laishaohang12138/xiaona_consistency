# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from core.qa_pipeline import (
    calibrate_quality_thresholds,
    create_runtime,
    load_anchor_set,
    load_thresholds_from_file,
    main as pipeline_main,
    print_runtime_config,
    run_pipeline,
)
from core.qa_runtime import (
    AnchorSet,
    EngineState,
    FaceFeat,
    PoseFeat,
    ProjectPaths,
    QualityThresholds,
    RuntimeConfig,
    RuntimeContext,
    save_thresholds_to_file,
)

BASE_DIR = Path(__file__).resolve().parent
main = pipeline_main


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_cli_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_json_dict(raw_text: str, label: str) -> Dict[str, Any]:
    try:
        node = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(node, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return node


def _load_threshold_override(args: argparse.Namespace, base_dir: Path) -> Optional[Dict[str, Any]]:
    override: Dict[str, Any] = {}
    if args.threshold_override_file is not None:
        override_path = _resolve_cli_path(args.threshold_override_file, base_dir)
        if not override_path.exists():
            raise ValueError(f"threshold override file does not exist: {override_path}")
        file_node = _load_json_dict(
            override_path.read_text(encoding="utf-8"),
            f"threshold override file {override_path}",
        )
        override = _deep_merge_dict(override, file_node)
    if args.threshold_override_json:
        json_node = _load_json_dict(args.threshold_override_json, "--threshold-override-json")
        override = _deep_merge_dict(override, json_node)
    return override or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Xiaona consistency QA or calibration from the command line.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Project base directory used to resolve configs, anchors, input, and outputs.",
    )
    parser.add_argument(
        "--profile",
        help="Override the active task profile for QA mode, for example the BODY GOLD front-core profile.",
    )
    parser.add_argument(
        "--mode",
        choices=["qa", "calibrate", "benchmark", "optuna"],
        help="Override runtime.config.run_mode for this invocation.",
    )
    parser.add_argument(
        "--auto-load-thresholds",
        action="store_true",
        help="Load outputs/quality_thresholds.json before running QA.",
    )
    parser.add_argument(
        "--threshold-override-file",
        type=Path,
        help="Path to a JSON object whose values override runtime thresholds in memory for this run only.",
    )
    parser.add_argument(
        "--threshold-override-json",
        help="Inline JSON object whose values override runtime thresholds in memory for this run only.",
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="Report JSON used by benchmark replay mode. Defaults to outputs/qa_report.json.",
    )
    parser.add_argument(
        "--benchmark-labels",
        type=Path,
        help="Benchmark label JSON for replay mode.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help="Optional output path for benchmark metrics JSON.",
    )
    parser.add_argument(
        "--benchmark-template-out",
        type=Path,
        help="Export a benchmark label template from the chosen report file and exit.",
    )
    parser.add_argument(
        "--optuna-search-space",
        type=Path,
        help="Optuna search-space JSON used by offline benchmark tuning mode.",
    )
    parser.add_argument(
        "--optuna-output",
        type=Path,
        help="Optional output JSON path for the Optuna study summary.",
    )
    parser.add_argument(
        "--optuna-best-override-out",
        type=Path,
        help="Optional output JSON path for the best threshold override found by Optuna.",
    )
    parser.add_argument(
        "--optuna-study-name",
        help="Optional study name override for Optuna tuning mode.",
    )
    parser.add_argument(
        "--optuna-storage-path",
        type=Path,
        help="Optional sqlite storage path for persisting the Optuna study.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        help="Optional trial-count override for Optuna tuning mode.",
    )
    return parser


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base_dir = _resolve_cli_path(args.base_dir, BASE_DIR)
    try:
        threshold_override = _load_threshold_override(args, base_dir)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode == "optuna":
        if args.optuna_search_space is None:
            parser.error("optuna mode requires --optuna-search-space")
        if args.benchmark_labels is None:
            parser.error("optuna mode requires --benchmark-labels")

        from core.qa_optuna import run_optuna_search

        result = run_optuna_search(
            base_dir=base_dir,
            report_path=_resolve_cli_path(args.benchmark_report, base_dir)
            if args.benchmark_report
            else (base_dir / "outputs" / "qa_report.json").resolve(),
            labels_path=_resolve_cli_path(args.benchmark_labels, base_dir),
            search_space_path=_resolve_cli_path(args.optuna_search_space, base_dir),
            cli_fixed_override=threshold_override,
            output_path=_resolve_cli_path(args.optuna_output, base_dir) if args.optuna_output else None,
            best_override_out=_resolve_cli_path(args.optuna_best_override_out, base_dir)
            if args.optuna_best_override_out
            else None,
            study_name_override=args.optuna_study_name,
            storage_path=_resolve_cli_path(args.optuna_storage_path, base_dir) if args.optuna_storage_path else None,
            trials_override=args.optuna_trials,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    pipeline_main(
        base_dir=base_dir,
        profile_name=args.profile,
        run_mode=args.mode,
        auto_load_thresholds=True if args.auto_load_thresholds else None,
        threshold_override=threshold_override,
        benchmark_report_path=_resolve_cli_path(args.benchmark_report, base_dir) if args.benchmark_report else None,
        benchmark_labels_path=_resolve_cli_path(args.benchmark_labels, base_dir) if args.benchmark_labels else None,
        benchmark_output_path=_resolve_cli_path(args.benchmark_output, base_dir) if args.benchmark_output else None,
        benchmark_template_out=_resolve_cli_path(args.benchmark_template_out, base_dir) if args.benchmark_template_out else None,
    )
    return 0


__all__ = [
    "AnchorSet",
    "EngineState",
    "FaceFeat",
    "PoseFeat",
    "ProjectPaths",
    "QualityThresholds",
    "RuntimeConfig",
    "RuntimeContext",
    "calibrate_quality_thresholds",
    "cli",
    "create_runtime",
    "load_anchor_set",
    "load_thresholds_from_file",
    "main",
    "print_runtime_config",
    "run_pipeline",
    "save_thresholds_to_file",
]


if __name__ == "__main__":
    raise SystemExit(cli())
