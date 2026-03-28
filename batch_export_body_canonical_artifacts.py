from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from export_body_canonical_artifact import _build_artifact, _write_artifact


def _iter_files_by_globs(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _normalize_match_key(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"(\.png|\.jpg|\.jpeg|\.webp|\.bmp)$", "", name)
    name = re.sub(r"(_result|_results|_hmr2|_pred|_output)$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def _build_image_index(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in paths:
        key = _normalize_match_key(path)
        if not key:
            continue
        index.setdefault(key, []).append(path)
    return index


def _choose_match(export_path: Path, image_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], str]:
    key = _normalize_match_key(export_path)
    matches = list(image_index.get(key) or [])
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        return None, f"multiple image matches for export {export_path.name}: {[str(item) for item in matches]}"
    return None, f"no image match for export {export_path.name}"


def _build_conversion_namespace(
    *,
    input_path: Path,
    output_path: Path,
    source_image: Optional[Path],
    source_role: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_path,
        output=output_path,
        source_image=source_image,
        source_role=source_role,
        index=args.index,
        track_id=args.track_id,
        beta_key=args.beta_key,
        pose_key=args.pose_key,
        global_orient_key=args.global_orient_key,
        confidence_key=args.confidence_key,
        coverage_key=args.coverage_key,
        measurements_key=args.measurements_key,
        measurement=list(args.measurement or []),
        measurement_scale=list(args.measurement_scale or []),
        default_measurement_scale=args.default_measurement_scale,
        provider_name=args.provider_name,
        provider_version=args.provider_version,
        model_id=args.model_id,
        notes=args.notes,
    )


def _candidate_output_path(image_path: Path, args: argparse.Namespace) -> Path:
    if args.output_mode == "adjacent":
        return image_path.with_name(f"{image_path.name}{args.sidecar_suffix}")
    if args.output_dir is None:
        raise ValueError("--output-dir is required when --output-mode=output_dir")
    return args.output_dir / f"{image_path.name}{args.sidecar_suffix}"


def _convert_one(
    *,
    input_path: Path,
    output_path: Path,
    source_image: Optional[Path],
    source_role: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> Dict[str, Any]:
    namespace = _build_conversion_namespace(
        input_path=input_path,
        output_path=output_path,
        source_image=source_image,
        source_role=source_role,
        args=args,
    )
    artifact = _build_artifact(namespace)
    if not dry_run:
        _write_artifact(output_path, artifact)
    shape_beta = artifact.get("shape_beta")
    shape_beta_dim = 0
    if isinstance(shape_beta, np.ndarray):
        shape_beta_dim = int(shape_beta.reshape(-1).shape[0])
    elif isinstance(shape_beta, (list, tuple)):
        shape_beta_dim = len(shape_beta)
    return {
        "ok": True,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "source_role": source_role,
        "source_image": str(source_image.resolve()) if source_image else "",
        "shape_beta_dim": shape_beta_dim,
        "measurement_count": len(dict(artifact.get("canonical_measurements") or {})),
        "fit_confidence": artifact.get("fit_confidence"),
        "coverage": artifact.get("coverage"),
        "dry_run": dry_run,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量把 HMR2/4D-Humans 导出转换成 xiaona_consistency 可读取的 body canonical artifact。"
    )
    parser.add_argument("--candidate-export-dir", type=Path, help="候选图对应的 HMR2 导出目录。")
    parser.add_argument("--image-dir", type=Path, help="候选原图目录，例如 input。")
    parser.add_argument(
        "--export-glob",
        action="append",
        default=["*.pkl", "*.pickle", "*.json", "*.npz", "*.npy"],
        help="候选导出文件匹配模式，可重复传入。",
    )
    parser.add_argument(
        "--image-glob",
        action="append",
        default=["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"],
        help="候选原图匹配模式，可重复传入。",
    )
    parser.add_argument(
        "--output-mode",
        choices=["adjacent", "output_dir"],
        default="adjacent",
        help="候选 sidecar 输出方式：adjacent 表示写到原图旁边，output_dir 表示集中写到目录。",
    )
    parser.add_argument("--output-dir", type=Path, help="当 output_mode=output_dir 时使用的输出目录。")
    parser.add_argument(
        "--sidecar-suffix",
        default=".body_canonical.json",
        help="候选 sidecar 后缀，默认写成 <image>.body_canonical.json",
    )
    parser.add_argument("--master-input", type=Path, help="116-1 的 HMR2 导出文件。")
    parser.add_argument("--master-output", type=Path, help="116-1 master truth artifact 输出路径。")
    parser.add_argument("--master-source-image", type=Path, help="116-1 原图路径。")
    parser.add_argument("--index", type=int, default=0, help="从 list-like 导出中选择的记录索引。")
    parser.add_argument("--track-id", default="", help="当导出文件包含多 track 时指定 track id。")
    parser.add_argument("--beta-key", default="", help="betas 字段的 dotted path。")
    parser.add_argument("--pose-key", default="", help="完整 pose vector 的 dotted path。")
    parser.add_argument("--global-orient-key", default="", help="global orient 的 dotted path。")
    parser.add_argument("--confidence-key", default="", help="fit confidence 的 dotted path。")
    parser.add_argument("--coverage-key", default="", help="coverage 的 dotted path。")
    parser.add_argument("--measurements-key", default="", help="canonical measurements 的 dotted path。")
    parser.add_argument("--measurement", action="append", default=[], help="追加 measurement，格式 KEY=VALUE。")
    parser.add_argument("--measurement-scale", action="append", default=[], help="追加 measurement scale，格式 KEY=VALUE。")
    parser.add_argument("--default-measurement-scale", type=float, default=0.08, help="未显式指定时使用的 measurement scale。")
    parser.add_argument("--provider-name", default="hmr2", help="写入 artifact 的 provider_name。")
    parser.add_argument("--provider-version", default="hmr2_export_v1", help="写入 artifact 的 provider_version。")
    parser.add_argument("--model-id", default="4d_humans_hmr2", help="写入 artifact 的 model_id。")
    parser.add_argument("--notes", default="", help="写入 artifact 的自由备注。")
    parser.add_argument("--dry-run", action="store_true", help="只检查匹配和转换，不真正写文件。")
    parser.add_argument(
        "--strict-missing-match",
        action="store_true",
        help="只要有候选导出找不到原图或存在多重匹配，就直接返回非 0。",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    has_master = args.master_input is not None or args.master_output is not None or args.master_source_image is not None
    has_candidates = args.candidate_export_dir is not None or args.image_dir is not None
    if not has_master and not has_candidates:
        raise ValueError("至少需要提供 master 转换参数，或候选批量转换参数。")
    if has_master and (args.master_input is None or args.master_output is None):
        raise ValueError("启用 master 转换时，--master-input 和 --master-output 必须同时提供。")
    if has_candidates and (args.candidate_export_dir is None or args.image_dir is None):
        raise ValueError("启用候选批量转换时，--candidate-export-dir 和 --image-dir 必须同时提供。")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    summary: Dict[str, Any] = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "master": None,
        "candidate_exports": {
            "converted": [],
            "skipped": [],
            "matched_count": 0,
            "skipped_count": 0,
        },
    }

    if args.master_input is not None:
        summary["master"] = _convert_one(
            input_path=args.master_input,
            output_path=args.master_output,
            source_image=args.master_source_image,
            source_role="master_truth",
            args=args,
            dry_run=bool(args.dry_run),
        )

    strict_failed = False
    if args.candidate_export_dir is not None:
        export_paths = sorted(_iter_files_by_globs(args.candidate_export_dir, args.export_glob))
        image_paths = sorted(_iter_files_by_globs(args.image_dir, args.image_glob))
        image_index = _build_image_index(image_paths)

        for export_path in export_paths:
            image_path, reason = _choose_match(export_path, image_index)
            if image_path is None:
                summary["candidate_exports"]["skipped"].append(
                    {
                        "export": str(export_path.resolve()),
                        "reason": reason,
                    }
                )
                strict_failed = strict_failed or bool(args.strict_missing_match)
                continue

            output_path = _candidate_output_path(image_path, args)
            converted = _convert_one(
                input_path=export_path,
                output_path=output_path,
                source_image=image_path,
                source_role="candidate",
                args=args,
                dry_run=bool(args.dry_run),
            )
            summary["candidate_exports"]["converted"].append(converted)

        summary["candidate_exports"]["matched_count"] = len(summary["candidate_exports"]["converted"])
        summary["candidate_exports"]["skipped_count"] = len(summary["candidate_exports"]["skipped"])

    summary["ok"] = summary["ok"] and not strict_failed
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
