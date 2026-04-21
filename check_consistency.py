# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PIPELINE_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    from core.qa_pipeline import (
        calibrate_quality_thresholds,
        create_runtime,
        load_anchor_set,
        load_thresholds_from_file,
        main as pipeline_main,
        print_runtime_config,
        run_pipeline,
    )
except ModuleNotFoundError as exc:
    _PIPELINE_IMPORT_ERROR = exc

    def _pipeline_unavailable(*args: Any, **kwargs: Any) -> Any:
        missing_name = str(getattr(_PIPELINE_IMPORT_ERROR, "name", "") or "unknown")
        raise RuntimeError(
            f"QA pipeline dependencies are unavailable because module '{missing_name}' is missing. "
            "Install the full runtime dependencies before running shot_review or advanced CLI modes."
        ) from _PIPELINE_IMPORT_ERROR

    calibrate_quality_thresholds = _pipeline_unavailable
    create_runtime = _pipeline_unavailable
    load_anchor_set = _pipeline_unavailable
    load_thresholds_from_file = _pipeline_unavailable
    pipeline_main = _pipeline_unavailable
    print_runtime_config = _pipeline_unavailable
    run_pipeline = _pipeline_unavailable
_RUNTIME_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
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
except ModuleNotFoundError as exc:
    _RUNTIME_IMPORT_ERROR = exc

    def _runtime_unavailable(*args: Any, **kwargs: Any) -> Any:
        missing_name = str(getattr(_RUNTIME_IMPORT_ERROR, "name", "") or "unknown")
        raise RuntimeError(
            f"QA runtime dependencies are unavailable because module '{missing_name}' is missing. "
            "Install the runtime stack before running calibration or pipeline-backed modes."
        ) from _RUNTIME_IMPORT_ERROR

    AnchorSet = Any
    EngineState = Any
    FaceFeat = Any
    PoseFeat = Any
    ProjectPaths = Any
    QualityThresholds = Any
    RuntimeConfig = Any
    RuntimeContext = Any
    save_thresholds_to_file = _runtime_unavailable

BASE_DIR = Path(__file__).resolve().parent
main = pipeline_main

HEAVY_PROVIDER_UI: Dict[str, Dict[str, str]] = {
    "segformer_body_fusion": {
        "label": "工业默认融合",
        "summary": "同时启用服装边界证据和体态几何证据，适合日常批次复核。",
        "scene": "日常 shot_review / 没有 canonical 产物时的稳定默认模式",
    },
    "segformer_body_truth_fusion": {
        "label": "真相融合模式",
        "summary": "在默认融合上叠加 116-1 canonical body truth，适合验证绝对身材真相是否到位。",
        "scene": "已经准备好 116-1 master artifact 和候选 sidecar 时",
    },
    "segformer_parser": {
        "label": "边界证据模式",
        "summary": "只看服装边界、领口、肩线和可见肢体边界。",
        "scene": "优先检查服装边界、领口、肩线稳定性",
    },
    "body_measure_lite": {
        "label": "体态几何模式",
        "summary": "只看轻量体态、3D 和空间结构证据。",
        "scene": "优先判断站姿、比例、侧后轮廓是否稳定",
    },
    "body_canonical_hmr2": {
        "label": "116-1 真相直连模式",
        "summary": "优先尝试本地 HMR2/4D-Humans 直连导出并使用显卡，失败时继续读取现有 canonical body truth 产物。",
        "scene": "单独验证 116-1 身材真相链，或给 truth fusion 提供 body canonical 证据",
    },
    "disabled": {
        "label": "关闭重证据",
        "summary": "只保留轻量 QA 主链路，不做任何重型补证。",
        "scene": "纯轻量回放、排查基础分数链路时",
    },
}

VIEW_CLASSIFIER_UI: Dict[str, str] = {
    "view_classifier_lite": "轻量视角辅助分类器（只做 shadow 对照，不改主路由）",
    "disabled": "关闭视角辅助分类器",
}

FACE_CANONICAL_UI: Dict[str, str] = {
    "face_pose_canonical_3ddfa": "脸部 canonical 直连 3DDFA-V3（优先直连外部仓库，失败自动回退 bridge）",
    "face_pose_canonical_bridge": "脸部 canonical 桥接（shadow-only，面向 0 号脸规范化对照）",
    "disabled": "关闭脸部 canonical 辅助",
}

WORKFLOW_UI: Dict[str, Dict[str, str]] = {
    "shot_review": {
        "label": "批次复核",
        "summary": "对当前 input 图集运行 QA，输出排序、review packet 和 winner 候选。",
    },
    "preflight_batch": {
        "label": "批次前置预检",
        "summary": "在重型 QA 前先看 lane 纯度、混批风险和 prompt intent 元数据是否齐全。",
    },
    "prepare_input_manifest": {
        "label": "准备输入清单",
        "summary": "为当前 input 目录生成或更新 input_manifest.json，给 prompt_id / seed / intended_view 留标准入口。",
    },
    "fill_input_manifest_defaults": {
        "label": "回填输入清单默认值",
        "summary": "给当前 input_manifest.json 批量补录共享字段，例如 prompt_id、seed、anchor_source。", 
    },
    "merge_input_manifest_metadata": {
        "label": "合并输入清单逐图元数据",
        "summary": "把外部 image->fields 映射文件合并进当前 input_manifest.json，适合补录逐图 seed 和 prompt_id。",
    },
    "refresh_review_run_index": {
        "label": "刷新运行索引",
        "summary": "扫描 outputs 和 outputs_snapshots，生成统一运行索引与 clean batch 指针。",
    },
    "prepare_front_bootstrap_review": {
        "label": "准备 front bootstrap 复审表",
        "summary": "从运行索引读取推荐的 front clean batch，生成 top-3 人工复审表。",
    },
    "refresh_review_status_board": {
        "label": "刷新总控状态板",
        "summary": "汇总当前 outputs、clean snapshot、front top-3、winner bank 和 manifest 状态。",
    },
    "prepare_split_batch_plan": {
        "label": "准备拆批方案",
        "summary": "按 observed lane 汇总当前批次，生成 front / three_quarter / side / back 拆批方案。",
    },
    "materialize_split_batches": {
        "label": "落盘拆批目录",
        "summary": "按 batch_split_plan 把当前 input 复制到 input_split/<lane>/，并为每个子批次生成 manifest 模板。",
    },
    "refresh_review_artifacts": {
        "label": "刷新复核产物",
        "summary": "基于现有 qa_report 与缓存，回填 topology 字段并重建 review packet / GPT 分析包。",
    },
    "inspect_review_packet": {
        "label": "查看复核摘要",
        "summary": "直接读取最近一次 review packet，快速看批次状态、Top1、风险和人工提示。",
    },
    "prepare_winner_bank_review": {
        "label": "准备 winner 复审包",
        "summary": "汇总 winner candidate、batch preflight 和当前 bank 状态，生成一份人工确认用 review packet。",
    },
    "promote_winner": {
        "label": "确认 winner",
        "summary": "把人工确认通过的 winner 写入 winner bank，不等于主训练集自动准入。",
    },
    "seal_training_admission": {
        "label": "封印训练准入",
        "summary": "把通过 release gate 的候选写入 training admission manifest，和 winner bank 彻底分开。",
    },
    "winner_bank_status": {
        "label": "查看 winner bank",
        "summary": "查看已确认样本、最新漂移和下一步人工动作。",
    },
    "training_admission_status": {
        "label": "查看训练准入",
        "summary": "查看 training admission manifest 的已封印样本、bucket 分布和最近 seal 记录。",
    },
    "setup_external_models": {
        "label": "准备外部模型",
        "summary": "自动执行 external 仓库拉取/定位、补丁应用和本地 helper 同步，供 3DDFA/HMR2 真相链使用。",
    },
    "advanced_cli": {
        "label": "高级工程模式",
        "summary": "进入 qa / benchmark / optuna / calibrate 等工程入口。",
    },
}


def _heavy_provider_ui(provider_name: Any) -> Dict[str, str]:
    name = str(provider_name or "").strip()
    return HEAVY_PROVIDER_UI.get(
        name,
        {
            "label": name or "未知重证据模式",
            "summary": "暂无中文说明。",
            "scene": "请结合 provider 名称排查。",
        },
    )


def _heavy_provider_title(provider_name: Any) -> str:
    name = str(provider_name or "").strip()
    ui = _heavy_provider_ui(name)
    return f"{ui['label']} [{name}]" if name else ui["label"]


def _print_heavy_provider_explanation(provider_name: Any) -> None:
    name = str(provider_name or "").strip()
    ui = _heavy_provider_ui(name)
    print(f"[交互引导] 已选择重型证据模式：{ui['label']} [{name}]")
    print(f"[交互引导] 适用场景：{ui['scene']}")
    print(f"[交互引导] 模式说明：{ui['summary']}")
    if name in {"segformer_body_truth_fusion", "body_canonical_hmr2"}:
        print("[交互引导] 前置条件：需要 116-1 的 canonical master artifact，以及候选图 sidecar。")
        print("[交互引导] 产物说明见 docs/20_body_canonical_artifact_bridge.md")
        print("[交互引导] 如需直连 HMR2，请继续看 docs/23_body_canonical_hmr2_integration.md")


def _print_workflow_explanation(workflow_name: Any) -> None:
    name = str(workflow_name or "").strip()
    ui = WORKFLOW_UI.get(name)
    if not ui:
        return
    print(f"[交互引导] 已选择任务：{ui['label']} [{name}]")
    print(f"[交互引导] 任务说明：{ui['summary']}")


def _configure_console_encoding() -> None:
    for stream_name in ["stdin", "stdout", "stderr"]:
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


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


def _prompt_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ")
    except EOFError as exc:
        raise ValueError("交互模式需要可读取的标准输入") from exc
    raw = raw.replace("\ufeff", "").strip()
    return raw or default


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        choice = _prompt_text(f"{prompt} ({default_text})", "")
        if not choice:
            return default
        normalized = choice.strip().lower()
        if normalized in {"y", "yes", "1", "true", "on"}:
            return True
        if normalized in {"n", "no", "0", "false", "off"}:
            return False
        print(f"[交互引导] 无法识别的是/否输入: {choice}")


def _select_choice(
    prompt: str,
    options: Sequence[Tuple[str, str]],
    *,
    default: str,
) -> str:
    print("[交互引导] 可选项:")
    for index, (value, description) in enumerate(options, start=1):
        print(f"  {index}. {description} [{value}]")
    valid_names = {value for value, _ in options}
    default_value = default if default in valid_names else options[0][0]
    while True:
        choice = _prompt_text(prompt, default_value)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1][0]
        matched = [value for value, _ in options if value == choice]
        if matched:
            return matched[0]
        print(f"[交互引导] 无法识别的选项: {choice}")


def _prompt_path(
    prompt: str,
    *,
    base_dir: Path,
    default: str = "",
    must_exist: bool = False,
) -> Path:
    while True:
        raw = _prompt_text(prompt, default)
        if not raw:
            print("[交互引导] 路径不能为空")
            continue
        candidate = _resolve_cli_path(Path(raw), base_dir)
        if must_exist and not candidate.exists():
            print(f"[交互引导] 路径不存在: {candidate}")
            continue
        return candidate


def _load_json_file(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return payload


def _select_workflow_interactively(default: str = "shot_review") -> str:
    return _select_choice(
        "请选择当前要完成的任务",
        [
            ("shot_review", "审一轮 shot 批次，输出 QA、排序和 review packet"),
            ("preflight_batch", "在 heavy QA 前先做 lane 纯度、混批和元数据预检"),
            ("prepare_input_manifest", "为当前 input 目录生成或更新 input_manifest.json 模板"),
            ("fill_input_manifest_defaults", "给当前 input_manifest.json 批量补录 prompt_id / seed / anchor_source"),
            ("merge_input_manifest_metadata", "把外部逐图 metadata 合并进 input_manifest.json"),
            ("refresh_review_run_index", "扫描 outputs 和 outputs_snapshots，生成统一运行索引"),
            ("prepare_front_bootstrap_review", "从运行索引读取 front clean batch，生成 top-3 人工复审表"),
            ("refresh_review_status_board", "汇总当前主结果、front 基线、winner bank 和 manifest 状态"),
            ("prepare_split_batch_plan", "按 observed lane 生成当前批次的拆批方案"),
            ("materialize_split_batches", "把拆批方案落到 input_split/<lane>/ 目录"),
            ("refresh_review_artifacts", "基于现有 qa_report 和缓存刷新 review packet / GPT 包"),
            ("inspect_review_packet", "查看最近一次 review packet 的批次摘要和复核提示"),
            ("prepare_winner_bank_review", "生成 winner bank 人工确认包，先看能不能 promote"),
            ("promote_winner", "把人工确认的 winner 写入 winner bank"),
            ("seal_training_admission", "把通过 release gate 的候选写入 training admission manifest"),
            ("winner_bank_status", "查看 winner bank 状态与最新跨批次漂移报告"),
            ("training_admission_status", "查看 training admission manifest 的最新封印状态"),
            ("setup_external_models", "自动准备 external/3DDFA-V3 与 external/4D-Humans 及补丁"),
            ("advanced_cli", "进入高级工程模式（qa / benchmark / optuna / calibrate）"),
        ],
        default=default,
    )


def _select_review_profile_interactively(default: str = "body_gold_fullbody") -> str:
    return _select_choice(
        "请选择本轮 shot 批次最接近的训练层",
        [
            ("body_gold_fullbody", "BODY GOLD 前向/常规主体批次"),
            ("bridge_simple_outfit", "简单穿搭 / BRIDGE 训练准入批次"),
            ("body_gold_side90_shadow", "90 度侧身 shadow 观察批次"),
            ("body_gold_back180_shadow", "180 度背身 shadow 观察批次"),
            ("full_body_outfit", "通用穿搭稳定性批次"),
        ],
        default=default,
    )


def _select_heavy_provider_interactively(default: str = "segformer_body_fusion") -> str:
    options = [
        "segformer_body_fusion",
        "segformer_body_truth_fusion",
        "segformer_parser",
        "body_measure_lite",
        "body_canonical_hmr2",
        "disabled",
    ]
    print("[交互引导] 本轮可选的重型证据模式：")
    for index, value in enumerate(options, start=1):
        ui = _heavy_provider_ui(value)
        print(f"  {index}. {ui['label']} [{value}]")
        print(f"     适用: {ui['scene']}")
        print(f"     说明: {ui['summary']}")
    valid_names = set(options)
    default_value = default if default in valid_names else options[0]
    while True:
        choice = _prompt_text("请选择本轮使用的重型证据模式（输入编号或内部代号）", default_value)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        if choice in valid_names:
            return choice
        print(f"[交互引导] 无法识别的选项: {choice}")


def _parse_heavy_provider_compare_targets(values: Optional[Sequence[str]]) -> List[str]:
    allowed = {
        "segformer_body_fusion",
        "segformer_body_truth_fusion",
        "segformer_parser",
        "body_measure_lite",
        "body_canonical_hmr2",
        "disabled",
    }
    parsed: List[str] = []
    for raw_value in values or []:
        for chunk in str(raw_value).split(","):
            value = chunk.strip()
            if not value:
                continue
            if value not in allowed:
                raise ValueError(
                    "unknown heavy provider in compare list: "
                    f"{value}. allowed={sorted(allowed)}"
                )
            if value not in parsed:
                parsed.append(value)
    return parsed


def _prompt_heavy_provider_compare_targets(
    default: Sequence[str] = (
        "segformer_body_fusion",
        "segformer_body_truth_fusion",
        "segformer_parser",
        "body_measure_lite",
        "body_canonical_hmr2",
    ),
) -> List[str]:
    print("[交互引导] 可对比的重型证据模式：")
    compare_order = list(default) + ["disabled"]
    for index, value in enumerate(compare_order, start=1):
        ui = _heavy_provider_ui(value)
        print(f"  {index}. {ui['label']} [{value}]")
        print(f"     适用: {ui['scene']}")
        print(f"     说明: {ui['summary']}")
    default_text = ",".join(default)
    while True:
        raw = _prompt_text("请输入需要对比的 heavy provider（逗号分隔）", default_text)
        try:
            parsed = _parse_heavy_provider_compare_targets([raw])
        except ValueError as exc:
            print(f"[交互引导] {exc}")
            continue
        if len(parsed) == 0:
            print("[交互引导] 至少需要选择 1 个 heavy provider")
            continue
        return parsed
def _resolve_artifacts_dir_arg(artifacts_dir: Optional[Path], base_dir: Path) -> Path:
    if artifacts_dir is None:
        return (base_dir / "outputs").resolve()
    return _resolve_cli_path(artifacts_dir, base_dir)


def _default_review_paths(base_dir: Path, artifacts_dir: Optional[Path] = None) -> Dict[str, Path]:
    output_dir = _resolve_artifacts_dir_arg(artifacts_dir, base_dir)
    return {
        "preflight_batch": output_dir / "preflight_batch.json",
        "batch_split_plan": output_dir / "batch_split_plan.json",
        "materialized_batch_split": output_dir / "materialized_batch_split.json",
        "review_run_index": output_dir / "review_run_index.json",
        "front_bootstrap_review_sheet": output_dir / "front_bootstrap_review_sheet.json",
        "review_status_board": output_dir / "review_status_board.json",
        "qa_report": output_dir / "qa_report.json",
        "ranked_candidates": output_dir / "ranked_candidates.json",
        "review_packet": output_dir / "review_packet.json",
        "gpt_review_packet": output_dir / "gpt_review_packet.json",
        "review_artifacts": output_dir / "review_artifacts.json",
        "winner_bank_candidate": output_dir / "winner_bank_candidate.json",
        "winner_bank_report": output_dir / "winner_bank_report.json",
        "winner_bank_review_packet": output_dir / "winner_bank_review_packet.json",
        "winner_bank": output_dir / "winner_bank.json",
        "training_admission_manifest": output_dir / "training_admission_manifest.json",
    }


def _resolve_input_dir_arg(input_dir: Optional[Path], base_dir: Path) -> Path:
    if input_dir is None:
        return (base_dir / "input").resolve()
    return _resolve_cli_path(input_dir, base_dir)


def _override_runtime_input_dir(runtime: Any, input_dir: Path) -> None:
    runtime.config.paths = replace(runtime.config.paths, dir_input=input_dir)


def _powershell_executable() -> str:
    return os.environ.get("ComSpec", "").lower().endswith("cmd.exe") and "powershell" or "powershell"


def _run_local_powershell_script(
    script_path: Path,
    *,
    base_dir: Path,
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    if not script_path.exists():
        raise ValueError(f"script does not exist: {script_path}")
    command = [
        _powershell_executable(),
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        *(list(extra_args or [])),
    ]
    subprocess.run(command, cwd=str(base_dir), check=True)


def _external_setup_status(base_dir: Path) -> Dict[str, Any]:
    repo_3ddfa = (base_dir / "external" / "3DDFA-V3").resolve()
    repo_hmr2 = (base_dir / "external" / "4D-Humans").resolve()
    smpl_candidates = [
        base_dir / "external" / "4D-Humans" / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        base_dir / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        Path.home() / ".cache" / "4DHumans" / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
    ]
    asset_3ddfa = [
        repo_3ddfa / "assets" / "face_model.npy",
        repo_3ddfa / "assets" / "large_base_net.pth",
        repo_3ddfa / "assets" / "net_recon.pth",
        repo_3ddfa / "assets" / "retinaface_resnet50_2020-07-20_old_torch.pth",
        repo_3ddfa / "assets" / "similarity_Lm3D_all.mat",
    ]
    return {
        "repo_3ddfa_ready": repo_3ddfa.exists(),
        "repo_hmr2_ready": repo_hmr2.exists(),
        "smpl_ready": any(path.exists() for path in smpl_candidates),
        "smpl_path": str(next((path for path in smpl_candidates if path.exists()), smpl_candidates[0])),
        "assets_3ddfa_ready": all(path.exists() for path in asset_3ddfa),
        "missing_3ddfa_assets": [str(path) for path in asset_3ddfa if not path.exists()],
    }


def _print_external_setup_status(status: Dict[str, Any]) -> None:
    print("\n[外部模型状态]")
    print(
        "  仓库    : "
        f"3DDFA={'ok' if status.get('repo_3ddfa_ready') else 'missing'} "
        f"| 4D-Humans={'ok' if status.get('repo_hmr2_ready') else 'missing'}"
    )
    print(
        "  资产    : "
        f"3DDFA_assets={'ok' if status.get('assets_3ddfa_ready') else 'missing'} "
        f"| SMPL={'ok' if status.get('smpl_ready') else 'missing'}"
    )
    if status.get("smpl_ready"):
        print(f"  SMPL 路径: {status.get('smpl_path')}")
    missing_assets = list(status.get("missing_3ddfa_assets") or [])
    if missing_assets:
        print("  缺少 3DDFA 资产:")
        for item in missing_assets[:5]:
            print(f"    - {item}")


def _print_review_packet_summary(packet: Dict[str, Any]) -> None:
    batch = packet.get("batch_summary") or {}
    selection = batch.get("selection") or {}
    batch_gate = batch.get("batch_gate") or {}
    engine_status = batch.get("engine_status") or {}
    anchor_truth = batch.get("anchor_truth") or {}
    heavy_provider_status = batch.get("heavy_provider_status") or {}
    view_classifier_status = batch.get("view_classifier_status") or {}
    face_canonical_status = batch.get("face_canonical_status") or {}
    heavy_evidence = batch.get("heavy_evidence_summary") or {}
    canonical_truth = batch.get("canonical_truth_summary") or {}
    lane_risk_focus = batch.get("lane_risk_focus") or {}
    master_truth_artifact_dir = batch.get("master_truth_artifact_dir")
    artifact_manifest_file = batch.get("artifact_manifest_file")
    artifact_manifest_summary = batch.get("artifact_manifest_summary") or {}
    identity = batch.get("identity_summary") or {}
    geometry = batch.get("geometry_summary") or {}
    admission = batch.get("admission_advice") or {}
    release_gate = batch.get("release_gate") or admission.get("release_gate") or {}
    batch_preflight = batch.get("batch_preflight") or admission.get("batch_preflight") or {}
    evidence_completeness = batch.get("evidence_completeness") or admission.get("evidence_completeness") or {}
    training_admission = packet.get("training_admission_status") or batch.get("training_admission_governance") or {}
    training_manifest_summary = training_admission.get("manifest_summary") or {}
    ranked = packet.get("ranked_review_packet") or {}
    groups = list(ranked.get("groups") or [])
    primary_group = groups[0] if len(groups) > 0 and isinstance(groups[0], dict) else {}
    shortlist = list(primary_group.get("shortlist") or [])
    top_shortlist = shortlist[0] if len(shortlist) > 0 and isinstance(shortlist[0], dict) else {}
    print("\n[复核摘要]")
    print(f"  训练层: {batch.get('target_profile')}")
    print(f"  状态  : {batch.get('run_status')}")
    print(f"  图片数: {batch.get('input_count')}")
    if engine_status:
        print(
            "  引擎  : "
            f"face={engine_status.get('face_mode')} "
            f"| pose={engine_status.get('pose_mode')} "
            f"| fatal={engine_status.get('fatal')} "
            f"| classic_cv={engine_status.get('classic_cv_fallback_active')}"
        )
        if engine_status.get("fatal_reasons"):
            print(f"  致命原因: {engine_status.get('fatal_reasons')}")
    if anchor_truth:
        print(
            "  真相锚点: "
            f"face={((anchor_truth.get('face_truth_anchor') or {}).get('anchor_id'))} "
            f"| body={((anchor_truth.get('body_truth_anchor') or {}).get('anchor_id'))} "
            f"| upper={((anchor_truth.get('upper_support_anchor') or {}).get('anchor_id'))}"
        )
    if heavy_provider_status:
        component_names = ",".join(
            _heavy_provider_ui(node.get("provider_name")).get("label")
            for node in heavy_provider_status.get("component_providers", [])
            if isinstance(node, dict) and str(node.get("provider_name") or "").strip()
        )
        print(
            "  重证据策略: "
            f"请求={_heavy_provider_title(heavy_provider_status.get('requested_heavy_evidence'))} "
            f"| 生效={_heavy_provider_title(heavy_provider_status.get('provider_name'))} "
            f"| enabled={heavy_provider_status.get('enabled')}"
            f"{f' | 组件={component_names}' if component_names else ''}"
        )
    if view_classifier_status:
        view_name = str(view_classifier_status.get("provider_name") or "").strip()
        print(
            "  视角辅助: "
            f"请求={VIEW_CLASSIFIER_UI.get(str(view_classifier_status.get('requested_view_classifier') or '').strip(), view_classifier_status.get('requested_view_classifier'))} "
            f"| 生效={VIEW_CLASSIFIER_UI.get(view_name, view_name)} "
            f"| enabled={view_classifier_status.get('enabled')}"
        )
    if face_canonical_status:
        face_canonical_name = str(face_canonical_status.get("provider_name") or "").strip()
        requested_name = str(face_canonical_status.get("requested_face_canonical") or "").strip()
        print(
            "  脸部辅助: "
            f"请求={FACE_CANONICAL_UI.get(requested_name, requested_name)} "
            f"| 生效={FACE_CANONICAL_UI.get(face_canonical_name, face_canonical_name)} "
            f"| enabled={face_canonical_status.get('enabled')}"
        )
    if artifact_manifest_file:
        print(
            "  资产索引: "
            f"entries={artifact_manifest_summary.get('total_entries')} "
            f"| file={artifact_manifest_file}"
        )
    print(f"  第一名: {selection.get('top_ranked_image')}")
    print(f"  复核窗: top {selection.get('manual_review_window')}")
    print(f"  批次闸门: {batch_gate.get('status')} | reasons={batch_gate.get('reasons') or []}")
    if batch_preflight:
        print(
            "  批次预检: "
            f"status={batch_preflight.get('status')} "
            f"| purity={batch_preflight.get('lane_purity_score')} "
            f"| dominant={batch_preflight.get('dominant_lane_family')} "
            f"| inside_gate={batch_preflight.get('inside_release_gate_share')} "
            f"| split={batch_preflight.get('split_batch_recommended')}"
        )
        if batch_preflight.get("governance_lane_source"):
            print(
                "  预检治理: "
                f"lane_source={batch_preflight.get('governance_lane_source')} "
                f"| prompt_is_weak_prior={batch_preflight.get('prompt_intent_is_weak_prior')}"
            )
        if batch_preflight.get("intended_lane_coverage") is not None:
            print(
                "  意图对照: "
                f"coverage={batch_preflight.get('intended_lane_coverage')} "
                f"| dominant_intended={batch_preflight.get('dominant_intended_lane_family')} "
                f"| observed_match={batch_preflight.get('intended_observed_lane_match_share')} "
                f"| center_dist_mean={batch_preflight.get('observed_lane_center_distance_mean_deg')}"
            )
        if batch_preflight.get("reasons"):
            print(f"  预检原因: {batch_preflight.get('reasons')}")
        if batch_preflight.get("recommended_action"):
            print(f"  预检动作: {batch_preflight.get('recommended_action')}")
    if release_gate:
        print(
            "  Release Gate: "
            f"bucket={release_gate.get('target_bucket')} "
            f"| state={release_gate.get('release_state')} "
            f"| ceiling={release_gate.get('machine_status_ceiling')} "
            f"| seal_allowed={release_gate.get('training_admission_allowed')}"
        )
    if training_admission:
        print(
            "  准入封印: "
            f"entries={training_manifest_summary.get('entry_count')} "
            f"| last={training_manifest_summary.get('last_sealed_at_utc')} "
            f"| file={training_admission.get('manifest_file')}"
        )
    if heavy_evidence:
        heavy_summary = heavy_evidence.get("summary") or {}
        heavy_ui = _heavy_provider_ui(heavy_evidence.get("provider_name"))
        component_names = ",".join(
            _heavy_provider_ui(node.get("provider_name")).get("label")
            for node in heavy_summary.get("component_providers", [])
            if isinstance(node, dict) and str(node.get("provider_name") or "").strip()
        )
        print(
            "  重证据结果: "
            f"模式={_heavy_provider_title(heavy_evidence.get('provider_name'))} "
            f"| 可用={heavy_evidence.get('available')} "
            f"| 置信度={heavy_evidence.get('confidence')} "
            f"| 覆盖度={heavy_evidence.get('coverage')} "
            f"| cache_hit={heavy_summary.get('cache_hit_count')} "
            f"| cache_write={heavy_summary.get('cache_write_count')}"
            f"{f' | 组件={component_names}' if component_names else ''}"
        )
        print(f"  说明  : {heavy_ui.get('summary')}")
    if evidence_completeness:
        print(
            "  证据完备: "
            f"status={evidence_completeness.get('status')} "
            f"| completeness={evidence_completeness.get('completeness_score')} "
            f"| replay_ready={evidence_completeness.get('replay_ready')} "
            f"| gpt_ready={evidence_completeness.get('gpt_review_ready')}"
        )
        coverage = evidence_completeness.get("coverage") or {}
        if coverage:
            print(
                "  证据覆盖: "
                f"face_canonical={coverage.get('face_canonical_coverage')} "
                f"| body_canonical={coverage.get('body_canonical_coverage')} "
                f"| surface={coverage.get('surface_evidence_coverage')} "
                f"| master={coverage.get('master_consistency_coverage')}"
            )
        if evidence_completeness.get("reasons"):
            print(f"  证据原因: {evidence_completeness.get('reasons')}")
    show_batch_truth = any(
        canonical_truth.get(key) is not None
        for key in [
            "canonical_truth_available",
            "body_shape_truth_alignment",
            "body_shape_beta_similarity",
            "body_topology_signature_similarity",
            "canonical_measurement_similarity",
            "body_mesh_fit_confidence",
        ]
    ) or str(heavy_evidence.get("provider_name") or "") in {"segformer_body_truth_fusion", "body_canonical_hmr2"}
    if show_batch_truth:
        provider_state = canonical_truth.get("body_canonical_provider_state") or {}
        print(
            "  身材真相: "
            f"available={canonical_truth.get('canonical_truth_available')} "
            f"| align={canonical_truth.get('body_shape_truth_alignment')} "
            f"| beta={canonical_truth.get('body_shape_beta_similarity')} "
            f"| topo={canonical_truth.get('body_topology_signature_similarity')} "
            f"| meas={canonical_truth.get('canonical_measurement_similarity')} "
            f"| fit={canonical_truth.get('body_mesh_fit_confidence')}"
            + (f" | provider={provider_state.get('provider_name')}" if provider_state else "")
        )
        if canonical_truth.get("canonical_truth_available") is not True:
            print("  说明  : 当前还没有可用的 116-1 canonical body truth 证据，系统会退化为边界/体态几何复核。")
            if master_truth_artifact_dir:
                print(f"  下一步: 先准备 {master_truth_artifact_dir} 下的 master artifact 和候选 sidecar")
            if str((heavy_provider_status or {}).get("integration_state") or "") == "missing_smpl_model":
                print("  阻塞  : 本地 HMR2 已接通，但缺少 SMPL neutral 模型文件。")
                print("  参考  : docs/23_body_canonical_hmr2_integration.md")
            else:
                print("  参考  : docs/20_body_canonical_artifact_bridge.md")
    if top_shortlist:
        top_face_canonical = top_shortlist.get("face_canonical_summary") or {}
        top_truth = top_shortlist.get("canonical_truth_summary") or {}
        show_top_truth = any(
            top_truth.get(key) is not None
            for key in [
                "canonical_truth_available",
                "body_shape_truth_alignment",
                "body_shape_beta_similarity",
                "body_topology_signature_similarity",
                "canonical_measurement_similarity",
                "body_mesh_fit_confidence",
            ]
        )
        if show_top_truth:
            print(
                "  第一名真相: "
                f"available={top_truth.get('canonical_truth_available')} "
                f"| align={top_truth.get('body_shape_truth_alignment')} "
                f"| beta={top_truth.get('body_shape_beta_similarity')} "
                f"| topo={top_truth.get('body_topology_signature_similarity')} "
                f"| meas={top_truth.get('canonical_measurement_similarity')} "
                f"| fit={top_truth.get('body_mesh_fit_confidence')}"
            )
        show_top_face = any(
            top_face_canonical.get(key) is not None
            for key in [
                "available",
                "face_pose_normalization_confidence",
                "canonical_face_landmark_similarity",
                "canonical_face_identity_similarity",
                "canonical_face_topology_similarity",
            ]
        ) or str((face_canonical_status or {}).get("provider_name") or "") in {"face_pose_canonical_bridge", "face_pose_canonical_3ddfa"}
        if show_top_face:
            print(
                "  第一名脸辅: "
                f"available={top_face_canonical.get('available')} "
                f"| normalize={top_face_canonical.get('face_pose_normalization_confidence')} "
                f"| landmark={top_face_canonical.get('canonical_face_landmark_similarity')} "
                f"| identity={top_face_canonical.get('canonical_face_identity_similarity')} "
                f"| topo={top_face_canonical.get('canonical_face_topology_similarity')} "
                f"| pose_delta={top_face_canonical.get('pose_delta_deg')}"
            )
    print(
        "  身份凝聚: "
        f"face={identity.get('batch_identity_cohesion')} "
        f"clothfree={identity.get('batch_clothfree_identity_cohesion')} "
        f"hybrid={identity.get('batch_hybrid_identity_cohesion')}"
    )
    print(
        "  几何对齐: "
        f"body={geometry.get('body_under_clothes_continuity')} "
        f"3d={geometry.get('batch_3d_cohesion')} "
        f"world3d={geometry.get('batch_world3d_cohesion')}"
    )
    if lane_risk_focus:
        print(
            "  Lane 风险: "
            f"family={lane_risk_focus.get('dominant_lane_family')} "
            f"| note={lane_risk_focus.get('note')}"
        )
        print(f"  主要风险: {lane_risk_focus.get('primary_risks') or []}")
        if lane_risk_focus.get("suppressed_noise"):
            print(f"  降噪项  : {lane_risk_focus.get('suppressed_noise')}")
        if lane_risk_focus.get("review_focus"):
            print(f"  关注点  : {lane_risk_focus.get('review_focus')}")
    else:
        print(f"  主要风险: {batch.get('primary_risks') or []}")
    print(
        f"  复核建议: target={admission.get('target_bucket')} "
        f"| action={admission.get('suggested_action')} "
        f"| seal={admission.get('eligible_for_training_seal')} "
        f"| blockers={admission.get('blockers') or []}"
    )
    print(f"  人工提示: {batch.get('review_guidance') or []}")


def _print_winner_bank_summary(report: Dict[str, Any]) -> None:
    print("\n[Winner Bank 摘要]")
    print(f"  状态    : {report.get('status')}")
    print(f"  已确认库: {report.get('curated_bank_available')} | entries={report.get('curated_entry_count')}")
    print(f"  候选数  : {report.get('candidate_entry_count')}")
    print(f"  漂移行数: {report.get('drift_row_count')}")
    print(f"  下一步  : {report.get('manual_next_step')}")
    top_risks = list(report.get("top_drift_risks") or [])
    if top_risks:
        print(f"  主要风险: {top_risks[:4]}")
    else:
        print("  漂移画像: 当前还没有稳定的 drift 风险画像，先以当前批次 review_packet 为主。")
    drift_rows = list(report.get("drift_rows") or [])
    for row in drift_rows[:2]:
        print(
            f"  漂移样本: {row.get('image')} | severity={row.get('drift_severity')} "
            f"| flags={list(row.get('drift_flags') or [])[:3]}"
        )
        focus = list(row.get("manual_focus") or [])[:2]
        if focus:
            print(f"  复核重点: {focus}")


def _print_training_admission_summary(summary: Dict[str, Any]) -> None:
    print("\n[Training Admission 摘要]")
    print(f"  可用    : {summary.get('available')} | entries={summary.get('entry_count')}")
    print(f"  原因    : {summary.get('reason')}")
    print(f"  文件    : {summary.get('manifest_file')}")
    print(f"  最近封印: {summary.get('last_sealed_at_utc')}")
    bucket_counts = summary.get("bucket_counts") or {}
    if bucket_counts:
        print(f"  Bucket 分布: {bucket_counts}")
    recent_entries = list(summary.get("recent_entries") or [])
    for row in recent_entries[:3]:
        print(
            f"  已封印样本: {row.get('image')} | bucket={row.get('target_bucket')} "
            f"| owner={row.get('owner')} | at={row.get('sealed_at_utc')}"
        )


def _describe_alignment_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "暂无结论"
    if numeric >= 0.82:
        return "稳定贴近真相"
    if numeric >= 0.76:
        return "进入人工复核区"
    if numeric >= 0.72:
        return "已经出现可见漂移"
    return "偏离明显"


def _describe_percentage_bucket(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "暂无结论"
    if numeric >= 0.85:
        return "稳定"
    if numeric >= 0.65:
        return "可用"
    if numeric >= 0.35:
        return "偏弱"
    return "很弱"


def _describe_canonical_truth_state(packet: Dict[str, Any]) -> None:
    batch = packet.get("batch_summary") or {}
    admission = batch.get("admission_advice") or {}
    canonical_truth = batch.get("canonical_truth_summary") or {}
    face_canonical_status = batch.get("face_canonical_status") or {}
    shortlist = _review_packet_shortlist_entries(packet)
    top_item = shortlist[0] if shortlist else {}
    master_card = top_item.get("master_consistency_card") or {}
    top_truth = top_item.get("canonical_truth_summary") or {}
    top_face = top_item.get("face_canonical_summary") or {}
    provider_status = batch.get("heavy_provider_status") or {}
    provider_name = str(provider_status.get("provider_name") or "").strip()
    lane_family = str(master_card.get("lane_family") or "").strip()
    route_action = str(admission.get("suggested_action") or "").strip()
    print("\n[运行结论]")
    if lane_family:
        print(f"  批次定位: 当前 Top1 更接近 {lane_family} lane，应按对应 lane 标准理解分数。")
    if route_action:
        print(f"  流程动作: {route_action}")

    canonical_available = top_truth.get("canonical_truth_available")
    if canonical_available is True:
        align = top_truth.get("body_shape_truth_alignment")
        beta = top_truth.get("body_shape_beta_similarity")
        topology = top_truth.get("body_topology_signature_similarity")
        measurement = top_truth.get("canonical_measurement_similarity")
        fit = top_truth.get("body_mesh_fit_confidence")
        face_topology = top_face.get("canonical_face_topology_similarity")
        print(
            "  116-1 真相链: 已接入 canonical body truth "
            f"| align={align}（{_describe_alignment_bucket(align)}）"
            f" | beta={beta} | topo={topology} | meas={measurement} | fit={fit}"
        )
        if face_topology is not None:
            print(f"  脸部拓扑: canonical face topology={face_topology}")
        return

    batch_canonical_available = canonical_truth.get("canonical_truth_available")
    if provider_name in {"segformer_body_truth_fusion", "body_canonical_hmr2"}:
        print("  116-1 真相链: 已切到 canonical 模式，但当前批次还没有可用的 HMR2 真相产物。")
        master_truth_artifact_dir = batch.get("master_truth_artifact_dir")
        if master_truth_artifact_dir:
            print(f"  下一步    : 先补齐 {master_truth_artifact_dir} 下的 master artifact 和候选 sidecar。")
        if str(provider_status.get("integration_state") or "") == "missing_smpl_model":
            print("  阻塞      : 本地 HMR2 已接通 GPU，但缺少 SMPL neutral 模型文件。")
            print("  参考文档  : docs/23_body_canonical_hmr2_integration.md")
        else:
            print("  参考文档  : docs/20_body_canonical_artifact_bridge.md")
    elif batch_canonical_available is not True:
        print("  116-1 真相链: 当前仍在轻量代理模式，尚未接入 canonical body truth。")
        print("  建议模式  : 如需严格按 116-1 身材真相复核，请改用 segformer_body_truth_fusion。")

    light_align = master_card.get("body_truth_alignment")
    if light_align is not None:
        print(
            "  轻量真相代理: "
            f"Top1 body_truth_alignment={light_align}（{_describe_alignment_bucket(light_align)}）"
        )
        print("  说明      : 这是当前轻量主链对 116-1 的代理读数，不等于 canonical HMR2 真相结论。")

    face_provider_name = str(face_canonical_status.get("provider_name") or "").strip()
    if face_provider_name in {"face_pose_canonical_bridge", "face_pose_canonical_3ddfa"}:
        if top_face.get("available") is True:
            print(
                "  0 号脸辅助: 已接入 canonical face shadow "
                f"| normalize={top_face.get('face_pose_normalization_confidence')} "
                f"| landmark={top_face.get('canonical_face_landmark_similarity')} "
                f"| identity={top_face.get('canonical_face_identity_similarity')}"
            )
        else:
            print("  0 号脸辅助: 已启用 canonical shadow，但当前没有可用的 face canonical 证据。")
            if face_provider_name == "face_pose_canonical_3ddfa":
                print("  下一步    : clone 3DDFA-V3 到 external/3DDFA-V3，或先补 face canonical sidecar。")
                print("  参考文档  : docs/21_face_pose_canonical_artifact_bridge.md / docs/22_face_pose_canonical_3ddfa_integration.md")


def _print_benchmark_replay_summary(payload: Dict[str, Any]) -> None:
    metrics = payload.get("metrics") or {}
    heavy_metrics = payload.get("heavy_evidence_metrics") or {}
    canonical_metrics = payload.get("canonical_truth_metrics") or {}
    shadow_metrics = payload.get("shadow_view_classifier_metrics") or {}
    face_canonical_metrics = payload.get("face_canonical_metrics") or {}
    lane_focus = payload.get("lane_focus") or {}
    print("\n[Benchmark 回放摘要]")
    print(
        f"  样本数  : report={payload.get('num_report_items')} "
        f"| labels={payload.get('num_labeled_items')} | benchmark={payload.get('num_benchmarked_items')}"
    )
    print(
        "  核心指标: "
        f"release_safety={metrics.get('release_safety_score')} "
        f"| macro_f1={metrics.get('macro_f1')} "
        f"| false_pass={metrics.get('false_pass_rate')} "
        f"| exact={metrics.get('exact_accuracy')}"
    )
    if metrics:
        print(
            f"  结果解读: 当前规则回放整体{_describe_percentage_bucket(metrics.get('release_safety_score'))}，"
            f" false_pass_rate={metrics.get('false_pass_rate')}。"
        )
    if heavy_metrics:
        providers = ",".join(list(heavy_metrics.get("providers") or []))
        print(
            "  重证据  : "
            f"providers={providers or '无'} "
            f"| available={heavy_metrics.get('available_weight_ratio')} "
            f"| confidence={heavy_metrics.get('confidence_mean')} "
            f"| coverage={heavy_metrics.get('coverage_mean')}"
        )
    if canonical_metrics:
        print(
            "  真相证据: "
            f"available={canonical_metrics.get('body_shape_truth_available_weight_ratio')} "
            f"| align={canonical_metrics.get('body_shape_truth_alignment_mean')} "
            f"| beta={canonical_metrics.get('body_shape_beta_similarity_mean')} "
            f"| readiness={canonical_metrics.get('canonical_truth_readiness_score')}"
        )
        if float(canonical_metrics.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0) <= 0.0:
            print("  说明    : 这次 benchmark 还没有真正评到 canonical body truth，当前只是在评轻量或边界证据。")
    if shadow_metrics:
        print(
            "  视角辅助: "
            f"available={shadow_metrics.get('available_weight_ratio')} "
            f"| lane_acc={shadow_metrics.get('lane_accuracy', shadow_metrics.get('shadow_view_lane_accuracy'))} "
            f"| detail_acc={shadow_metrics.get('lane_detail_accuracy', shadow_metrics.get('shadow_view_lane_detail_accuracy'))} "
            f"| agreement={shadow_metrics.get('primary_lane_agreement', shadow_metrics.get('shadow_primary_lane_agreement'))}"
        )
    if face_canonical_metrics:
        providers = ",".join(list(face_canonical_metrics.get("providers") or []))
        print(
            "  脸部辅助: "
            f"providers={providers or '无'} "
            f"| available={face_canonical_metrics.get('available_weight_ratio')} "
            f"| truth={face_canonical_metrics.get('truth_available_weight_ratio')} "
            f"| normalize={face_canonical_metrics.get('face_pose_normalization_confidence_mean')} "
            f"| landmark={face_canonical_metrics.get('canonical_face_landmark_similarity_mean')} "
            f"| readiness={face_canonical_metrics.get('face_canonical_readiness_score')}"
        )
        if float(face_canonical_metrics.get("available_weight_ratio", 0.0) or 0.0) <= 0.0:
            print("  说明    : 这次 benchmark 还没有真正评到 0 号脸 canonical sidecar，当前仍主要依赖原始 face 主链。")
    if lane_focus:
        print(
            "  Lane 解读: "
            f"family={lane_focus.get('dominant_lane_family')} "
            f"| note={lane_focus.get('note')}"
        )
        print(f"  主要风险: {lane_focus.get('primary_risks') or []}")
        if lane_focus.get("suppressed_noise"):
            print(f"  降噪项  : {lane_focus.get('suppressed_noise')}")
        if lane_focus.get("review_focus"):
            print(f"  关注点  : {lane_focus.get('review_focus')}")


def _print_benchmark_heavy_compare_summary(payload: Dict[str, Any]) -> None:
    comparison = payload.get("comparison") or {}
    ranking_generic = list(
        comparison.get("ranking_by_generic_readiness")
        or comparison.get("ranking_by_evidence_readiness")
        or []
    )
    ranking_truth = list(comparison.get("ranking_by_truth_readiness") or [])
    baseline = str(comparison.get("baseline_provider") or "").strip()
    face_canonical_metrics = payload.get("face_canonical_metrics") or {}
    lane_focus = payload.get("lane_focus") or {}
    print("\n[Heavy Compare 摘要]")
    print(f"  基线模式: {_heavy_provider_title(baseline)}")
    note = ((payload.get("comparison_scope") or {}).get("note") or "").strip()
    if note:
        print(f"  说明    : {note}")
    print("  通用榜  :")
    for index, row in enumerate(ranking_generic[:4], start=1):
        if not isinstance(row, dict):
            continue
        provider_name = str(row.get("provider_name") or "").strip()
        print(
            f"    {index}. {_heavy_provider_title(provider_name)} "
            f"| readiness={row.get('evidence_readiness_score')} "
            f"| truth={row.get('canonical_truth_readiness_score')} "
            f"| available={row.get('available_weight_ratio')} "
            f"| confidence={row.get('confidence_mean')} "
            f"| coverage={row.get('coverage_mean')}"
        )
    truth_enabled_rows = [
        row for row in ranking_truth
        if float(row.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0) > 0.0
    ]
    if truth_enabled_rows:
        print("  真相榜  :")
        for index, row in enumerate(truth_enabled_rows[:4], start=1):
            provider_name = str(row.get("provider_name") or "").strip()
            print(
                f"    {index}. {_heavy_provider_title(provider_name)} "
                f"| truth={row.get('canonical_truth_readiness_score')} "
                f"| align={row.get('body_shape_truth_alignment_mean')} "
                f"| beta={row.get('body_shape_beta_similarity_mean')} "
                f"| available={row.get('body_shape_truth_available_weight_ratio')}"
            )
    providers = payload.get("providers") or {}
    top_row = ranking_generic[0] if ranking_generic else {}
    top_provider_name = str(top_row.get("provider_name") or "").strip()
    top_provider = providers.get(top_provider_name) if isinstance(providers, dict) else {}
    top_canonical = (top_provider or {}).get("canonical_truth_metrics") or {}
    if face_canonical_metrics:
        providers_text = ",".join(list(face_canonical_metrics.get("providers") or []))
        print(
            "  脸部辅助: "
            f"providers={providers_text or '无'} "
            f"| available={face_canonical_metrics.get('available_weight_ratio')} "
            f"| truth={face_canonical_metrics.get('truth_available_weight_ratio')} "
            f"| normalize={face_canonical_metrics.get('face_pose_normalization_confidence_mean')} "
            f"| readiness={face_canonical_metrics.get('face_canonical_readiness_score')}"
        )
        print("  说明    : 脸部 canonical 属于固定 shadow 证据，本次 heavy compare 不拿它参与 provider 排名。")
    if lane_focus:
        print(
            "  Lane 解读: "
            f"family={lane_focus.get('dominant_lane_family')} "
            f"| note={lane_focus.get('note')}"
        )
        print(f"  主要风险: {lane_focus.get('primary_risks') or []}")
        if lane_focus.get("suppressed_noise"):
            print(f"  降噪项  : {lane_focus.get('suppressed_noise')}")
        if lane_focus.get("review_focus"):
            print(f"  关注点  : {lane_focus.get('review_focus')}")
    if top_provider_name:
        if float(top_canonical.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0) > 0.0:
            print(
                "  结论    : 当前通用榜第一已经真正接入 canonical body truth，"
                f" align={top_canonical.get('body_shape_truth_alignment_mean')}。"
            )
        elif truth_enabled_rows:
            best_truth = truth_enabled_rows[0]
            print(
                "  结论    : 通用榜当前仍由基础边界/体态几何模式领先，"
                "但 canonical body truth 已进入对比；"
                f"最佳真相模式={_heavy_provider_title(best_truth.get('provider_name'))} "
                f"| truth={best_truth.get('canonical_truth_readiness_score')} "
                f"| align={best_truth.get('body_shape_truth_alignment_mean')}。"
            )
        else:
            print("  结论    : 当前排名主要仍由边界/体态几何证据决定，canonical body truth 还没有真正进入对比。")


def _review_packet_shortlist_entries(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = list((packet.get("ranked_review_packet") or {}).get("groups") or [])
    if len(groups) == 0:
        return []
    return list((groups[0] or {}).get("shortlist") or [])


def _print_shortlist_review_for_promotion(packet: Dict[str, Any]) -> None:
    shortlist = _review_packet_shortlist_entries(packet)
    if len(shortlist) == 0:
        return
    print("\n[Shortlist]")
    for row in shortlist:
        master = row.get("master_consistency_card") or {}
        admission = row.get("admission_advice") or {}
        print(
            f"  rank {row.get('rank')}: {row.get('image')} | score={row.get('selection_score')} "
            f"| master={master.get('hybrid_master_alignment')} | admit={admission.get('suggestion')}"
        )
        winner_reasons = list(row.get("winner_reasons") or [])[:2]
        cautions = list((master.get("cautions") or []))[:2]
        if winner_reasons:
            print(f"    strengths: {winner_reasons}")
        if cautions:
            print(f"    cautions : {cautions}")
        focus = list(master.get("manual_focus") or [])[:2]
        if focus:
            print(f"    focus    : {focus}")
    groups = list((packet.get("ranked_review_packet") or {}).get("groups") or [])
    pairwise = list((groups[0] if groups else {}).get("pairwise_compare_cards") or [])
    if pairwise:
        print("\n[Pairwise Focus]")
        for card in pairwise[:2]:
            print(
                f"  top1={card.get('top_image')} vs rank{card.get('candidate_rank')}={card.get('candidate_image')} "
                f"| focus={list(card.get('combined_manual_focus') or card.get('manual_focus') or [])[:3]}"
            )
            prompts = list(card.get("manual_review_prompts") or [])[:2]
            if prompts:
                print(f"    prompts : {prompts}")


def _select_winner_candidate_by_rank(entries: Sequence[Dict[str, Any]], rank_value: int) -> Dict[str, Any]:
    for entry in entries:
        if int(entry.get("rank") or 0) == int(rank_value):
            return dict(entry)
    raise ValueError(f"winner shortlist rank not found: {rank_value}")


def _select_winner_candidate_interactively(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    print("[交互引导] 可晋升到 winner bank 的候选:")
    for index, entry in enumerate(entries, start=1):
        master = entry.get("master_consistency_card") or {}
        print(
            f"  {index}. rank={entry.get('rank')} {entry.get('image')} | score={entry.get('selection_score')} "
            f"| profile={entry.get('target_profile')} | reasons={list(entry.get('winner_reasons') or [])[:2]}"
        )
        if master:
            print(
                f"     master={master.get('hybrid_master_alignment')} "
                f"lane={master.get('lane_validity')} cautions={list(master.get('cautions') or [])[:2]}"
            )
            focus = list(master.get("manual_focus") or [])[:2]
            if focus:
                print(f"     focus={focus}")
    while True:
        choice = _prompt_text("请选择要写入 winner bank 的候选编号", "1")
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(entries):
                return dict(entries[index - 1])
        print(f"[交互引导] 无法识别的候选编号: {choice}")


def _maybe_enable_interactive_wizard(args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    if args.interactive or len(raw_argv) > 0:
        return
    try:
        args.interactive = _prompt_yes_no("是否进入交互式引导", default=True)
    except ValueError:
        args.interactive = False


def _select_run_mode_interactively(default: str = "qa") -> str:
    return _select_choice(
        "请选择运行模式",
        [
            ("qa", "运行当前 input 图片集质检，适合日常候选筛选和人工初筛"),
            ("benchmark", "回放现有 qa_report 与标签集，适合看规则效果、分组指标与回归表现"),
            ("optuna", "在冻结 benchmark 上做离线参数拟合，不会重跑视觉模型"),
            ("calibrate", "使用校准图集重算质量阈值，仅用于阈值标定场景"),
        ],
        default=default,
    )


def _select_benchmark_action_interactively(default: str = "replay") -> str:
    return _select_choice(
        "请选择 benchmark 子动作",
        [
            ("replay", "用标签文件回放已保存的 qa_report，输出评测指标"),
            ("compare_heavy", "在同一批标签上对比 heavy provider 的可用率、置信度和覆盖率"),
            ("template", "根据当前 qa_report 导出标签模板，便于后续人工补标"),
            ("seal", "给已有标签文件补齐冻结元数据，供 Optuna 拟合前使用"),
        ],
        default=default,
    )

def _prepare_interactive_args(args: argparse.Namespace, base_dir: Path) -> None:
    workflow = str(args.workflow or "").strip()
    effective_mode = str(args.mode or "qa")
    if not args.interactive:
        return

    if args.workflow is None and args.mode is None:
        args.workflow = _select_workflow_interactively(default="shot_review")
        workflow = str(args.workflow or "").strip()
        _print_workflow_explanation(workflow)

    if workflow == "shot_review":
        args.mode = "qa"
        if args.profile is None:
            args.profile = _select_review_profile_interactively(default=str(args.profile or "body_gold_fullbody"))
        if getattr(args, "heavy_provider", None) is None:
            args.heavy_provider = _select_heavy_provider_interactively(default="segformer_body_fusion")
        _print_heavy_provider_explanation(args.heavy_provider)
        return

    if workflow == "setup_external_models":
        return

    if workflow in {
        "preflight_batch",
        "prepare_input_manifest",
        "fill_input_manifest_defaults",
        "merge_input_manifest_metadata",
        "refresh_review_run_index",
        "prepare_front_bootstrap_review",
        "refresh_review_status_board",
        "prepare_split_batch_plan",
        "materialize_split_batches",
        "refresh_review_artifacts",
        "inspect_review_packet",
        "prepare_winner_bank_review",
        "promote_winner",
        "seal_training_admission",
        "winner_bank_status",
        "training_admission_status",
    }:
        return

    if args.mode is None:
        args.mode = _select_run_mode_interactively(default=effective_mode)
        effective_mode = str(args.mode)
    if effective_mode == "qa" and getattr(args, "heavy_provider", None) is None:
        args.heavy_provider = _select_heavy_provider_interactively(default="segformer_body_fusion")
        _print_heavy_provider_explanation(args.heavy_provider)

    if effective_mode == "benchmark":
        has_action = (
            args.benchmark_template_out is not None
            or args.benchmark_seal_labels
            or args.benchmark_labels is not None
            or bool(getattr(args, "benchmark_compare_heavy_providers", None))
        )
        if not has_action:
            action = _select_benchmark_action_interactively(default="replay")
            if action == "template":
                args.benchmark_template_out = _prompt_path(
                    "请输入 benchmark 模板输出路径",
                    base_dir=base_dir,
                    default="outputs/benchmark_labels.interactive.json",
                    must_exist=False,
                )
            elif action == "seal":
                args.benchmark_seal_labels = True
                args.benchmark_labels = _prompt_path(
                    "请输入需要封板的 benchmark 标签文件路径",
                    base_dir=base_dir,
                    default="outputs/benchmark_labels.interactive.json",
                    must_exist=True,
                )
            elif action == "compare_heavy":
                args.benchmark_labels = _prompt_path(
                    "请输入 benchmark 标签文件路径",
                    base_dir=base_dir,
                    default="outputs/benchmark_labels_verify.json",
                    must_exist=True,
                )
                args.benchmark_compare_heavy_providers = _prompt_heavy_provider_compare_targets()
                args.benchmark_image_root = _prompt_path(
                    "请输入重型证据回放使用的原图根目录",
                    base_dir=base_dir,
                    default="input",
                    must_exist=True,
                )
                if args.benchmark_output is None:
                    args.benchmark_output = _resolve_cli_path(
                        Path("outputs/benchmark_heavy_compare.interactive.json"),
                        base_dir,
                    )
            else:
                args.benchmark_labels = _prompt_path(
                    "请输入 benchmark 标签文件路径",
                    base_dir=base_dir,
                    default="outputs/benchmark_labels_verify.json",
                    must_exist=True,
                )
                if args.benchmark_output is None:
                    args.benchmark_output = _resolve_cli_path(
                        Path("outputs/benchmark_replay.interactive.json"),
                        base_dir,
                    )

        if args.benchmark_template_out is not None and args.benchmark_report is None:
            args.benchmark_report = _prompt_path(
                "请输入用于导出模板的 QA 报告路径",
                base_dir=base_dir,
                default="outputs/qa_report.json",
                must_exist=True,
            )
        if (
            args.benchmark_template_out is None
            and not args.benchmark_seal_labels
            and args.benchmark_labels is None
        ):
            args.benchmark_labels = _prompt_path(
                "请输入 benchmark 标签文件路径",
                base_dir=base_dir,
                default="outputs/benchmark_labels_verify.json",
                must_exist=True,
            )
        if (
            args.benchmark_template_out is None
            and not args.benchmark_seal_labels
            and args.benchmark_report is None
        ):
            args.benchmark_report = _prompt_path(
                "请输入 QA 报告路径",
                base_dir=base_dir,
                default="outputs/qa_report.json",
                must_exist=True,
            )

    if effective_mode == "optuna" and args.benchmark_labels is None:
        args.benchmark_labels = _prompt_path(
            "请输入冻结 benchmark 标签文件路径",
            base_dir=base_dir,
            default="outputs/benchmark_labels.interactive.json",
            must_exist=True,
        )


def _handle_workflow_action(args: argparse.Namespace, base_dir: Path) -> Optional[int]:
    workflow = str(args.workflow or "").strip()
    if workflow in {"", "shot_review", "advanced_cli"}:
        return None

    paths = _default_review_paths(base_dir, args.artifacts_dir)
    if workflow == "prepare_input_manifest":
        from core.qa_input_manifest import create_or_update_input_manifest

        input_dir = _resolve_input_dir_arg(args.input_dir, base_dir)
        result = create_or_update_input_manifest(
            input_dir=input_dir,
            manifest_path=args.input_manifest,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("[交互引导] 下一步：补齐 prompt_id / seed / anchor_source / intended_view，然后再跑 preflight_batch。")
        return 0

    if workflow == "fill_input_manifest_defaults":
        from core.qa_input_manifest import fill_input_manifest_defaults

        input_dir = _resolve_input_dir_arg(args.input_dir, base_dir)
        result = fill_input_manifest_defaults(
            input_dir=input_dir,
            manifest_path=args.input_manifest,
            prompt_id=args.manifest_prompt_id,
            seed=args.manifest_seed,
            anchor_source=args.manifest_anchor_source,
            generator_name=args.manifest_generator_name,
            generator_version=args.manifest_generator_version,
            prompt_pack=args.manifest_prompt_pack,
            note=args.manifest_note,
            missing_only=not bool(args.manifest_overwrite_existing),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("[交互引导] 下一步：重新跑 preflight_batch，确认 prompt intent 字段覆盖率是否达标。")
        return 0

    if workflow == "merge_input_manifest_metadata":
        from core.qa_input_manifest import merge_input_manifest_item_metadata

        input_dir = _resolve_input_dir_arg(args.input_dir, base_dir)
        if args.manifest_metadata_file is None:
            raise ValueError("merge_input_manifest_metadata requires --manifest-metadata-file")
        result = merge_input_manifest_item_metadata(
            input_dir=input_dir,
            metadata_file=_resolve_cli_path(args.manifest_metadata_file, base_dir),
            manifest_path=args.input_manifest,
            missing_only=not bool(args.manifest_overwrite_existing),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("[交互引导] 下一步：重新跑 preflight_batch，确认逐图 metadata 合并后的字段覆盖率。")
        return 0

    if workflow == "refresh_review_run_index":
        from core.qa_run_index import build_review_run_index

        result = build_review_run_index(
            base_dir=base_dir,
            output_file=paths["review_run_index"],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[运行索引] {paths['review_run_index']}")
        print("[交互引导] 先看 recommended_runs.front_bootstrap_snapshot，再决定下一轮人工 winner 复审。")
        return 0

    if workflow == "prepare_front_bootstrap_review":
        from core.qa_run_index import build_review_run_index
        from core.qa_front_bootstrap_review import build_front_bootstrap_review_sheet

        run_index_path = paths["review_run_index"]
        build_review_run_index(
            base_dir=base_dir,
            output_file=run_index_path,
        )
        result = build_front_bootstrap_review_sheet(
            run_index_file=run_index_path,
            output_file=paths["front_bootstrap_review_sheet"],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[front bootstrap 复审表] {paths['front_bootstrap_review_sheet']}")
        print("[交互引导] 先在 top_candidates 里做人工只选一的结论，再考虑 promote_winner。")
        return 0

    if workflow == "refresh_review_status_board":
        from core.qa_front_bootstrap_review import build_front_bootstrap_review_sheet
        from core.qa_run_index import build_review_run_index
        from core.qa_status_board import build_review_status_board

        run_index_path = paths["review_run_index"]
        build_review_run_index(
            base_dir=base_dir,
            output_file=run_index_path,
        )
        build_front_bootstrap_review_sheet(
            run_index_file=run_index_path,
            output_file=paths["front_bootstrap_review_sheet"],
        )
        result = build_review_status_board(
            base_dir=base_dir,
            output_file=paths["review_status_board"],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[总控状态板] {paths['review_status_board']}")
        print("[交互引导] 先看 next_actions，再决定补 manifest 还是进入 front bootstrap 人工复审。")
        return 0

    if workflow == "preflight_batch":
        try:
            from core.qa_preflight import create_lightweight_preflight_config, run_preflight_batch
        except ModuleNotFoundError as exc:
            missing_name = str(getattr(exc, "name", "") or "unknown")
            raise RuntimeError(
                f"preflight_batch requires the runtime stack because module '{missing_name}' is missing. "
                "Run it with .venv\\Scripts\\python.exe or install the full QA dependencies."
            ) from exc

        runtime = None
        config = create_lightweight_preflight_config(base_dir)
        try:
            runtime = create_runtime(base_dir)
            config = runtime.config
        except Exception:
            runtime = None
        input_dir = _resolve_input_dir_arg(args.input_dir, base_dir)
        if runtime is not None:
            _override_runtime_input_dir(runtime, input_dir)
            config = runtime.config
        target_profile = str(args.profile or config.review.active_profile or "").strip()
        if not target_profile:
            raise ValueError("preflight_batch could not resolve an active profile")

        result = run_preflight_batch(
            runtime,
            config=config,
            input_dir=input_dir,
            target_profile=target_profile,
            manifest_path=args.input_manifest,
        )
        paths["preflight_batch"].write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_preflight_summary(result)
        print(f"[预检文件] {paths['preflight_batch']}")
        print("[交互引导] 如果这里已经 WARN/FAIL，先拆批或补齐 input manifest，再跑 shot_review。")
        return 0

    if workflow == "prepare_split_batch_plan":
        from core.qa_batch_split import build_batch_split_plan

        payload = build_batch_split_plan(
            review_packet_file=paths["review_packet"],
            preflight_file=paths["preflight_batch"],
            output_file=paths["batch_split_plan"],
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        _print_batch_split_plan_summary(payload)
        print(f"[拆批方案] {paths['batch_split_plan']}")
        print("[交互引导] 先按 lane_family 拆到独立 input 目录，再对每个子批次分别跑 preflight_batch / shot_review。")
        return 0

    if workflow == "materialize_split_batches":
        from core.qa_batch_split import materialize_split_batches

        payload = materialize_split_batches(
            plan_file=paths["batch_split_plan"],
            input_dir=(base_dir / "input").resolve(),
            output_root=(base_dir / "input_split").resolve(),
        )
        paths["materialized_batch_split"].write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        _print_materialized_split_summary(payload)
        print(f"[拆批落盘摘要] {paths['materialized_batch_split']}")
        print("[交互引导] 下一步：进入对应 input_split/<lane>/，先跑 preflight_batch，再跑 shot_review。")
        return 0

    if workflow == "setup_external_models":
        status_before = _external_setup_status(base_dir)
        _print_external_setup_status(status_before)
        skip_checkout = False
        if args.interactive:
            skip_checkout = _prompt_yes_no(
                "是否跳过 external 仓库 checkout 到固定提交（仅同步补丁，不切换版本）",
                default=False,
            )
        bootstrap_script = (base_dir / "bootstrap_external_models.ps1").resolve()
        apply_script = (base_dir / "apply_external_patches.ps1").resolve()
        bootstrap_args = ["-SkipCheckout"] if skip_checkout else []
        print(f"[交互引导] 执行 external bootstrap: {bootstrap_script}")
        _run_local_powershell_script(bootstrap_script, base_dir=base_dir, extra_args=bootstrap_args)
        print(f"[交互引导] 执行 external patch apply: {apply_script}")
        _run_local_powershell_script(apply_script, base_dir=base_dir)
        status_after = _external_setup_status(base_dir)
        _print_external_setup_status(status_after)
        print("[交互引导] 下一步：")
        print("  1. 运行批次复核，确认 3DDFA/HMR2 真相链状态。")
        print("  2. 如 3DDFA 仍缺资产，请按上面的缺失清单补齐 external/3DDFA-V3/assets。")
        print("  3. 如 HMR2 仍缺 SMPL，请确认 basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 路径。")
        return 0

    if workflow == "refresh_review_artifacts":
        from core.qa_run_index import build_review_run_index
        from core.qa_review_refresh import rebuild_review_artifacts_from_report

        result = rebuild_review_artifacts_from_report(paths["qa_report"], output_dir=paths["qa_report"].parent)
        build_review_run_index(
            base_dir=base_dir,
            output_file=paths["review_run_index"],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        packet = _load_json_file(paths["review_packet"], "review packet")
        _print_review_packet_summary(packet)
        _describe_canonical_truth_state(packet)
        print(f"[报告] {paths['qa_report']}")
        print(f"[排序] {paths['ranked_candidates']}")
        print(f"[复核摘要文件] {paths['review_packet']}")
        print(f"[GPT 分析包] {paths['gpt_review_packet']}")
        print(f"[运行索引] {paths['review_run_index']}")
        print("[交互引导] 当整轮 heavy QA 超时，或只想回填 topology 字段时，优先用这个工作流刷新复审产物。")
        return 0

    if workflow == "inspect_review_packet":
        packet = _load_json_file(paths["review_packet"], "review packet")
        _print_review_packet_summary(packet)
        _describe_canonical_truth_state(packet)
        print(f"[复核摘要文件] {paths['review_packet']}")
        if paths["gpt_review_packet"].exists():
            print(f"[GPT 分析包] {paths['gpt_review_packet']}")
            if paths["review_artifacts"].exists():
                print(f"[产物索引] {paths['review_artifacts']}")
            print("[交互引导] 默认发 gpt_review_packet.json 给 GPT 做批次分析。")
            print("[交互引导] 只有做跨批次漂移分析时，再额外附带 winner_bank_report.json。")
        else:
            print("[交互引导] 当前尚未生成 gpt_review_packet.json，请先重跑本轮 QA。")
        return 0

    if workflow == "prepare_winner_bank_review":
        from core.qa_winner_bank_review import build_winner_bank_review_packet

        payload = build_winner_bank_review_packet(
            candidate_file=paths["winner_bank_candidate"],
            winner_bank_report_file=paths["winner_bank_report"],
            review_packet_file=paths["review_packet"],
            preflight_file=paths["preflight_batch"],
            output_file=paths["winner_bank_review_packet"],
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[Winner Bank 复审包] {paths['winner_bank_review_packet']}")
        if payload.get("promotion_ready"):
            print("[交互引导] 当前批次允许进入 winner bank 人工确认。先看 recommended_candidate，再决定是否执行 promote_winner。")
        else:
            print("[交互引导] 当前批次不建议 promote winner。先处理 promotion_blockers，再重新生成 review packet。")
        return 0

    if workflow == "winner_bank_status":
        report = _load_json_file(paths["winner_bank_report"], "winner bank report")
        _print_winner_bank_summary(report)
        print(f"[Winner Bank 报告] {paths['winner_bank_report']}")
        if paths["winner_bank_review_packet"].exists():
            print(f"[Winner Bank 复审包] {paths['winner_bank_review_packet']}")
        return 0

    if workflow == "training_admission_status":
        from core.qa_training_admission import load_training_admission_manifest_summary

        summary = load_training_admission_manifest_summary(paths["training_admission_manifest"])
        _print_training_admission_summary(summary)
        print(f"[Training Admission 清单] {paths['training_admission_manifest']}")
        return 0

    if workflow == "promote_winner":
        from core.qa_winner_bank import load_winner_bank_candidates, promote_winner_entry

        candidate_payload = load_winner_bank_candidates(paths["winner_bank_candidate"])
        if not candidate_payload.get("available"):
            raise ValueError(
                f"winner bank candidate file is not ready: {paths['winner_bank_candidate']} "
                f"({candidate_payload.get('reason')})"
            )
        entries = list(candidate_payload.get("entries") or [])
        review_packet = _load_json_file(paths["review_packet"], "review packet") if paths["review_packet"].exists() else {}
        if review_packet:
            _print_review_packet_summary(review_packet)
            _describe_canonical_truth_state(review_packet)
            _print_shortlist_review_for_promotion(review_packet)
        selected_entry: Optional[Dict[str, Any]] = None
        if args.winner_rank is not None:
            selected_entry = _select_winner_candidate_by_rank(entries, args.winner_rank)
        elif args.winner_image:
            needle = str(args.winner_image).strip()
            for entry in entries:
                if needle in {
                    str(entry.get("image") or "").strip(),
                    str(entry.get("record_key") or "").strip(),
                }:
                    selected_entry = dict(entry)
                    break
            if selected_entry is None:
                raise ValueError(f"winner candidate not found: {needle}")
        elif len(entries) == 1 and not args.interactive:
            selected_entry = dict(entries[0])
        elif args.interactive:
            selected_entry = _select_winner_candidate_interactively(entries)
        else:
            raise ValueError("promote_winner requires --winner-rank or --winner-image when multiple candidates exist")

        manual_note = args.winner_note
        if args.interactive and not manual_note:
            manual_note = _prompt_text("可选：为这次人工确认写一句备注", "")

        result = promote_winner_entry(
            selected_entry,
            paths["winner_bank"],
            manual_note=manual_note,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("[交互引导] 已写入 winner bank。建议下一次重新跑 shot review，让系统用新的 curated bank 做跨批次漂移检查。")
        return 0

    if workflow == "seal_training_admission":
        from core.qa_training_admission import (
            load_training_admission_manifest_summary,
            seal_training_admission_entry,
        )
        from core.qa_winner_bank import load_winner_bank_candidates

        candidate_payload = load_winner_bank_candidates(paths["winner_bank_candidate"])
        if not candidate_payload.get("available"):
            raise ValueError(
                f"winner bank candidate file is not ready: {paths['winner_bank_candidate']} "
                f"({candidate_payload.get('reason')})"
            )
        entries = list(candidate_payload.get("entries") or [])
        review_packet = _load_json_file(paths["review_packet"], "review packet") if paths["review_packet"].exists() else {}
        if review_packet:
            _print_review_packet_summary(review_packet)
            _describe_canonical_truth_state(review_packet)
            _print_shortlist_review_for_promotion(review_packet)
        selected_entry: Optional[Dict[str, Any]] = None
        if args.winner_rank is not None:
            selected_entry = _select_winner_candidate_by_rank(entries, args.winner_rank)
        elif args.winner_image:
            needle = str(args.winner_image).strip()
            for entry in entries:
                if needle in {
                    str(entry.get("image") or "").strip(),
                    str(entry.get("record_key") or "").strip(),
                }:
                    selected_entry = dict(entry)
                    break
            if selected_entry is None:
                raise ValueError(f"winner candidate not found: {needle}")
        elif len(entries) == 1 and not args.interactive:
            selected_entry = dict(entries[0])
        elif args.interactive:
            selected_entry = _select_winner_candidate_interactively(entries)
        else:
            raise ValueError(
                "seal_training_admission requires --winner-rank or --winner-image when multiple candidates exist"
            )

        manual_owner = str(args.admission_owner or "").strip()
        if args.interactive and not manual_owner:
            manual_owner = _prompt_text("请输入这次 training admission seal 的负责人", os.environ.get("USERNAME", ""))
        if not manual_owner:
            raise ValueError("seal_training_admission requires --admission-owner")

        manual_note = args.admission_note
        if args.interactive and not manual_note:
            manual_note = _prompt_text("可选：为这次 training admission seal 写一句备注", "")

        batch_summary = (review_packet.get("batch_summary") or {}) if review_packet else {}
        admission = batch_summary.get("admission_advice") or {}
        report_meta = ((review_packet.get("debug") or {}).get("report_meta") or {}) if review_packet else {}
        result = seal_training_admission_entry(
            selected_entry,
            paths["training_admission_manifest"],
            release_gate=admission.get("release_gate") or batch_summary.get("release_gate") or {},
            admission_advice=admission,
            batch_preflight=batch_summary.get("batch_preflight") or admission.get("batch_preflight") or {},
            evidence_completeness=batch_summary.get("evidence_completeness") or admission.get("evidence_completeness") or {},
            threshold_hash=((report_meta.get("threshold_snapshot") or {}).get("hash")),
            anchor_snapshot=report_meta.get("anchor_registry_snapshot") or {},
            source_batch={
                "target_profile": batch_summary.get("target_profile"),
                "review_packet_generated_at_utc": review_packet.get("generated_at_utc") if review_packet else None,
                "report_generated_at_utc": report_meta.get("generated_at_utc"),
                "top_ranked_image": ((batch_summary.get("selection") or {}).get("top_ranked_image")),
            },
            source_files=(review_packet.get("source_files") or {}) if review_packet else {},
            manual_owner=manual_owner,
            manual_note=manual_note,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") != "ok":
            print("[交互引导] 当前批次不满足 training admission 硬门。请先看 review packet 的 release gate、batch_preflight、evidence_completeness 和 blockers。")
            return 1
        summary = load_training_admission_manifest_summary(paths["training_admission_manifest"])
        _print_training_admission_summary(summary)
        print("[交互引导] 已写入 training admission manifest。winner bank 与正式训练准入现已物理分开。")
        return 0

    raise ValueError(f"unsupported workflow: {workflow}")


def _print_preflight_summary(payload: Dict[str, Any]) -> None:
    batch = payload.get("batch_preflight") or {}
    manifest = payload.get("manifest_summary") or {}
    print("\n[批次预检]")
    print(f"  训练层: {payload.get('target_profile')}")
    print(f"  图片数: {payload.get('input_count')}")
    print(
        "  Manifest: "
        f"available={manifest.get('available')} "
        f"| coverage={manifest.get('matched_image_share')} "
        f"| path={manifest.get('path')}"
    )
    print(
        "  预检治理: "
        f"status={batch.get('status')} "
        f"| lane_source={batch.get('governance_lane_source')} "
        f"| dominant={batch.get('dominant_lane_family')} "
        f"| share={batch.get('dominant_lane_share')} "
        f"| purity={batch.get('lane_purity_score')}"
    )
    print(
        "  意图对照: "
        f"source={batch.get('prompt_intent_source')} "
        f"| weak_prior={batch.get('prompt_intent_is_weak_prior')} "
        f"| coverage={batch.get('intended_lane_coverage')} "
        f"| match={batch.get('intended_observed_lane_match_share')}"
    )
    if batch.get("manifest_required_field_coverage"):
        print(f"  Manifest 字段: {batch.get('manifest_required_field_coverage')}")
    if batch.get("lane_counts"):
        print(f"  观测 lane: {batch.get('lane_counts')}")
    if batch.get("intended_lane_counts"):
        print(f"  意图 lane: {batch.get('intended_lane_counts')}")
    if batch.get("reasons"):
        print(f"  风险  : {batch.get('reasons')}")
    print(f"  建议动作: {batch.get('recommended_action')}")


def _print_batch_split_plan_summary(payload: Dict[str, Any]) -> None:
    print("\n[拆批方案]")
    print(f"  lane_source: {payload.get('lane_source')}")
    print(f"  图片数     : {payload.get('input_count')}")
    print(f"  split_required: {payload.get('split_required')}")
    print(f"  建议动作   : {payload.get('recommended_action')}")
    lane_groups = list(payload.get("lane_groups") or [])
    for group in lane_groups[:6]:
        print(
            f"  - {group.get('lane_family')}: count={group.get('count')} "
            f"share={group.get('share')} profile={group.get('suggested_profile')} "
            f"dir={group.get('suggested_input_dir')}"
        )


def _print_materialized_split_summary(payload: Dict[str, Any]) -> None:
    print("\n[拆批落盘]")
    print(f"  output_root: {payload.get('output_root')}")
    print(f"  lane_group_count: {payload.get('lane_group_count')}")
    print(f"  copied_images: {payload.get('total_copied_images')}")
    print(f"  copied_sidecars: {payload.get('total_copied_sidecars')}")
    for group in list(payload.get("lane_groups") or [])[:6]:
        print(
            f"  - {group.get('lane_family')}: images={group.get('copied_images')} "
            f"sidecars={group.get('copied_sidecars')} manifest={group.get('manifest_path')}"
        )
        missing_files = list(group.get("missing_files") or [])
        if missing_files:
            print(f"    missing={missing_files[:5]}")


def _run_shot_review_preflight(
    *,
    base_dir: Path,
    target_profile: Optional[str],
    input_manifest: Optional[Path],
    input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from core.qa_preflight import create_lightweight_preflight_config, run_preflight_batch

    runtime = None
    config = create_lightweight_preflight_config(base_dir)
    resolved_input_dir = input_dir or config.paths.dir_input
    try:
        runtime = create_runtime(base_dir)
        config = runtime.config
        _override_runtime_input_dir(runtime, resolved_input_dir)
        config = runtime.config
    except Exception:
        runtime = None
    config.paths = replace(config.paths, dir_input=resolved_input_dir)

    resolved_profile = str(target_profile or config.review.active_profile or "").strip()
    if not resolved_profile:
        raise ValueError("shot_review preflight could not resolve an active profile")

    result = run_preflight_batch(
        runtime,
        config=config,
        input_dir=resolved_input_dir,
        target_profile=resolved_profile,
        manifest_path=input_manifest,
    )
    paths = _default_review_paths(base_dir)
    paths["preflight_batch"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _select_preset_interactively(
    *,
    base_dir: Path,
    presets_path: Optional[Path],
    fit_only: bool,
) -> Dict[str, Any]:
    from core.qa_optuna import list_optuna_mode_presets, resolve_optuna_mode_preset

    bundle = list_optuna_mode_presets(base_dir=base_dir, presets_path=presets_path)
    preset_rows = [
        row
        for row in bundle.get("presets", [])
        if (not fit_only) or bool(row.get("fit_enabled", False))
    ]
    if len(preset_rows) == 0:
        raise ValueError("当前没有可供交互选择的 Optuna 预设")

    print("[交互引导] 可选预设:")
    for index, row in enumerate(preset_rows, start=1):
        profile = str(row.get("recommended_runtime_profile", "")).strip() or "-"
        fit_tag = "可拟合" if bool(row.get("fit_enabled", False)) else "仅评估"
        print(f"  {index}. {row['name']} | {row.get('label', row['name'])} | {fit_tag} | profile={profile}")
        description = str(row.get("description", "")).strip()
        if description:
            print(f"     {description}")

    default_name = str(bundle.get("default_preset", "")).strip()
    if fit_only and default_name:
        fit_names = {row["name"] for row in preset_rows}
        if default_name not in fit_names:
            default_name = "front_core_fit" if "front_core_fit" in fit_names else preset_rows[0]["name"]
    elif fit_only:
        fit_names = {row["name"] for row in preset_rows}
        default_name = "front_core_fit" if "front_core_fit" in fit_names else preset_rows[0]["name"]
    elif not default_name:
        default_name = preset_rows[0]["name"]

    while True:
        choice = _prompt_text("请选择预设（输入编号或名称）", default_name)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(preset_rows):
                return resolve_optuna_mode_preset(
                    base_dir=base_dir,
                    preset_name=preset_rows[index - 1]["name"],
                    presets_path=presets_path,
                )
        matched = [row for row in preset_rows if row["name"] == choice]
        if matched:
            return resolve_optuna_mode_preset(
                base_dir=base_dir,
                preset_name=matched[0]["name"],
                presets_path=presets_path,
            )
        print(f"[交互引导] 无法识别的预设: {choice}")


def _resolve_optuna_preset_info(
    *,
    base_dir: Path,
    preset_name: Optional[str],
    presets_path: Optional[Path],
    interactive: bool,
    fit_only: bool,
) -> Optional[Dict[str, Any]]:
    from core.qa_optuna import resolve_optuna_mode_preset

    if preset_name:
        return resolve_optuna_mode_preset(
            base_dir=base_dir,
            preset_name=preset_name,
            presets_path=presets_path,
        )
    if interactive:
        return _select_preset_interactively(
            base_dir=base_dir,
            presets_path=presets_path,
            fit_only=fit_only,
        )
    return None


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
        "--workflow",
        choices=[
            "shot_review",
            "preflight_batch",
            "prepare_input_manifest",
            "fill_input_manifest_defaults",
            "merge_input_manifest_metadata",
            "refresh_review_run_index",
            "prepare_front_bootstrap_review",
            "refresh_review_status_board",
            "prepare_split_batch_plan",
            "materialize_split_batches",
            "refresh_review_artifacts",
            "inspect_review_packet",
            "prepare_winner_bank_review",
            "promote_winner",
            "seal_training_admission",
            "winner_bank_status",
            "training_admission_status",
            "setup_external_models",
            "advanced_cli",
        ],
        help="User-facing workflow entry. Prefer this over --mode when you want task-oriented review actions.",
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
        "--interactive",
        action="store_true",
        help="Enable interactive preset selection and benchmark metadata prompts.",
    )
    parser.add_argument(
        "--heavy-provider",
        choices=[
            "segformer_body_fusion",
            "segformer_body_truth_fusion",
            "segformer_parser",
            "body_measure_lite",
            "body_canonical_hmr2",
            "disabled",
        ],
        help="Override the heavy evidence provider for QA mode.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Optional input manifest JSON. Defaults to input/input_manifest.json when present.",
    )
    parser.add_argument(
        "--manifest-prompt-id",
        help="Shared prompt_id written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-seed",
        help="Shared seed written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-anchor-source",
        help="Shared anchor_source written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-generator-name",
        help="Shared generator_name written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-generator-version",
        help="Shared generator_version written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-prompt-pack",
        help="Shared prompt_pack written into input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-note",
        help="Optional note appended to input_manifest items by fill_input_manifest_defaults.",
    )
    parser.add_argument(
        "--manifest-overwrite-existing",
        action="store_true",
        help="Allow fill_input_manifest_defaults to overwrite non-empty manifest fields instead of only filling missing values.",
    )
    parser.add_argument(
        "--manifest-metadata-file",
        type=Path,
        help="External JSON file keyed by image name, used by merge_input_manifest_metadata to fill per-item manifest fields.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Override the input directory for preflight_batch, prepare_input_manifest, and qa shot_review.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Override the review artifact directory for inspect_review_packet, prepare_winner_bank_review, and other review-oriented workflows.",
    )
    parser.add_argument(
        "--allow-preflight-fail",
        action="store_true",
        help="Allow QA shot_review to continue even when preflight_batch returns FAIL.",
    )
    parser.add_argument(
        "--winner-image",
        help="Image name or record_key selected in promote_winner / seal_training_admission workflow.",
    )
    parser.add_argument(
        "--winner-rank",
        type=int,
        help="Shortlist rank selected in promote_winner / seal_training_admission workflow.",
    )
    parser.add_argument(
        "--winner-note",
        help="Optional manual note attached when promoting a winner into outputs/winner_bank.json.",
    )
    parser.add_argument(
        "--admission-owner",
        help="Manual owner recorded when sealing a candidate into outputs/training_admission_manifest.json.",
    )
    parser.add_argument(
        "--admission-note",
        help="Optional manual note attached when sealing a candidate into outputs/training_admission_manifest.json.",
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
        "--benchmark-compare-heavy-providers",
        nargs="+",
        help="Replay heavy evidence on the labeled benchmark set for one or more providers, for example segformer_body_fusion segformer_body_truth_fusion segformer_parser body_measure_lite body_canonical_hmr2.",
    )
    parser.add_argument(
        "--benchmark-image-root",
        type=Path,
        help="Optional image root used to resolve report items back to source images during heavy evidence benchmark compare.",
    )
    parser.add_argument(
        "--benchmark-template-out",
        type=Path,
        help="Export a benchmark label template from the chosen report file and exit.",
    )
    parser.add_argument(
        "--benchmark-preset",
        help="Named preset used to drive runtime profile and benchmark label metadata for QA/benchmark flows.",
    )
    parser.add_argument(
        "--benchmark-seal-labels",
        action="store_true",
        help="Update a benchmark label file to the chosen preset metadata without manual JSON editing.",
    )
    parser.add_argument(
        "--benchmark-id",
        help="Optional benchmark identifier written into the label file/template.",
    )
    parser.add_argument(
        "--benchmark-freeze-tag",
        help="Optional freeze tag written into the label file/template.",
    )
    parser.add_argument(
        "--optuna-search-space",
        type=Path,
        help="Optuna search-space JSON used by offline benchmark tuning mode.",
    )
    parser.add_argument(
        "--optuna-preset",
        help="Named Optuna preset from configs/optuna_mode_presets.json, for example front_core_fit.",
    )
    parser.add_argument(
        "--optuna-presets-file",
        type=Path,
        help="Optional Optuna preset registry JSON. Defaults to configs/optuna_mode_presets.json.",
    )
    parser.add_argument(
        "--optuna-list-presets",
        action="store_true",
        help="Print available Optuna presets and exit.",
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
    parser.add_argument(
        "--optuna-guard-path",
        type=Path,
        help="Optional Optuna guard JSON. Defaults to configs/optuna_guard.json and is enforced by default.",
    )
    return parser


def cli(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console_encoding()
    parser = build_arg_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    _maybe_enable_interactive_wizard(args, raw_argv)
    base_dir = _resolve_cli_path(args.base_dir, BASE_DIR)
    _prepare_interactive_args(args, base_dir)
    try:
        args.benchmark_compare_heavy_providers = _parse_heavy_provider_compare_targets(
            args.benchmark_compare_heavy_providers
        )
    except ValueError as exc:
        parser.error(str(exc))
    if str(args.workflow or "").strip() == "shot_review" and args.mode is None:
        args.mode = "qa"
    effective_mode = str(args.mode or "qa")
    presets_path = _resolve_cli_path(args.optuna_presets_file, base_dir) if args.optuna_presets_file else None
    try:
        threshold_override = _load_threshold_override(args, base_dir)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        workflow_result = _handle_workflow_action(args, base_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if workflow_result is not None:
        return workflow_result

    if args.optuna_list_presets:
        from core.qa_optuna import list_optuna_mode_presets

        result = list_optuna_mode_presets(
            base_dir=base_dir,
            presets_path=presets_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    benchmark_preset_info: Optional[Dict[str, Any]] = None
    if effective_mode in {"qa", "benchmark"} and (args.benchmark_preset or args.interactive):
        try:
            benchmark_preset_info = _resolve_optuna_preset_info(
                base_dir=base_dir,
                preset_name=args.benchmark_preset,
                presets_path=presets_path,
                interactive=args.interactive and not bool(args.benchmark_preset),
                fit_only=False,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if benchmark_preset_info is not None:
            print(
                f"[交互引导] 已选择评测预设：{benchmark_preset_info['name']} "
                f"| 建议运行 profile={benchmark_preset_info.get('recommended_runtime_profile', '-')}"
            )

    if args.mode == "optuna":
        if args.benchmark_labels is None:
            parser.error("optuna mode requires --benchmark-labels")

        preset_info: Optional[Dict[str, Any]] = None
        search_space_path: Optional[Path] = None
        guard_path: Optional[Path] = None
        if args.optuna_preset or (args.interactive and args.optuna_search_space is None and args.optuna_guard_path is None):
            if args.optuna_search_space is not None or args.optuna_guard_path is not None:
                parser.error(
                    "optuna mode does not allow mixing --optuna-preset with --optuna-search-space or --optuna-guard-path"
                )
            from core.qa_optuna import run_optuna_search

            try:
                preset_info = _resolve_optuna_preset_info(
                    base_dir=base_dir,
                    preset_name=args.optuna_preset,
                    presets_path=presets_path,
                    interactive=args.interactive and not bool(args.optuna_preset),
                    fit_only=True,
                )
            except ValueError as exc:
                parser.error(str(exc))
            if not bool(preset_info.get("fit_enabled", False)):
                parser.error(
                    f"optuna preset {preset_info.get('name', args.optuna_preset)!r} is review-only and does not enable fitting"
                )
            print(
                f"[交互引导] 已选择拟合预设：{preset_info['name']} "
                f"| 建议运行 profile={preset_info.get('recommended_runtime_profile', '-')}"
            )
            search_space_path = Path(str(preset_info["search_space_path"]))
            guard_path = Path(str(preset_info["guard_path"]))
        else:
            if args.optuna_search_space is None:
                parser.error("optuna mode requires --optuna-search-space or --optuna-preset")
            from core.qa_optuna import run_optuna_search

            search_space_path = _resolve_cli_path(args.optuna_search_space, base_dir)
            guard_path = _resolve_cli_path(args.optuna_guard_path, base_dir) if args.optuna_guard_path else None

        result = run_optuna_search(
            base_dir=base_dir,
            report_path=_resolve_cli_path(args.benchmark_report, base_dir)
            if args.benchmark_report
            else (base_dir / "outputs" / "qa_report.json").resolve(),
            labels_path=_resolve_cli_path(args.benchmark_labels, base_dir),
            search_space_path=search_space_path,
            cli_fixed_override=threshold_override,
            output_path=_resolve_cli_path(args.optuna_output, base_dir) if args.optuna_output else None,
            best_override_out=_resolve_cli_path(args.optuna_best_override_out, base_dir)
            if args.optuna_best_override_out
            else None,
            study_name_override=args.optuna_study_name,
            storage_path=_resolve_cli_path(args.optuna_storage_path, base_dir) if args.optuna_storage_path else None,
            trials_override=args.optuna_trials,
            guard_path=guard_path,
            preset=preset_info,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    selected_profile = args.profile
    if selected_profile is None and benchmark_preset_info is not None:
        recommended_profile = str(benchmark_preset_info.get("recommended_runtime_profile", "")).strip()
        if recommended_profile:
            selected_profile = recommended_profile

    benchmark_dataset_role: Optional[str] = None
    benchmark_optuna_ready: Optional[bool] = None
    benchmark_id = args.benchmark_id
    benchmark_freeze_tag = args.benchmark_freeze_tag
    if benchmark_preset_info is not None:
        benchmark_dataset_role = str(benchmark_preset_info.get("recommended_dataset_role", "")).strip() or None
        if args.benchmark_template_out is not None:
            benchmark_optuna_ready = False
        if args.benchmark_seal_labels:
            benchmark_optuna_ready = bool(benchmark_preset_info.get("fit_enabled", False))
        if benchmark_id is None:
            benchmark_id = str(benchmark_preset_info.get("name", "")).strip() or None
    elif args.benchmark_seal_labels:
        benchmark_optuna_ready = True

    if args.interactive and (args.benchmark_template_out is not None or args.benchmark_seal_labels):
        benchmark_id = _prompt_text(
            "请输入 Benchmark ID（建议使用 lane/版本号，便于后续回溯）",
            benchmark_id or "",
        )
        default_freeze_tag = benchmark_freeze_tag or date.today().isoformat()
        benchmark_freeze_tag = _prompt_text(
            "请输入 Freeze Tag（建议使用日期或里程碑标记）",
            default_freeze_tag,
        )

    if effective_mode == "qa":
        preflight_payload = _run_shot_review_preflight(
            base_dir=base_dir,
            target_profile=selected_profile,
            input_manifest=_resolve_cli_path(args.input_manifest, base_dir) if args.input_manifest else None,
            input_dir=_resolve_input_dir_arg(args.input_dir, base_dir),
        )
        _print_preflight_summary(preflight_payload)
        print(f"[预检文件] {_default_review_paths(base_dir, args.artifacts_dir)['preflight_batch']}")
        preflight_status = str((preflight_payload.get("batch_preflight") or {}).get("status") or "").upper()
        if preflight_status == "FAIL" and not args.allow_preflight_fail:
            print("[交互引导] 当前 batch preflight 失败，已阻断 shot_review。")
            print("[交互引导] 先拆批或补齐 input manifest；如确需继续，可显式追加 --allow-preflight-fail。")
            return 2
        if preflight_status == "FAIL" and args.allow_preflight_fail:
            print("[交互引导] 已显式允许 preflight FAIL，继续执行 shot_review。")

    if effective_mode == "qa" and args.input_dir is not None:
        runtime = create_runtime(base_dir)
        _override_runtime_input_dir(runtime, _resolve_input_dir_arg(args.input_dir, base_dir))
        if args.mode is not None:
            runtime.config.run_mode = str(args.mode)
        if args.heavy_provider is not None:
            runtime.config.provider_policy["heavy_evidence"] = str(args.heavy_provider)
            if runtime.providers is not None:
                from core.providers import build_provider_bundle

                runtime.providers = build_provider_bundle(runtime.config.provider_policy)
        if selected_profile is not None:
            runtime.config.review.active_profile = str(selected_profile)
        if args.auto_load_thresholds:
            runtime.config.auto_load_thresholds = True
        print_runtime_config(runtime)
        run_pipeline(runtime, profile_name=selected_profile, threshold_override=threshold_override)
    else:
        pipeline_main(
            base_dir=base_dir,
            profile_name=selected_profile,
            run_mode=args.mode,
            heavy_evidence_provider=args.heavy_provider,
            auto_load_thresholds=True if args.auto_load_thresholds else None,
            threshold_override=threshold_override,
            benchmark_report_path=_resolve_cli_path(args.benchmark_report, base_dir) if args.benchmark_report else None,
            benchmark_labels_path=_resolve_cli_path(args.benchmark_labels, base_dir) if args.benchmark_labels else None,
            benchmark_output_path=_resolve_cli_path(args.benchmark_output, base_dir) if args.benchmark_output else None,
            benchmark_template_out=_resolve_cli_path(args.benchmark_template_out, base_dir) if args.benchmark_template_out else None,
            benchmark_compare_heavy_providers=args.benchmark_compare_heavy_providers,
            benchmark_image_root=_resolve_cli_path(args.benchmark_image_root, base_dir) if args.benchmark_image_root else None,
            benchmark_dataset_role=benchmark_dataset_role,
            benchmark_optuna_ready=benchmark_optuna_ready,
            benchmark_id=benchmark_id,
            benchmark_freeze_tag=benchmark_freeze_tag,
            benchmark_update_labels=bool(args.benchmark_seal_labels),
        )
    if effective_mode == "qa":
        review_packet_path = _default_review_paths(base_dir, args.artifacts_dir)["review_packet"]
        try:
            packet = _load_json_file(review_packet_path, "review packet")
        except ValueError:
            packet = None
        if packet is not None:
            _print_review_packet_summary(packet)
            _describe_canonical_truth_state(packet)
            print(f"[复核摘要文件] {review_packet_path}")
    elif effective_mode == "benchmark" and args.benchmark_output is not None:
        benchmark_output_path = _resolve_cli_path(args.benchmark_output, base_dir)
        try:
            benchmark_payload = _load_json_file(benchmark_output_path, "benchmark output")
        except ValueError:
            benchmark_payload = None
        if benchmark_payload is not None:
            schema_version = str(benchmark_payload.get("schema_version") or "").strip()
            if schema_version == "qa_benchmark_heavy_compare_v1":
                _print_benchmark_heavy_compare_summary(benchmark_payload)
            else:
                _print_benchmark_replay_summary(benchmark_payload)
            print(f"[Benchmark 输出文件] {benchmark_output_path}")
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


