# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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
        print(f"[交互引导] 无法识别的是/否输入：{choice}")


def _select_choice(
    prompt: str,
    options: Sequence[Tuple[str, str]],
    *,
    default: str,
) -> str:
    print("[交互引导] 可选项：")
    for index, (value, description) in enumerate(options, start=1):
        print(f"  {index}. {value} | {description}")
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
        print(f"[交互引导] 无法识别的选项：{choice}")


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
            print(f"[交互引导] 路径不存在：{candidate}")
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
            ("shot_review", "审一个 shot 批次，输出 QA、排序和 review packet"),
            ("inspect_review_packet", "查看最近一次 review packet 的批次摘要和复核提示"),
            ("promote_winner", "把人工确认的 winner 写入 winner bank"),
            ("winner_bank_status", "查看 winner bank 状态与最新跨批次漂移报告"),
            ("advanced_cli", "进入高级工程模式（qa / benchmark / optuna / calibrate）"),
        ],
        default=default,
    )


def _select_review_profile_interactively(default: str = "body_gold_fullbody") -> str:
    return _select_choice(
        "请选择本轮 shot 批次最接近的量纲体系",
        [
            ("body_gold_fullbody", "BODY GOLD 前向/普通主体批次"),
            ("bridge_simple_outfit", "简单穿搭 / BRIDGE 训练准入批次"),
            ("body_gold_side90_shadow", "90 度侧面 shadow 观察批次"),
            ("body_gold_back180_shadow", "180 度背部 shadow 观察批次"),
            ("full_body_outfit", "通用穿搭稳定性批次"),
        ],
        default=default,
    )


def _default_review_paths(base_dir: Path) -> Dict[str, Path]:
    output_dir = (base_dir / "outputs").resolve()
    return {
        "review_packet": output_dir / "review_packet.json",
        "winner_bank_candidate": output_dir / "winner_bank_candidate.json",
        "winner_bank_report": output_dir / "winner_bank_report.json",
        "winner_bank": output_dir / "winner_bank.json",
    }


def _print_review_packet_summary(packet: Dict[str, Any]) -> None:
    batch = packet.get("batch_summary") or {}
    selection = batch.get("selection") or {}
    batch_gate = batch.get("batch_gate") or {}
    identity = batch.get("identity_summary") or {}
    geometry = batch.get("geometry_summary") or {}
    admission = batch.get("admission_advice") or {}
    print("\n[Review Packet]")
    print(f"  Profile: {batch.get('target_profile')}")
    print(f"  Images : {batch.get('input_count')}")
    print(f"  Top1   : {selection.get('top_ranked_image')}")
    print(f"  Window : top {selection.get('manual_review_window')}")
    print(f"  Gate   : {batch_gate.get('status')} | reasons={batch_gate.get('reasons') or []}")
    print(
        "  Identity: "
        f"face={identity.get('batch_identity_cohesion')} "
        f"clothfree={identity.get('batch_clothfree_identity_cohesion')} "
        f"hybrid={identity.get('batch_hybrid_identity_cohesion')}"
    )
    print(
        "  Geometry: "
        f"body={geometry.get('body_under_clothes_continuity')} "
        f"3d={geometry.get('batch_3d_cohesion')} "
        f"world3d={geometry.get('batch_world3d_cohesion')}"
    )
    print(f"  Risks  : {batch.get('primary_risks') or []}")
    print(
        f"  Admission: target={admission.get('target_bucket')} "
        f"| action={admission.get('suggested_action')} | blockers={admission.get('blockers') or []}"
    )
    print(f"  Guidance: {batch.get('review_guidance') or []}")


def _print_winner_bank_summary(report: Dict[str, Any]) -> None:
    print("\n[Winner Bank]")
    print(f"  Status     : {report.get('status')}")
    print(f"  Curated    : {report.get('curated_bank_available')} | entries={report.get('curated_entry_count')}")
    print(f"  Candidates : {report.get('candidate_entry_count')}")
    print(f"  Drift rows : {report.get('drift_row_count')}")
    print(f"  Next step  : {report.get('manual_next_step')}")
    top_risks = list(report.get("top_drift_risks") or [])
    if top_risks:
        print(f"  Top risks  : {top_risks[:4]}")
    drift_rows = list(report.get("drift_rows") or [])
    for row in drift_rows[:2]:
        print(
            f"  Drift      : {row.get('image')} | severity={row.get('drift_severity')} "
            f"| flags={list(row.get('drift_flags') or [])[:3]}"
        )
        focus = list(row.get("manual_focus") or [])[:2]
        if focus:
            print(f"               focus={focus}")


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
    print("[交互引导] 可晋升的候选 winner：")
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
        print(f"[交互引导] 无法识别的候选编号：{choice}")


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
            ("qa", "运行当前 input 图集质检，适合日常候选筛选与人工初筛"),
            ("benchmark", "回放已有 qa_report 与标签集，适合看规则效果、分组指标与回归表现"),
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

    if workflow == "shot_review":
        args.mode = "qa"
        if args.profile is None:
            args.profile = _select_review_profile_interactively(default=str(args.profile or "body_gold_fullbody"))
        return

    if workflow in {"inspect_review_packet", "promote_winner", "winner_bank_status"}:
        return

    if args.mode is None:
        args.mode = _select_run_mode_interactively(default=effective_mode)
        effective_mode = str(args.mode)

    if effective_mode == "benchmark":
        has_action = args.benchmark_template_out is not None or args.benchmark_seal_labels or args.benchmark_labels is not None
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
            else:
                args.benchmark_labels = _prompt_path(
                    "请输入 benchmark 标签文件路径",
                    base_dir=base_dir,
                    default="outputs/benchmark_labels_verify.json",
                    must_exist=True,
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

    paths = _default_review_paths(base_dir)
    if workflow == "inspect_review_packet":
        packet = _load_json_file(paths["review_packet"], "review packet")
        _print_review_packet_summary(packet)
        print(f"[Review Packet File] {paths['review_packet']}")
        print("[交互引导] 如需让 GPT 深入分析，请直接读取这份 review_packet.json。")
        return 0

    if workflow == "winner_bank_status":
        report = _load_json_file(paths["winner_bank_report"], "winner bank report")
        _print_winner_bank_summary(report)
        print(f"[Winner Bank Report] {paths['winner_bank_report']}")
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
        print(
            "[交互引导] 已写入 winner bank。建议下一次重新跑 shot review，"
            "让系统用新的 curated bank 做跨批次漂移检查。"
        )
        return 0

    raise ValueError(f"unsupported workflow: {workflow}")


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

    print("[交互引导] 可选预设：")
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
        print(f"[交互引导] 无法识别的预设：{choice}")


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
        choices=["shot_review", "inspect_review_packet", "promote_winner", "winner_bank_status", "advanced_cli"],
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
        "--winner-image",
        help="Image name or record_key promoted into outputs/winner_bank.json in promote_winner workflow.",
    )
    parser.add_argument(
        "--winner-rank",
        type=int,
        help="Shortlist rank promoted into outputs/winner_bank.json in promote_winner workflow.",
    )
    parser.add_argument(
        "--winner-note",
        help="Optional manual note attached when promoting a winner into outputs/winner_bank.json.",
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
    parser = build_arg_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    _maybe_enable_interactive_wizard(args, raw_argv)
    base_dir = _resolve_cli_path(args.base_dir, BASE_DIR)
    _prepare_interactive_args(args, base_dir)
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

    pipeline_main(
        base_dir=base_dir,
        profile_name=selected_profile,
        run_mode=args.mode,
        auto_load_thresholds=True if args.auto_load_thresholds else None,
        threshold_override=threshold_override,
        benchmark_report_path=_resolve_cli_path(args.benchmark_report, base_dir) if args.benchmark_report else None,
        benchmark_labels_path=_resolve_cli_path(args.benchmark_labels, base_dir) if args.benchmark_labels else None,
        benchmark_output_path=_resolve_cli_path(args.benchmark_output, base_dir) if args.benchmark_output else None,
        benchmark_template_out=_resolve_cli_path(args.benchmark_template_out, base_dir) if args.benchmark_template_out else None,
        benchmark_dataset_role=benchmark_dataset_role,
        benchmark_optuna_ready=benchmark_optuna_ready,
        benchmark_id=benchmark_id,
        benchmark_freeze_tag=benchmark_freeze_tag,
        benchmark_update_labels=bool(args.benchmark_seal_labels),
    )
    if effective_mode == "qa":
        review_packet_path = _default_review_paths(base_dir)["review_packet"]
        try:
            packet = _load_json_file(review_packet_path, "review packet")
        except ValueError:
            packet = None
        if packet is not None:
            _print_review_packet_summary(packet)
            print(f"[Review Packet File] {review_packet_path}")
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
