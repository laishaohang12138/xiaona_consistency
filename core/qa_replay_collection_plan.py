from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .qa_input_manifest import load_input_manifest_index, required_prompt_intent_fields
from .qa_io import atomic_write_json

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_SIDE_BACK_SPECS: Dict[str, Dict[str, Any]] = {
    "side": {
        "input_dir": Path("input_split") / "side",
        "target_profile": "body_gold_side90_shadow",
        "intended_view": "side_90",
        "intended_lane_family": "side",
        "intended_view_center_deg": 90,
        "recommended_min_images": 12,
        "recommended_target_images": 24,
    },
    "back": {
        "input_dir": Path("input_split") / "back",
        "target_profile": "body_gold_back180_shadow",
        "intended_view": "back_180",
        "intended_lane_family": "back",
        "intended_view_center_deg": 180,
        "recommended_min_images": 12,
        "recommended_target_images": 24,
    },
}

_LIGHTING_VARIANT_PRIORITY = {
    "neutral_base": 0,
    "bright_exposure": 1,
    "dim_exposure": 2,
    "warm_cast": 3,
    "cool_cast": 4,
}

_OUTER_FAMILY_PRIORITY = {
    "blazer": 0,
    "collarless_jacket": 1,
    "overshirt": 2,
    "cardigan": 3,
    "trench": 4,
    "rigid_coat": 5,
    "short_jacket": 6,
    "long_panel_coat": 7,
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _rel_path(path: Any, base_dir: Path) -> str:
    text = _safe_text(path)
    if not text:
        return ""
    resolved = Path(text)
    try:
        return str(resolved.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return text.replace("\\", "/")


def _image_count(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)


def _collection_status(image_count: int, minimum: int, target: int) -> str:
    if image_count <= 0:
        return "empty_needs_collection"
    if image_count < minimum:
        return "below_minimum_needs_more_images"
    if image_count < target:
        return "minimum_ready_target_incomplete"
    return "target_ready"


def _seed_or_unavailable_ready(item: Dict[str, Any]) -> bool:
    return item.get("seed") is not None or bool(_safe_text(item.get("seed_unavailable_reason")))


def _manifest_state(input_dir: Path) -> Dict[str, Any]:
    manifest = load_input_manifest_index(input_dir)
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    required = required_prompt_intent_fields()
    missing_counts: Dict[str, int] = {}
    for field in required:
        if field == "seed":
            missing_counts[field] = sum(1 for item in entries if not _seed_or_unavailable_ready(item))
        elif field == "intended_view":
            missing_counts[field] = sum(
                1
                for item in entries
                if not _safe_text(item.get("intended_view") or item.get("view_expected"))
            )
        else:
            missing_counts[field] = sum(1 for item in entries if not _safe_text(item.get(field)))
    missing_fields = [field for field, count in missing_counts.items() if count > 0]
    return {
        "available": bool(manifest.get("available")),
        "manifest_path": manifest.get("path"),
        "item_count": len(entries),
        "required_fields": required,
        "missing_counts": missing_counts,
        "missing_fields": missing_fields,
        "required_field_ready": bool(entries) and not missing_fields,
    }


def _command(
    *,
    workflow: str,
    input_dir: str,
    profile: str,
    artifacts_dir: str = "",
    heavy_provider: str = "",
) -> str:
    command = (
        f".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow {workflow} "
        f"--input-dir {input_dir} --profile {profile}"
    )
    if artifacts_dir:
        command += f" --artifacts-dir {artifacts_dir}"
    if workflow == "shot_review" and heavy_provider:
        command += f" --heavy-provider {heavy_provider}"
    return command


def _artifact_dir_for_input(input_dir: str) -> str:
    normalized = _safe_text(input_dir).replace("\\", "/").strip("/")
    if normalized.startswith("input_replay/"):
        normalized = normalized[len("input_replay/") :]
    elif normalized.startswith("input_split/"):
        normalized = "split/" + normalized[len("input_split/") :]
    elif normalized.startswith("input/"):
        normalized = "input/" + normalized[len("input/") :]
    normalized = normalized or "manual"
    return f"outputs/replay/{normalized}"


def _task(
    *,
    priority_rank: int,
    priority_group: str,
    area: str,
    task_type: str,
    title: str,
    status: str,
    input_dir: str = "",
    target_profile: str = "",
    image_count: int = 0,
    recommended_min_images: int = 0,
    recommended_target_images: int = 0,
    details: Optional[Dict[str, Any]] = None,
    run_commands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "priority_rank": priority_rank,
        "priority_group": priority_group,
        "area": area,
        "task_type": task_type,
        "title": title,
        "status": status,
        "input_dir": input_dir,
        "target_profile": target_profile,
        "image_count": image_count,
        "recommended_min_images": recommended_min_images,
        "recommended_target_images": recommended_target_images,
        "images_needed_for_minimum": max(0, recommended_min_images - image_count),
        "images_needed_for_target": max(0, recommended_target_images - image_count),
        "details": details or {},
        "run_commands": run_commands or [],
    }


def _manifest_completion_tasks(*, base_dir: Path, manifest_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    splits = manifest_plan.get("splits") if isinstance(manifest_plan.get("splits"), list) else []
    for idx, raw_split in enumerate(splits):
        split = raw_split if isinstance(raw_split, dict) else {}
        if bool(split.get("ready_for_clean_replay")):
            continue
        label = _safe_text(split.get("split")) or f"split_{idx + 1}"
        tasks.append(
            _task(
                priority_rank=10 + idx,
                priority_group="P0",
                area="manifest_metadata",
                task_type="complete_clean_lane_manifest",
                title=f"Complete {label} split prompt intent metadata",
                status="metadata_incomplete",
                input_dir=_rel_path(split.get("input_dir"), base_dir),
                image_count=_safe_int(split.get("item_count")),
                details={
                    "split": label,
                    "manifest_path": split.get("manifest_path"),
                    "missing_fields": split.get("missing_fields") if isinstance(split.get("missing_fields"), list) else [],
                    "missing_counts": split.get("missing_counts") if isinstance(split.get("missing_counts"), dict) else {},
                    "field_policy": split.get("field_policy") if isinstance(split.get("field_policy"), dict) else {},
                    "example_fill_command": _safe_text(split.get("example_fill_command")),
                },
            )
        )
    return tasks


def _lighting_replay_tasks(
    *,
    base_dir: Path,
    lighting_pack: Dict[str, Any],
    invariance_status: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not lighting_pack:
        return [
            _task(
                priority_rank=30,
                priority_group="P0",
                area="lighting_invariance",
                task_type="prepare_lighting_replay_pack",
                title="Prepare controlled lighting replay pack",
                status="pack_missing",
                run_commands=[
                    ".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow prepare_lighting_replay_pack"
                ],
            )
        ]

    gates = invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {}
    lighting_gate = gates.get("lighting_invariance") if isinstance(gates.get("lighting_invariance"), dict) else {}
    metrics = lighting_gate.get("metrics") if isinstance(lighting_gate.get("metrics"), dict) else {}
    warn_rates = (
        metrics.get("skin_lighting_risk_warn_rate")
        if isinstance(metrics.get("skin_lighting_risk_warn_rate"), dict)
        else {}
    )

    tasks: List[Dict[str, Any]] = []
    lanes = lighting_pack.get("lanes") if isinstance(lighting_pack.get("lanes"), list) else []
    for raw_lane in lanes:
        lane = raw_lane if isinstance(raw_lane, dict) else {}
        lane_name = _safe_text(lane.get("lane"))
        lane_warn_rate = _safe_float(warn_rates.get(lane_name))
        lane_priority = 0 if lane_warn_rate >= 0.35 else 10
        if lane_name == "front":
            lane_priority -= 2
        target_profile = _safe_text(lane.get("target_profile"))
        variants = lane.get("variants") if isinstance(lane.get("variants"), list) else []
        for raw_variant in variants:
            variant = raw_variant if isinstance(raw_variant, dict) else {}
            variant_key = _safe_text(variant.get("variant_key"))
            input_dir = _rel_path(variant.get("variant_dir"), base_dir)
            artifacts_dir = _artifact_dir_for_input(input_dir)
            count = _safe_int(variant.get("image_count"))
            minimum = _safe_int(variant.get("recommended_min_images"), 6)
            target = _safe_int(variant.get("recommended_target_images"), 8)
            status = _collection_status(count, minimum, target)
            tasks.append(
                _task(
                    priority_rank=30 + lane_priority + _LIGHTING_VARIANT_PRIORITY.get(variant_key, 9),
                    priority_group="P0" if count < minimum else "P1",
                    area="lighting_invariance",
                    task_type="collect_lighting_variant",
                    title=f"Collect {lane_name} lighting variant {variant_key}",
                    status=status,
                    input_dir=input_dir,
                    target_profile=target_profile,
                    image_count=count,
                    recommended_min_images=minimum,
                    recommended_target_images=target,
                    details={
                        "lane": lane_name,
                        "variant_key": variant_key,
                        "capture_hint": variant.get("capture_hint"),
                        "expected_delta": variant.get("expected_delta"),
                        "lane_lighting_warn_rate": _round_or_none(lane_warn_rate),
                        "manifest_path": _rel_path(variant.get("manifest_path"), base_dir),
                        "metadata_template_path": _rel_path(variant.get("metadata_template_path"), base_dir),
                        "artifacts_dir": artifacts_dir,
                    },
                    run_commands=[
                        ".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow prepare_lighting_replay_pack",
                        _command(
                            workflow="preflight_batch",
                            input_dir=input_dir,
                            profile=target_profile,
                            artifacts_dir=artifacts_dir,
                        ),
                        _command(
                            workflow="shot_review",
                            input_dir=input_dir,
                            profile=target_profile,
                            artifacts_dir=artifacts_dir,
                        ),
                    ],
                )
            )
    return tasks


def _outer_prompt_number(prompt_id: str) -> int:
    text = _safe_text(prompt_id).upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return _safe_int(digits, 99)


def _outer_prompt_priority(prompt_id: str) -> int:
    number = _outer_prompt_number(prompt_id)
    if number == 1:
        return 0
    if number == 3:
        return 1
    if number == 2:
        return 2
    return 5


def _outer_replay_tasks(*, base_dir: Path, outer_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not outer_pack:
        return [
            _task(
                priority_rank=50,
                priority_group="P0",
                area="clothing_invariance",
                task_type="prepare_outer_replay_pack",
                title="Prepare governed OUTER occlusion replay pack",
                status="pack_missing",
                run_commands=[
                    ".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow prepare_outer_replay_pack"
                ],
            )
        ]

    tasks: List[Dict[str, Any]] = []
    lanes = outer_pack.get("lanes") if isinstance(outer_pack.get("lanes"), list) else []
    for raw_lane in lanes:
        lane = raw_lane if isinstance(raw_lane, dict) else {}
        lane_name = _safe_text(lane.get("lane"))
        lane_priority = 0 if lane_name == "front" else 10
        target_profile = _safe_text(lane.get("target_profile"))
        families = lane.get("families") if isinstance(lane.get("families"), list) else []
        for raw_family in families:
            family = raw_family if isinstance(raw_family, dict) else {}
            family_name = _safe_text(family.get("family"))
            family_priority = _OUTER_FAMILY_PRIORITY.get(family_name, 20)
            prompts = family.get("prompts") if isinstance(family.get("prompts"), list) else []
            for raw_prompt in prompts:
                prompt = raw_prompt if isinstance(raw_prompt, dict) else {}
                prompt_id = _safe_text(prompt.get("prompt_id"))
                input_dir = _rel_path(prompt.get("variant_dir"), base_dir)
                artifacts_dir = _artifact_dir_for_input(input_dir)
                count = _safe_int(prompt.get("image_count"))
                minimum = _safe_int(prompt.get("recommended_min_images"), 4)
                target = _safe_int(prompt.get("recommended_target_images"), 6)
                prompt_priority = _outer_prompt_priority(prompt_id)
                status = _collection_status(count, minimum, target)
                tasks.append(
                    _task(
                        priority_rank=50 + lane_priority + family_priority * 3 + prompt_priority,
                        priority_group="P0" if count < minimum and prompt_priority <= 1 else "P1",
                        area="clothing_invariance",
                        task_type="collect_outer_prompt_leaf",
                        title=f"Collect OUTER {lane_name} {family_name} {prompt_id}",
                        status=status,
                        input_dir=input_dir,
                        target_profile=target_profile,
                        image_count=count,
                        recommended_min_images=minimum,
                        recommended_target_images=target,
                        details={
                            "lane": lane_name,
                            "family": family_name,
                            "prompt_id": prompt_id,
                            "prompt_file": prompt.get("prompt_file"),
                            "manifest_path": _rel_path(prompt.get("manifest_path"), base_dir),
                            "metadata_template_path": _rel_path(prompt.get("metadata_template_path"), base_dir),
                            "starter_wave": prompt_priority <= 1,
                            "artifacts_dir": artifacts_dir,
                        },
                        run_commands=[
                            ".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow prepare_outer_replay_pack",
                            _command(
                                workflow="preflight_batch",
                                input_dir=input_dir,
                                profile=target_profile,
                                artifacts_dir=artifacts_dir,
                            ),
                            _command(
                                workflow="shot_review",
                                input_dir=input_dir,
                                profile=target_profile,
                                artifacts_dir=artifacts_dir,
                            ),
                        ],
                    )
                )
    return tasks


def _iter_runs(run_index: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    current = run_index.get("current_outputs") if isinstance(run_index.get("current_outputs"), dict) else {}
    if current:
        yield current
    snapshots = run_index.get("snapshot_runs") if isinstance(run_index.get("snapshot_runs"), list) else []
    for raw_run in snapshots:
        if isinstance(raw_run, dict):
            yield raw_run


def _truth_fusion_run_for_lane(run_index: Dict[str, Any], lane: str, profile: str) -> Optional[Dict[str, Any]]:
    for run in _iter_runs(run_index):
        if _safe_text(run.get("dominant_lane_family")) != lane:
            continue
        if _safe_text(run.get("target_profile")) != profile:
            continue
        if _safe_text(run.get("preflight_status")).upper() != "PASS":
            continue
        if _safe_text(run.get("evidence_status")).upper() != "PASS":
            continue
        if "truth_fusion" not in _safe_text(run.get("active_heavy_provider")).lower():
            continue
        return run
    return None


def _side_back_topology_tasks(*, base_dir: Path, run_index: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for idx, (lane_name, spec) in enumerate(_SIDE_BACK_SPECS.items()):
        input_dir = (base_dir / spec["input_dir"]).resolve()
        input_rel = _rel_path(input_dir, base_dir)
        artifacts_dir = _artifact_dir_for_input(input_rel)
        image_count = _image_count(input_dir)
        minimum = _safe_int(spec.get("recommended_min_images"), 12)
        target = _safe_int(spec.get("recommended_target_images"), 24)
        profile = _safe_text(spec.get("target_profile"))
        run = _truth_fusion_run_for_lane(run_index, lane_name, profile)
        manifest_state = _manifest_state(input_dir)
        if run is not None:
            status = "truth_fusion_validation_run_available"
        else:
            status = _collection_status(image_count, minimum, target)
            if image_count >= minimum:
                status = "ready_to_run_truth_fusion_review"
        tasks.append(
            _task(
                priority_rank=90 + idx,
                priority_group="P1" if run is None else "P2",
                area="topology_consistency",
                task_type="validate_side_back_truth_fusion_topology",
                title=f"Validate {lane_name} topology with segformer_body_truth_fusion",
                status=status,
                input_dir=input_rel,
                target_profile=profile,
                image_count=image_count,
                recommended_min_images=minimum,
                recommended_target_images=target,
                details={
                    "lane": lane_name,
                    "intended_view": spec.get("intended_view"),
                    "intended_lane_family": spec.get("intended_lane_family"),
                    "intended_view_center_deg": spec.get("intended_view_center_deg"),
                    "manifest_state": manifest_state,
                    "existing_truth_fusion_run": {
                        "artifact_root": run.get("artifact_root"),
                        "top_ranked_image": run.get("top_ranked_image"),
                        "completeness_score": run.get("completeness_score"),
                    }
                    if run is not None
                    else {},
                    "truth_policy": {
                        "face_truth": "A-Core_01_0deg_MASTER.png",
                        "body_truth": "Task-63987060-116-1.png",
                        "body_read": "pose_gait_aware_absolute_116_1",
                    },
                    "artifacts_dir": artifacts_dir,
                },
                run_commands=[
                    _command(
                        workflow="preflight_batch",
                        input_dir=input_rel,
                        profile=profile,
                        artifacts_dir=artifacts_dir,
                    ),
                    _command(
                        workflow="shot_review",
                        input_dir=input_rel,
                        profile=profile,
                        artifacts_dir=artifacts_dir,
                    ),
                ],
            )
        )
    return tasks


def _topology_replay_tasks(
    *,
    base_dir: Path,
    run_index: Dict[str, Any],
    topology_pack: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not topology_pack:
        return _side_back_topology_tasks(base_dir=base_dir, run_index=run_index)

    tasks: List[Dict[str, Any]] = []
    lanes = topology_pack.get("lanes") if isinstance(topology_pack.get("lanes"), list) else []
    for raw_lane in lanes:
        lane = raw_lane if isinstance(raw_lane, dict) else {}
        lane_name = _safe_text(lane.get("lane"))
        lane_priority = 0 if lane_name == "side" else 10
        variants = lane.get("variants") if isinstance(lane.get("variants"), list) else []
        for idx, raw_variant in enumerate(variants):
            variant = raw_variant if isinstance(raw_variant, dict) else {}
            input_dir = _rel_path(variant.get("variant_dir"), base_dir)
            artifacts_dir = _artifact_dir_for_input(input_dir)
            profile = _safe_text(variant.get("target_profile"))
            count = _safe_int(variant.get("image_count"))
            minimum = _safe_int(variant.get("recommended_min_images"), 8)
            target = _safe_int(variant.get("recommended_target_images"), 12)
            run = _truth_fusion_run_for_lane(run_index, lane_name, profile)
            manifest_state = _manifest_state(base_dir / input_dir)
            if run is not None and count > 0:
                status = "truth_fusion_validation_run_available"
            else:
                status = _collection_status(count, minimum, target)
                if count >= minimum:
                    status = "ready_to_run_truth_fusion_review"
            tasks.append(
                _task(
                    priority_rank=90 + lane_priority + idx,
                    priority_group="P1" if status != "truth_fusion_validation_run_available" else "P2",
                    area="topology_consistency",
                    task_type="collect_topology_variant",
                    title=f"Collect {lane_name} topology variant {variant.get('variant_key')}",
                    status=status,
                    input_dir=input_dir,
                    target_profile=profile,
                    image_count=count,
                    recommended_min_images=minimum,
                    recommended_target_images=target,
                    details={
                        "lane": lane_name,
                        "variant_key": variant.get("variant_key"),
                        "intended_view": variant.get("intended_view"),
                        "intended_lane_family": variant.get("intended_lane_family"),
                        "intended_view_center_deg": variant.get("intended_view_center_deg"),
                        "intended_view_side": variant.get("intended_view_side"),
                        "pose_control_bucket": variant.get("pose_control_bucket"),
                        "capture_hint": variant.get("capture_hint"),
                        "manifest_path": _rel_path(variant.get("manifest_path"), base_dir),
                        "metadata_template_path": _rel_path(variant.get("metadata_template_path"), base_dir),
                        "manifest_state": manifest_state,
                        "existing_truth_fusion_run": {
                            "artifact_root": run.get("artifact_root"),
                            "top_ranked_image": run.get("top_ranked_image"),
                            "completeness_score": run.get("completeness_score"),
                        }
                        if run is not None
                        else {},
                        "truth_policy": {
                            "face_truth": "A-Core_01_0deg_MASTER.png",
                            "body_truth": "Task-63987060-116-1.png",
                            "body_read": "pose_gait_aware_absolute_116_1",
                        },
                        "artifacts_dir": artifacts_dir,
                    },
                    run_commands=[
                        ".\\.venv\\Scripts\\python.exe .\\check_consistency.py --workflow prepare_topology_replay_pack",
                        _command(
                            workflow="preflight_batch",
                            input_dir=input_dir,
                            profile=profile,
                            artifacts_dir=artifacts_dir,
                        ),
                        _command(
                            workflow="shot_review",
                            input_dir=input_dir,
                            profile=profile,
                            artifacts_dir=artifacts_dir,
                        ),
                    ],
                )
            )
    return tasks


def _queue(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actionable_status = {
        "metadata_incomplete",
        "pack_missing",
        "empty_needs_collection",
        "below_minimum_needs_more_images",
        "minimum_ready_target_incomplete",
        "ready_to_run_truth_fusion_review",
    }
    return sorted(
        [task for task in tasks if _safe_text(task.get("status")) in actionable_status],
        key=lambda item: (
            _safe_int(item.get("priority_rank"), 999),
            _safe_text(item.get("area")),
            _safe_text(item.get("title")),
        ),
    )


def _summary(tasks: List[Dict[str, Any]], queue: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_area: Dict[str, Dict[str, int]] = {}
    for task in tasks:
        area = _safe_text(task.get("area")) or "unknown"
        status = _safe_text(task.get("status")) or "unknown"
        by_area.setdefault(area, {})
        by_area[area][status] = by_area[area].get(status, 0) + 1
    return {
        "total_task_count": len(tasks),
        "actionable_task_count": len(queue),
        "p0_actionable_count": sum(1 for task in queue if _safe_text(task.get("priority_group")) == "P0"),
        "p1_actionable_count": sum(1 for task in queue if _safe_text(task.get("priority_group")) == "P1"),
        "status_counts_by_area": by_area,
    }


def build_replay_collection_plan(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    manifest_plan = _load_json(outputs_dir / "input_manifest_completion_plan.json")
    lighting_pack = _load_json(outputs_dir / "lighting_replay_pack.json")
    outer_pack = _load_json(outputs_dir / "outer_replay_pack.json")
    topology_pack = _load_json(outputs_dir / "topology_replay_pack.json")
    invariance_status = _load_json(outputs_dir / "review_invariance_status.json")
    run_index = _load_json(outputs_dir / "review_run_index.json")

    tasks: List[Dict[str, Any]] = []
    tasks.extend(_manifest_completion_tasks(base_dir=base_dir, manifest_plan=manifest_plan))
    tasks.extend(
        _lighting_replay_tasks(
            base_dir=base_dir,
            lighting_pack=lighting_pack,
            invariance_status=invariance_status,
        )
    )
    tasks.extend(_outer_replay_tasks(base_dir=base_dir, outer_pack=outer_pack))
    tasks.extend(
        _topology_replay_tasks(
            base_dir=base_dir,
            run_index=run_index,
            topology_pack=topology_pack,
        )
    )

    collection_queue = _queue(tasks)
    summary = _summary(tasks, collection_queue)
    overall_status = "READY_FOR_REPLAY_EXECUTION" if not collection_queue else "NEEDS_COLLECTION_OR_METADATA"
    payload = {
        "schema_version": "replay_collection_plan_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Turn review-only invariance gaps into concrete metadata, collection, and replay-run tasks. "
            "This is screening evidence only, not final training admission."
        ),
        "overall_status": overall_status,
        "project_scope": {
            "role": "screening_and_evidence_only",
            "training_admission_participation": False,
            "final_training_decision_owner": "external_training_decision_flow",
        },
        "truth_policy": {
            "face_truth": "A-Core_01_0deg_MASTER.png",
            "body_truth": "Task-63987060-116-1.png",
            "body_truth_read": "pose_gait_aware_absolute_116_1",
            "winner_bank_state": "mutable_review_memory_not_frozen",
            "parameter_fitting_allowed": False,
        },
        "source_files": {
            "manifest_completion_plan": _rel_path(outputs_dir / "input_manifest_completion_plan.json", base_dir),
            "review_invariance_status": _rel_path(outputs_dir / "review_invariance_status.json", base_dir),
            "lighting_replay_pack": _rel_path(outputs_dir / "lighting_replay_pack.json", base_dir),
            "outer_replay_pack": _rel_path(outputs_dir / "outer_replay_pack.json", base_dir),
            "topology_replay_pack": _rel_path(outputs_dir / "topology_replay_pack.json", base_dir),
            "review_run_index": _rel_path(outputs_dir / "review_run_index.json", base_dir),
        },
        "phase_order": [
            "Complete clean-lane manifest metadata before rerunning front / three_quarter replay.",
            "Collect controlled lighting variants, prioritizing lanes with high lighting-warning rates.",
            "Collect governed OUTER prompt leaves as clothing-occlusion evidence, starting from starter-wave prompts.",
            "Collect controlled side/back topology variants, then validate them with profile-default segformer_body_truth_fusion.",
        ],
        "summary": summary,
        "immediate_operator_queue": collection_queue[:12],
        "collection_queue": collection_queue,
        "all_tasks": sorted(tasks, key=lambda item: (_safe_int(item.get("priority_rank"), 999), _safe_text(item.get("title")))),
        "operator_holds": [
            "Do not freeze winner_bank from this plan.",
            "Do not use this plan for final training-set admission.",
            "Do not use this plan to decide final image-set membership.",
            "Do not run parameter fitting before project optimization is complete.",
            "Do not treat pose/gait topology-margin rows as body drift without manual structure review.",
        ],
    }
    _write_json(output_file, payload)
    return payload
