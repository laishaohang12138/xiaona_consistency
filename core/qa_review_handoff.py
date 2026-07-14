from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .qa_io import atomic_write_json


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _dedupe(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _rel(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _compact_gate_rows(gates: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, raw_gate in gates.items():
        gate = raw_gate if isinstance(raw_gate, dict) else {}
        metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
        rows.append(
            {
                "gate": name,
                "status": str(gate.get("status") or "").strip(),
                "key_metrics": metrics,
                "reasons": [
                    str(item).strip()
                    for item in (gate.get("reasons") or [])
                    if str(item).strip()
                ],
            }
        )
    return rows


def _run_pointer(
    label: str,
    run: Dict[str, Any],
    *,
    base_dir: Path,
) -> Dict[str, Any]:
    root_text = str(run.get("artifact_root") or "").strip()
    root = Path(root_text) if root_text else Path("")
    gpt_packet = root / "gpt_review_packet.json" if root_text else Path("")
    review_packet = root / "review_packet.json" if root_text else Path("")
    screening_risks: List[str] = []
    for item in (run.get("screening_risks") or run.get("admission_blockers") or []):
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("RELEASE_GATE_") or "TRAINING_ADMISSION" in text:
            continue
        screening_risks.append(text)
    return {
        "label": label,
        "artifact_root": root_text,
        "gpt_review_packet": _rel(gpt_packet, base_dir) if root_text else "",
        "review_packet": _rel(review_packet, base_dir) if root_text else "",
        "target_profile": str(run.get("target_profile") or "").strip(),
        "dominant_lane_family": str(run.get("dominant_lane_family") or "").strip(),
        "input_count": _safe_int(run.get("input_count")),
        "top_ranked_image": str(run.get("top_ranked_image") or "").strip(),
        "review_only_pass_count": _safe_int(run.get("review_only_pass_count")),
        "review_only_warn_count": _safe_int(run.get("review_only_warn_count")),
        "preflight_status": str(run.get("preflight_status") or "").strip(),
        "evidence_status": str(run.get("evidence_status") or "").strip(),
        "completeness_score": _round_or_none(run.get("completeness_score")),
        "active_heavy_provider": str(run.get("active_heavy_provider") or "").strip(),
        "release_state": str(run.get("release_state") or "").strip(),
        "local_decision_authority": "NONE",
        "external_review_route": str(run.get("external_review_route") or "").strip(),
        "training_admission_allowed": False,
        "legacy_admission_fields_state": "DEPRECATED_FORCED_FALSE",
        "screening_risks": screening_risks,
        "legacy_admission_blockers": [
            str(item).strip()
            for item in (run.get("admission_blockers") or [])
            if str(item).strip()
        ],
    }


def _recommended_run_pointers(
    run_index: Dict[str, Any],
    *,
    base_dir: Path,
) -> List[Dict[str, Any]]:
    recommended = run_index.get("recommended_runs") if isinstance(run_index.get("recommended_runs"), dict) else {}
    rows: List[Dict[str, Any]] = []
    current = run_index.get("current_outputs") if isinstance(run_index.get("current_outputs"), dict) else {}
    if current:
        rows.append(_run_pointer("current_outputs", current, base_dir=base_dir))
    front = recommended.get("front_bootstrap_snapshot")
    if isinstance(front, dict):
        rows.append(_run_pointer("front_bootstrap_snapshot", front, base_dir=base_dir))
    three_quarter = recommended.get("three_quarter_clean_snapshot")
    if isinstance(three_quarter, dict):
        rows.append(_run_pointer("three_quarter_clean_snapshot", three_quarter, base_dir=base_dir))
    return rows


def _blocking_factors(status_board: Dict[str, Any], invariance_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []
    gates = invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {}
    for name, raw_gate in gates.items():
        gate = raw_gate if isinstance(raw_gate, dict) else {}
        status = str(gate.get("status") or "").strip().upper()
        if status == "PASS":
            continue
        reasons = [
            str(item).strip()
            for item in (gate.get("reasons") or [])
            if str(item).strip()
        ]
        factors.append(
            {
                "area": name,
                "status": status,
                "why_it_matters": {
                    "angle_invariance": "Angle noise from generation can be mistaken for identity drift.",
                    "clothing_invariance": "Outerwear and occlusion can hide body truth and create false failures.",
                    "lighting_invariance": "Exposure or skin-tone changes can be mistaken for face/body drift.",
                    "topology_consistency": "Face/body topology support is the hard evidence for cross-pose identity.",
                }.get(name, "This gate is not mature enough for promotion."),
                "reasons": reasons,
            }
        )
    input_manifests = status_board.get("input_manifests") if isinstance(status_board.get("input_manifests"), dict) else {}
    for name, raw_manifest in input_manifests.items():
        manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
        if bool(manifest.get("required_field_ready")):
            continue
        factors.append(
            {
                "area": name,
                "status": "WARN",
                "why_it_matters": "Missing prompt_id / seed-or-unavailable / anchor_source makes batch replay and drift diagnosis weaker.",
                "reasons": [
                    f"MISSING_{field.upper()}"
                    for field in (manifest.get("missing_fields") or [])
                    if str(field).strip()
                ],
            }
        )
    pose_gait = (
        status_board.get("pose_gait_body_truth")
        if isinstance(status_board.get("pose_gait_body_truth"), dict)
        else {}
    )
    read_counts = pose_gait.get("read_counts") if isinstance(pose_gait.get("read_counts"), dict) else {}
    non_consistent_count = _safe_int(pose_gait.get("non_consistent_count"))
    if non_consistent_count > 0:
        factors.append(
            {
                "area": "pose_gait_body_truth",
                "status": "REVIEW",
                "why_it_matters": "116-1 is the absolute body truth, but stance and gait can explain some measurement deltas.",
                "reasons": [
                    f"{read}={count}"
                    for read, count in read_counts.items()
                    if str(read).strip() != "pose_gait_consistent" and _safe_int(count) > 0
                ],
            }
        )
    return factors


def _compact_pose_gait_body_truth(status_board: Dict[str, Any]) -> Dict[str, Any]:
    summary = (
        status_board.get("pose_gait_body_truth")
        if isinstance(status_board.get("pose_gait_body_truth"), dict)
        else {}
    )
    examples = summary.get("review_examples") if isinstance(summary.get("review_examples"), list) else []
    return {
        "available": bool(summary.get("available")),
        "truth_anchor": str(summary.get("truth_anchor") or "").strip(),
        "face_truth_anchor": str(summary.get("face_truth_anchor") or "").strip(),
        "policy": str(summary.get("policy") or "").strip(),
        "sample_count": _safe_int(summary.get("sample_count")),
        "read_counts": summary.get("read_counts") if isinstance(summary.get("read_counts"), dict) else {},
        "non_consistent_count": _safe_int(summary.get("non_consistent_count")),
        "metric_means": summary.get("metric_means") if isinstance(summary.get("metric_means"), dict) else {},
        "review_examples": examples[:3],
        "action_hint": str(summary.get("action_hint") or "").strip(),
    }


def _compact_optimization_focus(status_board: Dict[str, Any]) -> Dict[str, Any]:
    focus = (
        status_board.get("optimization_focus")
        if isinstance(status_board.get("optimization_focus"), dict)
        else {}
    )
    compact: Dict[str, Any] = {}
    for key in ["clothing_invariance", "gait_invariance", "topology_consistency"]:
        row = focus.get(key) if isinstance(focus.get(key), dict) else {}
        if not row:
            continue
        compact[key] = {
            "state": str(row.get("state") or "").strip(),
            "gate_status": str(row.get("gate_status") or "").strip(),
            "blocking_reasons": row.get("blocking_reasons") if isinstance(row.get("blocking_reasons"), list) else [],
            "read_counts": row.get("read_counts") if isinstance(row.get("read_counts"), dict) else {},
            "next_action": str(row.get("next_action") or "").strip(),
            "holds": row.get("holds") if isinstance(row.get("holds"), list) else [],
        }
    return compact


def _compact_project_scope(status_board: Dict[str, Any]) -> Dict[str, Any]:
    scope = (
        status_board.get("project_scope")
        if isinstance(status_board.get("project_scope"), dict)
        else {}
    )
    return {
        "role": str(scope.get("role") or "screening_and_evidence_only").strip(),
        "machine_role": str(scope.get("machine_role") or "rank_candidates_explain_risks_and_package_review_evidence").strip(),
        "training_admission_participation": bool(scope.get("training_admission_participation", False)),
        "image_set_decision_participation": bool(scope.get("image_set_decision_participation", False)),
        "training_admission_status": str(scope.get("training_admission_status") or "out_of_scope_for_this_project").strip(),
        "final_training_decision_owner": str(scope.get("final_training_decision_owner") or "external_training_decision_flow").strip(),
        "final_image_set_decision_owner": str(scope.get("final_image_set_decision_owner") or "external_dataset_curation_flow").strip(),
    }


def _compact_replay_collection_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not plan:
        return {
            "available": False,
            "overall_status": "",
            "summary": {},
            "immediate_operator_queue": [],
        }
    queue = plan.get("immediate_operator_queue") if isinstance(plan.get("immediate_operator_queue"), list) else []
    compact_queue: List[Dict[str, Any]] = []
    for raw_task in queue[:8]:
        if not isinstance(raw_task, dict):
            continue
        compact_queue.append(
            {
                "priority_group": str(raw_task.get("priority_group") or "").strip(),
                "area": str(raw_task.get("area") or "").strip(),
                "task_type": str(raw_task.get("task_type") or "").strip(),
                "title": str(raw_task.get("title") or "").strip(),
                "status": str(raw_task.get("status") or "").strip(),
                "input_dir": str(raw_task.get("input_dir") or "").strip(),
                "target_profile": str(raw_task.get("target_profile") or "").strip(),
                "image_count": _safe_int(raw_task.get("image_count")),
                "images_needed_for_minimum": _safe_int(raw_task.get("images_needed_for_minimum")),
                "run_commands": raw_task.get("run_commands") if isinstance(raw_task.get("run_commands"), list) else [],
            }
        )
    return {
        "available": True,
        "overall_status": str(plan.get("overall_status") or "").strip(),
        "summary": plan.get("summary") if isinstance(plan.get("summary"), dict) else {},
        "immediate_operator_queue": compact_queue,
    }


def _compact_consistency_confidence(matrix: Dict[str, Any]) -> Dict[str, Any]:
    if not matrix:
        return {
            "available": False,
            "overall_status": "",
            "item_count": 0,
            "weakest_axes": [],
            "top_unresolved_evidence_gaps": [],
            "top_review_queue": [],
        }
    batch = matrix.get("batch_confidence") if isinstance(matrix.get("batch_confidence"), dict) else {}
    queue = matrix.get("top_review_queue") if isinstance(matrix.get("top_review_queue"), list) else []
    compact_queue: List[Dict[str, Any]] = []
    for raw_row in queue[:5]:
        if not isinstance(raw_row, dict):
            continue
        compact_queue.append(
            {
                "rank": raw_row.get("rank"),
                "image": str(raw_row.get("image") or "").strip(),
                "review_only_status": str(raw_row.get("review_only_status") or "").strip(),
                "evidence_confidence_score": _round_or_none(raw_row.get("evidence_confidence_score")),
                "evidence_confidence_band": str(raw_row.get("evidence_confidence_band") or "").strip(),
                "review_priority": str(raw_row.get("review_priority") or "").strip(),
                "body_truth_pose_gait_read": str(raw_row.get("body_truth_pose_gait_read") or "").strip(),
                "unresolved_evidence_gaps": raw_row.get("unresolved_evidence_gaps")
                if isinstance(raw_row.get("unresolved_evidence_gaps"), list)
                else [],
            }
        )
    return {
        "available": True,
        "overall_status": str(batch.get("overall_status") or "").strip(),
        "item_count": _safe_int(batch.get("item_count")),
        "evidence_confidence_band_counts": batch.get("evidence_confidence_band_counts")
        if isinstance(batch.get("evidence_confidence_band_counts"), dict)
        else {},
        "review_priority_counts": batch.get("review_priority_counts")
        if isinstance(batch.get("review_priority_counts"), dict)
        else {},
        "pose_gait_read_counts": batch.get("pose_gait_read_counts")
        if isinstance(batch.get("pose_gait_read_counts"), dict)
        else {},
        "weakest_axes": batch.get("weakest_axes") if isinstance(batch.get("weakest_axes"), list) else [],
        "top_unresolved_evidence_gaps": batch.get("top_unresolved_evidence_gaps")
        if isinstance(batch.get("top_unresolved_evidence_gaps"), list)
        else [],
        "ranking_stability": batch.get("ranking_stability")
        if isinstance(batch.get("ranking_stability"), dict)
        else {},
        "top_review_queue": compact_queue,
    }


def _compact_pose_gait_margin_review(sheet: Dict[str, Any]) -> Dict[str, Any]:
    if not sheet:
        return {
            "available": False,
            "review_row_count": 0,
            "p0_review_row_count": 0,
            "category_counts": {},
            "p0_review_queue": [],
        }
    summary = sheet.get("summary") if isinstance(sheet.get("summary"), dict) else {}
    queue = sheet.get("p0_review_queue") if isinstance(sheet.get("p0_review_queue"), list) else []
    compact_queue: List[Dict[str, Any]] = []
    for raw_row in queue[:6]:
        if not isinstance(raw_row, dict):
            continue
        compact_queue.append(
            {
                "priority": str(raw_row.get("priority") or "").strip(),
                "rank": raw_row.get("rank"),
                "image": str(raw_row.get("image") or "").strip(),
                "review_category": str(raw_row.get("review_category") or "").strip(),
                "body_truth_pose_gait_read": str(raw_row.get("body_truth_pose_gait_read") or "").strip(),
                "evidence_confidence_score": _round_or_none(raw_row.get("evidence_confidence_score")),
                "axis_scores": raw_row.get("axis_scores") if isinstance(raw_row.get("axis_scores"), dict) else {},
                "review_focus": raw_row.get("review_focus") if isinstance(raw_row.get("review_focus"), list) else [],
            }
        )
    return {
        "available": True,
        "review_row_count": _safe_int(summary.get("review_row_count")),
        "p0_review_row_count": _safe_int(summary.get("p0_review_row_count")),
        "category_counts": summary.get("category_counts") if isinstance(summary.get("category_counts"), dict) else {},
        "priority_counts": summary.get("priority_counts") if isinstance(summary.get("priority_counts"), dict) else {},
        "body_truth_pose_gait_read_counts": summary.get("body_truth_pose_gait_read_counts")
        if isinstance(summary.get("body_truth_pose_gait_read_counts"), dict)
        else {},
        "axis_means": summary.get("axis_means") if isinstance(summary.get("axis_means"), dict) else {},
        "current_primary_limit": str(summary.get("current_primary_limit") or "").strip(),
        "p0_review_queue": compact_queue,
    }


def build_review_handoff_packet(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = base_dir / "outputs"
    status_board = _load_json(outputs_dir / "review_status_board.json")
    invariance_status = _load_json(outputs_dir / "review_invariance_status.json")
    manifest_completion = _load_json(outputs_dir / "input_manifest_completion_plan.json")
    replay_collection = _load_json(outputs_dir / "replay_collection_plan.json")
    consistency_matrix = _load_json(outputs_dir / "consistency_confidence_matrix.json")
    pose_gait_margin_sheet = _load_json(outputs_dir / "pose_gait_margin_review_sheet.json")
    run_index = _load_json(outputs_dir / "review_run_index.json")
    front_sheet = _load_json(outputs_dir / "front_bootstrap_review_sheet.json")
    gates = invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {}
    winner_policy = (
        status_board.get("winner_bank_bootstrap_policy")
        if isinstance(status_board.get("winner_bank_bootstrap_policy"), dict)
        else {}
    )
    training_admission = (
        status_board.get("training_admission")
        if isinstance(status_board.get("training_admission"), dict)
        else {}
    )
    top_candidates = (
        (status_board.get("front_bootstrap_review") or {}).get("top_candidates")
        if isinstance(status_board.get("front_bootstrap_review"), dict)
        else []
    )
    if not top_candidates:
        top_candidates = front_sheet.get("top_candidates") if isinstance(front_sheet.get("top_candidates"), list) else []

    next_actions = _dedupe(status_board.get("next_actions") or invariance_status.get("next_actions") or [])
    pose_gait_body_truth = _compact_pose_gait_body_truth(status_board)
    optimization_focus = _compact_optimization_focus(status_board)
    project_scope = _compact_project_scope(status_board)
    payload = {
        "schema_version": "review_handoff_packet_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compact packet for GPT/manual review. Use this before opening large raw QA artifacts.",
        "project_scope": project_scope,
        "decision_state": {
            "current_phase": "review_only_invariance_hardening",
            "machine_role": project_scope["role"],
            "final_training_decision_owner": project_scope["final_training_decision_owner"],
            "final_image_set_decision_owner": project_scope["final_image_set_decision_owner"],
            "review_invariance_overall_status": str(invariance_status.get("overall_status") or "").strip(),
            "winner_bank_bootstrap_allowed": bool(invariance_status.get("winner_bank_bootstrap_allowed")),
            "winner_bank_freeze_allowed": bool(invariance_status.get("winner_bank_freeze_allowed")),
            "winner_bank_mutable_memory_allowed": bool(invariance_status.get("winner_bank_mutable_memory_allowed", True)),
            "winner_bank_policy_state": str(winner_policy.get("state") or "").strip(),
            "winner_bank_freeze_state": str(winner_policy.get("freeze_state") or "").strip(),
            "parameter_fitting_allowed": bool(invariance_status.get("parameter_fitting_allowed")),
            "training_admission_participation": bool(project_scope["training_admission_participation"]),
            "image_set_decision_participation": bool(project_scope["image_set_decision_participation"]),
            "training_admission_status": project_scope["training_admission_status"],
            "legacy_or_external_training_manifest_available": bool(training_admission.get("available")),
            "training_admission_allowed_now": False,
        },
        "send_policy": {
            "send_first": [_rel(output_file, base_dir)],
            "send_if_gpt_needs_batch_detail": [
                "outputs/review_status_board.json",
                "outputs/review_invariance_status.json",
                "outputs/lighting_replay_pack.json",
                "outputs/outer_replay_pack.json",
                "outputs/topology_replay_pack.json",
                "outputs/replay_collection_plan.json",
                "outputs/input_manifest_completion_plan.json",
                "outputs/consistency_confidence_matrix.json",
                "outputs/pose_gait_margin_review_sheet.json",
                "outputs/review_run_index.json",
                "outputs/body_topology_truth_fusion_compare.json",
                "outputs/front_bootstrap_review_sheet.json",
            ],
            "send_if_gpt_needs_candidate_level_detail": [
                "outputs/gpt_review_packet.json",
                "outputs_snapshots/2026-04-18_front_segformer_body_fusion/gpt_review_packet.json",
                "outputs_snapshots/2026-04-23_three_quarter_segformer_body_truth_fusion_topology_v3/gpt_review_packet.json",
            ],
            "do_not_send_by_default": [
                {"file": "outputs/qa_report.json", "reason": "large raw machine report"},
                {"file": "outputs/ranked_candidates.json", "reason": "large ranking dump"},
                {"file": "outputs/review_packet.json", "reason": "verbose review artifact"},
                {"file": "outputs/winner_bank_candidate.json", "reason": "mutable winner-bank candidate export"},
            ],
        },
        "manifest_completion": {
            "overall_status": manifest_completion.get("overall_status"),
            "ready_for_clean_replay": bool(manifest_completion.get("ready_for_clean_replay")),
            "blocked_splits": manifest_completion.get("blocked_splits")
            if isinstance(manifest_completion.get("blocked_splits"), list)
            else [],
        },
        "replay_collection_plan": _compact_replay_collection_plan(replay_collection),
        "consistency_confidence_matrix": _compact_consistency_confidence(consistency_matrix),
        "pose_gait_margin_review": _compact_pose_gait_margin_review(pose_gait_margin_sheet),
        "gate_summary": _compact_gate_rows(gates),
        "pose_gait_body_truth": pose_gait_body_truth,
        "optimization_focus": optimization_focus,
        "blocking_factors": _blocking_factors(status_board, invariance_status),
        "recommended_run_pointers": _recommended_run_pointers(run_index, base_dir=base_dir),
        "diagnostic_front_top_candidates": [
            {
                "rank": item.get("rank"),
                "image": item.get("image"),
                "selection_score": _round_or_none(item.get("selection_score")),
                "face_master_alignment": _round_or_none(item.get("face_master_alignment")),
                "body_truth_alignment": _round_or_none(item.get("body_truth_alignment")),
            }
            for item in (top_candidates or [])[:3]
            if isinstance(item, dict)
        ],
        "next_actions": next_actions,
        "explicit_holds": [
            "Do not freeze winner_bank until review_invariance_overall_status is READY.",
            "Training admission is outside this project's scope; route evidence to the external training-decision flow.",
            "Final image-set decisions are outside this project's scope; route evidence to the external dataset-curation flow.",
            "Do not treat mutable winner_bank entries or front diagnostic top candidates as identity truth.",
            "Do not run parameter fitting before project optimization is complete.",
        ],
        "questions_for_gpt_or_manual_review": [
            "Which blocker should be cleared first: manifest metadata, angle replay, lighting replay, or OUTER occlusion replay?",
            "Which candidates should be routed as strong screening candidates, review-needed candidates, or diagnostic-only examples?",
            "Which optional detail file is needed for the next decision, if any?",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_file, payload)
    return payload
