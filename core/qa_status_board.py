from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_input_manifest import load_input_manifest_index, required_prompt_intent_fields
from .qa_io import atomic_write_json
from .qa_winner_bank_policy import winner_bank_bootstrap_policy


def _project_scope() -> Dict[str, Any]:
    return {
        "schema_version": "project_scope_v1",
        "role": "screening_and_evidence_only",
        "machine_role": "rank_candidates_explain_risks_route_review_priority_and_package_evidence",
        "training_admission_participation": False,
        "image_set_decision_participation": False,
        "training_admission_status": "out_of_scope_for_this_project",
        "final_training_decision_owner": "external_training_decision_flow",
        "final_image_set_decision_owner": "external_dataset_curation_flow",
        "outputs_are": [
            "candidate screening",
            "risk routing",
            "review-priority ranking",
            "review evidence packets",
            "mutable winner-bank review memory",
        ],
        "outputs_are_not": [
            "final training-set admission",
            "final image-set decision",
            "dataset membership decision",
            "training sample seal",
            "frozen identity truth",
            "parameter-fitting data",
        ],
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _first_text(value: Any) -> str:
    return str(value or "").strip()


def _seed_or_unavailable_ready(item: Dict[str, Any]) -> bool:
    return item.get("seed") is not None or bool(_first_text(item.get("seed_unavailable_reason")))


def _manifest_required_field_coverage(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(entries))
    return {
        "prompt_id": float(sum(1 for item in entries if _first_text(item.get("prompt_id"))) / total),
        "seed": float(sum(1 for item in entries if _seed_or_unavailable_ready(item)) / total),
        "seed_available": float(sum(1 for item in entries if item.get("seed") is not None) / total),
        "seed_unavailable_reason": float(
            sum(1 for item in entries if _first_text(item.get("seed_unavailable_reason"))) / total
        ),
        "anchor_source": float(sum(1 for item in entries if _first_text(item.get("anchor_source"))) / total),
        "intended_view": float(sum(1 for item in entries if _first_text(item.get("intended_view"))) / total),
    }


def _manifest_board_entry(input_dir: Path) -> Dict[str, Any]:
    manifest_index = load_input_manifest_index(input_dir)
    entries = manifest_index.get("entries") if isinstance(manifest_index.get("entries"), list) else []
    coverage = _manifest_required_field_coverage(entries)
    missing_fields = [field for field in required_prompt_intent_fields() if float(coverage.get(field) or 0.0) < 1.0]
    return {
        "input_dir": str(input_dir.resolve()),
        "available": bool(manifest_index.get("available")),
        "manifest_path": manifest_index.get("path"),
        "item_count": len(entries),
        "required_fields": required_prompt_intent_fields(),
        "field_policy": {
            "seed": "ready when seed is present, or seed_unavailable_reason documents why the generator did not expose it",
        },
        "required_field_coverage": {key: _round_or_none(value) for key, value in coverage.items()},
        "required_field_ready": not missing_fields and bool(entries),
        "missing_fields": missing_fields,
    }


def _manifest_missing_action(label: str, manifest_state: Dict[str, Any]) -> str:
    missing_fields = [
        str(field or "").strip()
        for field in (manifest_state.get("missing_fields") or [])
        if str(field or "").strip()
    ]
    if not missing_fields:
        return f"verify {label} split manifest prompt intent metadata"
    field_text = " / ".join(missing_fields)
    return f"fill {label} split manifest fields: {field_text}"


def _training_admission_summary(manifest_file: Path) -> Dict[str, Any]:
    payload = _load_json(manifest_file)
    scope = _project_scope()
    if not payload:
        return {
            "scope": "external_final_decision_out_of_scope",
            "participates_in_final_admission": bool(scope.get("training_admission_participation")),
            "availability_is_blocker": False,
            "available": False,
            "manifest_file": str(manifest_file.resolve()),
            "entry_count": 0,
            "last_recorded_at_utc": None,
            "last_sealed_at_utc": None,
            "legacy_seal_fields_state": "DEPRECATED_FORCED_EMPTY",
            "status": "not_required_by_screening_project",
        }
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    recent_entry = entries[-1] if entries else {}
    external_audit = (
        recent_entry.get("external_audit")
        if isinstance(recent_entry, dict) and isinstance(recent_entry.get("external_audit"), dict)
        else {}
    )
    legacy_seal = (
        recent_entry.get("human_seal")
        if isinstance(recent_entry, dict) and isinstance(recent_entry.get("human_seal"), dict)
        else {}
    )
    return {
        "scope": "external_final_decision_out_of_scope",
        "participates_in_final_admission": bool(scope.get("training_admission_participation")),
        "availability_is_blocker": False,
        "available": True,
        "manifest_file": str(manifest_file.resolve()),
        "entry_count": len(entries),
        "last_recorded_at_utc": external_audit.get("recorded_at_utc")
        or legacy_seal.get("sealed_at_utc"),
        "last_sealed_at_utc": None,
        "legacy_seal_fields_state": "DEPRECATED_FORCED_EMPTY",
        "status": "legacy_or_external_manifest_present_for_audit_only",
    }


def _replay_collection_summary(plan_file: Path) -> Dict[str, Any]:
    payload = _load_json(plan_file)
    if not payload:
        return {
            "available": False,
            "plan_file": str(plan_file.resolve()),
            "overall_status": "",
            "summary": {},
            "immediate_operator_queue": [],
        }
    queue = payload.get("immediate_operator_queue") if isinstance(payload.get("immediate_operator_queue"), list) else []
    compact_queue: List[Dict[str, Any]] = []
    for raw_task in queue[:6]:
        if not isinstance(raw_task, dict):
            continue
        compact_queue.append(
            {
                "priority_group": _first_text(raw_task.get("priority_group")),
                "area": _first_text(raw_task.get("area")),
                "task_type": _first_text(raw_task.get("task_type")),
                "title": _first_text(raw_task.get("title")),
                "status": _first_text(raw_task.get("status")),
                "input_dir": _first_text(raw_task.get("input_dir")),
                "image_count": _safe_int(raw_task.get("image_count")),
                "images_needed_for_minimum": _safe_int(raw_task.get("images_needed_for_minimum")),
            }
        )
    return {
        "available": True,
        "plan_file": str(plan_file.resolve()),
        "overall_status": _first_text(payload.get("overall_status")),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "immediate_operator_queue": compact_queue,
    }


def _consistency_confidence_summary(matrix_file: Path) -> Dict[str, Any]:
    payload = _load_json(matrix_file)
    if not payload:
        return {
            "available": False,
            "matrix_file": str(matrix_file.resolve()),
            "overall_status": "",
            "item_count": 0,
            "weakest_axes": [],
            "top_unresolved_evidence_gaps": [],
            "ranking_stability": {},
        }
    batch = payload.get("batch_confidence") if isinstance(payload.get("batch_confidence"), dict) else {}
    return {
        "available": True,
        "matrix_file": str(matrix_file.resolve()),
        "overall_status": _first_text(batch.get("overall_status")),
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
    }


def _pose_gait_margin_review_summary(sheet_file: Path) -> Dict[str, Any]:
    payload = _load_json(sheet_file)
    if not payload:
        return {
            "available": False,
            "sheet_file": str(sheet_file.resolve()),
            "review_row_count": 0,
            "p0_review_row_count": 0,
            "category_counts": {},
            "priority_counts": {},
            "current_primary_limit": "",
        }
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "available": True,
        "sheet_file": str(sheet_file.resolve()),
        "review_row_count": _safe_int(summary.get("review_row_count")),
        "p0_review_row_count": _safe_int(summary.get("p0_review_row_count")),
        "category_counts": summary.get("category_counts") if isinstance(summary.get("category_counts"), dict) else {},
        "priority_counts": summary.get("priority_counts") if isinstance(summary.get("priority_counts"), dict) else {},
        "body_truth_pose_gait_read_counts": summary.get("body_truth_pose_gait_read_counts")
        if isinstance(summary.get("body_truth_pose_gait_read_counts"), dict)
        else {},
        "axis_means": summary.get("axis_means") if isinstance(summary.get("axis_means"), dict) else {},
        "current_primary_limit": _first_text(summary.get("current_primary_limit")),
    }


def _screening_risks(run: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    for item in (run.get("screening_risks") or run.get("admission_blockers") or []):
        text = str(item).strip()
        if not text:
            continue
        if text.startswith("RELEASE_GATE_") or "TRAINING_ADMISSION" in text:
            continue
        risks.append(text)
    return risks


def _candidate_rows_for_pose_gait(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    queue = packet.get("priority_review_queue") if isinstance(packet.get("priority_review_queue"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for bucket, raw_items in queue.items():
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            item["_pose_gait_source_bucket"] = str(bucket or "").strip()
            rows.append(item)
    if rows:
        return rows

    top_candidates = packet.get("top_candidates") if isinstance(packet.get("top_candidates"), list) else []
    for raw_item in top_candidates:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item["_pose_gait_source_bucket"] = "top_candidates"
        rows.append(item)
    return rows


def _pose_gait_body_truth_summary(gpt_packet_file: Path) -> Dict[str, Any]:
    packet = _load_json(gpt_packet_file)
    if not packet:
        return {
            "available": False,
            "source_file": str(gpt_packet_file.resolve()),
            "truth_anchor": "Task-63987060-116-1.png",
            "face_truth_anchor": "A-Core_01_0deg_MASTER.png",
            "policy": "pose_gait_aware_absolute_116_1",
            "sample_count": 0,
            "read_counts": {},
            "metric_means": {},
            "review_examples": [],
            "action_hint": "refresh gpt_review_packet before pose/gait body truth review",
        }

    rows = _candidate_rows_for_pose_gait(packet)
    read_counts: Dict[str, int] = {}
    metric_names = [
        "body_pose_independent_truth_alignment",
        "body_gait_tolerant_topology_similarity",
        "body_core_measurement_similarity",
        "body_pose_sensitive_measurement_similarity",
        "body_topology_partition_mean_similarity",
        "body_topology_weakest_part_similarity",
        "body_pose_explained_delta_score",
        "body_pose_measurement_gap",
    ]
    metric_sums: Dict[str, float] = {name: 0.0 for name in metric_names}
    metric_counts: Dict[str, int] = {name: 0 for name in metric_names}
    consistent_examples: List[Dict[str, Any]] = []
    non_consistent_examples: List[Dict[str, Any]] = []
    non_consistent_reads = {
        "gait_tolerant_topology_margin_review",
        "manual_review_required",
        "pose_sensitive_noise_possible",
        "pose_explained_delta_possible",
        "unexplained_body_drift_risk",
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        summary = (
            row.get("canonical_truth_summary")
            if isinstance(row.get("canonical_truth_summary"), dict)
            else {}
        )
        read = _first_text(summary.get("body_truth_pose_gait_read")) or "unavailable"
        read_counts[read] = read_counts.get(read, 0) + 1
        for name in metric_names:
            number = _safe_float(summary.get(name))
            if number is None:
                continue
            metric_sums[name] += number
            metric_counts[name] += 1

        example = {
            "source_bucket": _first_text(row.get("_pose_gait_source_bucket")),
            "rank": row.get("rank"),
            "image": row.get("image"),
            "review_only_status": row.get("review_only_status"),
            "body_truth_pose_gait_read": read,
            "body_pose_independent_truth_alignment": _round_or_none(
                summary.get("body_pose_independent_truth_alignment")
            ),
            "body_gait_tolerant_topology_similarity": _round_or_none(
                summary.get("body_gait_tolerant_topology_similarity")
            ),
            "body_core_measurement_similarity": _round_or_none(
                summary.get("body_core_measurement_similarity")
            ),
            "body_pose_sensitive_measurement_similarity": _round_or_none(
                summary.get("body_pose_sensitive_measurement_similarity")
            ),
            "body_topology_partition_mean_similarity": _round_or_none(
                summary.get("body_topology_partition_mean_similarity")
            ),
            "body_topology_weakest_part": _first_text(summary.get("body_topology_weakest_part")),
            "body_topology_weakest_part_similarity": _round_or_none(
                summary.get("body_topology_weakest_part_similarity")
            ),
            "body_pose_explained_delta_score": _round_or_none(
                summary.get("body_pose_explained_delta_score")
            ),
            "body_pose_measurement_gap": _round_or_none(summary.get("body_pose_measurement_gap")),
        }
        if read in non_consistent_reads:
            if len(non_consistent_examples) < 6:
                non_consistent_examples.append(example)
        elif len(consistent_examples) < 3:
            consistent_examples.append(example)

    sorted_counts = dict(sorted(read_counts.items(), key=lambda item: (-item[1], item[0])))
    metric_means = {
        name: _round_or_none(metric_sums[name] / metric_counts[name])
        for name in metric_names
        if metric_counts[name] > 0
    }
    non_consistent_count = sum(
        count for read, count in read_counts.items() if read != "pose_gait_consistent"
    )
    if not rows:
        action_hint = "no candidate rows found for pose/gait body truth review"
    elif non_consistent_count:
        action_hint = "review non-consistent pose/gait rows before treating body deltas as identity drift"
    else:
        action_hint = "pose/gait body truth reads are consistent in the current review sample"

    return {
        "available": bool(rows),
        "source_file": str(gpt_packet_file.resolve()),
        "truth_anchor": "Task-63987060-116-1.png",
        "face_truth_anchor": "A-Core_01_0deg_MASTER.png",
        "policy": "pose_gait_aware_absolute_116_1",
        "sample_count": len(rows),
        "read_counts": sorted_counts,
        "non_consistent_count": non_consistent_count,
        "metric_means": metric_means,
        "review_examples": (non_consistent_examples + consistent_examples)[:6],
        "action_hint": action_hint,
    }


def _gate_reasons(gate: Dict[str, Any]) -> List[str]:
    return [
        str(item).strip()
        for item in (gate.get("reasons") or [])
        if str(item).strip()
    ]


def _optimization_focus(
    invariance_status: Dict[str, Any],
    pose_gait_body_truth: Dict[str, Any],
) -> Dict[str, Any]:
    gates = invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {}
    clothing_gate = gates.get("clothing_invariance") if isinstance(gates.get("clothing_invariance"), dict) else {}
    topology_gate = gates.get("topology_consistency") if isinstance(gates.get("topology_consistency"), dict) else {}
    clothing_metrics = (
        clothing_gate.get("metrics")
        if isinstance(clothing_gate.get("metrics"), dict)
        else {}
    )
    topology_metrics = (
        topology_gate.get("metrics")
        if isinstance(topology_gate.get("metrics"), dict)
        else {}
    )
    clothing_reasons = _gate_reasons(clothing_gate)
    topology_reasons = _gate_reasons(topology_gate)
    read_counts = (
        pose_gait_body_truth.get("read_counts")
        if isinstance(pose_gait_body_truth.get("read_counts"), dict)
        else {}
    )
    margin_count = _safe_int(read_counts.get("gait_tolerant_topology_margin_review"))
    manual_count = _safe_int(read_counts.get("manual_review_required"))
    drift_risk_count = _safe_int(read_counts.get("unexplained_body_drift_risk"))

    if "OUTER_OCCLUSION_REPLAY_NOT_COLLECTED" in clothing_reasons:
        clothing_state = "outer_replay_evidence_gap"
        clothing_next = "collect governed OUTER replay images before changing clothing gates or activating OUTER runtime assets"
    elif clothing_reasons:
        clothing_state = "lane_metric_review_required"
        clothing_next = "inspect clothing lane metrics and candidate examples before any threshold change"
    else:
        clothing_state = "simple_and_outer_evidence_ready"
        clothing_next = "keep clothing gate as evidence-only until winner-bank freeze governance reopens"

    if drift_risk_count > 0:
        gait_state = "body_drift_risk_review_required"
        gait_next = "inspect unexplained drift rows against 116-1 before reviewing ranking changes"
    elif margin_count > 0:
        gait_state = "gait_tolerant_topology_margin_review"
        gait_next = "review margin rows as gait/topology cases; do not treat them as body drift without structural evidence"
    elif manual_count > 0:
        gait_state = "manual_pose_gait_review_required"
        gait_next = "classify remaining manual rows by pose/gait, clothing, and topology support"
    else:
        gait_state = "pose_gait_reads_consistent"
        gait_next = "keep pose/gait reads as review evidence; do not feed them into fitting"

    topology_compare = (
        topology_metrics.get("three_quarter_truth_fusion_compare")
        if isinstance(topology_metrics.get("three_quarter_truth_fusion_compare"), dict)
        else {}
    )
    topology_pack = (
        topology_metrics.get("side_back_topology_replay_pack")
        if isinstance(topology_metrics.get("side_back_topology_replay_pack"), dict)
        else {}
    )
    topology_status = str(topology_gate.get("status") or "").strip().upper()
    if topology_status == "PASS" and bool(topology_compare.get("resolved_for_three_quarter_review")):
        if not bool(topology_pack.get("prepared")):
            topology_state = "three_quarter_resolved_prepare_side_back_topology_pack"
            topology_next = "prepare controlled side/back topology replay pack before promotion"
        elif _safe_int(topology_pack.get("total_current_images")) <= 0:
            topology_state = "three_quarter_resolved_side_back_collection_needed"
            topology_next = "collect controlled side/back topology variants into input_replay/topology before promotion"
        else:
            topology_state = "three_quarter_resolved_run_side_back_topology_replay"
            topology_next = "run side/back replay with profile-default truth-fusion before promotion"
    elif topology_status == "PASS":
        topology_state = "front_three_quarter_pass"
        topology_next = "keep topology monitoring active while clearing clothing and pose/gait review queues"
    else:
        topology_state = "topology_support_review_required"
        topology_next = "tighten topology evidence coverage before any winner-bank freeze"

    return {
        "clothing_invariance": {
            "state": clothing_state,
            "gate_status": str(clothing_gate.get("status") or "").strip(),
            "blocking_reasons": clothing_reasons,
            "simple_outfit_metrics_by_lane": clothing_metrics.get("clothing_metrics_by_lane")
            if isinstance(clothing_metrics.get("clothing_metrics_by_lane"), dict)
            else {},
            "outer_replay_pack": clothing_metrics.get("outer_replay_pack")
            if isinstance(clothing_metrics.get("outer_replay_pack"), dict)
            else {},
            "next_action": clothing_next,
            "holds": [
                "do not activate OUTER runtime pack from mixed production evidence",
                "do not change clothing thresholds before controlled OUTER replay exists",
            ],
        },
        "gait_invariance": {
            "state": gait_state,
            "truth_anchor": str(pose_gait_body_truth.get("truth_anchor") or "").strip(),
            "policy": str(pose_gait_body_truth.get("policy") or "").strip(),
            "read_counts": read_counts,
            "metric_means": pose_gait_body_truth.get("metric_means")
            if isinstance(pose_gait_body_truth.get("metric_means"), dict)
            else {},
            "review_examples": pose_gait_body_truth.get("review_examples")
            if isinstance(pose_gait_body_truth.get("review_examples"), list)
            else [],
            "next_action": gait_next,
            "holds": [
                "do not treat gait/topology margin rows as body drift without manual structure review",
                "do not fit pose/gait thresholds from current candidate-review data",
            ],
        },
        "topology_consistency": {
            "state": topology_state,
            "gate_status": str(topology_gate.get("status") or "").strip(),
            "blocking_reasons": topology_reasons,
            "body_topology_top3_mean": _round_or_none(topology_metrics.get("body_topology_top3_mean")),
            "body_topology_top3_mean_by_lane": topology_metrics.get("body_topology_top3_mean_by_lane")
            if isinstance(topology_metrics.get("body_topology_top3_mean_by_lane"), dict)
            else {},
            "three_quarter_truth_fusion_compare": topology_compare,
            "side_back_topology_replay_pack": topology_pack,
            "next_action": topology_next,
            "holds": [
                "do not spend another optimization loop on three_quarter topology unless a new replay regresses",
                "do validate the same truth-fusion topology chain on side/back lanes",
            ],
        },
    }


def build_review_status_board(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    run_index = _load_json(outputs_dir / "review_run_index.json")
    front_sheet = _load_json(outputs_dir / "front_bootstrap_review_sheet.json")
    invariance_status = _load_json(outputs_dir / "review_invariance_status.json")
    winner_bank_report = _load_json(outputs_dir / "winner_bank_report.json")
    winner_policy = winner_bank_bootstrap_policy()
    training_admission = _training_admission_summary(outputs_dir / "training_admission_manifest.json")
    replay_collection = _replay_collection_summary(outputs_dir / "replay_collection_plan.json")
    consistency_confidence = _consistency_confidence_summary(outputs_dir / "consistency_confidence_matrix.json")
    pose_gait_margin_review = _pose_gait_margin_review_summary(outputs_dir / "pose_gait_margin_review_sheet.json")
    pose_gait_body_truth = _pose_gait_body_truth_summary(outputs_dir / "gpt_review_packet.json")
    optimization_focus = _optimization_focus(invariance_status, pose_gait_body_truth)
    project_scope = _project_scope()

    current_outputs = run_index.get("current_outputs") if isinstance(run_index.get("current_outputs"), dict) else {}
    recommended_runs = run_index.get("recommended_runs") if isinstance(run_index.get("recommended_runs"), dict) else {}
    front_bootstrap = (
        recommended_runs.get("front_bootstrap_snapshot")
        if isinstance(recommended_runs.get("front_bootstrap_snapshot"), dict)
        else {}
    )
    three_quarter_clean = (
        recommended_runs.get("three_quarter_clean_snapshot")
        if isinstance(recommended_runs.get("three_quarter_clean_snapshot"), dict)
        else {}
    )
    top_candidates = front_sheet.get("top_candidates") if isinstance(front_sheet.get("top_candidates"), list) else []
    front_top3 = [
        {
            "rank": item.get("rank"),
            "image": item.get("image"),
            "selection_score": _round_or_none(item.get("selection_score")),
            "face_master_alignment": _round_or_none(item.get("face_master_alignment")),
            "body_truth_alignment": _round_or_none(item.get("body_truth_alignment")),
        }
        for item in top_candidates[:3]
        if isinstance(item, dict)
    ]

    manifest_states = {
        "input_split_front": _manifest_board_entry(base_dir / "input_split" / "front"),
        "input_split_three_quarter": _manifest_board_entry(base_dir / "input_split" / "three_quarter"),
    }

    next_actions: List[str] = []
    invariance_next_actions = (
        invariance_status.get("next_actions")
        if isinstance(invariance_status.get("next_actions"), list)
        else []
    )
    if not manifest_states["input_split_front"]["required_field_ready"]:
        next_actions.append(_manifest_missing_action("front", manifest_states["input_split_front"]))
    if not manifest_states["input_split_three_quarter"]["required_field_ready"]:
        next_actions.append(_manifest_missing_action("three_quarter", manifest_states["input_split_three_quarter"]))
    if str(winner_policy.get("state") or "") == "deferred":
        next_actions.append("defer winner_bank bootstrap until review-only invariance and 3D topology consistency mature")
        for action in invariance_next_actions:
            action_text = str(action or "").strip()
            if action_text and action_text not in next_actions:
                next_actions.append(action_text)
        next_actions.append("use front_bootstrap_review_sheet top-3 for diagnostic review only")
    elif bool(front_sheet) and not bool(front_sheet.get("promotion_ready")):
        next_actions.append("manually review front_bootstrap_review_sheet top-3 as mutable candidate memory only")
    if not bool(winner_bank_report.get("curated_bank_available")) and str(winner_policy.get("state") or "") != "deferred":
        next_actions.append("record mutable winner_bank entry only after manual review resolves current blockers")
    next_actions.append("run prepare_replay_collection_plan and follow immediate_operator_queue for controlled replay collection")
    if bool(consistency_confidence.get("available")):
        weakest_axes = consistency_confidence.get("weakest_axes") if isinstance(consistency_confidence.get("weakest_axes"), list) else []
        if weakest_axes:
            axis = _first_text((weakest_axes[0] or {}).get("axis") if isinstance(weakest_axes[0], dict) else "")
            if axis:
                next_actions.append(f"review consistency_confidence_matrix weakest axis first: {axis}")
    if _safe_int(pose_gait_margin_review.get("p0_review_row_count")) > 0:
        next_actions.append("review pose_gait_margin_review_sheet P0 rows before calling body drift")
    next_actions.append("route screened candidates and evidence packets to the external training-decision flow")

    payload = {
        "schema_version": "review_status_board_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_scope": project_scope,
        "current_outputs": {
            "artifact_root": current_outputs.get("artifact_root"),
            "target_profile": current_outputs.get("target_profile"),
            "dominant_lane_family": current_outputs.get("dominant_lane_family"),
            "top_ranked_image": current_outputs.get("top_ranked_image"),
            "evidence_status": current_outputs.get("evidence_status"),
            "preflight_status": current_outputs.get("preflight_status"),
            "active_heavy_provider": current_outputs.get("active_heavy_provider"),
        },
        "recommended_runs": {
            "front_bootstrap_snapshot": {
                "artifact_root": front_bootstrap.get("artifact_root"),
                "top_ranked_image": front_bootstrap.get("top_ranked_image"),
                "evidence_status": front_bootstrap.get("evidence_status"),
                "completeness_score": front_bootstrap.get("completeness_score"),
                "active_heavy_provider": front_bootstrap.get("active_heavy_provider"),
                "screening_risks": _screening_risks(front_bootstrap),
                "legacy_admission_blockers": front_bootstrap.get("admission_blockers"),
            },
            "three_quarter_clean_snapshot": {
                "artifact_root": three_quarter_clean.get("artifact_root"),
                "top_ranked_image": three_quarter_clean.get("top_ranked_image"),
                "evidence_status": three_quarter_clean.get("evidence_status"),
                "completeness_score": three_quarter_clean.get("completeness_score"),
                "active_heavy_provider": three_quarter_clean.get("active_heavy_provider"),
                "screening_risks": _screening_risks(three_quarter_clean),
                "legacy_admission_blockers": three_quarter_clean.get("admission_blockers"),
            },
        },
        "front_bootstrap_review": {
            "promotion_ready": bool(front_sheet.get("promotion_ready")),
            "promotion_blockers": front_sheet.get("promotion_blockers") or [],
            "top_candidates": front_top3,
        },
        "winner_bank": {
            "available": bool(winner_bank_report.get("curated_bank_available")),
            "entry_count": int(winner_bank_report.get("curated_entry_count") or 0),
            "status": str(winner_bank_report.get("status") or "").strip(),
            "freeze_state": str(winner_policy.get("freeze_state") or "").strip(),
            "manual_next_step": (
                str(winner_policy.get("reason") or "").strip()
                if str(winner_policy.get("state") or "") == "deferred"
                else (
                    "record/update only mutable winner_bank memory after manual review; do not freeze or use for training/fitting"
                    if str(winner_policy.get("freeze_state") or "").strip() == "not_frozen"
                    else str(winner_bank_report.get("manual_next_step") or "").strip()
                )
            ),
        },
        "winner_bank_bootstrap_policy": winner_policy,
        "review_invariance_status": {
            "overall_status": invariance_status.get("overall_status"),
            "winner_bank_bootstrap_allowed": bool(invariance_status.get("winner_bank_bootstrap_allowed")),
            "winner_bank_freeze_allowed": bool(invariance_status.get("winner_bank_freeze_allowed")),
            "winner_bank_mutable_memory_allowed": bool(invariance_status.get("winner_bank_mutable_memory_allowed", True)),
            "parameter_fitting_allowed": bool(invariance_status.get("parameter_fitting_allowed")),
            "gates": invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {},
        },
        "pose_gait_body_truth": pose_gait_body_truth,
        "pose_gait_margin_review": pose_gait_margin_review,
        "optimization_focus": optimization_focus,
        "consistency_confidence_matrix": consistency_confidence,
        "replay_collection_plan": replay_collection,
        "training_admission": training_admission,
        "input_manifests": manifest_states,
        "next_actions": next_actions,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_file, payload)
    return payload
