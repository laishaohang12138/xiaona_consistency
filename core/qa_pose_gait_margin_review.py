from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


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


def _mean(values: Iterable[Any]) -> Optional[float]:
    numbers: List[float] = []
    for value in values:
        number = _safe_float(value)
        if number is not None:
            numbers.append(number)
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


def _dedupe(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _axis_score(row: Dict[str, Any], axis: str) -> Optional[float]:
    axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
    axis_node = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
    return _safe_float(axis_node.get("score"))


def _axis_band(row: Dict[str, Any], axis: str) -> str:
    axes = row.get("axes") if isinstance(row.get("axes"), dict) else {}
    axis_node = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
    return _safe_text(axis_node.get("band"))


def _review_category(row: Dict[str, Any]) -> str:
    read = _safe_text(row.get("body_truth_pose_gait_read"))
    body_truth = _axis_score(row, "body_truth")
    pose_gait = _axis_score(row, "pose_gait_explanation")
    clothing = _axis_score(row, "clothing_independence")
    lighting = _axis_score(row, "lighting_robustness")
    lane_pose = _axis_score(row, "lane_pose_trace")
    gaps = {_safe_text(gap) for gap in (row.get("unresolved_evidence_gaps") or []) if _safe_text(gap)}

    if read == "manual_review_required":
        return "manual_body_truth_review"
    if "lighting_warn_risk_present" in gaps or (lighting is not None and lighting < 0.65):
        return "lighting_confounded_pose_review"
    if clothing is not None and clothing < 0.74:
        return "clothing_or_occlusion_confounded_review"
    if lane_pose is not None and lane_pose < 0.70 and pose_gait is not None and pose_gait >= 0.78:
        return "pose_lane_projection_review"
    if body_truth is not None and body_truth < 0.74 and (pose_gait is None or pose_gait < 0.80):
        return "structural_margin_review"
    if read == "gait_tolerant_topology_margin_review":
        return "gait_tolerant_review"
    return "general_pose_gait_review"


def _review_focus(row: Dict[str, Any], category: str) -> List[str]:
    focus = [
        "compare 116-1 torso compactness and waist containment before calling body drift",
        "separate stance/gait projection from true body-structure change",
    ]
    if category == "lighting_confounded_pose_review":
        focus.append("check whether exposure or skin lighting changed perceived face/body continuity")
    if category == "clothing_or_occlusion_confounded_review":
        focus.append("check whether garment boundary or visible-body coverage hides the body truth")
    if category == "pose_lane_projection_review":
        focus.append("check whether yaw/stance explains lower-limb and shoulder-pelvis deltas")
    if category == "structural_margin_review":
        focus.append("prioritize shoulder-neck, torso core, waist-pelvis, and leg-axis structure")
    if category == "manual_body_truth_review":
        focus.append("treat this row as unresolved review evidence, not as automatic body drift")
    if bool(row.get("inside_close_top_cluster")):
        focus.append("compare against nearby top-ranked candidates before trusting rank order")
    return focus


def _priority(row: Dict[str, Any], category: str) -> str:
    if category in {"manual_body_truth_review", "structural_margin_review"}:
        return "P0"
    if bool(row.get("inside_close_top_cluster")):
        return "P0"
    if category in {"lighting_confounded_pose_review", "pose_lane_projection_review"}:
        return "P1"
    return "P2"


def _candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    category = _review_category(row)
    return {
        "priority": _priority(row, category),
        "rank": row.get("rank"),
        "image": _safe_text(row.get("image")),
        "review_only_status": _safe_text(row.get("review_only_status")),
        "selection_score": _round_or_none(row.get("selection_score")),
        "evidence_confidence_score": _round_or_none(row.get("evidence_confidence_score")),
        "evidence_confidence_band": _safe_text(row.get("evidence_confidence_band")),
        "inside_close_top_cluster": bool(row.get("inside_close_top_cluster")),
        "body_truth_pose_gait_read": _safe_text(row.get("body_truth_pose_gait_read")),
        "review_category": category,
        "observed_lane_family": _safe_text(row.get("observed_lane_family")),
        "body_yaw_deg": _round_or_none(row.get("body_yaw_deg")),
        "face_pose_delta_deg": _round_or_none(row.get("face_pose_delta_deg")),
        "axis_scores": {
            "body_truth": _round_or_none(_axis_score(row, "body_truth")),
            "pose_gait_explanation": _round_or_none(_axis_score(row, "pose_gait_explanation")),
            "body_topology_partition": _round_or_none(_axis_score(row, "body_topology_partition")),
            "clothing_independence": _round_or_none(_axis_score(row, "clothing_independence")),
            "lighting_robustness": _round_or_none(_axis_score(row, "lighting_robustness")),
            "lane_pose_trace": _round_or_none(_axis_score(row, "lane_pose_trace")),
            "metadata_traceability": _round_or_none(_axis_score(row, "metadata_traceability")),
        },
        "axis_bands": {
            "body_truth": _axis_band(row, "body_truth"),
            "pose_gait_explanation": _axis_band(row, "pose_gait_explanation"),
            "body_topology_partition": _axis_band(row, "body_topology_partition"),
            "clothing_independence": _axis_band(row, "clothing_independence"),
            "lighting_robustness": _axis_band(row, "lighting_robustness"),
            "lane_pose_trace": _axis_band(row, "lane_pose_trace"),
            "metadata_traceability": _axis_band(row, "metadata_traceability"),
        },
        "unresolved_evidence_gaps": _dedupe(row.get("unresolved_evidence_gaps") or []),
        "review_focus": _review_focus(row, category),
        "manual_resolution": "",
        "manual_note": "",
    }


def _sort_key(row: Dict[str, Any]) -> tuple[int, int, float, str]:
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    category_order = {
        "manual_body_truth_review": 0,
        "structural_margin_review": 1,
        "pose_lane_projection_review": 2,
        "lighting_confounded_pose_review": 3,
        "clothing_or_occlusion_confounded_review": 4,
        "gait_tolerant_review": 5,
    }
    score = _safe_float(row.get("evidence_confidence_score"))
    return (
        priority_order.get(_safe_text(row.get("priority")), 9),
        category_order.get(_safe_text(row.get("review_category")), 9),
        -(score if score is not None else 0.0),
        _safe_text(row.get("image")),
    )


def build_pose_gait_margin_review_sheet(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    matrix_file = outputs_dir / "consistency_confidence_matrix.json"
    matrix = _load_json(matrix_file)
    rows = matrix.get("candidate_matrix") if isinstance(matrix.get("candidate_matrix"), list) else []

    review_rows: List[Dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        read = _safe_text(raw_row.get("body_truth_pose_gait_read"))
        gaps = {
            _safe_text(gap)
            for gap in (raw_row.get("unresolved_evidence_gaps") or [])
            if _safe_text(gap)
        }
        if read == "pose_gait_consistent" and "BODY_TOPOLOGY_PARTITION_NOT_IN_CURRENT_PACKET" not in gaps:
            continue
        if read in {
            "gait_tolerant_topology_margin_review",
            "manual_review_required",
            "pose_explained_delta_possible",
            "pose_sensitive_noise_possible",
            "unexplained_body_drift_risk",
        }:
            review_rows.append(_candidate_row(raw_row))

    review_rows.sort(key=_sort_key)
    category_counts = Counter(_safe_text(row.get("review_category")) for row in review_rows)
    priority_counts = Counter(_safe_text(row.get("priority")) for row in review_rows)
    read_counts = Counter(_safe_text(row.get("body_truth_pose_gait_read")) for row in review_rows)
    p0_rows = [row for row in review_rows if _safe_text(row.get("priority")) == "P0"]
    axis_means = {
        axis: _round_or_none(_mean((row.get("axis_scores") or {}).get(axis) for row in review_rows))
        for axis in [
            "body_truth",
            "pose_gait_explanation",
            "body_topology_partition",
            "clothing_independence",
            "lighting_robustness",
            "lane_pose_trace",
            "metadata_traceability",
        ]
    }

    payload = {
        "schema_version": "pose_gait_margin_review_sheet_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Review queue for body-truth rows where gait, stance, projection, lighting, "
            "or clothing can explain apparent deltas. This is review routing only."
        ),
        "project_scope": {
            "role": "screening_and_evidence_only",
            "training_admission_participation": False,
            "image_set_decision_participation": False,
            "final_training_decision_owner": "external_training_decision_flow",
            "final_image_set_decision_owner": "external_dataset_curation_flow",
        },
        "truth_policy": {
            "face_truth_anchor": "A-Core_01_0deg_MASTER.png",
            "body_truth_anchor": "Task-63987060-116-1.png",
            "body_truth_policy": "pose_gait_aware_absolute_116_1",
            "do_not_call_body_drift_without_structure_review": True,
        },
        "source_files": {
            "consistency_confidence_matrix": str(matrix_file.resolve()),
        },
        "summary": {
            "matrix_available": bool(matrix),
            "candidate_count": len(rows),
            "review_row_count": len(review_rows),
            "p0_review_row_count": len(p0_rows),
            "category_counts": dict(sorted(category_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
            "body_truth_pose_gait_read_counts": dict(sorted(read_counts.items())),
            "axis_means": axis_means,
            "current_primary_limit": (
                "body_topology_partition_not_available_in_current_packet"
                if axis_means.get("body_topology_partition") is None
                else "pose_gait_review_rows_need_manual_resolution"
            ),
        },
        "p0_review_queue": p0_rows[:12],
        "review_rows": review_rows,
        "manual_resolution_options": [
            "pose_or_gait_explains_delta",
            "clothing_or_occlusion_explains_delta",
            "lighting_confounds_read",
            "structural_body_review_needed",
            "unexplained_body_drift_risk",
            "insufficient_evidence",
        ],
        "next_actions": [
            "review P0 rows before treating gait margin as body drift",
            "refresh heavy evidence after body topology partition fields are available",
            "collect side/back topology replay before extending the body-truth read to new lanes",
            "do not use this sheet for final image-set membership or training admission",
        ],
        "explicit_holds": [
            "do not fit pose/gait thresholds from these rows",
            "do not freeze winner_bank from this sheet",
            "do not create a new body truth anchor from gait-margin rows",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
