from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .qa_admission import build_batch_admission_advice, build_candidate_admission_advice
from .qa_io import atomic_write_json


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _dedupe_keep_order(values: Sequence[str], limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _top_reason_counts(items: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    ignored = {
        "FULL_PASS",
        "FRAMING_OK",
        "FEET_IN_FRAME",
        "FACE_EMBEDDING_READY",
        "FACE_LANDMARKS_READY",
    }
    counts: Dict[str, int] = {}
    for item in items:
        for reason in item.get("reasons") or []:
            if not isinstance(reason, str) or reason in ignored or reason.endswith("_READY"):
                continue
            counts[reason] = counts.get(reason, 0) + 1
    rows = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return [{"reason": reason, "count": count} for reason, count in rows[:limit]]


def _lane_family_from_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "back" in text:
        return "back"
    if "side" in text:
        return "side"
    if "3q" in text or "quarter" in text:
        return "three_quarter"
    if "front" in text or "frontal" in text:
        return "front"
    return "unknown"


def _dominant_lane_family(items: Sequence[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        debug = item.get("debug") or {}
        lane_text = debug.get("view_lane_detail") or debug.get("view_lane")
        family = _lane_family_from_text(lane_text)
        if family == "unknown":
            continue
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))[0][0]


def _build_lane_risk_focus(items: Sequence[Dict[str, Any]], limit: int = 6) -> Dict[str, Any]:
    dominant_lane = _dominant_lane_family(items)
    suppressed_by_lane = {
        "side": {
            "UPPER_FAIL",
            "FACE_LOW_CONFIDENCE",
            "FACE_LOW_CONF_NEEDS_REVIEW",
            "FACE_TOO_SMALL",
            "FACE_DARKER_THAN_TONE_ANCHOR",
            "FACE_FLIP_CANONICALIZED",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_NO_RELIABLE_SIGNAL",
        },
        "back": {
            "UPPER_FAIL",
            "FACE_LOW_CONFIDENCE",
            "FACE_LOW_CONF_NEEDS_REVIEW",
            "FACE_TOO_SMALL",
            "FACE_DARKER_THAN_TONE_ANCHOR",
            "FACE_FLIP_CANONICALIZED",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_NO_RELIABLE_SIGNAL",
        },
        "three_quarter": {
            "UPPER_FAIL",
        },
    }
    review_focus_by_lane = {
        "front": ["脸部身份", "年龄感/表情", "肩颈与上身比例"],
        "three_quarter": ["脸型漂移", "年龄感", "肩颈与体态衔接"],
        "side": ["身材真相", "侧身轮廓", "depth3d/厚度", "腿线与站姿"],
        "back": ["后背轮廓", "肩线与骨盆", "腿轴与重心", "后侧体量"],
        "unknown": ["人工复核主图", "确认 lane 是否正确", "优先看明显漂移项"],
    }
    note_by_lane = {
        "front": "当前批次按 front 解释，脸部身份和上身结构是主信号。",
        "three_quarter": "当前批次按 3Q 解释，脸型漂移和年龄感比纯正脸更重要。",
        "side": "当前批次按 side 解释，脸部弱信号只做否决参考，主看身材真相、轮廓和空间结构。",
        "back": "当前批次按 back 解释，正脸相关 reason 会被降噪，主看后背轮廓、骨盆和腿轴。",
        "unknown": "当前批次 lane 不够稳定，先确认路由再解释风险。",
    }

    suppressed_reasons = suppressed_by_lane.get(dominant_lane, set())
    ignored = {
        "FULL_PASS",
        "FRAMING_OK",
        "FEET_IN_FRAME",
        "FACE_EMBEDDING_READY",
        "FACE_LANDMARKS_READY",
    }
    active_counts: Dict[str, int] = {}
    noise_counts: Dict[str, int] = {}
    for item in items:
        for reason in item.get("reasons") or []:
            if not isinstance(reason, str) or reason in ignored or reason.endswith("_READY"):
                continue
            if reason in suppressed_reasons:
                noise_counts[reason] = noise_counts.get(reason, 0) + 1
                continue
            active_counts[reason] = active_counts.get(reason, 0) + 1

    active_rows = sorted(active_counts.items(), key=lambda row: (-row[1], row[0]))
    if not active_rows:
        for row in _top_reason_counts(items, limit=limit):
            reason = str(row.get("reason") or "").strip()
            if reason:
                active_rows.append((reason, int(row.get("count") or 0)))
    noise_rows = sorted(noise_counts.items(), key=lambda row: (-row[1], row[0]))
    return {
        "dominant_lane_family": dominant_lane,
        "note": note_by_lane.get(dominant_lane, note_by_lane["unknown"]),
        "review_focus": review_focus_by_lane.get(dominant_lane, review_focus_by_lane["unknown"]),
        "primary_risks": [
            {"reason": reason, "count": count}
            for reason, count in active_rows[:limit]
        ],
        "suppressed_noise": [
            {"reason": reason, "count": count}
            for reason, count in noise_rows[: min(4, limit)]
        ],
        "suppressed_reason_codes": sorted(suppressed_reasons),
    }


def _status_counts(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _module_status_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    modules = ("face", "upper", "full")
    result: Dict[str, Dict[str, int]] = {}
    for module_name in modules:
        counts: Dict[str, int] = {}
        for item in items:
            state = str((item.get("module_state") or {}).get(module_name) or "").strip()
            if not state:
                continue
            counts[state] = counts.get(state, 0) + 1
        result[module_name] = dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))
    return result


def _lane_detail_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        debug = item.get("debug") or {}
        lane_detail = str(debug.get("view_lane_detail") or debug.get("view_lane") or "").strip()
        if not lane_detail:
            continue
        counts[lane_detail] = counts.get(lane_detail, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _candidate_lookup(shot_selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for group in shot_selection.get("groups") or []:
        for row in group.get("candidates") or []:
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            if record_key:
                lookup[record_key] = row
    return lookup


def _drift_flag_lookup(winner_bank_report: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    report = winner_bank_report or {}
    for row in report.get("drift_rows") or []:
        record_key = str(row.get("record_key") or row.get("image") or "").strip()
        if record_key:
            lookup[record_key] = list(row.get("drift_flags") or [])
    return lookup


def _load_json_if_exists(path_str: Any) -> Optional[Dict[str, Any]]:
    try:
        path = Path(str(path_str))
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        cleaned[key] = value
    return cleaned


def _compact_selection_comparison(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    production_method = str(
        data.get("production_method") or data.get("preferred_method") or ""
    ).strip()
    legacy_retired = bool(data.get("legacy_retired_for_production")) or production_method == "review_only_score_v2"
    comparison_mode = str(data.get("comparison_mode") or "").strip()
    if legacy_retired and comparison_mode == "unlabeled_truth_proxy":
        comparison_mode = "legacy_monitor_only"
    compact = {
        "comparison_mode": comparison_mode,
        "production_method": production_method,
        "legacy_retired_for_production": legacy_retired,
        "legacy_monitor_preferred_method": str(data.get("legacy_monitor_preferred_method") or "").strip(),
        "truth_proxy_coverage": _round_or_none(data.get("truth_proxy_coverage")),
        "decision_reasons": list(data.get("decision_reasons") or [])[:4],
    }
    return _clean_dict(compact)


def _compact_release_gate(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return _clean_dict(
        {
            "target_bucket": str(data.get("target_bucket") or "").strip(),
            "release_state": str(data.get("release_state") or "").strip(),
            "machine_status_ceiling": str(data.get("machine_status_ceiling") or "").strip(),
            "training_admission_allowed": bool(data.get("training_admission_allowed")),
            "required_lane_families": list(data.get("required_lane_families") or []),
            "notes": str(data.get("notes") or "").strip(),
        }
    )


def _compact_admission_advice(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    suggested_action = str(
        data.get("suggested_action") or data.get("suggestion") or ""
    ).strip()
    return _clean_dict(
        {
            "target_bucket": str(data.get("target_bucket") or "").strip(),
            "suggested_action": suggested_action,
            "machine_ceiling": str(data.get("machine_ceiling") or "").strip(),
            "eligible_for_training_seal": data.get("eligible_for_training_seal"),
            "blockers": list(data.get("blockers") or [])[:6],
            "supports": list(data.get("supports") or [])[:4],
        }
    )


def _strip_legacy_review_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(row)
    for key in (
        "selection_score_legacy",
        "legacy_review_only_status",
        "legacy_review_only_confidence",
        "rank_legacy",
        "review_bucket_legacy",
    ):
        cleaned.pop(key, None)
    delta_vs_top = cleaned.get("delta_vs_top")
    if isinstance(delta_vs_top, dict):
        delta_cleaned = dict(delta_vs_top)
        delta_cleaned.pop("selection_score_legacy", None)
        cleaned["delta_vs_top"] = delta_cleaned
    return cleaned


def _compact_heavy_review_summary(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    compact = {
        "provider_name": str(data.get("provider_name") or "").strip(),
        "provider_version": str(data.get("provider_version") or "").strip(),
        "cache_state": str(data.get("cache_state") or "").strip(),
        "coverage": _round_or_none(data.get("coverage")),
        "parser_confidence": _round_or_none(data.get("parser_confidence")),
        "parser_consensus_score": _round_or_none(data.get("parser_consensus_score")),
        "enhanced_selection_score": _round_or_none(data.get("enhanced_selection_score")),
        "rank_in_heavy_review": data.get("rank_in_heavy_review"),
    }
    return _clean_dict(compact)


def _pose_gait_body_truth_read(
    *,
    body_truth_alignment: Optional[float],
    body_topology_alignment: Optional[float],
    pose_sensitive_measurement: Optional[float],
    pose_measurement_gap: Optional[float],
) -> str:
    if body_truth_alignment is None and body_topology_alignment is None:
        return "unavailable"
    truth_signal = body_truth_alignment
    if truth_signal is None:
        truth_signal = body_topology_alignment
    topology_signal = body_topology_alignment
    if topology_signal is None:
        topology_signal = body_truth_alignment
    if (
        truth_signal is not None
        and float(truth_signal) >= 0.64
        and pose_sensitive_measurement is not None
        and float(pose_sensitive_measurement) < 0.46
    ):
        return "pose_sensitive_noise_possible"
    if (
        truth_signal is not None
        and float(truth_signal) >= 0.62
        and pose_measurement_gap is not None
        and float(pose_measurement_gap) > 0.10
    ):
        return "pose_explained_delta_possible"
    if (
        truth_signal is not None
        and topology_signal is not None
        and float(truth_signal) < 0.56
        and float(topology_signal) < 0.56
    ):
        return "unexplained_body_drift_risk"
    if (
        truth_signal is not None
        and topology_signal is not None
        and float(truth_signal) >= 0.74
        and 0.66 <= float(topology_signal) < 0.72
        and pose_sensitive_measurement is not None
        and float(pose_sensitive_measurement) >= 0.78
        and pose_measurement_gap is not None
        and abs(float(pose_measurement_gap)) <= 0.04
    ):
        return "gait_tolerant_topology_margin_review"
    if (
        truth_signal is not None
        and topology_signal is not None
        and float(truth_signal) >= 0.72
        and float(topology_signal) >= 0.72
    ):
        return "pose_gait_consistent"
    return "manual_review_required"


def _extract_canonical_truth_summary(heavy_evidence: Any) -> Dict[str, Any]:
    bundle = heavy_evidence if isinstance(heavy_evidence, dict) else {}
    metrics = list(bundle.get("metrics") or [])
    summary = dict(bundle.get("summary") or {}) if isinstance(bundle.get("summary"), dict) else {}
    component_rows = list(summary.get("component_providers") or [])
    body_canonical_summary = (
        dict(summary.get("body_canonical_summary") or {})
        if isinstance(summary.get("body_canonical_summary"), dict)
        else {}
    )

    def _metric_value(metric_name: str) -> Optional[float]:
        for row in metrics:
            if not isinstance(row, dict):
                continue
            if str(row.get("metric_name") or "").strip() != metric_name:
                continue
            value = row.get("metric_value")
            try:
                return float(value)
            except Exception:
                return None
        return None

    body_canonical_component = None
    for row in component_rows:
        if not isinstance(row, dict):
            continue
        provider_name = str(row.get("provider_name") or "").strip()
        component_key = str(row.get("component_key") or "").strip()
        if provider_name == "body_canonical_hmr2" or component_key == "body_canonical":
            body_canonical_component = dict(row)
            break

    available = None
    if isinstance(body_canonical_component, dict):
        available = bool(body_canonical_component.get("available"))
    elif str(bundle.get("provider_name") or "").strip() == "body_canonical_hmr2":
        available = bool(bundle.get("available")) if bundle.get("available") is not None else None
        body_canonical_component = {
            "provider_name": "body_canonical_hmr2",
            "available": available,
            "confidence": bundle.get("confidence"),
            "coverage": bundle.get("coverage"),
            "device": bundle.get("device"),
        }
        body_canonical_summary = summary

    body_pose_independent_truth_alignment = _round_or_none(_metric_value("body_pose_independent_truth_alignment"))
    body_shape_truth_alignment = _round_or_none(_metric_value("body_shape_truth_alignment"))
    body_gait_tolerant_topology_similarity = _round_or_none(_metric_value("body_gait_tolerant_topology_similarity"))
    if body_gait_tolerant_topology_similarity is None:
        body_gait_tolerant_topology_similarity = _round_or_none(_metric_value("body_topology_signature_similarity"))
    body_pose_sensitive_measurement_similarity = _round_or_none(
        _metric_value("body_pose_sensitive_measurement_similarity")
    )
    body_pose_measurement_gap = _round_or_none(body_canonical_summary.get("body_pose_measurement_gap"))
    body_topology_partition = (
        dict(body_canonical_summary.get("body_topology_partition") or {})
        if isinstance(body_canonical_summary.get("body_topology_partition"), dict)
        else {}
    )
    body_topology_partition_mean_similarity = _round_or_none(
        _metric_value("body_topology_partition_mean_similarity")
    )
    if body_topology_partition_mean_similarity is None:
        partition_mean_raw = body_canonical_summary.get("body_topology_partition_mean_similarity")
        if partition_mean_raw is None:
            partition_mean_raw = body_topology_partition.get("mean_similarity")
        body_topology_partition_mean_similarity = _round_or_none(
            partition_mean_raw
        )
    body_topology_weakest_part_similarity = _round_or_none(
        _metric_value("body_topology_weakest_part_similarity")
    )
    if body_topology_weakest_part_similarity is None:
        weakest_part_raw = body_canonical_summary.get("body_topology_weakest_part_similarity")
        if weakest_part_raw is None:
            weakest_part_raw = body_topology_partition.get("weakest_part_similarity")
        body_topology_weakest_part_similarity = _round_or_none(
            weakest_part_raw
        )
    body_topology_weakest_part = str(
        body_canonical_summary.get("body_topology_weakest_part")
        or body_topology_partition.get("weakest_part")
        or ""
    ).strip()

    def _body_partition_metric(metric_name: str, summary_key: str, partition_key: str) -> Optional[float]:
        value = _round_or_none(_metric_value(metric_name))
        if value is None:
            raw_value = body_canonical_summary.get(summary_key)
            if raw_value is None:
                raw_value = body_topology_partition.get(partition_key)
            value = _round_or_none(raw_value)
        return value

    return {
        "canonical_truth_available": available,
        "body_canonical_provider_state": body_canonical_component or {},
        "body_truth_policy": "pose_gait_aware_absolute_116_1",
        "body_truth_pose_gait_read": _pose_gait_body_truth_read(
            body_truth_alignment=(
                body_pose_independent_truth_alignment
                if body_pose_independent_truth_alignment is not None
                else body_shape_truth_alignment
            ),
            body_topology_alignment=body_gait_tolerant_topology_similarity,
            pose_sensitive_measurement=body_pose_sensitive_measurement_similarity,
            pose_measurement_gap=body_pose_measurement_gap,
        ),
        "body_pose_independent_truth_alignment": body_pose_independent_truth_alignment,
        "body_shape_truth_alignment": body_shape_truth_alignment,
        "body_shape_beta_similarity": _round_or_none(_metric_value("body_shape_beta_similarity")),
        "body_gait_tolerant_topology_similarity": body_gait_tolerant_topology_similarity,
        "body_topology_signature_similarity": _round_or_none(_metric_value("body_topology_signature_similarity")),
        "body_core_measurement_similarity": _round_or_none(_metric_value("body_core_measurement_similarity")),
        "canonical_measurement_similarity": _round_or_none(_metric_value("canonical_measurement_similarity")),
        "body_pose_sensitive_measurement_similarity": body_pose_sensitive_measurement_similarity,
        "body_pose_delta_similarity": _round_or_none(_metric_value("body_pose_delta_similarity")),
        "body_mesh_fit_confidence": _round_or_none(_metric_value("body_mesh_fit_confidence")),
        "body_pose_measurement_gap": body_pose_measurement_gap,
        "body_topology_partition": body_topology_partition,
        "body_topology_partition_mean_similarity": body_topology_partition_mean_similarity,
        "body_topology_weakest_part": body_topology_weakest_part,
        "body_topology_weakest_part_similarity": body_topology_weakest_part_similarity,
        "body_topology_torso_core_similarity": _body_partition_metric(
            "body_topology_torso_core_similarity",
            "body_topology_torso_core_similarity",
            "torso_core_similarity",
        ),
        "body_topology_shoulder_neck_frame_similarity": _body_partition_metric(
            "body_topology_shoulder_neck_frame_similarity",
            "body_topology_shoulder_neck_frame_similarity",
            "shoulder_neck_frame_similarity",
        ),
        "body_topology_waist_pelvis_similarity": _body_partition_metric(
            "body_topology_waist_pelvis_similarity",
            "body_topology_waist_pelvis_similarity",
            "waist_pelvis_similarity",
        ),
        "body_topology_leg_axis_similarity": _body_partition_metric(
            "body_topology_leg_axis_similarity",
            "body_topology_leg_axis_similarity",
            "leg_axis_similarity",
        ),
        "body_topology_lower_body_volume_similarity": _body_partition_metric(
            "body_topology_lower_body_volume_similarity",
            "body_topology_lower_body_volume_similarity",
            "lower_body_volume_similarity",
        ),
        "body_topology_gait_phase_similarity": _body_partition_metric(
            "body_topology_gait_phase_similarity",
            "body_topology_gait_phase_similarity",
            "gait_phase_similarity",
        ),
        "body_pose_explained_delta_score": _body_partition_metric(
            "body_pose_explained_delta_score",
            "body_pose_explained_delta_score",
            "pose_explained_delta_score",
        ),
        "body_core_measurement_coverage": _round_or_none(body_canonical_summary.get("body_core_measurement_coverage")),
        "body_pose_sensitive_measurement_coverage": _round_or_none(
            body_canonical_summary.get("body_pose_sensitive_measurement_coverage")
        ),
    }


def _extract_face_canonical_summary(debug: Any) -> Dict[str, Any]:
    payload = debug if isinstance(debug, dict) else {}
    shadow = payload.get("face_canonical_shadow") if isinstance(payload.get("face_canonical_shadow"), dict) else {}
    return {
        "available": bool(shadow.get("available")) if shadow else None,
        "provider_name": str(shadow.get("provider_name") or "") if shadow else "",
        "mode": str(shadow.get("mode") or "") if shadow else "",
        "face_pose_normalization_confidence": _round_or_none(shadow.get("face_pose_normalization_confidence")),
        "visible_face_coverage": _round_or_none(shadow.get("visible_face_coverage")),
        "frontalization_quality": _round_or_none(shadow.get("frontalization_quality")),
        "pose_fit_confidence": _round_or_none(shadow.get("pose_fit_confidence")),
        "canonical_face_landmark_similarity": _round_or_none(shadow.get("canonical_face_landmark_similarity")),
        "canonical_face_identity_similarity": _round_or_none(shadow.get("canonical_face_identity_similarity")),
        "canonical_face_topology_similarity": _round_or_none(shadow.get("canonical_face_topology_similarity")),
        "canonical_face_topology_delta": _round_or_none(shadow.get("canonical_face_topology_delta")),
        "head_topology_partition": dict(shadow.get("head_topology_partition") or {}),
        "head_topology_mean_similarity": _round_or_none(shadow.get("head_topology_mean_similarity")),
        "head_topology_weakest_part": str(shadow.get("head_topology_weakest_part") or ""),
        "head_topology_weakest_part_similarity": _round_or_none(
            shadow.get("head_topology_weakest_part_similarity")
        ),
        "head_topology_upper_face_similarity": _round_or_none(
            shadow.get("head_topology_upper_face_similarity")
        ),
        "head_topology_mid_face_similarity": _round_or_none(
            shadow.get("head_topology_mid_face_similarity")
        ),
        "head_topology_lower_face_similarity": _round_or_none(
            shadow.get("head_topology_lower_face_similarity")
        ),
        "head_topology_contour_similarity": _round_or_none(
            shadow.get("head_topology_contour_similarity")
        ),
        "head_topology_center_axis_similarity": _round_or_none(
            shadow.get("head_topology_center_axis_similarity")
        ),
        "head_topology_lateral_balance_similarity": _round_or_none(
            shadow.get("head_topology_lateral_balance_similarity")
        ),
        "pose_delta_similarity": _round_or_none(shadow.get("pose_delta_similarity")),
        "pose_delta_deg": _round_or_none(shadow.get("pose_delta_deg")),
        "guidance": list(shadow.get("guidance") or [])[:4] if shadow else [],
        "reasons": list(shadow.get("reasons") or [])[:6] if shadow else [],
    }


def _topology_review_focus(
    lane: Dict[str, Any],
    face_summary: Dict[str, Any],
    truth_summary: Dict[str, Any],
    breakdown: Dict[str, Any],
) -> Dict[str, List[str]]:
    lane_detail = str((lane or {}).get("view_lane_detail") or (lane or {}).get("view_lane") or "").strip().lower()
    manual_focus: List[str] = []
    prompts: List[str] = []

    face_topology = _round_or_none((face_summary or {}).get("canonical_face_topology_similarity"))
    head_topology_weakest = _round_or_none((face_summary or {}).get("head_topology_weakest_part_similarity"))
    head_topology_part = str((face_summary or {}).get("head_topology_weakest_part") or "").strip()
    body_topology_metric = _round_or_none((truth_summary or {}).get("body_gait_tolerant_topology_similarity"))
    if body_topology_metric is None:
        body_topology_metric = _round_or_none((truth_summary or {}).get("body_topology_signature_similarity"))
    body_truth_metric = _round_or_none((truth_summary or {}).get("body_pose_independent_truth_alignment"))
    if body_truth_metric is None:
        body_truth_metric = _round_or_none((truth_summary or {}).get("body_shape_truth_alignment"))
    body_measurement_metric = _round_or_none((truth_summary or {}).get("body_core_measurement_similarity"))
    if body_measurement_metric is None:
        body_measurement_metric = _round_or_none((truth_summary or {}).get("canonical_measurement_similarity"))
    body_pose_sensitive_metric = _round_or_none((truth_summary or {}).get("body_pose_sensitive_measurement_similarity"))
    body_pose_measurement_gap = _round_or_none((truth_summary or {}).get("body_pose_measurement_gap"))
    body_topology_partition_mean = _round_or_none((truth_summary or {}).get("body_topology_partition_mean_similarity"))
    body_topology_weakest_part = str((truth_summary or {}).get("body_topology_weakest_part") or "").strip()
    body_topology_weakest_part_similarity = _round_or_none(
        (truth_summary or {}).get("body_topology_weakest_part_similarity")
    )
    body_pose_explained_delta = _round_or_none((truth_summary or {}).get("body_pose_explained_delta_score"))
    body_topology_support = _round_or_none((breakdown or {}).get("body_topology_support"))
    angle_tolerance_score = _round_or_none((breakdown or {}).get("angle_tolerance_score"))
    body_angle_delta_deg = _round_or_none((breakdown or {}).get("body_angle_delta_deg"))
    face_angle_delta_deg = _round_or_none((breakdown or {}).get("face_angle_delta_deg"))
    projection_confidence = _round_or_none((breakdown or {}).get("same_truth_projection_confidence"))
    projection_uncertainty = _round_or_none((breakdown or {}).get("same_truth_projection_uncertainty"))
    projection_mode = str((breakdown or {}).get("same_truth_projection_mode") or "").strip()
    clothing_invariant_score = _round_or_none((breakdown or {}).get("clothing_invariant_score"))
    clothing_invariant_confidence = _round_or_none((breakdown or {}).get("clothing_invariant_confidence"))
    garment_occlusion_index = _round_or_none((breakdown or {}).get("garment_occlusion_index"))
    garment_boundary_risk = _round_or_none((breakdown or {}).get("garment_boundary_risk"))
    pose_gait_read = str((truth_summary or {}).get("body_truth_pose_gait_read") or "").strip()

    if face_topology is not None:
        face_floor = 0.64 if "side" in lane_detail else 0.70 if "three_quarter" in lane_detail else 0.74
        if float(face_topology) < face_floor:
            manual_focus.append("check nose bridge, lip-chin contour, and jawline continuity against A-Core_01")
            prompts.append("Face topology is weak. Judge the same head structure before trusting frontal resemblance.")
    if head_topology_weakest is not None:
        head_part_floor = 0.68 if "side" in lane_detail else 0.72 if "three_quarter" in lane_detail else 0.76
        if float(head_topology_weakest) < head_part_floor:
            part_text = head_topology_part or "weakest head partition"
            manual_focus.append(f"check {part_text}: this is the weakest canonical head topology partition")
            prompts.append("Head topology has a regional weak spot. Verify that this is pose/frontalization noise before calling the identity stable.")

    if body_topology_metric is not None or body_topology_support is not None:
        body_signal = body_topology_support if body_topology_support is not None else body_topology_metric
        body_floor = 0.68 if "side" in lane_detail else 0.72
        if body_signal is not None and float(body_signal) < body_floor:
            manual_focus.append("check shoulder-hip span, leg-to-torso ratio, and lower-body volume against 116-1")
            prompts.append("Body topology is weak. Compare 116-1 shape structure first, then treat pose/gait as secondary.")
    if body_topology_weakest_part_similarity is not None:
        body_part_floor = 0.60 if ("side" in lane_detail or "back" in lane_detail) else 0.64 if "three_quarter" in lane_detail else 0.68
        if float(body_topology_weakest_part_similarity) < body_part_floor:
            part_text = body_topology_weakest_part or "weakest body topology partition"
            manual_focus.append(f"check {part_text}: weakest body topology partition against 116-1")
            if (
                body_pose_explained_delta is not None
                and float(body_pose_explained_delta) >= 0.70
                and body_topology_weakest_part in {"leg_axis", "lower_body_volume"}
            ):
                prompts.append("Weak body partition may be gait or stance projection. Verify trunk and shoulder-hip structure before calling drift.")
            else:
                prompts.append("Body topology has a regional weak spot. Confirm whether this is structural drift or pose projection.")
    if (
        body_topology_partition_mean is not None
        and float(body_topology_partition_mean) >= 0.68
        and body_pose_explained_delta is not None
        and float(body_pose_explained_delta) >= 0.70
        and body_pose_sensitive_metric is not None
        and float(body_pose_sensitive_metric) < 0.50
    ):
        manual_focus.append("pose/gait explains much of the lower-body delta; keep review centered on stable structural partitions")
    if (
        body_pose_sensitive_metric is not None
        and float(body_pose_sensitive_metric) < 0.46
        and body_truth_metric is not None
        and float(body_truth_metric) >= 0.64
    ):
        manual_focus.append("allow gait asymmetry if trunk, shoulder, hip, and leg volume still match 116-1")
        prompts.append("Pose-sensitive leg balance is noisy here. Keep the decision on body structure, not left-right gait symmetry.")
    if (
        body_pose_measurement_gap is not None
        and float(body_pose_measurement_gap) > 0.10
        and body_measurement_metric is not None
        and float(body_measurement_metric) >= 0.62
    ):
        manual_focus.append("treat lower-limb asymmetry as pose noise when core body measurements still hold")
    if pose_gait_read == "gait_tolerant_topology_margin_review":
        manual_focus.append("review gait/topology margin: core body truth holds, but topology is just below pass band")
        prompts.append("Gait-tolerant topology is near the margin. Do not call body drift unless trunk, shoulder-hip span, or leg volume truly changes.")
    if angle_tolerance_score is not None and float(angle_tolerance_score) < 0.60:
        manual_focus.append("treat exact angle as noisy and decide by topology before pose neatness")
        prompts.append("Angle variation is noisy. Keep identity judgment centered on canonical topology, not exact pose bucket purity.")
    if body_angle_delta_deg is not None and float(body_angle_delta_deg) > 24.0:
        manual_focus.append("compare body topology first if this frame sits between lane centers")
    if face_angle_delta_deg is not None and float(face_angle_delta_deg) > 18.0:
        manual_focus.append("do not over-penalize face score when pose is between standard yaw buckets")
    if projection_uncertainty is not None and float(projection_uncertainty) > 0.50:
        manual_focus.append("review same-truth derived projection uncertainty before treating side/back as stable")
        prompts.append("Derived projection uncertainty is high. Use it as review priority evidence, not as a new truth anchor.")
    if projection_confidence is not None and float(projection_confidence) < 0.52 and projection_mode:
        manual_focus.append(f"check {projection_mode} against canonical face/body truth before ranking this candidate high")
    if clothing_invariant_score is not None and float(clothing_invariant_score) < 0.68:
        manual_focus.append("verify identity under clothing using body topology before trusting garment shape")
        prompts.append("Clothing-invariant support is weak. Treat garment silhouette as a risk, not as identity evidence.")
    if clothing_invariant_confidence is not None and float(clothing_invariant_confidence) < 0.56:
        manual_focus.append("check whether clothing hides waist, hip, shoulder, or leg evidence")
    if garment_occlusion_index is not None and float(garment_occlusion_index) > 0.72:
        manual_focus.append("high garment occlusion: do not use this as training proof without manual confirmation")
        prompts.append("Garment occlusion is high. Keep this in review unless face/body topology still clearly supports both absolute truths.")
    if garment_boundary_risk is not None and float(garment_boundary_risk) > 0.42:
        manual_focus.append("check whether garment boundary is inventing a new body silhouette")

    return {
        "manual_focus": _dedupe_keep_order(manual_focus, limit=4),
        "manual_review_prompts": _dedupe_keep_order(prompts, limit=3),
    }


def _primary_group(shot_selection: Dict[str, Any]) -> Dict[str, Any]:
    groups = list(shot_selection.get("groups") or [])
    if len(groups) == 0:
        return {}
    return groups[0] if isinstance(groups[0], dict) else {}


def _build_batch_summary(report_payload: Dict[str, Any]) -> Dict[str, Any]:
    report_meta = report_payload.get("report_meta") or {}
    collection_aggregates = report_payload.get("collection_aggregates") or {}
    shot_selection = report_payload.get("shot_selection") or {}
    items = list(report_payload.get("items") or [])
    primary_group = _primary_group(shot_selection)
    collection_summary = collection_aggregates.get("summary") or {}
    batch_gate = collection_aggregates.get("batch_gate") or {}
    batch_reference = primary_group.get("batch_reference") or {}

    return {
        "target_profile": report_meta.get("active_profile"),
        "run_status": report_meta.get("run_status"),
        "input_count": report_meta.get("input_count"),
        "engine_status": report_meta.get("engine_status") or {},
        "anchor_truth": report_meta.get("anchor_governance") or {},
        "master_truth_reference": report_meta.get("master_truth_reference") or {},
        "master_truth_artifact_dir": report_meta.get("master_truth_artifact_dir"),
        "artifact_manifest_file": report_meta.get("artifact_manifest_file"),
        "artifact_manifest_summary": report_meta.get("artifact_manifest_summary") or {},
        "heavy_provider_status": report_meta.get("heavy_provider_status") or {},
        "view_classifier_status": report_meta.get("view_classifier_status") or {},
        "face_canonical_status": report_meta.get("face_canonical_status") or {},
        "release_gate": report_meta.get("release_gate") or {},
        "training_admission_governance": report_meta.get("training_admission_governance") or {},
        "batch_preflight": report_meta.get("batch_preflight") or {},
        "evidence_completeness": report_meta.get("evidence_completeness") or {},
        "heavy_evidence_summary": shot_selection.get("heavy_evidence_summary") or {},
        "canonical_truth_summary": _extract_canonical_truth_summary(shot_selection.get("heavy_evidence_summary") or {}),
        "group_count": shot_selection.get("group_count"),
        "status_counts": _status_counts(items, "status"),
        "module_status_counts": _module_status_counts(items),
        "lane_detail_counts": _lane_detail_counts(items),
        "primary_risks": _top_reason_counts(items),
        "lane_risk_focus": _build_lane_risk_focus(items),
        "batch_gate": {
            "enabled": batch_gate.get("enabled"),
            "applied": batch_gate.get("applied"),
            "status": batch_gate.get("status"),
            "reasons": list(batch_gate.get("reasons") or []),
            "recommendation": batch_gate.get("recommendation"),
        },
        "selection": {
            "selection_method": primary_group.get("selection_method") or shot_selection.get("selection_method"),
            "top_ranked_image": primary_group.get("top_ranked_image"),
            "selection_gap_top2": primary_group.get("selection_gap_top2"),
            "manual_review_window": primary_group.get("manual_review_window"),
            "shortlist_size": primary_group.get("shortlist_size"),
            "selection_comparison": _compact_selection_comparison(
                primary_group.get("selection_comparison") or shot_selection.get("review_only_score_v2_summary") or {}
            ),
        },
        "identity_summary": {
            "identity_continuity": batch_reference.get("identity_continuity"),
            "batch_identity_cohesion": batch_reference.get("batch_identity_cohesion"),
            "batch_clothfree_identity_cohesion": batch_reference.get("batch_clothfree_identity_cohesion"),
            "batch_hybrid_identity_cohesion": batch_reference.get("batch_hybrid_identity_cohesion"),
        },
        "geometry_summary": {
            "body_under_clothes_continuity": batch_reference.get("body_under_clothes_continuity"),
            "batch_3d_cohesion": batch_reference.get("batch_3d_cohesion"),
            "batch_world3d_cohesion": batch_reference.get("batch_world3d_cohesion"),
            "routing_consistency": batch_reference.get("routing_consistency"),
        },
        "garment_summary": {
            "garment_boundary_stability": batch_reference.get("garment_boundary_stability"),
            "groupable_items": collection_summary.get("groupable_items"),
            "look_groups": collection_summary.get("look_groups"),
            "layer_groups": collection_summary.get("layer_groups"),
        },
        "review_guidance": list(primary_group.get("review_guidance") or []),
    }


def _build_ranked_review_packet(
    shot_selection: Dict[str, Any],
    item_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups_out: List[Dict[str, Any]] = []
    for group in shot_selection.get("groups") or []:
        shortlist_rows: List[Dict[str, Any]] = []
        for row in group.get("shortlist") or []:
            enriched = _strip_legacy_review_fields(dict(row))
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            item_row = item_lookup.get(record_key) or {}
            enriched["heavy_review"] = _compact_heavy_review_summary(enriched.get("heavy_review") or {})
            enriched.pop("heavy_evidence", None)
            enriched["canonical_truth_summary"] = _extract_canonical_truth_summary(
                row.get("heavy_evidence") or (item_row.get("canonical_truth_summary") if isinstance(item_row, dict) else {})
            )
            if item_row:
                enriched["master_consistency_card"] = item_row.get("master_consistency_card")
                enriched["admission_advice"] = item_row.get("admission_advice")
                enriched["face_canonical_summary"] = item_row.get("face_canonical_summary")
                enriched["review_focus"] = item_row.get("review_focus")
            shortlist_rows.append(enriched)
        pairwise_rows: List[Dict[str, Any]] = []
        for card in group.get("pairwise_compare_cards") or []:
            enriched_card = dict(card)
            top_row = item_lookup.get(str(card.get("top_image") or "").strip())
            cand_row = item_lookup.get(str(card.get("candidate_image") or "").strip())
            top_master = (top_row or {}).get("master_consistency_card") or {}
            cand_master = (cand_row or {}).get("master_consistency_card") or {}
            if top_row:
                enriched_card["top_master_consistency"] = top_master
            if cand_row:
                enriched_card["candidate_master_consistency"] = cand_master
                enriched_card["candidate_admission_advice"] = cand_row.get("admission_advice")
            enriched_card["combined_manual_focus"] = _dedupe_keep_order(
                list(card.get("manual_focus") or [])
                + list(top_master.get("manual_focus") or [])
                + list(cand_master.get("manual_focus") or []),
                limit=8,
            )
            enriched_card["manual_review_prompts"] = _dedupe_keep_order(
                list(top_master.get("manual_review_prompts") or [])
                + list(cand_master.get("manual_review_prompts") or []),
                limit=6,
            )
            pairwise_rows.append(enriched_card)
        groups_out.append(
            {
                "group_key": group.get("group_key"),
                "group_source": group.get("group_source"),
                "layer_tag": group.get("layer_tag"),
                "look_key": group.get("look_key"),
                "image_count": group.get("image_count"),
                "top_ranked_image": group.get("top_ranked_image"),
                "selection_method": group.get("selection_method") or shot_selection.get("selection_method"),
                "selection_gap_top2": group.get("selection_gap_top2"),
                "manual_review_window": group.get("manual_review_window"),
                "shortlist_size": group.get("shortlist_size"),
                "review_guidance": list(group.get("review_guidance") or []),
                "selection_comparison": _compact_selection_comparison(group.get("selection_comparison") or {}),
                "batch_reference": dict(group.get("batch_reference") or {}),
                "shortlist": shortlist_rows,
                "pairwise_compare_cards": pairwise_rows,
            }
        )
    top_candidates: List[Dict[str, Any]] = []
    if groups_out:
        primary_group = groups_out[0] if isinstance(groups_out[0], dict) else {}
        for row in list(primary_group.get("shortlist") or [])[:3]:
            if isinstance(row, dict):
                top_candidates.append(dict(row))
    return {
        "mode": shot_selection.get("mode"),
        "final_decision_owner": shot_selection.get("final_decision_owner"),
        "target_profile": shot_selection.get("target_profile"),
        "group_count": shot_selection.get("group_count"),
        "top_candidates": top_candidates,
        "groups": groups_out,
    }


def _summarize_item(
    item: Dict[str, Any],
    candidate_row: Optional[Dict[str, Any]],
    batch_admission: Dict[str, Any],
    drift_flags: Optional[Sequence[str]],
) -> Dict[str, Any]:
    debug = item.get("debug") or {}
    diagnostics = debug.get("collection_diagnostics") or {}
    garment = debug.get("garment_metrics") or {}
    depth_metrics = debug.get("depth_3d_metrics") or {}
    master_consistency_card = debug.get("master_consistency_card") or {}
    heavy_evidence = debug.get("heavy_evidence") or {}
    face_summary = _extract_face_canonical_summary(debug)
    truth_summary = _extract_canonical_truth_summary(heavy_evidence)
    breakdown = (candidate_row or {}).get("review_only_breakdown_v2") or {}
    topology_focus = _topology_review_focus(
        {
            "view_lane": debug.get("view_lane"),
            "view_lane_detail": debug.get("view_lane_detail"),
        },
        face_summary,
        truth_summary,
        breakdown,
    )
    record_key = (item.get("collection") or {}).get("input_relative_path") or item.get("image")
    row = {
        "image": item.get("image"),
        "record_key": record_key,
        "status": item.get("status"),
        "task_profile": item.get("task_profile"),
        "rank": (candidate_row or {}).get("rank"),
        "review_bucket": (candidate_row or {}).get("review_bucket"),
        "selection_method": (candidate_row or {}).get("selection_score_method"),
        "selection_score": (candidate_row or {}).get("selection_score"),
        "review_only_score_v2": (candidate_row or {}).get("review_only_score_v2"),
        "review_only_confidence_v2": (candidate_row or {}).get("review_only_confidence_v2"),
        "review_only_status_v2": (candidate_row or {}).get("review_only_status_v2"),
        "review_only_breakdown_v2": (candidate_row or {}).get("review_only_breakdown_v2"),
        "review_only_hard_vetoes_v2": list((candidate_row or {}).get("review_only_hard_vetoes_v2") or []),
        "review_only_soft_flags_v2": list((candidate_row or {}).get("review_only_soft_flags_v2") or []),
        "review_only_policy_note_v2": (candidate_row or {}).get("review_only_policy_note_v2"),
        "why_not_high_confidence_v2": list((candidate_row or {}).get("why_not_high_confidence_v2") or []),
        "delta_vs_top": (_strip_legacy_review_fields({"delta_vs_top": (candidate_row or {}).get("delta_vs_top")}).get("delta_vs_top")),
        "lane": {
            "intended_view": ((item.get("collection") or {}).get("view_expected")),
            "intended_lane_family": ((item.get("collection") or {}).get("view_expected_family")),
            "prompt_intent_is_weak_prior": ((item.get("collection") or {}).get("view_expected_is_weak_prior")),
            "view_lane": debug.get("view_lane"),
            "view_lane_detail": debug.get("view_lane_detail"),
            "observed_lane_family": (breakdown or {}).get("observed_lane_family"),
            "view_lane_detail_confidence": debug.get("view_lane_detail_confidence"),
            "view_lane_strictness_score": debug.get("view_lane_strictness_score"),
            "body_yaw_deg": ((debug.get("view_router_v2") or {}).get("body_yaw_deg")),
            "observed_lane_center_distance_deg": (breakdown or {}).get("observed_lane_center_distance_deg"),
            "observed_lane_source": (breakdown or {}).get("observed_lane_source"),
            "shadow_classifier": {
                "provider_name": ((debug.get("view_classifier_shadow") or {}).get("provider_name")),
                "lane": ((debug.get("view_classifier_shadow") or {}).get("lane")),
                "lane_detail": ((debug.get("view_classifier_shadow") or {}).get("lane_detail")),
                "confidence": ((debug.get("view_classifier_shadow") or {}).get("confidence")),
                "decision_margin": ((debug.get("view_classifier_shadow") or {}).get("decision_margin")),
                "disagrees_with_primary": debug.get("view_classifier_shadow_disagrees"),
            },
        },
        "face_canonical_summary": face_summary,
        "scores": {
            "face": (item.get("scores") or {}).get("face"),
            "upper": (item.get("scores") or {}).get("upper"),
            "full": (item.get("scores") or {}).get("full"),
            "constitution": (item.get("scores") or {}).get("constitution"),
            "depth_3d": (item.get("scores") or {}).get("depth_3d"),
        },
        "identity_alignment": {
            "batch_face_alignment": diagnostics.get("identity_centroid_similarity"),
            "body_alignment": diagnostics.get("body_identity_centroid_similarity"),
            "depth_alignment": diagnostics.get("depth_identity_centroid_similarity"),
            "world3d_alignment": diagnostics.get("world3d_identity_centroid_similarity"),
            "clothfree_alignment": diagnostics.get("clothfree_identity_alignment"),
            "hybrid_alignment": diagnostics.get("hybrid_identity_alignment"),
        },
        "garment": {
            "clothing_coverage_ratio": garment.get("clothing_coverage_ratio"),
            "upper_cloth_coverage": garment.get("upper_cloth_coverage"),
            "lower_cloth_coverage": garment.get("lower_cloth_coverage"),
            "neckline_openness": garment.get("neckline_openness"),
            "shoulder_exposure_balance": garment.get("shoulder_exposure_balance"),
            "confidence": garment.get("confidence"),
        },
        "depth3d": {
            "depth_3d_score": depth_metrics.get("depth_3d_score"),
            "primary_bottleneck": depth_metrics.get("primary_bottleneck"),
            "reasons": list(depth_metrics.get("reasons") or []),
        },
        "canonical_truth_summary": truth_summary,
        "outliers": {
            "score": diagnostics.get("outlier_score"),
            "reasons": list(diagnostics.get("outlier_reasons") or []),
        },
        "master_consistency_card": master_consistency_card,
        "review_focus": {
            "manual_focus": _dedupe_keep_order(
                list(master_consistency_card.get("manual_focus") or []) + list(topology_focus.get("manual_focus") or []),
                limit=8,
            ),
            "manual_review_prompts": _dedupe_keep_order(
                list(master_consistency_card.get("manual_review_prompts") or []) + list(topology_focus.get("manual_review_prompts") or []),
                limit=6,
            ),
        },
        "top_reasons": list(item.get("reasons") or [])[:10],
        "recommendations": list(item.get("recommendations") or [])[:6],
    }
    row["admission_advice"] = build_candidate_admission_advice(row, batch_admission, drift_flags=drift_flags)
    return row


def _build_item_analysis(
    report_payload: Dict[str, Any],
    batch_admission: Dict[str, Any],
    winner_bank_report: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    shot_selection = report_payload.get("shot_selection") or {}
    candidate_lookup = _candidate_lookup(shot_selection)
    drift_lookup = _drift_flag_lookup(winner_bank_report)
    rows: List[Dict[str, Any]] = []
    for item in report_payload.get("items") or []:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        rows.append(
            _summarize_item(
                item,
                candidate_lookup.get(record_key),
                batch_admission,
                drift_flags=drift_lookup.get(record_key),
            )
        )
    rows.sort(
        key=lambda row: (
            9999 if row.get("rank") is None else int(row.get("rank")),
            str(row.get("image") or ""),
        )
    )
    return rows


def _compact_face_canonical_for_gpt(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return _clean_dict(
        {
            "available": data.get("available"),
            "canonical_face_identity_similarity": _round_or_none(data.get("canonical_face_identity_similarity")),
            "canonical_face_landmark_similarity": _round_or_none(data.get("canonical_face_landmark_similarity")),
            "canonical_face_topology_similarity": _round_or_none(data.get("canonical_face_topology_similarity")),
            "head_topology_mean_similarity": _round_or_none(data.get("head_topology_mean_similarity")),
            "head_topology_weakest_part": str(data.get("head_topology_weakest_part") or ""),
            "head_topology_weakest_part_similarity": _round_or_none(
                data.get("head_topology_weakest_part_similarity")
            ),
            "head_topology_upper_face_similarity": _round_or_none(
                data.get("head_topology_upper_face_similarity")
            ),
            "head_topology_mid_face_similarity": _round_or_none(
                data.get("head_topology_mid_face_similarity")
            ),
            "head_topology_lower_face_similarity": _round_or_none(
                data.get("head_topology_lower_face_similarity")
            ),
            "head_topology_contour_similarity": _round_or_none(
                data.get("head_topology_contour_similarity")
            ),
            "head_topology_center_axis_similarity": _round_or_none(
                data.get("head_topology_center_axis_similarity")
            ),
            "head_topology_lateral_balance_similarity": _round_or_none(
                data.get("head_topology_lateral_balance_similarity")
            ),
            "pose_delta_deg": _round_or_none(data.get("pose_delta_deg")),
            "face_pose_normalization_confidence": _round_or_none(data.get("face_pose_normalization_confidence")),
        }
    )


def _compact_candidate_for_gpt(row: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = row.get("review_only_breakdown_v2") or {}
    lane = row.get("lane") or {}
    truth_center = {
        "face_truth_support": _round_or_none((breakdown or {}).get("face_truth_support")),
        "face_topology_support": _round_or_none((breakdown or {}).get("face_topology_support")),
        "body_topology_support": _round_or_none((breakdown or {}).get("body_topology_support")),
        "body_truth_support": _round_or_none((breakdown or {}).get("body_truth_support")),
        "truth_center_score": _round_or_none((breakdown or {}).get("truth_center_score")),
        "support_only_score": _round_or_none((breakdown or {}).get("support_only_score")),
        "angle_tolerance_score": _round_or_none((breakdown or {}).get("angle_tolerance_score")),
        "lane_membership_confidence": _round_or_none((breakdown or {}).get("lane_membership_confidence")),
    }
    same_truth_projection = {
        "mode": str((breakdown or {}).get("same_truth_projection_mode") or "").strip(),
        "policy": str((breakdown or {}).get("same_truth_projection_policy") or "").strip(),
        "confidence": _round_or_none((breakdown or {}).get("same_truth_projection_confidence")),
        "uncertainty": _round_or_none((breakdown or {}).get("same_truth_projection_uncertainty")),
        "reliability": _round_or_none((breakdown or {}).get("same_truth_projection_reliability")),
        "face_projection_confidence": _round_or_none((breakdown or {}).get("face_projection_confidence")),
        "body_projection_confidence": _round_or_none((breakdown or {}).get("body_projection_confidence")),
        "uncertainty_reasons": list((breakdown or {}).get("projection_uncertainty_reasons") or [])[:6],
    }
    clothing_invariant = {
        "clothing_invariant_score": _round_or_none((breakdown or {}).get("clothing_invariant_score")),
        "clothing_invariant_confidence": _round_or_none((breakdown or {}).get("clothing_invariant_confidence")),
        "garment_occlusion_index": _round_or_none((breakdown or {}).get("garment_occlusion_index")),
        "garment_boundary_risk": _round_or_none((breakdown or {}).get("garment_boundary_risk")),
        "surface_evidence_support": _round_or_none((breakdown or {}).get("surface_evidence_support")),
        "visible_body_surface_alignment": _round_or_none((breakdown or {}).get("visible_body_surface_alignment")),
        "visible_body_structure_score": _round_or_none((breakdown or {}).get("visible_body_structure_score")),
        "clothfree_identity_alignment": _round_or_none((breakdown or {}).get("clothfree_identity_alignment")),
        "body_under_clothes_continuity": _round_or_none((breakdown or {}).get("body_under_clothes_continuity")),
        "occlusion_adjusted_truth_score": _round_or_none((breakdown or {}).get("occlusion_adjusted_truth_score")),
    }
    compact = {
        "image": row.get("image"),
        "rank": row.get("rank"),
        "review_bucket": row.get("review_bucket"),
        "review_only_status": row.get("review_only_status_v2"),
        "selection_score": _round_or_none(row.get("selection_score")),
        "review_only_confidence": _round_or_none(row.get("review_only_confidence_v2")),
        "lane": _clean_dict(
            {
                "intended_view": str((lane or {}).get("intended_view") or "").strip(),
                "intended_lane_family": str((lane or {}).get("intended_lane_family") or "").strip(),
                "prompt_intent_is_weak_prior": (lane or {}).get("prompt_intent_is_weak_prior"),
                "view_lane": str((lane or {}).get("view_lane") or "").strip(),
                "view_lane_detail": str((lane or {}).get("view_lane_detail") or "").strip(),
                "observed_lane_family": str((lane or {}).get("observed_lane_family") or "").strip(),
                "view_lane_detail_confidence": _round_or_none((lane or {}).get("view_lane_detail_confidence")),
                "view_lane_strictness_score": _round_or_none((lane or {}).get("view_lane_strictness_score")),
                "body_yaw_deg": _round_or_none((lane or {}).get("body_yaw_deg")),
                "observed_lane_center_distance_deg": _round_or_none((lane or {}).get("observed_lane_center_distance_deg")),
                "observed_lane_source": str((lane or {}).get("observed_lane_source") or "").strip(),
            }
        ),
        "truth_center": _clean_dict(truth_center),
        "same_truth_projection": _clean_dict(same_truth_projection),
        "clothing_invariant": _clean_dict(clothing_invariant),
        "canonical_truth_summary": _clean_dict(dict(row.get("canonical_truth_summary") or {})),
        "face_canonical_summary": _compact_face_canonical_for_gpt(row.get("face_canonical_summary") or {}),
        "hard_vetoes": list(row.get("review_only_hard_vetoes_v2") or [])[:6],
        "soft_flags": list(row.get("review_only_soft_flags_v2") or [])[:6],
        "why_not_high_confidence": list(row.get("why_not_high_confidence_v2") or [])[:4],
        "review_focus": list(((row.get("review_focus") or {}).get("manual_focus")) or [])[:4],
        "top_reasons": list(row.get("top_reasons") or [])[:6],
        "admission_advice": _compact_admission_advice(row.get("admission_advice") or {}),
    }
    policy_note = str(row.get("review_only_policy_note_v2") or "").strip()
    if policy_note:
        compact["policy_note"] = policy_note
    return _clean_dict(compact)


def _first_rows_by_status(
    rows: Sequence[Dict[str, Any]],
    *,
    status: str,
    limit: int,
) -> List[Dict[str, Any]]:
    picked = [
        row
        for row in rows
        if str(row.get("review_only_status_v2") or "").strip().upper() == status
    ]
    picked.sort(
        key=lambda row: (
            9999 if row.get("rank") is None else int(row.get("rank")),
            str(row.get("image") or ""),
        )
    )
    return [_compact_candidate_for_gpt(row) for row in picked[:limit]]


def _build_gpt_review_packet(
    review_packet: Dict[str, Any],
    *,
    review_packet_file: Path,
    gpt_review_packet_file: Path,
) -> Dict[str, Any]:
    batch_summary = dict(review_packet.get("batch_summary") or {})
    ranked_review_packet = review_packet.get("ranked_review_packet") or {}
    items = list(review_packet.get("items") or [])
    selection = batch_summary.get("selection") or {}

    clothing_rows = [
        (row.get("review_only_breakdown_v2") or {})
        for row in items
        if isinstance(row, dict)
    ]
    surface_values = [value for value in (_round_or_none((row or {}).get("surface_evidence_support")) for row in clothing_rows) if value is not None]
    surface_alignment_values = [value for value in (_round_or_none((row or {}).get("visible_body_surface_alignment")) for row in clothing_rows) if value is not None]
    occlusion_values = [value for value in (_round_or_none((row or {}).get("garment_occlusion_index")) for row in clothing_rows) if value is not None]
    boundary_risk_values = [value for value in (_round_or_none((row or {}).get("garment_boundary_risk")) for row in clothing_rows) if value is not None]
    clothing_conf_values = [value for value in (_round_or_none((row or {}).get("clothing_invariant_confidence")) for row in clothing_rows) if value is not None]

    def _mean_or_none(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return _round_or_none(sum(float(value) for value in values) / max(1, len(values)))

    gpt_packet = {
        "schema_version": "gpt_review_packet_v1",
        "generated_at_utc": review_packet.get("generated_at_utc"),
        "system_role": review_packet.get("system_role"),
        "project_scope": review_packet.get("project_scope") or {},
        "final_decision_owner": review_packet.get("final_decision_owner"),
        "analysis_scope": {
            "default_send_to_gpt": ["gpt_review_packet.json"],
            "optional_companion_files": {
                "winner_bank_report.json": "only when analyzing cross-batch drift",
                "training_admission_manifest.json": "external/legacy audit only; not required for screening",
            },
            "do_not_send_by_default": [
                "qa_report.json",
                "ranked_candidates.json",
                "winner_bank_candidate.json",
                "outputs/heavy_evidence_cache/**",
            ],
        },
        "decision_boundary": {
            "packet_role": "compact_review_evidence_only",
            "does_not_decide": [
                "final training-set admission",
                "final image-set membership",
                "dataset assembly",
                "winner-bank freeze",
            ],
            "ranking_meaning": "review_priority_and_risk_routing_not_final_selection",
        },
        "source_files": _clean_dict(
            {
                "review_packet": str(review_packet_file),
                "gpt_review_packet": str(gpt_review_packet_file),
                "winner_bank_report": (review_packet.get("source_files") or {}).get("winner_bank_report"),
                "training_admission_manifest": (review_packet.get("source_files") or {}).get("training_admission_manifest"),
            }
        ),
        "batch": {
            "target_profile": batch_summary.get("target_profile"),
            "run_status": batch_summary.get("run_status"),
            "input_count": batch_summary.get("input_count"),
            "selection_method": selection.get("selection_method"),
            "top_ranked_image": selection.get("top_ranked_image"),
            "selection_gap_top2": selection.get("selection_gap_top2"),
            "manual_review_window": selection.get("manual_review_window"),
            "shortlist_size": selection.get("shortlist_size"),
            "selection_monitor": _compact_selection_comparison(selection.get("selection_comparison") or {}),
            "review_only_status_counts": _status_counts(items, "review_only_status_v2"),
            "main_status_counts": batch_summary.get("status_counts") or {},
            "lane_detail_counts": batch_summary.get("lane_detail_counts") or {},
            "primary_risks": list(batch_summary.get("primary_risks") or [])[:6],
            "clothing_invariant_summary": _clean_dict(
                {
                    "batch_clothfree_identity_cohesion": (batch_summary.get("identity_summary") or {}).get("batch_clothfree_identity_cohesion"),
                    "body_under_clothes_continuity": (batch_summary.get("geometry_summary") or {}).get("body_under_clothes_continuity"),
                    "garment_boundary_stability": (batch_summary.get("garment_summary") or {}).get("garment_boundary_stability"),
                    "surface_evidence_support_mean": _mean_or_none(surface_values),
                    "visible_body_surface_alignment_mean": _mean_or_none(surface_alignment_values),
                    "garment_occlusion_index_mean": _mean_or_none(occlusion_values),
                    "garment_boundary_risk_mean": _mean_or_none(boundary_risk_values),
                    "clothing_invariant_confidence_mean": _mean_or_none(clothing_conf_values),
                    "surface_evidence_coverage": _round_or_none(len(surface_values) / max(1, len(items))),
                }
            ),
            "lane_risk_focus": _clean_dict(dict(batch_summary.get("lane_risk_focus") or {})),
            "release_gate": _compact_release_gate(batch_summary.get("release_gate") or {}),
            "admission_advice": _compact_admission_advice(batch_summary.get("admission_advice") or {}),
            "dataset_curation_governance": _clean_dict(
                dict(review_packet.get("dataset_curation_status") or {})
            ),
            "batch_preflight": _clean_dict(dict(batch_summary.get("batch_preflight") or {})),
            "evidence_completeness": _clean_dict(dict(batch_summary.get("evidence_completeness") or {})),
            "lane_governance_note": "Use observed lane family for QA governance. Prompt/view intent remains a weak prior for diagnosis only.",
            "ranking_governance_note": "Top-ranked rows are review-priority evidence; this packet does not decide final image-set membership.",
        },
        "priority_review_queue": {
            "pass_candidates": _first_rows_by_status(items, status="PASS", limit=12),
            "warn_watchlist": _first_rows_by_status(items, status="WARN", limit=8),
            "fail_watchlist": _first_rows_by_status(items, status="FAIL", limit=8),
        },
        "top_candidates": [
            _compact_candidate_for_gpt(row)
            for row in list(ranked_review_packet.get("top_candidates") or [])[:5]
            if isinstance(row, dict)
        ],
    }
    return gpt_packet


def _build_review_artifacts_index(
    *,
    output_dir: Path,
    review_packet_file: Path,
    gpt_review_packet_file: Path,
    report_file: Path,
    ranked_candidates_file: Path,
    winner_bank_report_file: Any,
    training_admission_manifest_file: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": "review_artifacts_v1",
        "default_send_to_gpt": [
            {
                "file": str(gpt_review_packet_file),
                "purpose": "default compact packet for GPT batch analysis",
            }
        ],
        "human_deep_review": [
            {
                "file": str(review_packet_file),
                "purpose": "rich structured packet for human or deep GPT review",
            }
        ],
        "optional_companion_files": [
            {
                "file": str(winner_bank_report_file or ""),
                "purpose": "attach only for cross-batch drift analysis",
            },
            {
                "file": str(training_admission_manifest_file or ""),
                "purpose": "external/legacy audit only; final training admission is outside this project",
            },
        ],
        "internal_debug_only": [
            {
                "file": str(report_file),
                "purpose": "full internal replay/debug ledger",
            },
            {
                "file": str(ranked_candidates_file),
                "purpose": "full machine ranking and component details",
            },
            {
                "file": str((output_dir / "winner_bank_candidate.json")),
                "purpose": "machine candidate export, not for default GPT analysis",
            },
            {
                "file": str((output_dir / "heavy_evidence_cache")),
                "purpose": "heavy cache directory, never send by default",
            },
        ],
    }


def build_review_packet(
    report_payload: Dict[str, Any],
    output_dir: Path,
    report_file: Path,
    ranked_candidates_file: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_meta = report_payload.get("winner_bank_governance") or {}
    winner_bank_report = _load_json_if_exists(winner_meta.get("drift_report_file"))
    winner_bank_candidate = _load_json_if_exists(winner_meta.get("candidate_file"))
    project_scope = (report_payload.get("report_meta") or {}).get("project_scope") or {
        "role": "screening_and_evidence_only",
        "training_admission_participation": False,
        "image_set_decision_participation": False,
        "final_training_decision_owner": "external_training_decision_flow",
        "final_image_set_decision_owner": "external_dataset_curation_flow",
    }
    batch_summary = _build_batch_summary(report_payload)
    if not bool(project_scope.get("training_admission_participation", False)):
        batch_summary["training_admission_governance"] = {
            **dict(batch_summary.get("training_admission_governance") or {}),
            "enabled": False,
            "mode": "external_final_decision_out_of_scope",
            "participates_in_final_admission": False,
            "participates_in_final_image_set_decision": False,
            "final_decision_owner": "external_training_decision_flow",
            "final_image_set_decision_owner": "external_dataset_curation_flow",
            "local_role": "screening_evidence_only",
        }
    batch_admission = build_batch_admission_advice(batch_summary, {"report": winner_bank_report})
    item_rows = _build_item_analysis(report_payload, batch_admission, winner_bank_report)
    item_lookup = {
        str(row.get("record_key") or row.get("image") or "").strip(): row
        for row in item_rows
        if str(row.get("record_key") or row.get("image") or "").strip()
    }
    review_packet_file = output_dir / "review_packet.json"
    gpt_review_packet_file = output_dir / "gpt_review_packet.json"
    review_packet = {
        "schema_version": "review_packet_v1_7",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_role": "evidence_only",
        "project_scope": project_scope,
        "final_decision_owner": "external_training_decision_flow",
        "decision_boundary": {
            "local_role": "screening_risk_routing_review_priority_and_evidence_packaging",
            "local_role_is_not": [
                "final training-set admission",
                "final image-set membership decision",
                "dataset assembly decision",
                "winner-bank freeze decision",
            ],
            "external_owners": {
                "final_training_admission": "external_training_decision_flow",
                "final_image_set": "external_dataset_curation_flow",
            },
        },
        "usage_protocol": {
            "machine_role": "provide explainable screening metrics, shortlist ranking, and drift evidence only",
            "human_role": "custom_gpt_plus_human may review evidence; external flows own final training admission and final image-set construction",
            "manual_steps": [
                "review batch_summary first",
                "compare top candidates in ranked_review_packet and pairwise cards",
                "tag candidates as strong evidence, review-needed, diagnostic-only, or reroll-priority evidence",
                "optionally record a human-approved winner into mutable winner_bank memory",
                "send evidence packets to external training and dataset-curation flows when needed",
            ],
        },
        "source_files": {
            "qa_report": str(report_file),
            "ranked_candidates": str(ranked_candidates_file),
            "gpt_review_packet": str(gpt_review_packet_file),
            "winner_bank_candidate": winner_meta.get("candidate_file"),
            "winner_bank_report": winner_meta.get("drift_report_file"),
            "training_admission_manifest": ((report_payload.get("report_meta") or {}).get("training_admission_governance") or {}).get("manifest_file"),
        },
        "batch_summary": {
            **batch_summary,
            "admission_advice": batch_admission,
        },
        "ranked_review_packet": _build_ranked_review_packet(report_payload.get("shot_selection") or {}, item_lookup),
        "winner_bank_status": {
            "enabled": winner_meta.get("enabled"),
            "mode": winner_meta.get("mode"),
            "curated_bank_file": winner_meta.get("curated_bank_file"),
            "curated_bank_available": winner_meta.get("curated_bank_available"),
            "curated_entry_count": winner_meta.get("curated_entry_count"),
            "candidate_entry_count": winner_meta.get("candidate_entry_count"),
            "drift_row_count": winner_meta.get("drift_row_count"),
            "top_drift_risks": list((winner_bank_report or {}).get("top_drift_risks") or []),
            "drift_highlights": [
                {
                    "image": row.get("image"),
                    "drift_flags": list(row.get("drift_flags") or []),
                    "drift_severity": row.get("drift_severity"),
                    "manual_focus": list(row.get("manual_focus") or []),
                }
                for row in list((winner_bank_report or {}).get("drift_rows") or [])[:3]
            ],
            "report": winner_bank_report,
            "candidate": {
                "entry_count": (winner_bank_candidate or {}).get("entry_count"),
                "target_profile": (winner_bank_candidate or {}).get("target_profile"),
            },
        },
        "training_admission_status": (batch_summary.get("training_admission_governance") or {}),
        "dataset_curation_status": (
            (report_payload.get("report_meta") or {}).get("dataset_curation_governance")
            or {
                "enabled": False,
                "mode": "external_final_image_set_decision_out_of_scope",
                "participates_in_final_image_set_decision": False,
                "final_image_set_decision_owner": "external_dataset_curation_flow",
                "local_role": "screening_evidence_and_review_priority_only",
            }
        ),
        "items": item_rows,
        "debug": {
            "report_meta": report_payload.get("report_meta"),
            "collection_summary": (report_payload.get("collection_aggregates") or {}).get("summary"),
        },
    }
    atomic_write_json(review_packet_file, review_packet)
    gpt_review_packet = _build_gpt_review_packet(
        review_packet,
        review_packet_file=review_packet_file,
        gpt_review_packet_file=gpt_review_packet_file,
    )
    atomic_write_json(gpt_review_packet_file, gpt_review_packet)
    review_artifacts = _build_review_artifacts_index(
        output_dir=output_dir,
        review_packet_file=review_packet_file,
        gpt_review_packet_file=gpt_review_packet_file,
        report_file=report_file,
        ranked_candidates_file=ranked_candidates_file,
        winner_bank_report_file=winner_meta.get("drift_report_file"),
        training_admission_manifest_file=((report_payload.get("report_meta") or {}).get("training_admission_governance") or {}).get("manifest_file"),
    )
    review_artifacts_file = output_dir / "review_artifacts.json"
    atomic_write_json(review_artifacts_file, review_artifacts)
    return review_packet
