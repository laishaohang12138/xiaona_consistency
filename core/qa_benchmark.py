from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .providers import build_provider_bundle
from .qa_io import atomic_write_json
from .qa_consistency import (
    apply_consistency_soft_gate,
    compute_body_constitution_confidence,
    score_body_constitution_measurements,
    score_depth_3d_lite_geometry,
)
from .qa_heavy_review import normalize_heavy_evidence_bundle
from .qa_runtime import RuntimeContext
from .qa_scoring import classify_module, fuse_overall, get_profile_policy
from .qa_utils import dedupe_keep_order, get_face_size_bucket, get_quality_tolerances_by_face_size


VALID_STATUSES = ("PASS", "WARN", "FAIL")
BENCHMARK_LABEL_SCHEMA = "qa_benchmark_labels_v1"
HEAVY_PROVIDER_COMPARE_SCHEMA = "qa_benchmark_heavy_compare_v1"


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


def _benchmark_lane_focus(per_item: Sequence[Dict[str, Any]], limit: int = 6) -> Dict[str, Any]:
    lane_counts: Dict[str, int] = {}
    for row in per_item:
        family = _lane_family_from_text(row.get("view_lane_detail") or row.get("view_lane"))
        if family == "unknown":
            continue
        lane_counts[family] = lane_counts.get(family, 0) + 1
    dominant_lane = "unknown"
    if lane_counts:
        dominant_lane = sorted(lane_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

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
    note_by_lane = {
        "front": "当前 benchmark 主要是 front lane，脸部身份和上身结构是主解释信号。",
        "three_quarter": "当前 benchmark 主要是 3Q lane，优先看脸型漂移、年龄感和上身衔接。",
        "side": "当前 benchmark 主要是 side lane，脸部弱信号只做否决参考，主看身材真相、轮廓和空间结构。",
        "back": "当前 benchmark 主要是 back lane，正脸相关 reason 已降噪，主看后背轮廓、骨盆和腿轴。",
        "unknown": "当前 benchmark lane 不够稳定，先确认路由再解读风险。",
    }
    review_focus_by_lane = {
        "front": ["脸部身份", "年龄感/表情", "肩颈与上身比例"],
        "three_quarter": ["脸型漂移", "年龄感", "肩颈与体态衔接"],
        "side": ["身材真相", "侧身轮廓", "depth3d/厚度", "腿线与站姿"],
        "back": ["后背轮廓", "肩线与骨盆", "腿轴与重心", "后侧体量"],
        "unknown": ["确认 lane 是否正确", "优先看明显漂移项", "再看 route/threshold 是否合理"],
    }
    ignored = {
        "FULL_PASS",
        "FRAMING_OK",
        "FEET_IN_FRAME",
        "FACE_EMBEDDING_READY",
        "FACE_LANDMARKS_READY",
    }
    suppressed = suppressed_by_lane.get(dominant_lane, set())
    active_counts: Dict[str, int] = {}
    noise_counts: Dict[str, int] = {}
    for row in per_item:
        for reason in row.get("reasons") or []:
            if not isinstance(reason, str) or reason in ignored or reason.endswith("_READY"):
                continue
            if reason in suppressed:
                noise_counts[reason] = noise_counts.get(reason, 0) + 1
            else:
                active_counts[reason] = active_counts.get(reason, 0) + 1
    active_rows = sorted(active_counts.items(), key=lambda item: (-item[1], item[0]))
    noise_rows = sorted(noise_counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "dominant_lane_family": dominant_lane,
        "lane_counts": lane_counts,
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
        "suppressed_reason_codes": sorted(suppressed),
    }
DEFAULT_BENCHMARK_LABEL_ROLE = "candidate_review"
DEFAULT_BENCHMARK_FROZEN_ROLE = "benchmark_frozen"


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must decode to an object: {path}")
    return payload


def _write_json_object(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    atomic_write_json(path, payload)
    return payload


def export_benchmark_template(
    report_path: Path,
    output_path: Path,
    *,
    dataset_role: str = DEFAULT_BENCHMARK_LABEL_ROLE,
    optuna_ready: bool = False,
    benchmark_id: str = "",
    freeze_tag: str = "",
) -> Dict[str, Any]:
    payload = _read_json_object(report_path)
    items = payload.get("items", [])
    template = {
        "schema_version": BENCHMARK_LABEL_SCHEMA,
        "dataset_role": str(dataset_role).strip() or DEFAULT_BENCHMARK_LABEL_ROLE,
        "optuna_ready": bool(optuna_ready),
        "benchmark_id": str(benchmark_id).strip(),
        "freeze_tag": str(freeze_tag).strip(),
        "report_file": str(report_path),
        "items": {},
    }
    for item in items:
        image_name = str(item.get("image", "")).strip()
        if not image_name:
            continue
        debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
        shadow = debug.get("view_router_v2", {}) if isinstance(debug.get("view_router_v2", {}), dict) else {}
        shadow_classifier = (
            debug.get("view_classifier_shadow", {})
            if isinstance(debug.get("view_classifier_shadow", {}), dict)
            else {}
        )
        template["items"][image_name] = {
            "expected_status": "",
            "current_status": str(item.get("status", "")),
            "expected_task_profile": "",
            "current_task_profile": str(item.get("task_profile", "")),
            "expected_view_lane": "",
            "current_view_lane": str(debug.get("view_lane", "")),
            "expected_view_lane_detail": "",
            "current_view_lane_detail": str(
                debug.get("view_lane_detail", shadow.get("lane_detail", ""))
            ),
            "current_view_lane_detail_confidence": _safe_float(
                debug.get("view_lane_detail_confidence", shadow.get("lane_detail_confidence", 0.0)),
                0.0,
            ),
            "current_view_lane_strictness_score": _safe_float(
                debug.get("view_lane_strictness_score", shadow.get("lane_strictness_score", 0.0)),
                0.0,
            ),
            "current_shadow_view_lane": str(shadow_classifier.get("lane", "")),
            "current_shadow_view_lane_detail": str(shadow_classifier.get("lane_detail", "")),
            "current_shadow_view_confidence": _safe_float(shadow_classifier.get("confidence", 0.0), 0.0),
            "must_have_reasons": [],
            "must_not_have_reasons": [],
            "weight": 1.0,
            "notes": "",
        }
    return _write_json_object(output_path, template)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    return value


def _normalize_string_list(value: Any, field_name: str, image_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"benchmark label {image_name!r} field {field_name} must be a list")
    out: List[str] = []
    for raw in value:
        text = str(raw).strip()
        if not text:
            continue
        out.append(text)
    return dedupe_keep_order(out)


def _normalize_label_node(image_name: str, node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"benchmark label {image_name!r} must be an object")
    expected_status = str(node.get("expected_status", node.get("status", ""))).strip().upper()
    if expected_status not in VALID_STATUSES:
        raise ValueError(
            f"benchmark label {image_name!r} expected_status must be one of {list(VALID_STATUSES)}"
        )
    weight = _safe_float(node.get("weight", 1.0), 1.0)
    if not math.isfinite(weight):
        raise ValueError(f"benchmark label {image_name!r} weight must be finite")
    if weight < 0.0:
        raise ValueError(f"benchmark label {image_name!r} weight must be >= 0")
    return {
        "expected_status": expected_status,
        "weight": weight,
        "expected_task_profile": str(node.get("expected_task_profile", "")).strip(),
        "expected_view_lane": str(node.get("expected_view_lane", "")).strip(),
        "expected_view_lane_detail": str(node.get("expected_view_lane_detail", "")).strip(),
        "must_have_reasons": _normalize_string_list(node.get("must_have_reasons", []), "must_have_reasons", image_name),
        "must_not_have_reasons": _normalize_string_list(
            node.get("must_not_have_reasons", []), "must_not_have_reasons", image_name
        ),
        "notes": str(node.get("notes", "")),
    }


def load_benchmark_label_bundle(labels_path: Path) -> Dict[str, Any]:
    payload = _read_json_object(labels_path)
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != BENCHMARK_LABEL_SCHEMA:
        raise ValueError(
            f"benchmark labels schema_version must be {BENCHMARK_LABEL_SCHEMA!r}, got {schema_version!r}"
        )
    items = payload.get("items", payload.get("images", {}))
    if not isinstance(items, dict):
        raise ValueError("benchmark labels must contain an 'items' object")

    labels: Dict[str, Dict[str, Any]] = {}
    for raw_image_name, node in items.items():
        image_name = str(raw_image_name).strip()
        if not image_name:
            raise ValueError("benchmark labels contain an empty image key")
        if image_name in labels:
            raise ValueError(f"benchmark labels contain duplicate image key after normalization: {image_name!r}")
        labels[image_name] = _normalize_label_node(image_name, node)
    if len(labels) == 0:
        raise ValueError("benchmark labels file contains no valid expected_status entries")
    return {
        "schema_version": BENCHMARK_LABEL_SCHEMA,
        "dataset_role": str(payload.get("dataset_role", DEFAULT_BENCHMARK_LABEL_ROLE)).strip() or DEFAULT_BENCHMARK_LABEL_ROLE,
        "optuna_ready": _safe_bool(payload.get("optuna_ready", False), default=False),
        "benchmark_id": str(payload.get("benchmark_id", "")).strip(),
        "freeze_tag": str(payload.get("freeze_tag", "")).strip(),
        "items": labels,
    }


def load_benchmark_labels(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    return load_benchmark_label_bundle(labels_path)["items"]


def update_benchmark_label_metadata(
    labels_path: Path,
    *,
    dataset_role: Optional[str] = None,
    optuna_ready: Optional[bool] = None,
    benchmark_id: Optional[str] = None,
    freeze_tag: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _read_json_object(labels_path)
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != BENCHMARK_LABEL_SCHEMA:
        raise ValueError(
            f"benchmark labels schema_version must be {BENCHMARK_LABEL_SCHEMA!r}, got {schema_version!r}"
        )

    if dataset_role is not None:
        payload["dataset_role"] = str(dataset_role).strip() or DEFAULT_BENCHMARK_LABEL_ROLE
    elif "dataset_role" not in payload:
        payload["dataset_role"] = DEFAULT_BENCHMARK_LABEL_ROLE

    if optuna_ready is not None:
        payload["optuna_ready"] = bool(optuna_ready)
    elif "optuna_ready" not in payload:
        payload["optuna_ready"] = False

    if benchmark_id is not None:
        payload["benchmark_id"] = str(benchmark_id).strip()
    elif "benchmark_id" not in payload:
        payload["benchmark_id"] = ""

    if freeze_tag is not None:
        payload["freeze_tag"] = str(freeze_tag).strip()
    elif "freeze_tag" not in payload:
        payload["freeze_tag"] = ""

    return _write_json_object(labels_path, payload)


def _recompute_quality_flags(runtime: RuntimeContext, item: Dict[str, Any], face_conf: float) -> Tuple[List[str], Dict[str, Any]]:
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    quality_debug = debug.get("quality_ref_stats", {}) if isinstance(debug.get("quality_ref_stats", {}), dict) else {}
    flags: List[str] = []
    extra_debug: Dict[str, Any] = {}

    bbox_ratio = _safe_float(debug.get("candidate_face_bbox_area_ratio", 0.0), 0.0)
    if face_conf < runtime.config.review.face_no_signal_conf_th or bbox_ratio <= 0.0:
        return ["FACE_NO_RELIABLE_SIGNAL"], {"reason": "low_face_conf_or_missing_bbox"}

    quality_thresholds = runtime.config.quality_thresholds
    tolerances = get_quality_tolerances_by_face_size(bbox_ratio, quality_thresholds)
    extra_debug["face_size_bucket"] = get_face_size_bucket(bbox_ratio)
    extra_debug["bucket_quality_tolerances"] = tolerances

    cand_lab = debug.get("candidate_face_lab_mean", None)
    if isinstance(cand_lab, list) and len(cand_lab) >= 3:
        cand_l = _safe_float(cand_lab[0], 0.0)
        extra_debug["candidate_face_L"] = cand_l
        if cand_l < tolerances["abs_luma_warn"]:
            flags.append("FACE_UNDEREXPOSED_DARK")

        tone_l_mean = quality_debug.get("tone_face_L_mean_bucket", None)
        if tone_l_mean is not None:
            tone_l_mean = _safe_float(tone_l_mean, cand_l)
            extra_debug["tone_face_L_mean_bucket"] = tone_l_mean
            if cand_l < (tone_l_mean - tolerances["dark_delta_L"]):
                flags.append("FACE_DARKER_THAN_TONE_ANCHOR")
            elif cand_l > (tone_l_mean + tolerances["dark_delta_L"]):
                flags.append("FACE_BRIGHTER_THAN_TONE_ANCHOR")

    cand_lap_var = debug.get("candidate_face_lap_var", None)
    if cand_lap_var is not None:
        cand_lap_var = _safe_float(cand_lap_var, 0.0)
        extra_debug["candidate_face_lap_var"] = cand_lap_var
        if cand_lap_var < tolerances["abs_lap_warn"]:
            flags.append("FACE_TOO_SOFT_POSSIBLE_SMOOTHING")
        anchor_lap_mean = quality_debug.get("anchor_face_lap_mean_bucket", None)
        if anchor_lap_mean is not None and cand_lap_var < (_safe_float(anchor_lap_mean, 0.0) * tolerances["sharp_ratio_floor"]):
            flags.append("FACE_SOFTER_THAN_ANCHOR")

    cand_hf_energy = debug.get("candidate_face_hf_energy", None)
    if cand_hf_energy is not None:
        cand_hf_energy = _safe_float(cand_hf_energy, 0.0)
        extra_debug["candidate_face_hf_energy"] = cand_hf_energy
        if cand_hf_energy < tolerances["abs_hf_warn"]:
            flags.append("FACE_LOW_MICROTEXTURE")
        anchor_hf_mean = quality_debug.get("anchor_face_hf_mean_bucket", None)
        if anchor_hf_mean is not None and cand_hf_energy < (_safe_float(anchor_hf_mean, 0.0) * tolerances["texture_ratio_floor"]):
            flags.append("FACE_LOWER_TEXTURE_THAN_ANCHOR")

    return dedupe_keep_order(flags), extra_debug


def _recompute_body_constitution_metrics(
    runtime: RuntimeContext,
    raw_metrics: Dict[str, Any],
    view_bucket: str,
    view_lane_detail: str = "",
) -> Dict[str, Any]:
    metrics = copy.deepcopy(raw_metrics) if isinstance(raw_metrics, dict) else {}
    metrics.update(
        score_body_constitution_measurements(
            metrics,
            runtime.config.consistency.body_constitution_scoring,
            view_bucket=view_bucket,
            view_lane_detail=view_lane_detail,
        )
    )

    width_ready = metrics.get("width_ready", None)
    if width_ready is None:
        width_ready = sum(
            1
            for key in ["chest_width_px", "waist_width_px", "hip_width_px", "thigh_width_px", "calf_width_px"]
            if metrics.get(key, None) is not None
        )
        metrics["width_ready"] = width_ready

    pose_visibility = metrics.get("pose_visibility", None)
    torso_fill = metrics.get("torso_fill", None)
    if pose_visibility is not None:
        metrics["confidence"] = compute_body_constitution_confidence(
            runtime.config.consistency.body_constitution_scoring,
            view_bucket=view_bucket,
            width_ready=int(width_ready),
            pose_visibility=_safe_float(pose_visibility, 0.0),
            torso_fill=None if torso_fill is None else _safe_float(torso_fill, 0.0),
            view_lane_detail=view_lane_detail,
        )

    min_width_metrics = int(
        runtime.config.consistency.body_constitution_scoring.get("validity", {}).get("min_width_metrics", 3)
    )
    metrics["is_valid"] = (metrics.get("body_constitution_score", None) is not None) and (int(width_ready) >= min_width_metrics)
    base_reasons = [
        reason
        for reason in metrics.get("reasons", [])
        if str(reason) not in {"BODY_CONSTITUTION_READY", "BODY_CONSTITUTION_SCORE_EMPTY"}
    ]
    metrics["reasons"] = base_reasons + [
        "BODY_CONSTITUTION_READY" if metrics["is_valid"] else "BODY_CONSTITUTION_SCORE_EMPTY"
    ]
    return metrics


def _recompute_depth_metrics(
    runtime: RuntimeContext,
    item: Dict[str, Any],
    view_lane: str,
    view_lane_detail: str = "",
) -> Dict[str, Any]:
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    router_debug = debug.get("view_router_v2", {}) if isinstance(debug.get("view_router_v2", {}), dict) else {}
    raw_metrics = debug.get("depth_3d_metrics", {}) if isinstance(debug.get("depth_3d_metrics", {}), dict) else {}
    upper_geom = debug.get("candidate_upper_geom", {}) if isinstance(debug.get("candidate_upper_geom", {}), dict) else {}
    full_geom = debug.get("candidate_full_geom", {}) if isinstance(debug.get("candidate_full_geom", {}), dict) else {}
    yaw_proxy = _safe_float(debug.get("yaw_proxy", 0.0), 0.0)
    body_yaw_deg = _safe_float(router_debug.get("body_yaw_deg", debug.get("body_yaw_deg", 0.0)), 0.0)
    pose_frontal_strength = _safe_float(
        router_debug.get("pose_frontal_strength", debug.get("pose_frontal_strength", 0.0)),
        0.0,
    )
    lane_strictness_score = _safe_float(
        debug.get("view_lane_strictness_score", router_debug.get("lane_strictness_score", 0.0)),
        0.0,
    )
    mask_symmetry = router_debug.get("mask_symmetry", debug.get("mask_symmetry", None))
    head_skin_ratio = router_debug.get("head_skin_ratio", debug.get("head_skin_ratio", None))

    metrics = copy.deepcopy(raw_metrics)
    metrics.update(
        score_depth_3d_lite_geometry(
            upper_geom,
            full_geom,
            view_bucket=view_lane,
            yaw_proxy=yaw_proxy,
            body_yaw_deg=body_yaw_deg,
            pose_frontal_strength=pose_frontal_strength,
            lane_strictness_score=lane_strictness_score,
            mask_symmetry=None if mask_symmetry is None else _safe_float(mask_symmetry, 0.0),
            head_skin_ratio=None if head_skin_ratio is None else _safe_float(head_skin_ratio, 0.0),
            scoring=runtime.config.consistency.depth3d_scoring,
            view_lane_detail=view_lane_detail,
        )
    )
    metrics["is_valid"] = metrics.get("depth_3d_score", None) is not None
    base_reasons = [
        reason
        for reason in metrics.get("reasons", [])
        if str(reason) not in {"DEPTH_3D_LITE_READY", "DEPTH_3D_LITE_EMPTY"}
    ]
    metrics["reasons"] = base_reasons + [
        "DEPTH_3D_LITE_READY" if metrics["is_valid"] else "DEPTH_3D_LITE_EMPTY"
    ]
    return metrics


def _apply_profile_view_policy_like(
    target_profile: str,
    policy: Dict[str, Any],
    view_lane: str,
    final_status: str,
    overall_state: str,
    reasons_all: List[str],
) -> Tuple[str, str]:
    def downgrade(reason: str) -> None:
        nonlocal final_status, overall_state
        reasons_all.append(reason)
        if final_status == "PASS":
            final_status = "WARN"
            overall_state = "WARN"

    allowed_view_buckets = policy.get("allowed_view_buckets", [])
    if isinstance(allowed_view_buckets, list) and len(allowed_view_buckets) > 0:
        allowed = {str(item) for item in allowed_view_buckets}
        if view_lane not in allowed:
            downgrade("VIEW_LANE_NOT_ALLOWED_FOR_PROFILE")

    for bucket in policy.get("soft_review_buckets", []) or []:
        if view_lane == str(bucket):
            reasons_all.append("THREE_QUARTER_SOFT_REVIEW" if view_lane == "three_quarter" else "VIEW_LANE_SOFT_REVIEW")

    pass_cap_mode = str(policy.get("pass_cap_mode", "none"))
    if pass_cap_mode == "always_warn" and final_status == "PASS":
        downgrade("PROFILE_PASS_CAPPED_TO_WARN")
    elif pass_cap_mode == "warn_non_front" and view_lane != "front" and final_status == "PASS":
        downgrade("NON_FRONT_PASS_CAPPED_TO_WARN")
    elif pass_cap_mode == "body_gold_front_core":
        if view_lane == "side_90" and final_status == "PASS":
            downgrade("PROFILE_LIKE_NO_SIDE_ANCHOR_PASS_CAPPED")
        elif target_profile == "body_gold_fullbody" and view_lane == "unknown":
            reasons_all.append("BODY_GOLD_VIEW_LANE_UNKNOWN")

    return final_status, overall_state


def _extract_heavy_evidence(debug: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(debug.get("heavy_evidence", {}), dict):
        return normalize_heavy_evidence_bundle(debug.get("heavy_evidence", {}))
    if isinstance(debug.get("heavy_review", {}), dict):
        return normalize_heavy_evidence_bundle(debug.get("heavy_review", {}))
    return normalize_heavy_evidence_bundle({"ok": False, "reasons": ["HEAVY_EVIDENCE_MISSING"]})


def _extract_shadow_classifier(debug: Dict[str, Any]) -> Dict[str, Any]:
    shadow: Dict[str, Any] = {}
    if isinstance(debug.get("view_classifier_shadow", {}), dict):
        shadow = copy.deepcopy(debug.get("view_classifier_shadow", {}))
    elif isinstance(debug.get("view_router_v2", {}), dict):
        router_payload = debug.get("view_router_v2", {})
        if isinstance(router_payload.get("shadow_classifier", {}), dict):
            shadow = copy.deepcopy(router_payload.get("shadow_classifier", {}))

    if not shadow:
        shadow = {}

    flat_lane = str(debug.get("view_classifier_shadow_lane", "")).strip()
    flat_confidence = debug.get("view_classifier_shadow_confidence", None)
    flat_disagrees = debug.get("view_classifier_shadow_disagrees", None)
    flat_available = debug.get("view_classifier_shadow_available", None)

    if flat_lane and not str(shadow.get("lane", "")).strip():
        shadow["lane"] = flat_lane
    if flat_confidence is not None and not isinstance(shadow.get("confidence", None), (int, float)):
        shadow["confidence"] = _safe_float(flat_confidence, 0.0)
    if flat_disagrees is not None and "disagrees" not in shadow:
        shadow["disagrees"] = bool(flat_disagrees)
    if flat_available is not None and "enabled" not in shadow:
        shadow["enabled"] = bool(flat_available)

    if "lane_detail" not in shadow:
        shadow["lane_detail"] = ""
    if "provider_name" not in shadow:
        shadow["provider_name"] = ""
    if "provider_version" not in shadow:
        shadow["provider_version"] = ""
    if "enabled" not in shadow:
        shadow["enabled"] = False
    return shadow


def _extract_face_canonical(debug: Dict[str, Any]) -> Dict[str, Any]:
    shadow: Dict[str, Any] = {}
    if isinstance(debug.get("face_canonical_shadow", {}), dict):
        shadow = copy.deepcopy(debug.get("face_canonical_shadow", {}))

    if not shadow:
        shadow = {}

    flat_available = debug.get("face_canonical_shadow_available", None)
    flat_confidence = debug.get("face_canonical_shadow_confidence", None)

    if flat_available is not None and "available" not in shadow:
        shadow["available"] = bool(flat_available)
    if flat_confidence is not None and not isinstance(
        shadow.get("face_pose_normalization_confidence", None), (int, float)
    ):
        shadow["face_pose_normalization_confidence"] = _safe_float(flat_confidence, 0.0)

    shadow.setdefault("provider_name", "")
    shadow.setdefault("provider_version", "")
    shadow.setdefault("mode", "shadow_only")
    shadow.setdefault("available", False)
    shadow.setdefault("canonical_truth_available", False)
    shadow.setdefault("guidance", [])
    shadow.setdefault("reasons", [])
    return shadow


def replay_report_item(runtime: RuntimeContext, item: Dict[str, Any]) -> Dict[str, Any]:
    target_profile = str(item.get("task_profile", runtime.config.review.active_profile))
    if target_profile not in runtime.config.task_profiles:
        raise ValueError(f"Unknown task_profile in report item: {target_profile}")

    profile = runtime.config.task_profiles[target_profile]
    weights = profile["weights"]
    reqs = profile["require"]
    th = profile["thresholds"]
    policy = get_profile_policy(runtime, target_profile)
    scores = item.get("scores", {}) if isinstance(item.get("scores", {}), dict) else {}
    confs = item.get("confidence", {}) if isinstance(item.get("confidence", {}), dict) else {}
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}

    face_score = _safe_float(scores.get("face", 0.0), 0.0)
    upper_score = _safe_float(scores.get("upper", 0.0), 0.0)
    full_score = _safe_float(scores.get("full", 0.0), 0.0)
    face_conf = _safe_float(confs.get("face", 0.0), 0.0)
    upper_conf = _safe_float(confs.get("upper", 0.0), 0.0)
    full_conf = _safe_float(confs.get("full", 0.0), 0.0)
    view_bucket = str(debug.get("view_bucket", "front"))
    view_lane = str(debug.get("view_lane", view_bucket))
    view_lane_detail = str(debug.get("view_lane_detail", ""))
    view_lane_detail_confidence = _safe_float(debug.get("view_lane_detail_confidence", 0.0), 0.0)
    view_lane_strictness_score = _safe_float(debug.get("view_lane_strictness_score", 0.0), 0.0)
    heavy_evidence = _extract_heavy_evidence(debug)
    shadow_classifier = _extract_shadow_classifier(debug)
    face_canonical = _extract_face_canonical(debug)
    shadow_classifier_lane = str(shadow_classifier.get("lane", "")).strip()
    shadow_classifier_enabled = bool(shadow_classifier.get("enabled"))
    shadow_classifier_confidence = _safe_float(shadow_classifier.get("confidence", 0.0), 0.0)

    face_state, face_state_reasons = classify_module(
        runtime, face_score, face_conf, th["face_pass"], th["face_warn"], "face"
    )
    upper_state, upper_state_reasons = classify_module(
        runtime, upper_score, upper_conf, th["upper_pass"], th["upper_warn"], "upper"
    )
    full_state, full_state_reasons = classify_module(
        runtime, full_score, full_conf, th["full_pass"], th["full_warn"], "full"
    )

    overall_score = fuse_overall(
        {"face": face_score, "upper": upper_score, "full": full_score},
        {"face": face_conf, "upper": upper_conf, "full": full_conf},
        weights,
        scoring=runtime.config.consistency.score_fusion,
    )
    if overall_score >= th["overall_pass"]:
        overall_state = "PASS"
    elif overall_score >= th["overall_warn"]:
        overall_state = "WARN"
    else:
        overall_state = "FAIL"

    hard_fail = False
    hard_warn = False
    strict_fail_min_conf = runtime.config.review.min_conf_for_strict_fail

    if reqs.get("face", False):
        if face_state == "FAIL" and face_conf >= strict_fail_min_conf:
            hard_fail = True
        elif face_state != "PASS":
            hard_warn = True

    if reqs.get("upper", False):
        if upper_state == "FAIL" and upper_conf >= strict_fail_min_conf:
            hard_fail = True
        elif upper_state != "PASS":
            hard_warn = True

    if reqs.get("full", False):
        if full_state == "FAIL" and full_conf >= strict_fail_min_conf:
            hard_fail = True
        elif full_state != "PASS":
            hard_warn = True

    if hard_fail or overall_state == "FAIL":
        final_status = "FAIL"
    elif hard_warn or overall_state == "WARN":
        final_status = "WARN"
    else:
        final_status = "PASS"

    reasons_all = (
        face_state_reasons
        + upper_state_reasons
        + full_state_reasons
        + list(item.get("reasons_face", []))
        + list(item.get("reasons_upper", []))
        + list(item.get("reasons_full", []))
    )

    quality_flags, quality_debug = _recompute_quality_flags(runtime, item, face_conf)
    reasons_all.extend(quality_flags)
    reasons_all = dedupe_keep_order(reasons_all)

    constitution_metrics = _recompute_body_constitution_metrics(
        runtime,
        debug.get("constitution_metrics", {}),
        view_bucket=view_bucket,
        view_lane_detail=view_lane_detail,
    )
    skin_metrics = copy.deepcopy(debug.get("skin_metrics", {})) if isinstance(debug.get("skin_metrics", {}), dict) else {}
    depth_metrics = _recompute_depth_metrics(
        runtime,
        item,
        view_lane=view_lane,
        view_lane_detail=view_lane_detail,
    )

    reasons_all, final_status, overall_state, consistency_gate_debug = apply_consistency_soft_gate(
        runtime=runtime,
        reasons_all=reasons_all,
        final_status=final_status,
        overall_state=overall_state,
        constitution_metrics=constitution_metrics,
        skin_metrics=skin_metrics,
        depth_3d_metrics=depth_metrics,
        view_bucket=view_lane,
    )

    quality_thresholds = runtime.config.quality_thresholds
    hard_quality_flags = set(policy.get("hard_quality_flags", set()))
    soft_quality_flags = quality_thresholds.degrade_flags - hard_quality_flags
    hard_hits = sum(1 for reason in reasons_all if reason in hard_quality_flags)
    soft_hits = sum(1 for reason in reasons_all if reason in soft_quality_flags)
    soft_hit_limit = int(policy.get("soft_quality_hits_to_warn", 2))

    if final_status == "PASS" and (hard_hits >= 1 or soft_hits >= soft_hit_limit):
        final_status = "WARN"
        overall_state = "WARN"

    if "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE" in reasons_all and final_status == "PASS":
        final_status = "WARN"
        overall_state = "WARN"

    skin_sample_risk = _safe_float(skin_metrics.get("sample_risk_score", 0.0), 0.0)
    skin_lighting_risk = _safe_float(skin_metrics.get("lighting_risk_score", 0.0), 0.0)
    skin_risk_policy = runtime.config.consistency.skin_risk
    if final_status == "PASS":
        if bool(policy.get("skin_sample_high_caps_pass", False)) and skin_sample_risk >= skin_risk_policy.sample_high_th:
            final_status = "WARN"
            overall_state = "WARN"
            reasons_all.append("BODY_GOLD_SKIN_SAMPLE_RISK_PASS_CAPPED")
        if bool(policy.get("skin_lighting_high_caps_pass", False)) and skin_lighting_risk >= skin_risk_policy.lighting_high_th:
            final_status = "WARN"
            overall_state = "WARN"
            reasons_all.append("BODY_GOLD_SKIN_LIGHTING_RISK_PASS_CAPPED")

    final_status, overall_state = _apply_profile_view_policy_like(
        target_profile=target_profile,
        policy=policy,
        view_lane=view_lane,
        final_status=final_status,
        overall_state=overall_state,
        reasons_all=reasons_all,
    )
    reasons_all = dedupe_keep_order(reasons_all)

    return {
        "image": str(item.get("image", "")),
        "task_profile": target_profile,
        "predicted_status": final_status,
        "scores": {
            "face": round(face_score, 4),
            "upper": round(upper_score, 4),
            "full": round(full_score, 4),
            "overall": round(overall_score, 4),
            "constitution": None
            if constitution_metrics.get("body_constitution_score", None) is None
            else round(_safe_float(constitution_metrics.get("body_constitution_score", 0.0), 0.0), 4),
            "skin": None
            if skin_metrics.get("skin_uniformity_score", None) is None
            else round(_safe_float(skin_metrics.get("skin_uniformity_score", 0.0), 0.0), 4),
            "depth_3d": None
            if depth_metrics.get("depth_3d_score", None) is None
            else round(_safe_float(depth_metrics.get("depth_3d_score", 0.0), 0.0), 4),
        },
        "confidence": {
            "face": round(face_conf, 4),
            "upper": round(upper_conf, 4),
            "full": round(full_conf, 4),
            "constitution": round(_safe_float(constitution_metrics.get("confidence", 0.0), 0.0), 4),
            "skin": round(_safe_float(skin_metrics.get("confidence", 0.0), 0.0), 4),
            "depth_3d": round(_safe_float(depth_metrics.get("confidence", 0.0), 0.0), 4),
        },
        "module_state": {
            "face": face_state,
            "upper": upper_state,
            "full": full_state,
            "overall": overall_state,
        },
        "reasons": reasons_all,
        "debug": {
            "view_bucket": view_bucket,
            "view_lane": view_lane,
            "view_lane_detail": view_lane_detail,
            "view_lane_detail_confidence": round(view_lane_detail_confidence, 6),
            "view_lane_strictness_score": round(view_lane_strictness_score, 6),
            "view_classifier_shadow": shadow_classifier,
            "view_classifier_shadow_available": shadow_classifier_enabled,
            "view_classifier_shadow_lane": shadow_classifier_lane,
            "view_classifier_shadow_confidence": round(shadow_classifier_confidence, 6),
            "view_classifier_shadow_disagrees": shadow_classifier_enabled and shadow_classifier_lane != view_lane,
            "face_canonical_shadow": face_canonical,
            "face_canonical_shadow_available": bool(face_canonical.get("available")),
            "face_canonical_shadow_confidence": face_canonical.get("face_pose_normalization_confidence"),
            "heavy_evidence": heavy_evidence,
            "quality_flags": quality_flags,
            "quality_debug": quality_debug,
            "constitution_metrics": constitution_metrics,
            "skin_metrics": skin_metrics,
            "depth_3d_metrics": depth_metrics,
            "consistency_gate": consistency_gate_debug,
            "quality_gate_soft_hits": soft_hits,
            "quality_gate_hard_hits": hard_hits,
        },
    }


def _resolve_heavy_image_root(
    runtime: RuntimeContext,
    report_payload: Dict[str, Any],
    image_root: Optional[Path],
) -> Optional[Path]:
    if image_root is not None:
        return image_root.resolve()

    report_meta = report_payload.get("report_meta", {}) if isinstance(report_payload.get("report_meta", {}), dict) else {}
    input_source = report_meta.get("input_source", {}) if isinstance(report_meta.get("input_source", {}), dict) else {}
    root_dir = str(input_source.get("root_dir", "")).strip()
    if root_dir:
        candidate = Path(root_dir)
        if candidate.exists():
            return candidate.resolve()

    runtime_input = getattr(getattr(runtime.config, "paths", None), "dir_input", None)
    if runtime_input is not None and Path(runtime_input).exists():
        return Path(runtime_input).resolve()
    return None


def _resolve_report_item_image_path(
    item: Dict[str, Any],
    *,
    default_image_root: Optional[Path],
) -> Tuple[Optional[Path], str]:
    if not isinstance(item, dict):
        return None, "invalid_report_item"

    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    collection = (
        debug.get("collection_metadata", {})
        if isinstance(debug.get("collection_metadata", {}), dict)
        else {}
    )

    absolute_candidates: List[Tuple[str, str]] = []
    for key in ["source_path", "image_path", "input_path"]:
        top_value = str(item.get(key, "")).strip()
        if top_value:
            absolute_candidates.append((top_value, f"item.{key}"))
        debug_value = str(debug.get(key, "")).strip()
        if debug_value:
            absolute_candidates.append((debug_value, f"debug.{key}"))

    for raw_path, source in absolute_candidates:
        candidate = Path(raw_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve(), source

    relative_candidates: List[Tuple[str, str]] = []
    input_relative = str(collection.get("input_relative_path", "")).strip()
    if input_relative:
        relative_candidates.append((input_relative, "collection_metadata.input_relative_path"))
    image_name = str(item.get("image", "")).strip()
    if image_name:
        relative_candidates.append((image_name, "item.image"))

    if default_image_root is not None:
        for rel_path, source in relative_candidates:
            candidate = (default_image_root / rel_path).resolve()
            if candidate.exists():
                return candidate, f"{source}@image_root"

    for raw_path, source in absolute_candidates:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate.resolve(), source

    if len(relative_candidates) > 0 and default_image_root is None:
        return None, "missing_image_root"
    if len(relative_candidates) > 0:
        return None, "image_not_found_under_root"
    return None, "image_path_missing"


def _new_heavy_metric_state() -> Dict[str, Any]:
    return {
        "num_items": 0,
        "total_weight": 0.0,
        "resolved_image_weight": 0.0,
        "available_weight": 0.0,
        "confidence_sum": 0.0,
        "confidence_weight": 0.0,
        "coverage_sum": 0.0,
        "coverage_weight": 0.0,
        "cache_hit_weight": 0.0,
        "cache_write_weight": 0.0,
        "cache_miss_weight": 0.0,
        "failure_reasons": {},
        "metric_stats": {},
    }


def _new_face_canonical_state() -> Dict[str, Any]:
    return {
        "total_weight": 0.0,
        "available_weight": 0.0,
        "truth_available_weight": 0.0,
        "normalization_sum": 0.0,
        "normalization_weight": 0.0,
        "landmark_sum": 0.0,
        "landmark_weight": 0.0,
        "identity_sum": 0.0,
        "identity_weight": 0.0,
        "pose_delta_sum": 0.0,
        "pose_delta_weight": 0.0,
        "coverage_sum": 0.0,
        "coverage_weight": 0.0,
        "frontalization_sum": 0.0,
        "frontalization_weight": 0.0,
        "fit_conf_sum": 0.0,
        "fit_conf_weight": 0.0,
        "providers": set(),
    }


def _update_face_canonical_state(
    state: Dict[str, Any],
    *,
    face_canonical: Dict[str, Any],
    weight: float,
) -> None:
    state["total_weight"] += float(weight)
    available = bool(face_canonical.get("available"))
    truth_available = bool(face_canonical.get("canonical_truth_available"))
    if available:
        state["available_weight"] += float(weight)
    if truth_available:
        state["truth_available_weight"] += float(weight)

    provider_name = str(face_canonical.get("provider_name", "")).strip()
    if provider_name:
        state["providers"].add(provider_name)

    for key, sum_key, weight_key in [
        ("face_pose_normalization_confidence", "normalization_sum", "normalization_weight"),
        ("canonical_face_landmark_similarity", "landmark_sum", "landmark_weight"),
        ("canonical_face_identity_similarity", "identity_sum", "identity_weight"),
        ("pose_delta_similarity", "pose_delta_sum", "pose_delta_weight"),
        ("visible_face_coverage", "coverage_sum", "coverage_weight"),
        ("frontalization_quality", "frontalization_sum", "frontalization_weight"),
        ("pose_fit_confidence", "fit_conf_sum", "fit_conf_weight"),
    ]:
        value = face_canonical.get(key, None)
        if isinstance(value, (int, float)):
            state[sum_key] += float(value) * float(weight)
            state[weight_key] += float(weight)


def _finalize_face_canonical_state(state: Dict[str, Any]) -> Dict[str, Any]:
    total_weight = float(state.get("total_weight", 0.0) or 0.0)
    normalization_mean = (
        None
        if float(state.get("normalization_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("normalization_sum", 0.0) or 0.0),
                float(state.get("normalization_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    landmark_mean = (
        None
        if float(state.get("landmark_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("landmark_sum", 0.0) or 0.0),
                float(state.get("landmark_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    identity_mean = (
        None
        if float(state.get("identity_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("identity_sum", 0.0) or 0.0),
                float(state.get("identity_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    pose_delta_similarity_mean = (
        None
        if float(state.get("pose_delta_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("pose_delta_sum", 0.0) or 0.0),
                float(state.get("pose_delta_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    coverage_mean = (
        None
        if float(state.get("coverage_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("coverage_sum", 0.0) or 0.0),
                float(state.get("coverage_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    frontalization_mean = (
        None
        if float(state.get("frontalization_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("frontalization_sum", 0.0) or 0.0),
                float(state.get("frontalization_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    fit_conf_mean = (
        None
        if float(state.get("fit_conf_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("fit_conf_sum", 0.0) or 0.0),
                float(state.get("fit_conf_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        )
    )
    readiness_score = round(
        0.30 * float(_safe_div(float(state.get("available_weight", 0.0) or 0.0), total_weight, default=0.0))
        + 0.15 * float(_safe_div(float(state.get("truth_available_weight", 0.0) or 0.0), total_weight, default=0.0))
        + 0.20 * float(normalization_mean or 0.0)
        + 0.15 * float(landmark_mean or 0.0)
        + 0.10 * float(identity_mean or 0.0)
        + 0.10 * float(fit_conf_mean or 0.0),
        6,
    )
    return {
        "available_weight_ratio": round(
            _safe_div(float(state.get("available_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "truth_available_weight_ratio": round(
            _safe_div(float(state.get("truth_available_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "face_pose_normalization_confidence_mean": normalization_mean,
        "canonical_face_landmark_similarity_mean": landmark_mean,
        "canonical_face_identity_similarity_mean": identity_mean,
        "pose_delta_similarity_mean": pose_delta_similarity_mean,
        "visible_face_coverage_mean": coverage_mean,
        "frontalization_quality_mean": frontalization_mean,
        "pose_fit_confidence_mean": fit_conf_mean,
        "face_canonical_readiness_score": readiness_score,
        "face_canonical_readiness_formula": "0.30*available + 0.15*truth_available + 0.20*normalize + 0.15*landmark + 0.10*identity + 0.10*fit_conf",
        "providers": sorted(state.get("providers", set())),
    }


def _update_heavy_metric_state(
    state: Dict[str, Any],
    *,
    heavy_evidence: Dict[str, Any],
    weight: float,
    image_resolved: bool,
) -> None:
    state["num_items"] += 1
    state["total_weight"] += weight
    if image_resolved:
        state["resolved_image_weight"] += weight
    if bool(heavy_evidence.get("available")):
        state["available_weight"] += weight

    confidence = heavy_evidence.get("confidence")
    if isinstance(confidence, (int, float)):
        state["confidence_sum"] += float(confidence) * weight
        state["confidence_weight"] += weight

    coverage = heavy_evidence.get("coverage")
    if isinstance(coverage, (int, float)):
        state["coverage_sum"] += float(coverage) * weight
        state["coverage_weight"] += weight

    cache_state = str(heavy_evidence.get("cache_state", "")).strip()
    if cache_state == "hit":
        state["cache_hit_weight"] += weight
    elif cache_state == "write":
        state["cache_write_weight"] += weight
    elif cache_state == "miss":
        state["cache_miss_weight"] += weight

    failure_reason = str(heavy_evidence.get("failure_reason", "")).strip()
    if failure_reason:
        reasons = state["failure_reasons"]
        reasons[failure_reason] = int(reasons.get(failure_reason, 0)) + 1

    metric_stats = state["metric_stats"]
    for metric in heavy_evidence.get("metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric_name = str(metric.get("metric_name", "")).strip()
        if not metric_name:
            continue
        metric_state = metric_stats.setdefault(
            metric_name,
            {
                "num_items": 0,
                "present_weight": 0.0,
                "value_sum": 0.0,
                "value_weight": 0.0,
                "confidence_sum": 0.0,
                "confidence_weight": 0.0,
                "coverage_sum": 0.0,
                "coverage_weight": 0.0,
            },
        )
        metric_state["num_items"] += 1
        metric_state["present_weight"] += weight

        metric_value = metric.get("metric_value")
        if isinstance(metric_value, (int, float)):
            metric_state["value_sum"] += float(metric_value) * weight
            metric_state["value_weight"] += weight

        metric_conf = metric.get("confidence")
        if isinstance(metric_conf, (int, float)):
            metric_state["confidence_sum"] += float(metric_conf) * weight
            metric_state["confidence_weight"] += weight

        metric_cov = metric.get("coverage")
        if isinstance(metric_cov, (int, float)):
            metric_state["coverage_sum"] += float(metric_cov) * weight
            metric_state["coverage_weight"] += weight


def _finalize_heavy_metric_state(state: Dict[str, Any]) -> Dict[str, Any]:
    total_weight = float(state.get("total_weight", 0.0) or 0.0)
    metric_means = {}
    metric_stats = state.get("metric_stats", {})
    if isinstance(metric_stats, dict):
        for metric_name, metric_state in sorted(metric_stats.items()):
            if not isinstance(metric_state, dict):
                continue
            metric_means[str(metric_name)] = {
                "num_items": int(metric_state.get("num_items", 0) or 0),
                "present_weight_ratio": round(
                    _safe_div(
                        float(metric_state.get("present_weight", 0.0) or 0.0),
                        total_weight,
                        default=0.0,
                    ),
                    6,
                ),
                "value_mean": None
                if float(metric_state.get("value_weight", 0.0) or 0.0) <= 0.0
                else round(
                    _safe_div(
                        float(metric_state.get("value_sum", 0.0) or 0.0),
                        float(metric_state.get("value_weight", 0.0) or 0.0),
                        default=0.0,
                    ),
                    6,
                ),
                "confidence_mean": None
                if float(metric_state.get("confidence_weight", 0.0) or 0.0) <= 0.0
                else round(
                    _safe_div(
                        float(metric_state.get("confidence_sum", 0.0) or 0.0),
                        float(metric_state.get("confidence_weight", 0.0) or 0.0),
                        default=0.0,
                    ),
                    6,
                ),
                "coverage_mean": None
                if float(metric_state.get("coverage_weight", 0.0) or 0.0) <= 0.0
                else round(
                    _safe_div(
                        float(metric_state.get("coverage_sum", 0.0) or 0.0),
                        float(metric_state.get("coverage_weight", 0.0) or 0.0),
                        default=0.0,
                    ),
                    6,
                ),
            }

    return {
        "num_items": int(state.get("num_items", 0) or 0),
        "weight_sum": round(total_weight, 6),
        "resolved_image_weight_ratio": round(
            _safe_div(float(state.get("resolved_image_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "available_weight_ratio": round(
            _safe_div(float(state.get("available_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "confidence_mean": None
        if float(state.get("confidence_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("confidence_sum", 0.0) or 0.0),
                float(state.get("confidence_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        ),
        "coverage_mean": None
        if float(state.get("coverage_weight", 0.0) or 0.0) <= 0.0
        else round(
            _safe_div(
                float(state.get("coverage_sum", 0.0) or 0.0),
                float(state.get("coverage_weight", 0.0) or 0.0),
                default=0.0,
            ),
            6,
        ),
        "cache_hit_weight_ratio": round(
            _safe_div(float(state.get("cache_hit_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "cache_write_weight_ratio": round(
            _safe_div(float(state.get("cache_write_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "cache_miss_weight_ratio": round(
            _safe_div(float(state.get("cache_miss_weight", 0.0) or 0.0), total_weight, default=0.0),
            6,
        ),
        "failure_reasons": {
            str(reason): int(count)
            for reason, count in sorted((state.get("failure_reasons", {}) or {}).items(), key=lambda item: (-item[1], item[0]))
        },
        "metric_means": metric_means,
    }


def _heavy_metric_mean_value(metrics: Dict[str, Any], metric_name: str, field_name: str) -> Optional[float]:
    metric_means = metrics.get("metric_means", {}) if isinstance(metrics, dict) else {}
    if not isinstance(metric_means, dict):
        return None
    metric_row = metric_means.get(metric_name, {})
    if not isinstance(metric_row, dict):
        return None
    value = metric_row.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_canonical_truth_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    body_truth_available_ratio = _heavy_metric_mean_value(
        metrics,
        "body_pose_independent_truth_alignment",
        "present_weight_ratio",
    )
    if body_truth_available_ratio is None:
        body_truth_available_ratio = _heavy_metric_mean_value(
            metrics,
            "body_shape_truth_alignment",
            "present_weight_ratio",
        )
    body_truth_mean = _heavy_metric_mean_value(
        metrics,
        "body_pose_independent_truth_alignment",
        "value_mean",
    )
    if body_truth_mean is None:
        body_truth_mean = _heavy_metric_mean_value(
            metrics,
            "body_shape_truth_alignment",
            "value_mean",
        )
    body_truth_mean_legacy = _heavy_metric_mean_value(
        metrics,
        "body_shape_truth_alignment",
        "value_mean",
    )
    body_beta_mean = _heavy_metric_mean_value(
        metrics,
        "body_shape_beta_similarity",
        "value_mean",
    )
    topology_mean = _heavy_metric_mean_value(
        metrics,
        "body_gait_tolerant_topology_similarity",
        "value_mean",
    )
    if topology_mean is None:
        topology_mean = _heavy_metric_mean_value(
            metrics,
            "body_topology_signature_similarity",
            "value_mean",
        )
    measurement_mean = _heavy_metric_mean_value(
        metrics,
        "body_core_measurement_similarity",
        "value_mean",
    )
    if measurement_mean is None:
        measurement_mean = _heavy_metric_mean_value(
            metrics,
            "canonical_measurement_similarity",
            "value_mean",
        )
    measurement_mean_legacy = _heavy_metric_mean_value(
        metrics,
        "canonical_measurement_similarity",
        "value_mean",
    )
    pose_delta_mean = _heavy_metric_mean_value(
        metrics,
        "body_pose_sensitive_measurement_similarity",
        "value_mean",
    )
    if pose_delta_mean is None:
        pose_delta_mean = _heavy_metric_mean_value(
            metrics,
            "body_pose_delta_similarity",
            "value_mean",
        )
    pose_delta_mean_legacy = _heavy_metric_mean_value(
        metrics,
        "body_pose_delta_similarity",
        "value_mean",
    )
    mesh_fit_mean = _heavy_metric_mean_value(
        metrics,
        "body_mesh_fit_confidence",
        "value_mean",
    )
    readiness_terms = [
        0.45 * float(body_truth_available_ratio or 0.0),
        0.30 * float(body_truth_mean or 0.0),
        0.15 * float(measurement_mean or 0.0),
        0.10 * float(mesh_fit_mean or 0.0),
    ]
    canonical_truth_readiness_score = round(sum(readiness_terms), 6)
    return {
        "body_shape_truth_available_weight_ratio": round(float(body_truth_available_ratio or 0.0), 6),
        "body_pose_independent_truth_available_weight_ratio": round(float(body_truth_available_ratio or 0.0), 6),
        "body_shape_truth_alignment_mean": None
        if body_truth_mean is None
        else round(body_truth_mean, 6),
        "body_shape_truth_alignment_mean_legacy": None
        if body_truth_mean_legacy is None
        else round(body_truth_mean_legacy, 6),
        "body_pose_independent_truth_alignment_mean": None
        if body_truth_mean is None
        else round(body_truth_mean, 6),
        "body_shape_beta_similarity_mean": None
        if body_beta_mean is None
        else round(body_beta_mean, 6),
        "body_gait_tolerant_topology_similarity_mean": None
        if topology_mean is None
        else round(topology_mean, 6),
        "canonical_measurement_similarity_mean": None
        if measurement_mean is None
        else round(measurement_mean, 6),
        "canonical_measurement_similarity_mean_legacy": None
        if measurement_mean_legacy is None
        else round(measurement_mean_legacy, 6),
        "body_core_measurement_similarity_mean": None
        if measurement_mean is None
        else round(measurement_mean, 6),
        "body_pose_delta_similarity_mean": None
        if pose_delta_mean is None
        else round(pose_delta_mean, 6),
        "body_pose_delta_similarity_mean_legacy": None
        if pose_delta_mean_legacy is None
        else round(pose_delta_mean_legacy, 6),
        "body_pose_sensitive_measurement_similarity_mean": None
        if pose_delta_mean is None
        else round(pose_delta_mean, 6),
        "body_mesh_fit_confidence_mean": None
        if mesh_fit_mean is None
        else round(mesh_fit_mean, 6),
        "canonical_truth_readiness_score": canonical_truth_readiness_score,
        "canonical_truth_readiness_formula": (
            "0.45*body_pose_independent_truth_available + 0.30*body_pose_independent_truth_alignment "
            "+ 0.15*body_core_measurement_similarity + 0.10*body_mesh_fit_confidence"
        ),
    }


def benchmark_heavy_provider_compare(
    runtime: RuntimeContext,
    report_path: Path,
    labels_path: Path,
    heavy_providers: Iterable[str],
    *,
    image_root: Optional[Path] = None,
) -> Dict[str, Any]:
    report_payload = _read_json_object(report_path)
    report_items = report_payload.get("items", [])
    label_bundle = load_benchmark_label_bundle(labels_path)
    labels = label_bundle["items"]
    items_by_name = {str(item.get("image", "")): item for item in report_items if str(item.get("image", "")).strip()}
    compare_targets = dedupe_keep_order([str(name).strip() for name in heavy_providers if str(name).strip()])
    if len(compare_targets) == 0:
        raise ValueError("heavy provider compare requires at least one provider name")

    default_image_root = _resolve_heavy_image_root(runtime, report_payload, image_root)
    report_meta = report_payload.get("report_meta", {}) if isinstance(report_payload.get("report_meta", {}), dict) else {}
    original_provider_policy = copy.deepcopy(runtime.config.provider_policy)
    original_provider_bundle = runtime.providers
    face_canonical_state = _new_face_canonical_state()

    for image_name, label in labels.items():
        report_item = items_by_name.get(image_name)
        if report_item is None:
            continue
        report_debug = report_item.get("debug", {}) if isinstance(report_item.get("debug", {}), dict) else {}
        face_canonical = _extract_face_canonical(report_debug)
        weight = _safe_float(label.get("weight", 1.0), 1.0)
        _update_face_canonical_state(
            face_canonical_state,
            face_canonical=face_canonical,
            weight=weight,
        )
    face_canonical_metrics = _finalize_face_canonical_state(face_canonical_state)

    provider_results: Dict[str, Any] = {}
    try:
        for provider_name in compare_targets:
            provider_policy = copy.deepcopy(original_provider_policy)
            provider_policy["heavy_evidence"] = str(provider_name)
            runtime.config.provider_policy = provider_policy
            runtime.providers = build_provider_bundle(provider_policy)
            provider_status = (
                runtime.providers.describe_heavy_evidence()
                if hasattr(runtime.providers, "describe_heavy_evidence")
                else {}
            )

            aggregate_state = _new_heavy_metric_state()
            by_expected_status = {status: _new_heavy_metric_state() for status in VALID_STATUSES}
            missing_from_report: List[str] = []
            missing_images: List[Dict[str, Any]] = []
            per_item: List[Dict[str, Any]] = []

            for image_name, label in labels.items():
                report_item = items_by_name.get(image_name)
                if report_item is None:
                    missing_from_report.append(image_name)
                    continue

                report_debug = report_item.get("debug", {}) if isinstance(report_item.get("debug", {}), dict) else {}
                face_canonical = _extract_face_canonical(report_debug)
                weight = _safe_float(label.get("weight", 1.0), 1.0)
                expected_status = str(label.get("expected_status", "")).strip().upper()
                resolved_image_path, resolution_source = _resolve_report_item_image_path(
                    report_item,
                    default_image_root=default_image_root,
                )

                if resolved_image_path is None:
                    heavy_evidence = normalize_heavy_evidence_bundle(
                        {
                            "ok": False,
                            "provider_name": str(provider_status.get("provider_name") or provider_name),
                            "provider_family": str(provider_status.get("provider_family") or "heavy_evidence"),
                            "provider_version": str(provider_status.get("provider_version") or "unknown"),
                            "reasons": [f"HEAVY_EVIDENCE_IMAGE_UNRESOLVED:{resolution_source}"],
                            "summary": {"guidance": ["benchmark 对比缺少原图路径，无法回放重型证据。"]},
                        }
                    )
                    missing_images.append(
                        {
                            "image": image_name,
                            "reason": resolution_source,
                            "reported_source_path": str(report_item.get("source_path", "")).strip() or None,
                            "reported_input_relative_path": str(
                                ((report_item.get("debug", {}) if isinstance(report_item.get("debug", {}), dict) else {})
                                 .get("collection_metadata", {}) if isinstance(
                                     (report_item.get("debug", {}) if isinstance(report_item.get("debug", {}), dict) else {}).get("collection_metadata", {}),
                                     dict,
                                 ) else {}
                                ).get("input_relative_path", "")
                            ).strip()
                            or None,
                        }
                    )
                else:
                    heavy_evidence = normalize_heavy_evidence_bundle(
                        runtime.providers.get_heavy_evidence(runtime, resolved_image_path)
                    )

                _update_heavy_metric_state(
                    aggregate_state,
                    heavy_evidence=heavy_evidence,
                    weight=weight,
                    image_resolved=resolved_image_path is not None,
                )
                if expected_status in by_expected_status:
                    _update_heavy_metric_state(
                        by_expected_status[expected_status],
                        heavy_evidence=heavy_evidence,
                        weight=weight,
                        image_resolved=resolved_image_path is not None,
                    )

                per_item.append(
                    {
                        "image": image_name,
                        "expected_status": expected_status,
                        "weight": weight,
                        "view_lane": str((report_debug.get("view_lane") or "")).strip() or "unknown",
                        "view_lane_detail": str((report_debug.get("view_lane_detail") or report_debug.get("view_lane") or "")).strip() or "unknown",
                        "reasons": list(report_item.get("reasons") or []),
                        "source_path": str(resolved_image_path) if resolved_image_path is not None else None,
                        "source_resolution": resolution_source,
                        "heavy_evidence": {
                            "available": bool(heavy_evidence.get("available")),
                            "provider_name": heavy_evidence.get("provider_name"),
                            "provider_version": heavy_evidence.get("provider_version"),
                            "confidence": heavy_evidence.get("confidence"),
                            "coverage": heavy_evidence.get("coverage"),
                            "cache_state": heavy_evidence.get("cache_state"),
                            "failure_reason": heavy_evidence.get("failure_reason"),
                            "summary": heavy_evidence.get("summary", {}),
                        },
                        "face_canonical": {
                            "available": bool(face_canonical.get("available")),
                            "provider_name": face_canonical.get("provider_name"),
                            "provider_version": face_canonical.get("provider_version"),
                            "face_pose_normalization_confidence": face_canonical.get("face_pose_normalization_confidence"),
                            "canonical_face_landmark_similarity": face_canonical.get("canonical_face_landmark_similarity"),
                            "canonical_face_identity_similarity": face_canonical.get("canonical_face_identity_similarity"),
                            "pose_delta_deg": face_canonical.get("pose_delta_deg"),
                        },
                    }
                )

            heavy_metrics = _finalize_heavy_metric_state(aggregate_state)
            canonical_truth_summary = _extract_canonical_truth_summary(heavy_metrics)
            evidence_readiness_score = round(
                0.50 * float(heavy_metrics.get("available_weight_ratio", 0.0) or 0.0)
                + 0.25 * float(heavy_metrics.get("confidence_mean", 0.0) or 0.0)
                + 0.25 * float(heavy_metrics.get("coverage_mean", 0.0) or 0.0),
                6,
            )

            provider_results[provider_name] = {
                "provider_status": _json_ready(provider_status),
                "comparison_scope": {
                    "qa_role": "evidence_only",
                    "status_replayed_from_saved_scores": True,
                    "heavy_provider_changes_final_status": False,
                    "heavy_provider_compare_focus": [
                        "available_weight_ratio",
                        "confidence_mean",
                        "coverage_mean",
                        "metric_means",
                    ],
                },
                "derived_scores": {
                    "evidence_readiness_score": evidence_readiness_score,
                    "evidence_readiness_formula": "0.50*available + 0.25*confidence + 0.25*coverage",
                    **canonical_truth_summary,
                },
                "num_report_items": len(report_items),
                "num_labeled_items": len(labels),
                "num_compared_items": len(per_item),
                "missing_from_report": sorted(missing_from_report),
                "missing_images": missing_images,
                "heavy_evidence_metrics": heavy_metrics,
                "canonical_truth_metrics": canonical_truth_summary,
                "group_metrics": {
                    status: _finalize_heavy_metric_state(state)
                    for status, state in by_expected_status.items()
                },
                "items": per_item,
            }
    finally:
        runtime.config.provider_policy = original_provider_policy
        runtime.providers = original_provider_bundle

    baseline_provider = compare_targets[0]
    baseline_metrics = (
        provider_results.get(baseline_provider, {}).get("heavy_evidence_metrics", {})
        if isinstance(provider_results.get(baseline_provider, {}), dict)
        else {}
    )
    ranking = []
    deltas_vs_baseline = {}
    for provider_name in compare_targets:
        metrics = provider_results.get(provider_name, {}).get("heavy_evidence_metrics", {})
        canonical_metrics = provider_results.get(provider_name, {}).get("canonical_truth_metrics", {})
        ranking.append(
            {
                "provider_name": provider_name,
                "evidence_readiness_score": provider_results.get(provider_name, {}).get("derived_scores", {}).get(
                    "evidence_readiness_score"
                ),
                "canonical_truth_readiness_score": canonical_metrics.get("canonical_truth_readiness_score"),
                "available_weight_ratio": metrics.get("available_weight_ratio"),
                "confidence_mean": metrics.get("confidence_mean"),
                "coverage_mean": metrics.get("coverage_mean"),
                "body_shape_truth_available_weight_ratio": canonical_metrics.get(
                    "body_shape_truth_available_weight_ratio"
                ),
                "body_shape_truth_alignment_mean": canonical_metrics.get("body_shape_truth_alignment_mean"),
                "body_shape_beta_similarity_mean": canonical_metrics.get("body_shape_beta_similarity_mean"),
            }
        )
        if provider_name == baseline_provider:
            continue
        baseline_canonical = provider_results.get(baseline_provider, {}).get("canonical_truth_metrics", {})
        deltas_vs_baseline[provider_name] = {
            "available_weight_ratio_delta": round(
                float(metrics.get("available_weight_ratio", 0.0) or 0.0)
                - float(baseline_metrics.get("available_weight_ratio", 0.0) or 0.0),
                6,
            ),
            "confidence_mean_delta": round(
                float(metrics.get("confidence_mean", 0.0) or 0.0)
                - float(baseline_metrics.get("confidence_mean", 0.0) or 0.0),
                6,
            ),
            "coverage_mean_delta": round(
                float(metrics.get("coverage_mean", 0.0) or 0.0)
                - float(baseline_metrics.get("coverage_mean", 0.0) or 0.0),
                6,
            ),
            "evidence_readiness_score_delta": round(
                float(provider_results.get(provider_name, {}).get("derived_scores", {}).get("evidence_readiness_score", 0.0) or 0.0)
                - float(provider_results.get(baseline_provider, {}).get("derived_scores", {}).get("evidence_readiness_score", 0.0) or 0.0),
                6,
            ),
            "canonical_truth_readiness_score_delta": round(
                float(canonical_metrics.get("canonical_truth_readiness_score", 0.0) or 0.0)
                - float(baseline_canonical.get("canonical_truth_readiness_score", 0.0) or 0.0),
                6,
            ),
            "body_shape_truth_available_weight_ratio_delta": round(
                float(canonical_metrics.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0)
                - float(baseline_canonical.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0),
                6,
            ),
            "body_shape_truth_alignment_mean_delta": round(
                float(canonical_metrics.get("body_shape_truth_alignment_mean", 0.0) or 0.0)
                - float(baseline_canonical.get("body_shape_truth_alignment_mean", 0.0) or 0.0),
                6,
            ),
        }

    ranking_generic = list(ranking)
    ranking_generic.sort(
        key=lambda row: (
            -float(row.get("evidence_readiness_score", 0.0) or 0.0),
            -float(row.get("canonical_truth_readiness_score", 0.0) or 0.0),
            -float(row.get("available_weight_ratio", 0.0) or 0.0),
            -float(row.get("coverage_mean", 0.0) or 0.0),
            str(row.get("provider_name", "")),
        )
    )
    ranking_truth = list(ranking)
    ranking_truth.sort(
        key=lambda row: (
            -float(row.get("canonical_truth_readiness_score", 0.0) or 0.0),
            -float(row.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0),
            -float(row.get("body_shape_truth_alignment_mean", 0.0) or 0.0),
            -float(row.get("body_shape_beta_similarity_mean", 0.0) or 0.0),
            -float(row.get("evidence_readiness_score", 0.0) or 0.0),
            str(row.get("provider_name", "")),
        )
    )
    best_generic_provider = str((ranking_generic[0] if ranking_generic else {}).get("provider_name") or "")
    truth_enabled_rows = [
        row for row in ranking_truth
        if float(row.get("body_shape_truth_available_weight_ratio", 0.0) or 0.0) > 0.0
    ]
    best_truth_provider = str((truth_enabled_rows[0] if truth_enabled_rows else {}).get("provider_name") or "")
    baseline_items = []
    if isinstance(provider_results.get(baseline_provider, {}), dict):
        baseline_items = list(provider_results.get(baseline_provider, {}).get("items") or [])
    lane_focus = _benchmark_lane_focus(baseline_items)

    return {
        "schema_version": HEAVY_PROVIDER_COMPARE_SCHEMA,
        "report_file": str(report_path),
        "labels_file": str(labels_path),
        "report_schema_version": str(report_meta.get("schema_version", "")),
        "label_bundle": {
            "dataset_role": str(label_bundle.get("dataset_role", DEFAULT_BENCHMARK_LABEL_ROLE)),
            "optuna_ready": bool(label_bundle.get("optuna_ready", False)),
            "benchmark_id": str(label_bundle.get("benchmark_id", "")),
            "freeze_tag": str(label_bundle.get("freeze_tag", "")),
        },
        "image_resolution": {
            "requested_image_root": str(image_root.resolve()) if image_root is not None else None,
            "default_image_root": str(default_image_root) if default_image_root is not None else None,
        },
        "comparison_scope": {
            "qa_role": "evidence_only",
            "heavy_provider_changes_final_status": False,
            "compare_mode": "heavy_evidence_replay",
            "note": "该对比只量化重型证据，不改写最终准入判断。",
        },
        "face_canonical_metrics": face_canonical_metrics,
        "lane_focus": lane_focus,
        "providers": provider_results,
        "comparison": {
            "baseline_provider": baseline_provider,
            "best_generic_provider": best_generic_provider,
            "best_truth_provider": best_truth_provider or None,
            "ranking_by_evidence_readiness": ranking_generic,
            "ranking_by_generic_readiness": ranking_generic,
            "ranking_by_truth_readiness": ranking_truth,
            "deltas_vs_baseline": deltas_vs_baseline,
        },
    }


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den <= 0:
        return float(default)
    return float(num / den)


def _precision_recall_f1(confusion: Dict[Tuple[str, str], float], label: str) -> Dict[str, float]:
    tp = confusion.get((label, label), 0.0)
    fp = sum(weight for (expected, predicted), weight in confusion.items() if predicted == label and expected != label)
    fn = sum(weight for (expected, predicted), weight in confusion.items() if expected == label and predicted != label)
    precision = _safe_div(tp, tp + fp, default=1.0)
    recall = _safe_div(tp, tp + fn, default=0.0)
    f1 = _safe_div(2.0 * precision * recall, precision + recall, default=0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def _optional_accuracy(matched_weight: float, checked_weight: float) -> Optional[float]:
    if checked_weight <= 0.0:
        return None
    return round(_safe_div(matched_weight, checked_weight, default=0.0), 6)


def _new_aggregate_state() -> Dict[str, Any]:
    return {
        "num_items": 0,
        "confusion": {},
        "total_weight": 0.0,
        "matched_weight": 0.0,
        "false_pass_weight": 0.0,
        "predicted_pass_weight": 0.0,
        "expected_pass_weight": 0.0,
        "task_profile_checked_weight": 0.0,
        "task_profile_matched_weight": 0.0,
        "view_lane_checked_weight": 0.0,
        "view_lane_matched_weight": 0.0,
        "view_lane_detail_checked_weight": 0.0,
        "view_lane_detail_matched_weight": 0.0,
        "shadow_view_lane_checked_weight": 0.0,
        "shadow_view_lane_matched_weight": 0.0,
        "shadow_view_lane_detail_checked_weight": 0.0,
        "shadow_view_lane_detail_matched_weight": 0.0,
        "shadow_primary_lane_agreement_checked_weight": 0.0,
        "shadow_primary_lane_agreement_matched_weight": 0.0,
        "reason_constraint_checked_weight": 0.0,
        "reason_constraint_matched_weight": 0.0,
    }


def _update_aggregate_state(
    state: Dict[str, Any],
    *,
    expected: str,
    predicted: str,
    weight: float,
    task_profile_match: Optional[bool],
    view_lane_match: Optional[bool],
    view_lane_detail_match: Optional[bool],
    shadow_view_lane_match: Optional[bool],
    shadow_view_lane_detail_match: Optional[bool],
    shadow_primary_lane_agreement: Optional[bool],
    reason_constraint_match: Optional[bool],
) -> None:
    confusion = state["confusion"]
    state["num_items"] += 1
    state["total_weight"] += weight
    confusion[(expected, predicted)] = confusion.get((expected, predicted), 0.0) + weight
    if expected == predicted:
        state["matched_weight"] += weight
    if predicted == "PASS":
        state["predicted_pass_weight"] += weight
    if expected == "PASS":
        state["expected_pass_weight"] += weight
    if predicted == "PASS" and expected != "PASS":
        state["false_pass_weight"] += weight
    if task_profile_match is not None:
        state["task_profile_checked_weight"] += weight
        if task_profile_match:
            state["task_profile_matched_weight"] += weight
    if view_lane_match is not None:
        state["view_lane_checked_weight"] += weight
        if view_lane_match:
            state["view_lane_matched_weight"] += weight
    if view_lane_detail_match is not None:
        state["view_lane_detail_checked_weight"] += weight
        if view_lane_detail_match:
            state["view_lane_detail_matched_weight"] += weight
    if shadow_view_lane_match is not None:
        state["shadow_view_lane_checked_weight"] += weight
        if shadow_view_lane_match:
            state["shadow_view_lane_matched_weight"] += weight
    if shadow_view_lane_detail_match is not None:
        state["shadow_view_lane_detail_checked_weight"] += weight
        if shadow_view_lane_detail_match:
            state["shadow_view_lane_detail_matched_weight"] += weight
    if shadow_primary_lane_agreement is not None:
        state["shadow_primary_lane_agreement_checked_weight"] += weight
        if shadow_primary_lane_agreement:
            state["shadow_primary_lane_agreement_matched_weight"] += weight
    if reason_constraint_match is not None:
        state["reason_constraint_checked_weight"] += weight
        if reason_constraint_match:
            state["reason_constraint_matched_weight"] += weight


def _finalize_aggregate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    confusion = state["confusion"]
    class_metrics = {
        status: _precision_recall_f1(confusion, status)
        for status in VALID_STATUSES
    }
    active_statuses = [
        status
        for status in VALID_STATUSES
        if any(expected == status or predicted == status for expected, predicted in confusion.keys())
    ]
    if len(active_statuses) == 0:
        active_statuses = list(VALID_STATUSES)
    macro_f1 = sum(class_metrics[status]["f1"] for status in active_statuses) / len(active_statuses)
    exact_accuracy = _safe_div(state["matched_weight"], state["total_weight"], default=0.0)
    pass_precision = class_metrics["PASS"]["precision"]
    pass_recall = class_metrics["PASS"]["recall"]
    false_pass_rate = _safe_div(state["false_pass_weight"], state["total_weight"], default=0.0)
    release_safety_score = 0.45 * macro_f1 + 0.35 * pass_precision + 0.20 * (1.0 - false_pass_rate)

    return {
        "num_items": int(state["num_items"]),
        "weight_sum": round(state["total_weight"], 6),
        "metrics": {
            "exact_accuracy": round(exact_accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "pass_precision": round(pass_precision, 6),
            "pass_recall": round(pass_recall, 6),
            "false_pass_rate": round(false_pass_rate, 6),
            "release_safety_score": round(release_safety_score, 6),
            "predicted_pass_weight": round(state["predicted_pass_weight"], 6),
            "expected_pass_weight": round(state["expected_pass_weight"], 6),
        },
        "agreement_metrics": {
            "task_profile_accuracy": _optional_accuracy(
                state["task_profile_matched_weight"],
                state["task_profile_checked_weight"],
            ),
            "view_lane_accuracy": _optional_accuracy(
                state["view_lane_matched_weight"],
                state["view_lane_checked_weight"],
            ),
            "view_lane_detail_accuracy": _optional_accuracy(
                state["view_lane_detail_matched_weight"],
                state["view_lane_detail_checked_weight"],
            ),
            "shadow_view_lane_accuracy": _optional_accuracy(
                state["shadow_view_lane_matched_weight"],
                state["shadow_view_lane_checked_weight"],
            ),
            "shadow_view_lane_detail_accuracy": _optional_accuracy(
                state["shadow_view_lane_detail_matched_weight"],
                state["shadow_view_lane_detail_checked_weight"],
            ),
            "shadow_primary_lane_agreement": _optional_accuracy(
                state["shadow_primary_lane_agreement_matched_weight"],
                state["shadow_primary_lane_agreement_checked_weight"],
            ),
            "reason_constraint_accuracy": _optional_accuracy(
                state["reason_constraint_matched_weight"],
                state["reason_constraint_checked_weight"],
            ),
            "task_profile_checked_weight": round(state["task_profile_checked_weight"], 6),
            "view_lane_checked_weight": round(state["view_lane_checked_weight"], 6),
            "view_lane_detail_checked_weight": round(state["view_lane_detail_checked_weight"], 6),
            "shadow_view_lane_checked_weight": round(state["shadow_view_lane_checked_weight"], 6),
            "shadow_view_lane_detail_checked_weight": round(state["shadow_view_lane_detail_checked_weight"], 6),
            "shadow_primary_lane_agreement_checked_weight": round(
                state["shadow_primary_lane_agreement_checked_weight"], 6
            ),
            "reason_constraint_checked_weight": round(state["reason_constraint_checked_weight"], 6),
        },
        "class_metrics": {
            status: {key: round(value, 6) for key, value in metrics.items()}
            for status, metrics in class_metrics.items()
        },
        "confusion": {
            f"{expected}->{predicted}": round(weight, 6)
            for (expected, predicted), weight in sorted(confusion.items())
        },
    }


def benchmark_report(
    runtime: RuntimeContext,
    report_path: Path,
    labels_path: Path,
    threshold_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .qa_pipeline import _apply_threshold_override

    report_payload = _read_json_object(report_path)
    report_items = report_payload.get("items", [])
    label_bundle = load_benchmark_label_bundle(labels_path)
    labels = label_bundle["items"]
    items_by_name = {str(item.get("image", "")): item for item in report_items if str(item.get("image", "")).strip()}

    confusion: Dict[Tuple[str, str], float] = {}
    per_item: List[Dict[str, Any]] = []
    total_weight = 0.0
    matched_weight = 0.0
    false_pass_weight = 0.0
    predicted_pass_weight = 0.0
    expected_pass_weight = 0.0
    missing_from_report: List[str] = []
    task_profile_checked_weight = 0.0
    task_profile_matched_weight = 0.0
    view_lane_checked_weight = 0.0
    view_lane_matched_weight = 0.0
    view_lane_detail_checked_weight = 0.0
    view_lane_detail_matched_weight = 0.0
    shadow_view_lane_checked_weight = 0.0
    shadow_view_lane_matched_weight = 0.0
    shadow_view_lane_detail_checked_weight = 0.0
    shadow_view_lane_detail_matched_weight = 0.0
    shadow_primary_lane_agreement_checked_weight = 0.0
    shadow_primary_lane_agreement_matched_weight = 0.0
    reason_constraint_checked_weight = 0.0
    reason_constraint_matched_weight = 0.0
    heavy_evidence_available_weight = 0.0
    heavy_evidence_confidence_weighted_sum = 0.0
    heavy_evidence_confidence_weight = 0.0
    heavy_evidence_coverage_weighted_sum = 0.0
    heavy_evidence_coverage_weight = 0.0
    heavy_evidence_providers: set[str] = set()
    heavy_metric_state = _new_heavy_metric_state()
    face_canonical_state = _new_face_canonical_state()
    shadow_classifier_available_weight = 0.0
    shadow_classifier_confidence_weighted_sum = 0.0
    shadow_classifier_confidence_weight = 0.0
    shadow_classifier_providers: set[str] = set()
    aggregate_by_view_lane: Dict[str, Dict[str, Any]] = {}
    aggregate_by_view_lane_detail: Dict[str, Dict[str, Any]] = {}
    aggregate_by_task_profile: Dict[str, Dict[str, Any]] = {}

    for image_name, label in labels.items():
        report_item = items_by_name.get(image_name)
        if report_item is None:
            missing_from_report.append(image_name)
            continue

        original_config = copy.deepcopy(runtime.config)
        try:
            if threshold_override is not None:
                _apply_threshold_override(runtime, str(report_item.get("task_profile", runtime.config.review.active_profile)), threshold_override)
            replayed = replay_report_item(runtime, report_item)
        finally:
            runtime.config = original_config

        expected = str(label["expected_status"])
        predicted = str(replayed["predicted_status"])
        weight = _safe_float(label.get("weight", 1.0), 1.0)
        total_weight += weight
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0.0) + weight
        if expected == predicted:
            matched_weight += weight
        if predicted == "PASS":
            predicted_pass_weight += weight
        if expected == "PASS":
            expected_pass_weight += weight
        if predicted == "PASS" and expected != "PASS":
            false_pass_weight += weight

        replay_debug = replayed.get("debug", {}) if isinstance(replayed.get("debug", {}), dict) else {}
        heavy_evidence = _extract_heavy_evidence(replay_debug)
        shadow_classifier = _extract_shadow_classifier(replay_debug)
        face_canonical = _extract_face_canonical(replay_debug)
        predicted_view_lane = str(replay_debug.get("view_lane", ""))
        predicted_view_lane_detail = str(replay_debug.get("view_lane_detail", ""))
        predicted_shadow_view_lane = str(shadow_classifier.get("lane", "")).strip()
        predicted_shadow_view_lane_detail = str(shadow_classifier.get("lane_detail", "")).strip()
        predicted_reasons = set(str(reason) for reason in replayed.get("reasons", []))
        expected_task_profile = str(label.get("expected_task_profile", "")).strip()
        expected_view_lane = str(label.get("expected_view_lane", "")).strip()
        expected_view_lane_detail = str(label.get("expected_view_lane_detail", "")).strip()
        must_have_reasons = [str(reason) for reason in label.get("must_have_reasons", [])]
        must_not_have_reasons = [str(reason) for reason in label.get("must_not_have_reasons", [])]

        task_profile_match: Optional[bool] = None
        if expected_task_profile:
            task_profile_checked_weight += weight
            task_profile_match = replayed["task_profile"] == expected_task_profile
            if task_profile_match:
                task_profile_matched_weight += weight

        view_lane_match: Optional[bool] = None
        if expected_view_lane:
            view_lane_checked_weight += weight
            view_lane_match = predicted_view_lane == expected_view_lane
            if view_lane_match:
                view_lane_matched_weight += weight

        view_lane_detail_match: Optional[bool] = None
        if expected_view_lane_detail:
            view_lane_detail_checked_weight += weight
            view_lane_detail_match = predicted_view_lane_detail == expected_view_lane_detail
            if view_lane_detail_match:
                view_lane_detail_matched_weight += weight

        shadow_enabled = bool(shadow_classifier.get("enabled"))
        shadow_available = shadow_enabled and bool(predicted_shadow_view_lane)
        if shadow_available:
            shadow_classifier_available_weight += weight
        shadow_provider = str(shadow_classifier.get("provider_name", "")).strip()
        if shadow_provider:
            shadow_classifier_providers.add(shadow_provider)
        shadow_confidence = shadow_classifier.get("confidence", None)
        if shadow_available and isinstance(shadow_confidence, (int, float)):
            shadow_classifier_confidence_weighted_sum += float(shadow_confidence) * weight
            shadow_classifier_confidence_weight += weight

        shadow_view_lane_match: Optional[bool] = None
        if expected_view_lane and shadow_available:
            shadow_view_lane_checked_weight += weight
            shadow_view_lane_match = predicted_shadow_view_lane == expected_view_lane
            if shadow_view_lane_match:
                shadow_view_lane_matched_weight += weight

        shadow_view_lane_detail_match: Optional[bool] = None
        if expected_view_lane_detail and shadow_available and predicted_shadow_view_lane_detail:
            shadow_view_lane_detail_checked_weight += weight
            shadow_view_lane_detail_match = predicted_shadow_view_lane_detail == expected_view_lane_detail
            if shadow_view_lane_detail_match:
                shadow_view_lane_detail_matched_weight += weight

        shadow_primary_lane_agreement: Optional[bool] = None
        if shadow_available:
            shadow_primary_lane_agreement_checked_weight += weight
            shadow_primary_lane_agreement = predicted_shadow_view_lane == predicted_view_lane
            if shadow_primary_lane_agreement:
                shadow_primary_lane_agreement_matched_weight += weight

        missing_reasons = [reason for reason in must_have_reasons if reason not in predicted_reasons]
        unexpected_reasons = [reason for reason in must_not_have_reasons if reason in predicted_reasons]
        reason_constraint_match: Optional[bool] = None
        if must_have_reasons or must_not_have_reasons:
            reason_constraint_checked_weight += weight
            reason_constraint_match = (len(missing_reasons) == 0) and (len(unexpected_reasons) == 0)
            if reason_constraint_match:
                reason_constraint_matched_weight += weight

        constraint_values = [
            value
            for value in [task_profile_match, view_lane_match, view_lane_detail_match, reason_constraint_match]
            if value is not None
        ]
        all_constraints_match = all(constraint_values) if constraint_values else None

        if bool(heavy_evidence.get("available")):
            heavy_evidence_available_weight += weight
        heavy_provider = str(heavy_evidence.get("provider_name", "")).strip()
        if heavy_provider:
            heavy_evidence_providers.add(heavy_provider)
        heavy_confidence = heavy_evidence.get("confidence", None)
        if isinstance(heavy_confidence, (int, float)):
            heavy_evidence_confidence_weighted_sum += float(heavy_confidence) * weight
            heavy_evidence_confidence_weight += weight
        heavy_coverage = heavy_evidence.get("coverage", None)
        if isinstance(heavy_coverage, (int, float)):
            heavy_evidence_coverage_weighted_sum += float(heavy_coverage) * weight
            heavy_evidence_coverage_weight += weight
        _update_heavy_metric_state(
            heavy_metric_state,
            heavy_evidence=heavy_evidence,
            weight=weight,
            image_resolved=True,
        )
        _update_face_canonical_state(
            face_canonical_state,
            face_canonical=face_canonical,
            weight=weight,
        )

        view_lane_key = predicted_view_lane or "unknown"
        view_lane_detail_key = predicted_view_lane_detail or "unknown"
        task_profile_key = str(replayed["task_profile"] or "unknown")
        lane_state = aggregate_by_view_lane.setdefault(view_lane_key, _new_aggregate_state())
        lane_detail_state = aggregate_by_view_lane_detail.setdefault(view_lane_detail_key, _new_aggregate_state())
        profile_state = aggregate_by_task_profile.setdefault(task_profile_key, _new_aggregate_state())
        for state in [lane_state, lane_detail_state, profile_state]:
            _update_aggregate_state(
                state,
                expected=expected,
                predicted=predicted,
                weight=weight,
                task_profile_match=task_profile_match,
                view_lane_match=view_lane_match,
                view_lane_detail_match=view_lane_detail_match,
                shadow_view_lane_match=shadow_view_lane_match,
                shadow_view_lane_detail_match=shadow_view_lane_detail_match,
                shadow_primary_lane_agreement=shadow_primary_lane_agreement,
                reason_constraint_match=reason_constraint_match,
            )

        per_item.append(
            {
                "image": image_name,
                "expected_status": expected,
                "predicted_status": predicted,
                "view_lane": view_lane_key,
                "view_lane_detail": view_lane_detail_key,
                "shadow_view_lane": predicted_shadow_view_lane or "unknown",
                "shadow_view_lane_detail": predicted_shadow_view_lane_detail or "unknown",
                "weight": weight,
                "match": expected == predicted,
                "scores": replayed["scores"],
                "module_state": replayed["module_state"],
                "reasons": replayed["reasons"],
                "heavy_evidence": {
                    "available": bool(heavy_evidence.get("available")),
                    "provider_name": heavy_evidence.get("provider_name"),
                    "provider_version": heavy_evidence.get("provider_version"),
                    "confidence": heavy_evidence.get("confidence"),
                    "coverage": heavy_evidence.get("coverage"),
                    "failure_reason": heavy_evidence.get("failure_reason"),
                    "summary": heavy_evidence.get("summary", {}),
                },
                "shadow_classifier": {
                    "available": shadow_available,
                    "provider_name": shadow_classifier.get("provider_name"),
                    "provider_version": shadow_classifier.get("provider_version"),
                    "lane": predicted_shadow_view_lane,
                    "lane_detail": predicted_shadow_view_lane_detail,
                    "confidence": shadow_classifier.get("confidence"),
                    "decision_margin": shadow_classifier.get("decision_margin"),
                    "disagrees": shadow_primary_lane_agreement is False,
                },
                "face_canonical": {
                    "available": bool(face_canonical.get("available")),
                    "provider_name": face_canonical.get("provider_name"),
                    "provider_version": face_canonical.get("provider_version"),
                    "face_pose_normalization_confidence": face_canonical.get("face_pose_normalization_confidence"),
                    "canonical_face_landmark_similarity": face_canonical.get("canonical_face_landmark_similarity"),
                    "canonical_face_identity_similarity": face_canonical.get("canonical_face_identity_similarity"),
                    "pose_delta_deg": face_canonical.get("pose_delta_deg"),
                },
                "agreement": {
                    "task_profile_match": task_profile_match,
                    "view_lane_match": view_lane_match,
                    "view_lane_detail_match": view_lane_detail_match,
                    "shadow_view_lane_match": shadow_view_lane_match,
                    "shadow_view_lane_detail_match": shadow_view_lane_detail_match,
                    "shadow_primary_lane_agreement": shadow_primary_lane_agreement,
                    "missing_reasons": missing_reasons,
                    "unexpected_reasons": unexpected_reasons,
                    "reason_constraints_match": reason_constraint_match,
                    "all_constraints_match": all_constraints_match,
                },
                "notes": label.get("notes", ""),
            }
        )

    class_metrics = {
        status: _precision_recall_f1(confusion, status)
        for status in VALID_STATUSES
    }
    active_statuses = [
        status
        for status in VALID_STATUSES
        if any(expected == status or predicted == status for expected, predicted in confusion.keys())
    ]
    if len(active_statuses) == 0:
        active_statuses = list(VALID_STATUSES)
    macro_f1 = sum(class_metrics[status]["f1"] for status in active_statuses) / len(active_statuses)
    exact_accuracy = _safe_div(matched_weight, total_weight, default=0.0)
    pass_precision = class_metrics["PASS"]["precision"]
    pass_recall = class_metrics["PASS"]["recall"]
    false_pass_rate = _safe_div(false_pass_weight, total_weight, default=0.0)
    release_safety_score = 0.45 * macro_f1 + 0.35 * pass_precision + 0.20 * (1.0 - false_pass_rate)
    group_metrics = {
        "view_lane": {
            key: _finalize_aggregate_state(state)
            for key, state in sorted(aggregate_by_view_lane.items())
        },
        "view_lane_detail": {
            key: _finalize_aggregate_state(state)
            for key, state in sorted(aggregate_by_view_lane_detail.items())
        },
        "task_profile": {
            key: _finalize_aggregate_state(state)
            for key, state in sorted(aggregate_by_task_profile.items())
        },
    }
    heavy_evidence_metrics = _finalize_heavy_metric_state(heavy_metric_state)
    heavy_evidence_metrics["providers"] = sorted(heavy_evidence_providers)
    canonical_truth_metrics = _extract_canonical_truth_summary(heavy_evidence_metrics)
    heavy_evidence_metrics.update(canonical_truth_metrics)
    face_canonical_metrics = _finalize_face_canonical_state(face_canonical_state)
    lane_focus = _benchmark_lane_focus(per_item)

    return {
        "schema_version": "qa_benchmark_result_v1_1",
        "report_file": str(report_path),
        "labels_file": str(labels_path),
        "label_bundle": {
            "dataset_role": str(label_bundle.get("dataset_role", DEFAULT_BENCHMARK_LABEL_ROLE)),
            "optuna_ready": bool(label_bundle.get("optuna_ready", False)),
            "benchmark_id": str(label_bundle.get("benchmark_id", "")),
            "freeze_tag": str(label_bundle.get("freeze_tag", "")),
        },
        "num_report_items": len(report_items),
        "num_labeled_items": len(labels),
        "num_benchmarked_items": len(per_item),
        "missing_from_report": sorted(missing_from_report),
        "metrics": {
            "exact_accuracy": round(exact_accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "pass_precision": round(pass_precision, 6),
            "pass_recall": round(pass_recall, 6),
            "false_pass_rate": round(false_pass_rate, 6),
            "release_safety_score": round(release_safety_score, 6),
            "predicted_pass_weight": round(predicted_pass_weight, 6),
            "expected_pass_weight": round(expected_pass_weight, 6),
        },
        "agreement_metrics": {
            "task_profile_accuracy": _optional_accuracy(task_profile_matched_weight, task_profile_checked_weight),
            "view_lane_accuracy": _optional_accuracy(view_lane_matched_weight, view_lane_checked_weight),
            "view_lane_detail_accuracy": _optional_accuracy(
                view_lane_detail_matched_weight,
                view_lane_detail_checked_weight,
            ),
            "shadow_view_lane_accuracy": _optional_accuracy(
                shadow_view_lane_matched_weight,
                shadow_view_lane_checked_weight,
            ),
            "shadow_view_lane_detail_accuracy": _optional_accuracy(
                shadow_view_lane_detail_matched_weight,
                shadow_view_lane_detail_checked_weight,
            ),
            "shadow_primary_lane_agreement": _optional_accuracy(
                shadow_primary_lane_agreement_matched_weight,
                shadow_primary_lane_agreement_checked_weight,
            ),
            "reason_constraint_accuracy": _optional_accuracy(
                reason_constraint_matched_weight,
                reason_constraint_checked_weight,
            ),
            "task_profile_checked_weight": round(task_profile_checked_weight, 6),
            "view_lane_checked_weight": round(view_lane_checked_weight, 6),
            "view_lane_detail_checked_weight": round(view_lane_detail_checked_weight, 6),
            "shadow_view_lane_checked_weight": round(shadow_view_lane_checked_weight, 6),
            "shadow_view_lane_detail_checked_weight": round(shadow_view_lane_detail_checked_weight, 6),
            "shadow_primary_lane_agreement_checked_weight": round(
                shadow_primary_lane_agreement_checked_weight, 6
            ),
            "reason_constraint_checked_weight": round(reason_constraint_checked_weight, 6),
        },
        "shadow_view_classifier_metrics": {
            "available_weight_ratio": round(_safe_div(shadow_classifier_available_weight, total_weight, default=0.0), 6),
            "confidence_mean": None
            if shadow_classifier_confidence_weight <= 0.0
            else round(
                _safe_div(shadow_classifier_confidence_weighted_sum, shadow_classifier_confidence_weight, default=0.0),
                6,
            ),
            "lane_accuracy": _optional_accuracy(
                shadow_view_lane_matched_weight,
                shadow_view_lane_checked_weight,
            ),
            "lane_detail_accuracy": _optional_accuracy(
                shadow_view_lane_detail_matched_weight,
                shadow_view_lane_detail_checked_weight,
            ),
            "primary_lane_agreement": _optional_accuracy(
                shadow_primary_lane_agreement_matched_weight,
                shadow_primary_lane_agreement_checked_weight,
            ),
            "lane_checked_weight": round(shadow_view_lane_checked_weight, 6),
            "lane_detail_checked_weight": round(shadow_view_lane_detail_checked_weight, 6),
            "primary_lane_agreement_checked_weight": round(shadow_primary_lane_agreement_checked_weight, 6),
            "providers": sorted(shadow_classifier_providers),
        },
        "face_canonical_metrics": face_canonical_metrics,
        "heavy_evidence_metrics": heavy_evidence_metrics,
        "canonical_truth_metrics": canonical_truth_metrics,
        "lane_focus": lane_focus,
        "class_metrics": {
            status: {key: round(value, 6) for key, value in metrics.items()}
            for status, metrics in class_metrics.items()
        },
        "confusion": {
            f"{expected}->{predicted}": round(weight, 6)
            for (expected, predicted), weight in sorted(confusion.items())
        },
        "group_metrics": group_metrics,
        "items": per_item,
    }
