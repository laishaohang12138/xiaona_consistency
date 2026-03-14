from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .qa_consistency import (
    apply_consistency_soft_gate,
    compute_body_constitution_confidence,
    score_body_constitution_measurements,
    score_depth_3d_lite_geometry,
)
from .qa_runtime import RuntimeContext
from .qa_scoring import classify_module, fuse_overall, get_profile_policy
from .qa_utils import dedupe_keep_order, get_face_size_bucket, get_quality_tolerances_by_face_size


VALID_STATUSES = ("PASS", "WARN", "FAIL")
BENCHMARK_LABEL_SCHEMA = "qa_benchmark_labels_v1"


def export_benchmark_template(report_path: Path, output_path: Path) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    template = {
        "schema_version": BENCHMARK_LABEL_SCHEMA,
        "report_file": str(report_path),
        "items": {},
    }
    for item in items:
        image_name = str(item.get("image", "")).strip()
        if not image_name:
            continue
        debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
        template["items"][image_name] = {
            "expected_status": "",
            "current_status": str(item.get("status", "")),
            "expected_task_profile": "",
            "current_task_profile": str(item.get("task_profile", "")),
            "expected_view_lane": "",
            "current_view_lane": str(debug.get("view_lane", "")),
            "must_have_reasons": [],
            "must_not_have_reasons": [],
            "weight": 1.0,
            "notes": "",
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    return template


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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
        "must_have_reasons": _normalize_string_list(node.get("must_have_reasons", []), "must_have_reasons", image_name),
        "must_not_have_reasons": _normalize_string_list(
            node.get("must_not_have_reasons", []), "must_not_have_reasons", image_name
        ),
        "notes": str(node.get("notes", "")),
    }


def load_benchmark_labels(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
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
    return labels


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
) -> Dict[str, Any]:
    metrics = copy.deepcopy(raw_metrics) if isinstance(raw_metrics, dict) else {}
    metrics.update(
        score_body_constitution_measurements(
            metrics,
            runtime.config.consistency.body_constitution_scoring,
            view_bucket=view_bucket,
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


def _recompute_depth_metrics(runtime: RuntimeContext, item: Dict[str, Any], view_lane: str) -> Dict[str, Any]:
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    raw_metrics = debug.get("depth_3d_metrics", {}) if isinstance(debug.get("depth_3d_metrics", {}), dict) else {}
    upper_geom = debug.get("candidate_upper_geom", {}) if isinstance(debug.get("candidate_upper_geom", {}), dict) else {}
    full_geom = debug.get("candidate_full_geom", {}) if isinstance(debug.get("candidate_full_geom", {}), dict) else {}
    yaw_proxy = _safe_float(debug.get("yaw_proxy", 0.0), 0.0)

    metrics = copy.deepcopy(raw_metrics)
    metrics.update(
        score_depth_3d_lite_geometry(
            upper_geom,
            full_geom,
            view_bucket=view_lane,
            yaw_proxy=yaw_proxy,
            scoring=runtime.config.consistency.depth3d_scoring,
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
    )
    skin_metrics = copy.deepcopy(debug.get("skin_metrics", {})) if isinstance(debug.get("skin_metrics", {}), dict) else {}
    depth_metrics = _recompute_depth_metrics(runtime, item, view_lane=view_lane)

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


def benchmark_report(
    runtime: RuntimeContext,
    report_path: Path,
    labels_path: Path,
    threshold_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .qa_pipeline import _apply_threshold_override

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_items = report_payload.get("items", [])
    labels = load_benchmark_labels(labels_path)
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
    reason_constraint_checked_weight = 0.0
    reason_constraint_matched_weight = 0.0

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
        predicted_view_lane = str(replay_debug.get("view_lane", ""))
        predicted_reasons = set(str(reason) for reason in replayed.get("reasons", []))
        expected_task_profile = str(label.get("expected_task_profile", "")).strip()
        expected_view_lane = str(label.get("expected_view_lane", "")).strip()
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

        missing_reasons = [reason for reason in must_have_reasons if reason not in predicted_reasons]
        unexpected_reasons = [reason for reason in must_not_have_reasons if reason in predicted_reasons]
        reason_constraint_match: Optional[bool] = None
        if must_have_reasons or must_not_have_reasons:
            reason_constraint_checked_weight += weight
            reason_constraint_match = (len(missing_reasons) == 0) and (len(unexpected_reasons) == 0)
            if reason_constraint_match:
                reason_constraint_matched_weight += weight

        constraint_values = [value for value in [task_profile_match, view_lane_match, reason_constraint_match] if value is not None]
        all_constraints_match = all(constraint_values) if constraint_values else None

        per_item.append(
            {
                "image": image_name,
                "expected_status": expected,
                "predicted_status": predicted,
                "weight": weight,
                "match": expected == predicted,
                "scores": replayed["scores"],
                "module_state": replayed["module_state"],
                "reasons": replayed["reasons"],
                "agreement": {
                    "task_profile_match": task_profile_match,
                    "view_lane_match": view_lane_match,
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

    return {
        "schema_version": "qa_benchmark_result_v1",
        "report_file": str(report_path),
        "labels_file": str(labels_path),
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
            "reason_constraint_accuracy": _optional_accuracy(
                reason_constraint_matched_weight,
                reason_constraint_checked_weight,
            ),
            "task_profile_checked_weight": round(task_profile_checked_weight, 6),
            "view_lane_checked_weight": round(view_lane_checked_weight, 6),
            "reason_constraint_checked_weight": round(reason_constraint_checked_weight, 6),
        },
        "class_metrics": {
            status: {key: round(value, 6) for key, value in metrics.items()}
            for status, metrics in class_metrics.items()
        },
        "confusion": {
            f"{expected}->{predicted}": round(weight, 6)
            for (expected, predicted), weight in sorted(confusion.items())
        },
        "items": per_item,
    }
