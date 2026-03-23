from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .qa_consistency import extract_depth_3d_lite_metrics
from .qa_runtime import AnchorSet, FaceFeat, PoseFeat, RuntimeContext


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _weighted_mean(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _normalize_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


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


def _resolve_registry_path(runtime: RuntimeContext, raw_path: str) -> Path:
    expanded = str(raw_path).strip()
    expanded = expanded.replace("${PROJECT_ROOT}", str(runtime.config.paths.base_dir))
    expanded = expanded.replace("${CONFIG_DIR}", str(runtime.config.paths.config_dir))
    path = Path(expanded)
    if not path.is_absolute():
        path = (runtime.config.paths.base_dir / path).resolve()
    return path.resolve()


def _registry_ref_path(runtime: RuntimeContext, role: str) -> Optional[Path]:
    anchors_node = runtime.config.anchor_registry.get("anchors", {})
    if not isinstance(anchors_node, dict):
        return None
    for _, node in anchors_node.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("role", "")).upper() != role.upper():
            continue
        if str(node.get("anchor_tier", "")).lower() != "absolute":
            continue
        raw_path = str(node.get("path", "")).strip()
        if raw_path:
            return _resolve_registry_path(runtime, raw_path)
    return None


def _match_pose_anchor_by_path(anchors: AnchorSet, target_path: Optional[Path]) -> Optional[PoseFeat]:
    if target_path is None:
        return None
    full_paths = list(anchors.meta.get("full_paths") or [])
    for index, raw_path in enumerate(full_paths):
        try:
            resolved = Path(str(raw_path)).resolve()
        except Exception:
            continue
        if resolved == target_path and index < len(anchors.full_pose_feats):
            return anchors.full_pose_feats[index]
    return None


def _build_body_identity_signature(
    pose: PoseFeat,
    constitution_metrics: Dict[str, Any],
    depth_3d_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    upper = pose.upper_geom or {}
    full = pose.full_geom or {}
    feature_specs = [
        ("head_body_ratio", full.get("head_body_ratio"), 5.0, 2.0),
        ("leg_ratio", full.get("leg_ratio"), 0.47, 0.14),
        ("torso_len_norm", upper.get("torso_len_norm"), 0.24, 0.08),
        ("hip_shoulder_ratio", upper.get("hip_shoulder_ratio"), 0.58, 0.22),
        ("torso_compactness", upper.get("torso_compactness"), 0.92, 0.26),
        ("shoulder_hip_center_offset_norm", upper.get("shoulder_hip_center_offset_norm"), 0.02, 0.08),
        ("lower_limb_balance", full.get("lower_limb_balance"), 0.96, 0.12),
        ("foot_length_balance", full.get("foot_length_balance"), 0.93, 0.14),
        ("waist_to_torso_ratio", constitution_metrics.get("waist_to_torso_ratio"), 0.80, 0.22),
        ("hip_to_torso_ratio", constitution_metrics.get("hip_to_torso_ratio"), 1.00, 0.24),
    ]
    values: List[float] = []
    ready = 0
    for _, raw_value, center, scale in feature_specs:
        if raw_value is None:
            values.append(0.0)
            continue
        try:
            numeric = float(raw_value)
        except Exception:
            values.append(0.0)
            continue
        values.append((numeric - float(center)) / max(1e-6, float(scale)))
        ready += 1
    signature = _normalize_vector(values) if ready >= 4 else None
    return {"signature": signature, "ready_features": ready, "feature_count": len(feature_specs)}


def _build_depth_identity_signature(
    pose: PoseFeat,
    depth_3d_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    upper = pose.upper_geom or {}
    full = pose.full_geom or {}
    feature_specs = [
        ("shoulder_width_norm", upper.get("shoulder_width_norm"), 0.17, 0.05),
        ("hip_width_norm", upper.get("hip_width_norm"), 0.10, 0.04),
        ("hip_shoulder_ratio", upper.get("hip_shoulder_ratio"), 0.58, 0.22),
        ("torso_len_norm", upper.get("torso_len_norm"), 0.24, 0.08),
        ("torso_compactness", upper.get("torso_compactness"), 0.92, 0.26),
        ("shoulder_hip_center_offset_norm", upper.get("shoulder_hip_center_offset_norm"), 0.02, 0.08),
        ("spine_angle_deg", upper.get("spine_angle_deg"), 0.0, 8.0),
        ("lower_limb_balance", full.get("lower_limb_balance"), 0.96, 0.12),
        ("thigh_length_balance", full.get("thigh_length_balance"), 0.95, 0.10),
        ("calf_length_balance", full.get("calf_length_balance"), 0.95, 0.10),
        ("foot_length_balance", full.get("foot_length_balance"), 0.93, 0.14),
        ("ankle_gap_norm", full.get("ankle_gap_norm"), 0.09, 0.08),
        ("depth_3d_score", depth_3d_metrics.get("depth_3d_score"), 0.82, 0.16),
        ("turn_signal_score", depth_3d_metrics.get("turn_signal_score"), 0.75, 0.16),
        ("torso_volume_score", depth_3d_metrics.get("torso_volume_score"), 0.80, 0.16),
        ("side_profile_score", depth_3d_metrics.get("side_profile_score"), 0.84, 0.14),
        ("posterior_score", depth_3d_metrics.get("posterior_score"), 0.84, 0.14),
        ("torso_compactness_score", depth_3d_metrics.get("torso_compactness_score"), 0.80, 0.16),
    ]
    values: List[float] = []
    ready = 0
    for _, raw_value, center, scale in feature_specs:
        if raw_value is None:
            values.append(0.0)
            continue
        try:
            numeric = float(raw_value)
        except Exception:
            values.append(0.0)
            continue
        values.append((numeric - float(center)) / max(1e-6, float(scale)))
        ready += 1
    signature = _normalize_vector(values) if ready >= 6 else None
    return {"signature": signature, "ready_features": ready, "feature_count": len(feature_specs)}


def _build_world3d_identity_signature(pose: PoseFeat) -> Dict[str, Any]:
    xyz = getattr(pose, "lm_world", None)
    vis = getattr(pose, "lm_vis", None)
    if xyz is None or vis is None:
        return {"signature": None, "ready_features": 0, "feature_count": 0}

    def _dist(i: int, j: int) -> Optional[float]:
        if vis[i] <= 0.35 or vis[j] <= 0.35:
            return None
        return float(np.linalg.norm(xyz[i] - xyz[j]))

    def _balance(left_value: Optional[float], right_value: Optional[float]) -> Optional[float]:
        if left_value is None or right_value is None:
            return None
        hi = max(abs(float(left_value)), abs(float(right_value)))
        if hi <= 1e-6:
            return None
        return float(min(abs(float(left_value)), abs(float(right_value))) / hi)

    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_HEEL, R_HEEL = 29, 30
    L_FOOT, R_FOOT = 31, 32

    shoulder_span = _dist(L_SHOULDER, R_SHOULDER)
    hip_span = _dist(L_HIP, R_HIP)
    torso_len = None
    if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]):
        shoulder_mid = (xyz[L_SHOULDER] + xyz[R_SHOULDER]) / 2.0
        hip_mid = (xyz[L_HIP] + xyz[R_HIP]) / 2.0
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    leg_len = None
    if all(vis[idx] > 0.35 for idx in [L_HIP, R_HIP, L_ANKLE, R_ANKLE]):
        hip_mid = (xyz[L_HIP] + xyz[R_HIP]) / 2.0
        ankle_mid = (xyz[L_ANKLE] + xyz[R_ANKLE]) / 2.0
        leg_len = float(np.linalg.norm(hip_mid - ankle_mid))

    left_thigh = _dist(L_HIP, L_KNEE)
    right_thigh = _dist(R_HIP, R_KNEE)
    left_calf = _dist(L_KNEE, L_ANKLE)
    right_calf = _dist(R_KNEE, R_ANKLE)
    left_foot = _dist(L_HEEL, L_FOOT)
    right_foot = _dist(R_HEEL, R_FOOT)

    thigh_balance = _balance(left_thigh, right_thigh)
    calf_balance = _balance(left_calf, right_calf)
    foot_balance = _balance(left_foot, right_foot)
    shoulder_level = None
    shoulder_depth = None
    hip_level = None
    hip_depth = None
    torso_twist = None
    head_torso_ratio = None
    leg_ratio = None

    if shoulder_span is not None and shoulder_span > 1e-6 and all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER]):
        shoulder_level = abs(float(xyz[L_SHOULDER][1] - xyz[R_SHOULDER][1])) / shoulder_span
        shoulder_depth = abs(float(xyz[L_SHOULDER][2] - xyz[R_SHOULDER][2])) / shoulder_span
    if hip_span is not None and hip_span > 1e-6 and all(vis[idx] > 0.35 for idx in [L_HIP, R_HIP]):
        hip_level = abs(float(xyz[L_HIP][1] - xyz[R_HIP][1])) / hip_span
        hip_depth = abs(float(xyz[L_HIP][2] - xyz[R_HIP][2])) / hip_span
    if shoulder_span is not None and shoulder_span > 1e-6 and all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]):
        torso_twist = abs(
            (float(xyz[L_SHOULDER][2] - xyz[R_SHOULDER][2]))
            - (float(xyz[L_HIP][2] - xyz[R_HIP][2]))
        ) / shoulder_span
    if torso_len is not None and torso_len > 1e-6 and vis[NOSE] > 0.35 and all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER]):
        shoulder_mid = (xyz[L_SHOULDER] + xyz[R_SHOULDER]) / 2.0
        head_torso_ratio = float(np.linalg.norm(xyz[NOSE] - shoulder_mid)) / torso_len
    if torso_len is not None and torso_len > 1e-6 and leg_len is not None:
        leg_ratio = leg_len / torso_len

    feature_specs = [
        ("shoulder_span_world", shoulder_span, 0.32, 0.10),
        ("hip_span_world", hip_span, 0.24, 0.08),
        ("torso_len_world", torso_len, 0.42, 0.10),
        ("leg_ratio_world", leg_ratio, 2.25, 0.45),
        ("head_torso_ratio_world", head_torso_ratio, 0.45, 0.16),
        ("thigh_balance_world", thigh_balance, 0.95, 0.08),
        ("calf_balance_world", calf_balance, 0.95, 0.08),
        ("foot_balance_world", foot_balance, 0.94, 0.10),
        ("shoulder_level_world", shoulder_level, 0.02, 0.06),
        ("hip_level_world", hip_level, 0.02, 0.06),
        ("shoulder_depth_world", shoulder_depth, 0.08, 0.08),
        ("hip_depth_world", hip_depth, 0.08, 0.08),
        ("torso_twist_world", torso_twist, 0.10, 0.10),
    ]
    values: List[float] = []
    ready = 0
    for _, raw_value, center, scale in feature_specs:
        if raw_value is None:
            values.append(0.0)
            continue
        values.append((float(raw_value) - float(center)) / max(1e-6, float(scale)))
        ready += 1
    signature = _normalize_vector(values) if ready >= 6 else None
    return {"signature": signature, "ready_features": ready, "feature_count": len(feature_specs)}


def build_absolute_master_reference(runtime: RuntimeContext, anchors: AnchorSet) -> Dict[str, Any]:
    full_master_path = _registry_ref_path(runtime, "FULL_BODY_MASTER")
    pose = _match_pose_anchor_by_path(anchors, full_master_path)
    if pose is None:
        return {
            "full_master_path": str(full_master_path) if full_master_path else None,
            "body_signature": None,
            "depth_signature": None,
            "world3d_signature": None,
            "available": False,
        }

    constitution_metrics = {
        "waist_to_torso_ratio": None,
        "hip_to_torso_ratio": None,
        "confidence": float(getattr(pose, "confidence_full", 0.0) or 0.0),
    }
    empty_face = FaceFeat(ok=False)
    depth_3d_metrics = extract_depth_3d_lite_metrics(
        empty_face,
        pose,
        view_bucket="front",
        yaw_proxy=0.0,
        body_yaw_deg=0.0,
        view_lane_detail="front",
    )
    body_identity = _build_body_identity_signature(pose, constitution_metrics, depth_3d_metrics)
    depth_identity = _build_depth_identity_signature(pose, depth_3d_metrics)
    world3d_identity = _build_world3d_identity_signature(pose)
    return {
        "full_master_path": str(full_master_path) if full_master_path else None,
        "body_signature": body_identity.get("signature"),
        "depth_signature": depth_identity.get("signature"),
        "world3d_signature": world3d_identity.get("signature"),
        "available": any(
            ref is not None
            for ref in [
                body_identity.get("signature"),
                depth_identity.get("signature"),
                world3d_identity.get("signature"),
            ]
        ),
    }


def build_face_drift_diagnostics(
    face_score: Optional[float],
    face_conf: Optional[float],
    face_debug: Dict[str, Any],
    reasons: Sequence[str],
    *,
    lane_family: Optional[str] = None,
    quality_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    branch_key = "flipped" if bool(face_debug.get("flip_canonicalized")) and isinstance(face_debug.get("flipped"), dict) else "original"
    branch = face_debug.get(branch_key) or face_debug.get("original") or {}
    anchor_scores = list(branch.get("anchor_scores") or [])
    anchor_row = anchor_scores[0] if anchor_scores else {}
    metrics = anchor_row.get("identity_metrics") or {}
    embedding = metrics.get("embedding")
    geom = metrics.get("geom")
    texture = _weighted_mean(
        [
            (metrics.get("hog"), 0.4),
            (metrics.get("lbp"), 0.3),
            (metrics.get("ssim"), 0.3),
        ]
    )
    quality = quality_debug or {}
    candidate_l = _float_or_none(quality.get("candidate_face_L"))
    tone_l_mean = _float_or_none(quality.get("tone_face_L_mean_bucket"))
    candidate_lap = _float_or_none(quality.get("candidate_face_lap_var"))
    anchor_lap = _float_or_none(quality.get("anchor_face_lap_mean_bucket"))
    candidate_hf = _float_or_none(quality.get("candidate_face_hf_energy"))
    anchor_hf = _float_or_none(quality.get("anchor_face_hf_mean_bucket"))

    issues: List[str] = []
    bbox_area_ratio = branch.get("bbox_area_ratio")
    if isinstance(bbox_area_ratio, (int, float)) and float(bbox_area_ratio) < 0.01:
        issues.append("FACE_SIGNAL_TOO_SMALL")
    if isinstance(face_conf, (int, float)) and float(face_conf) < 0.12:
        issues.append("FACE_SIGNAL_CONF_WEAK")
    if isinstance(embedding, (int, float)) and float(embedding) < 0.18:
        issues.append("FACE_EMBEDDING_ALIGNMENT_WEAK")
    if isinstance(geom, (int, float)) and float(geom) < 0.72:
        issues.append("FACE_GEOMETRY_ALIGNMENT_WEAK")
    if isinstance(texture, (int, float)) and float(texture) < 0.72:
        issues.append("FACE_TEXTURE_ALIGNMENT_WEAK")
    if tone_l_mean is not None and candidate_l is not None and abs(candidate_l - tone_l_mean) >= 9.0:
        issues.append("FACE_TONE_EXPOSURE_DRIFT")
    if candidate_lap is not None and (candidate_lap < 11.0 or (anchor_lap is not None and candidate_lap < (anchor_lap * 0.84))):
        issues.append("FACE_SOFTNESS_RISK")
    if candidate_hf is not None and (candidate_hf < 1.15 or (anchor_hf is not None and candidate_hf < (anchor_hf * 0.84))):
        issues.append("FACE_MICROTEXTURE_RISK")
    if (
        "FACE_SOFTNESS_RISK" in issues
        and "FACE_MICROTEXTURE_RISK" in issues
        and (
            ("FACE_BRIGHTER_THAN_TONE_ANCHOR" in reasons)
            or (tone_l_mean is not None and candidate_l is not None and candidate_l > (tone_l_mean + 5.0))
        )
    ):
        issues.append("FACE_AGE_SOFTENING_RISK")
    if lane_family == "side" and isinstance(geom, (int, float)) and float(geom) < 0.80:
        issues.append("FACE_PROFILE_CONTOUR_AMBIGUOUS")
    if lane_family == "three_quarter" and isinstance(geom, (int, float)) and float(geom) < 0.76:
        issues.append("FACE_3Q_GEOMETRY_AMBIGUOUS")
    for reason in reasons:
        if reason in {
            "FACE_LOW_CONFIDENCE",
            "FACE_LOW_CONF_NEEDS_REVIEW",
            "FACE_TOO_SMALL",
            "FACE_DARKER_THAN_TONE_ANCHOR",
            "FACE_BRIGHTER_THAN_TONE_ANCHOR",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_LOWER_TEXTURE_THAN_ANCHOR",
        } and reason not in issues:
            issues.append(reason)

    primary = "stable"
    if "FACE_SIGNAL_TOO_SMALL" in issues or "FACE_SIGNAL_CONF_WEAK" in issues:
        primary = "low_signal"
    elif "FACE_AGE_SOFTENING_RISK" in issues:
        primary = "age_softening_risk"
    elif "FACE_EMBEDDING_ALIGNMENT_WEAK" in issues:
        primary = "embedding_drift"
    elif "FACE_TONE_EXPOSURE_DRIFT" in issues:
        primary = "tone_drift"
    elif "FACE_GEOMETRY_ALIGNMENT_WEAK" in issues:
        primary = "geometry_drift"
    elif "FACE_PROFILE_CONTOUR_AMBIGUOUS" in issues or "FACE_3Q_GEOMETRY_AMBIGUOUS" in issues:
        primary = "contour_ambiguity"
    elif "FACE_TEXTURE_ALIGNMENT_WEAK" in issues:
        primary = "texture_drift"
    elif isinstance(face_score, (int, float)) and float(face_score) < 0.28:
        primary = "weak_master_match"

    manual_focus: List[str] = []
    manual_review_prompts: List[str] = []
    if primary == "low_signal":
        manual_focus.extend(
            [
                "do not trust low-confidence face score alone",
                "check brow-eye exposure, nose-lip-chin contour, and jawline continuity",
            ]
        )
        manual_review_prompts.append(
            "Face signal is weak. Judge identity from stable contour cues instead of relying on raw face score."
        )
    if primary in {"geometry_drift", "contour_ambiguity"}:
        manual_focus.extend(
            [
                "check face shape, cheek fullness, and eye spacing impression",
                "verify nasal bridge / jawline contour consistency",
            ]
        )
        manual_review_prompts.append(
            "Face geometry is unstable. Compare overall face shape before trusting ranking deltas."
        )
    if primary in {"texture_drift", "age_softening_risk"}:
        manual_focus.extend(
            [
                "check over-smoothing, skin texture, and age impression",
                "watch for beauty-filter softness changing identity impression",
            ]
        )
        manual_review_prompts.append(
            "Texture/softness drift is present. Verify that skin smoothing is not making the face look younger or more generic."
        )
    if primary == "tone_drift":
        manual_focus.append("check exposure and facial tone drift before judging similarity")
        manual_review_prompts.append(
            "Tone drift is strong enough to bias similarity. Compare structure first, then tone."
        )
    if lane_family == "side":
        manual_focus.extend(
            [
                "check profile contour, jawline turn, and eye exposure",
                "verify nose bridge and lip-chin silhouette against the batch top candidates",
            ]
        )
        manual_review_prompts.append(
            "For side views, prioritize profile contour and jawline stability over frontal face similarity."
        )
    elif lane_family == "three_quarter":
        manual_focus.extend(
            [
                "check cheek volume, eye spacing impression, and age drift",
                "verify that 3/4 turn does not narrow or widen the face unnaturally",
            ]
        )
        manual_review_prompts.append(
            "For three-quarter views, compare age impression and cheek shape before trusting a small ranking gap."
        )

    return {
        "selected_branch": branch_key,
        "anchor_path": anchor_row.get("anchor_path"),
        "face_score": _round_or_none(face_score),
        "face_confidence": _round_or_none(face_conf),
        "embedding_alignment": _round_or_none(embedding),
        "geometry_alignment": _round_or_none(geom),
        "texture_alignment": _round_or_none(texture),
        "bbox_area_ratio": _round_or_none(bbox_area_ratio, digits=6),
        "candidate_face_luma": _round_or_none(candidate_l, digits=3),
        "anchor_face_luma_mean": _round_or_none(tone_l_mean, digits=3),
        "candidate_face_lap_var": _round_or_none(candidate_lap, digits=3),
        "anchor_face_lap_mean": _round_or_none(anchor_lap, digits=3),
        "candidate_face_hf_energy": _round_or_none(candidate_hf, digits=3),
        "anchor_face_hf_mean": _round_or_none(anchor_hf, digits=3),
        "primary_bottleneck": primary,
        "issues": issues[:10],
        "manual_focus": _dedupe_keep_order(manual_focus, limit=5),
        "manual_review_prompts": _dedupe_keep_order(manual_review_prompts, limit=4),
    }


def build_master_consistency_card(
    scores: Dict[str, Any],
    confidences: Dict[str, Any],
    face_debug: Dict[str, Any],
    reasons: Sequence[str],
    view_lane: Optional[str],
    view_lane_detail: Optional[str],
    lane_strictness_score: Optional[float],
    body_identity: Dict[str, Any],
    depth_identity: Dict[str, Any],
    world3d_identity: Dict[str, Any],
    master_reference: Dict[str, Any],
    *,
    quality_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lane_detail = str(view_lane_detail or view_lane or "").strip()
    if lane_detail.startswith("strict_side_90") or lane_detail.startswith("side_like"):
        lane_family = "side"
    elif lane_detail in {"strict_back_180", "back_like"}:
        lane_family = "back"
    elif lane_detail in {"front", "three_quarter"}:
        lane_family = lane_detail
    else:
        lane_family = str(view_lane or "front")
    face_drift = build_face_drift_diagnostics(
        scores.get("face"),
        confidences.get("face"),
        face_debug,
        reasons,
        lane_family=lane_family,
        quality_debug=quality_debug,
    )

    body_signature_alignment = _cosine(_normalize_vector(body_identity.get("signature")), master_reference.get("body_signature"))
    depth_signature_alignment = _cosine(_normalize_vector(depth_identity.get("signature")), master_reference.get("depth_signature"))
    world3d_signature_alignment = _cosine(_normalize_vector(world3d_identity.get("signature")), master_reference.get("world3d_signature"))

    if lane_family == "front":
        body_direct = _weighted_mean([(scores.get("full"), 0.55), (scores.get("constitution"), 0.30), (scores.get("upper"), 0.15)])
        hybrid_weights = {"face": 0.48, "body": 0.34, "world3d": 0.18}
    elif lane_family == "three_quarter":
        body_direct = _weighted_mean([(scores.get("full"), 0.48), (scores.get("constitution"), 0.32), (scores.get("upper"), 0.20)])
        hybrid_weights = {"face": 0.38, "body": 0.36, "world3d": 0.26}
    elif lane_family == "side":
        body_direct = _weighted_mean([(scores.get("full"), 0.36), (scores.get("constitution"), 0.42), (scores.get("upper"), 0.22)])
        hybrid_weights = {"face": 0.18, "body": 0.46, "world3d": 0.36}
    else:
        body_direct = _weighted_mean([(scores.get("full"), 0.42), (scores.get("constitution"), 0.38), (scores.get("upper"), 0.20)])
        hybrid_weights = {"face": 0.0, "body": 0.54, "world3d": 0.46}

    body_master_alignment = _weighted_mean(
        [
            (body_direct, 0.55),
            (body_signature_alignment, 0.30),
            (depth_signature_alignment, 0.15),
        ]
    )
    world3d_master_alignment = _weighted_mean(
        [
            (world3d_signature_alignment, 0.72),
            (depth_signature_alignment, 0.12),
            (scores.get("depth_3d"), 0.16),
        ]
    )
    surface_requested = str(face_debug.get("view_surface_requested") or "").strip()
    surface_used = str(face_debug.get("view_surface_used") or "").strip()
    surface_match = 1.0 if surface_requested == surface_used else 0.58
    lane_validity = _weighted_mean(
        [
            (lane_strictness_score, 0.78),
            (surface_match, 0.22),
        ]
    )
    hybrid_master_alignment = _weighted_mean(
        [
            (scores.get("face"), hybrid_weights["face"]),
            (body_master_alignment, hybrid_weights["body"]),
            (world3d_master_alignment, hybrid_weights["world3d"]),
        ]
    )

    highlights: List[str] = []
    cautions: List[str] = []
    manual_focus: List[str] = []
    manual_review_prompts: List[str] = list(face_drift.get("manual_review_prompts") or [])
    if isinstance(body_master_alignment, (int, float)) and float(body_master_alignment) >= 0.84:
        highlights.append("body/master alignment is stable")
    if isinstance(world3d_master_alignment, (int, float)) and float(world3d_master_alignment) >= 0.86:
        highlights.append("world3d structure stays close to the master")
    if isinstance(scores.get("face"), (int, float)) and float(scores.get("face")) >= 0.70:
        highlights.append("face stays close to the absolute master")
    if isinstance(lane_validity, (int, float)) and float(lane_validity) >= 0.80:
        highlights.append("current lane geometry is clean enough")

    if face_drift.get("primary_bottleneck") != "stable":
        cautions.append(f"face drift bottleneck: {face_drift.get('primary_bottleneck')}")
    if surface_requested and surface_used and surface_requested != surface_used:
        cautions.append(f"scoring surface still falls back to {surface_used}")
    if isinstance(lane_validity, (int, float)) and float(lane_validity) < 0.72:
        cautions.append("lane strictness is still below the review-safe range")
    if isinstance(hybrid_master_alignment, (int, float)) and float(hybrid_master_alignment) < 0.76:
        cautions.append("hybrid master consistency is still weak")

    if lane_family == "front":
        manual_focus.extend(
            [
                "check age impression, eye spacing impression, and mouth shape first",
                "use body/world3d only as support if the face gap is very small",
            ]
        )
    elif lane_family == "three_quarter":
        manual_focus.extend(
            [
                "check cheek fullness, eye spacing impression, and facial turn naturalness",
                "compare 3/4 turn against the master with age impression in mind",
            ]
        )
        manual_review_prompts.append(
            "Three-quarter view: if the top two are close, decide with cheek shape, eye spacing impression, and age drift."
        )
    elif lane_family == "side":
        manual_focus.extend(
            [
                "treat this as side-view advisory evidence, not front-core admission evidence",
                "check jawline, nose bridge, lip-chin contour, and neck-head connection",
            ]
        )
        manual_review_prompts.append(
            "Side view: judge profile contour and body continuity first; do not over-trust frontal face similarity."
        )
    else:
        manual_focus.extend(
            [
                "treat this as posterior/body evidence, not frontal identity proof",
                "check shoulder-pelvis axis and posterior contour before face-related cues",
            ]
        )
        manual_review_prompts.append(
            "Back view: use posterior structure as the main evidence and keep identity interpretation conservative."
        )

    manual_focus.extend(list(face_drift.get("manual_focus") or []))
    if isinstance(hybrid_master_alignment, (int, float)) and float(hybrid_master_alignment) < 0.76:
        manual_focus.append("top candidates are close enough that GPT/human should compare face and contour side by side")

    if lane_family in {"side", "back"}:
        advisory_status = "shadow_review_only"
    elif isinstance(hybrid_master_alignment, (int, float)) and isinstance(lane_validity, (int, float)) and float(hybrid_master_alignment) >= 0.84 and float(lane_validity) >= 0.80:
        advisory_status = "strong_manual_match"
    elif isinstance(hybrid_master_alignment, (int, float)) and float(hybrid_master_alignment) >= 0.78:
        advisory_status = "manual_review_match"
    else:
        advisory_status = "manual_caution"

    return {
        "lane_family": lane_family,
        "face_master_alignment": _round_or_none(scores.get("face")),
        "face_master_confidence": _round_or_none(confidences.get("face")),
        "body_master_alignment": _round_or_none(body_master_alignment),
        "world3d_master_alignment": _round_or_none(world3d_master_alignment),
        "hybrid_master_alignment": _round_or_none(hybrid_master_alignment),
        "lane_validity": _round_or_none(lane_validity),
        "view_surface_requested": surface_requested,
        "view_surface_used": surface_used,
        "face_drift": face_drift,
        "advisory_status": advisory_status,
        "highlights": highlights[:4],
        "cautions": cautions[:4],
        "manual_focus": _dedupe_keep_order(manual_focus, limit=6),
        "manual_review_prompts": _dedupe_keep_order(manual_review_prompts, limit=5),
    }
