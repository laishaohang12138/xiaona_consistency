from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .qa_runtime import FaceFeat, PoseFeat, RuntimeContext
from .qa_utils import canonicalize_view_lane, clamp, estimate_view_bucket_and_side


@dataclass
class ViewRouteResult:
    lane: str = "unknown"
    confidence: float = 0.0
    source: str = "none"
    body_yaw_deg: float = 0.0
    face_bucket: str = "unknown"
    face_side: str = "unknown"
    face_yaw_proxy: float = 0.0
    pose_bucket: str = "unknown"
    pose_profile_strength: float = 0.0
    pose_frontal_strength: float = 0.0
    mask_symmetry: Optional[float] = None
    head_skin_ratio: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    lane_scores: Dict[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["lane_scores"] = {
            key: round(float(value), 6) for key, value in sorted(self.lane_scores.items())
        }
        payload["confidence"] = round(float(self.confidence), 6)
        payload["body_yaw_deg"] = round(float(self.body_yaw_deg), 3)
        payload["pose_profile_strength"] = round(float(self.pose_profile_strength), 6)
        payload["pose_frontal_strength"] = round(float(self.pose_frontal_strength), 6)
        if self.mask_symmetry is not None:
            payload["mask_symmetry"] = round(float(self.mask_symmetry), 6)
        if self.head_skin_ratio is not None:
            payload["head_skin_ratio"] = round(float(self.head_skin_ratio), 6)
        payload["face_yaw_proxy"] = round(float(self.face_yaw_proxy), 6)
        return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _linear_map(x: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return float((x - low) / (high - low))


def _estimate_pose_lane_scores(pose_feat: PoseFeat) -> Tuple[str, float, float]:
    if not pose_feat.ok:
        return "unknown", 0.0, 0.0

    upper = pose_feat.upper_geom if isinstance(pose_feat.upper_geom, dict) else {}
    sw = upper.get("shoulder_width_norm", None)
    hw = upper.get("hip_width_norm", None)
    torso = upper.get("torso_compactness", None)

    profile_parts: List[float] = []
    frontal_parts: List[float] = []

    if sw is not None:
        sw = _safe_float(sw, 0.0)
        profile_parts.append(_linear_map(0.235 - sw, 0.0, 0.070))
        frontal_parts.append(_linear_map(sw, 0.18, 0.27))
    if hw is not None:
        hw = _safe_float(hw, 0.0)
        profile_parts.append(_linear_map(0.155 - hw, 0.0, 0.060))
        frontal_parts.append(_linear_map(hw, 0.10, 0.19))
    if torso is not None:
        torso = _safe_float(torso, 0.0)
        # Side-view torsos are usually visually tighter in the current MediaPipe geometry space.
        profile_parts.append(_linear_map(0.95 - torso, 0.0, 0.40))
        frontal_parts.append(_linear_map(torso, 0.72, 1.18))

    if len(profile_parts) == 0 and len(frontal_parts) == 0:
        return "unknown", 0.0, 0.0

    profile_strength = float(np.mean(np.array(profile_parts, dtype=np.float32))) if profile_parts else 0.0
    frontal_strength = float(np.mean(np.array(frontal_parts, dtype=np.float32))) if frontal_parts else 0.0
    pose_bucket = "side_like" if profile_strength > (frontal_strength + 0.08) else "frontal_like"
    return pose_bucket, clamp(profile_strength, 0.0, 1.0), clamp(frontal_strength, 0.0, 1.0)


def _subject_bbox(mask_u8: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if mask_u8 is None or mask_u8.size == 0:
        return None
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 24 or len(ys) < 24:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mask_symmetry_score(mask_u8: Optional[np.ndarray]) -> Optional[float]:
    bbox = _subject_bbox(mask_u8)
    if bbox is None or mask_u8 is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = mask_u8[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[1] < 12:
        return None
    mid = crop.shape[1] // 2
    left = crop[:, :mid]
    right = crop[:, crop.shape[1] - mid :]
    if left.size == 0 or right.size == 0:
        return None
    right_flipped = np.fliplr(right)
    overlap = np.logical_and(left > 0, right_flipped > 0).sum()
    union = np.logical_or(left > 0, right_flipped > 0).sum()
    if union <= 0:
        return None
    return float(overlap / union)


def _head_skin_ratio(mask_u8: Optional[np.ndarray], skin_u8: Optional[np.ndarray]) -> Optional[float]:
    bbox = _subject_bbox(mask_u8)
    if bbox is None or mask_u8 is None or skin_u8 is None:
        return None
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w < 18 or h < 32:
        return None

    hx1 = x1 + int(round(w * 0.32))
    hx2 = x2 - int(round(w * 0.32))
    hy1 = y1
    hy2 = y1 + max(6, int(round(h * 0.22)))
    if hx2 <= hx1 or hy2 <= hy1:
        return None

    head_subject = mask_u8[hy1:hy2, hx1:hx2] > 0
    if int(head_subject.sum()) < 16:
        return 0.0
    head_skin = skin_u8[hy1:hy2, hx1:hx2] > 0
    return float(np.logical_and(head_subject, head_skin).sum() / max(1, int(head_subject.sum())))


def route_view_lane(
    runtime: RuntimeContext,
    img_bgr: Optional[np.ndarray],
    face_feat: FaceFeat,
    pose_feat: PoseFeat,
) -> ViewRouteResult:
    face_bucket = "unknown"
    face_side = "unknown"
    face_yaw_proxy = 0.0
    legacy_lane = "unknown"
    if face_feat.ok and face_feat.kps5 is not None:
        face_bucket, face_side, face_yaw_proxy = estimate_view_bucket_and_side(face_feat)
        legacy_lane = canonicalize_view_lane(face_feat, face_bucket)

    pose_bucket, pose_profile_strength, pose_frontal_strength = _estimate_pose_lane_scores(pose_feat)

    subject_mask = None
    skin_mask = None
    if img_bgr is not None and getattr(runtime, "providers", None) is not None:
        try:
            subject_mask = runtime.providers.get_subject_mask(
                img_bgr,
                face_feat=face_feat,
                pose_feat=pose_feat,
            )
        except Exception:
            subject_mask = None
        try:
            skin_mask = runtime.providers.get_skin_region(
                img_bgr,
                face_feat=face_feat,
                pose_feat=pose_feat,
            )
        except Exception:
            skin_mask = None

    mask_symmetry = _mask_symmetry_score(subject_mask)
    head_skin_ratio = _head_skin_ratio(subject_mask, skin_mask)

    scores: Dict[str, float] = {
        "front": 0.0,
        "three_quarter": 0.0,
        "side_90": 0.0,
        "back_180": 0.0,
    }
    reasons: List[str] = []

    if legacy_lane != "unknown":
        face_lane_conf = clamp(
            0.28
            + (0.40 * _safe_float(face_feat.confidence, 0.0))
            + (0.32 * _linear_map(_safe_float(face_feat.bbox_area_ratio, 0.0), 0.004, 0.028)),
            0.0,
            1.0,
        )
        scores[legacy_lane] += 0.60 * face_lane_conf
        reasons.append(f"FACE_ROUTE_{legacy_lane.upper()}")
        if legacy_lane == "three_quarter":
            scores["front"] += 0.06 * face_lane_conf
            scores["side_90"] += 0.10 * face_lane_conf
        elif legacy_lane == "front":
            scores["three_quarter"] += 0.08 * face_lane_conf
        elif legacy_lane == "side_90":
            scores["three_quarter"] += 0.06 * face_lane_conf
    else:
        reasons.append("FACE_ROUTE_UNAVAILABLE")

    if pose_bucket == "side_like":
        scores["side_90"] += 0.28 * pose_profile_strength
        reasons.append("POSE_SIDE_LIKE")
    elif pose_bucket == "frontal_like":
        frontal_boost = 0.22 * pose_frontal_strength
        scores["front"] += 0.55 * frontal_boost
        scores["back_180"] += 0.45 * frontal_boost
        reasons.append("POSE_FRONTAL_LIKE")
    else:
        reasons.append("POSE_ROUTE_UNAVAILABLE")

    if mask_symmetry is not None:
        if mask_symmetry >= 0.72:
            scores["back_180"] += 0.16 * mask_symmetry
            scores["front"] += 0.10 * mask_symmetry
            reasons.append("MASK_HIGH_SYMMETRY")
        elif mask_symmetry <= 0.58:
            scores["side_90"] += 0.12 * (1.0 - mask_symmetry)
            reasons.append("MASK_PROFILE_ASYMMETRY")

    if head_skin_ratio is not None:
        if head_skin_ratio >= 0.085:
            scores["front"] += 0.08
            scores["three_quarter"] += 0.08
            if pose_profile_strength >= 0.45:
                scores["side_90"] += 0.06
            reasons.append("HEAD_SKIN_VISIBLE")
        elif head_skin_ratio <= 0.020:
            scores["back_180"] += 0.18
            reasons.append("HEAD_SKIN_ABSENT")

    if (not face_feat.ok) and pose_frontal_strength >= 0.55 and (head_skin_ratio is None or head_skin_ratio <= 0.020):
        scores["back_180"] += 0.20
        reasons.append("NO_FACE_FRONTAL_BODY_BACK_HINT")

    if (not face_feat.ok) and pose_profile_strength >= 0.60:
        scores["side_90"] += 0.12
        reasons.append("NO_FACE_PROFILE_BODY_SIDE_HINT")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_lane, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    positive_scores = {key: value for key, value in scores.items() if value > 0.0}
    evidence_coverage = clamp(len(positive_scores) / 4.0, 0.0, 1.0)
    confidence = clamp((best_score - second_score) + (0.22 * evidence_coverage) + 0.18, 0.0, 1.0)

    lane_to_yaw = {
        "front": 0.0,
        "three_quarter": 45.0,
        "side_90": 90.0,
        "back_180": 180.0,
    }
    if positive_scores:
        score_sum = float(sum(positive_scores.values()))
        body_yaw = float(
            sum(lane_to_yaw[key] * value for key, value in positive_scores.items()) / max(1e-6, score_sum)
        )
    else:
        body_yaw = 0.0

    if face_feat.ok and best_lane in {"front", "three_quarter", "side_90"}:
        source = "face_pose_mask_fusion"
    elif best_lane == "back_180":
        source = "pose_mask_fusion"
    elif pose_feat.ok:
        source = "pose_fusion"
    else:
        source = "fallback_unknown"

    return ViewRouteResult(
        lane=best_lane,
        confidence=confidence,
        source=source,
        body_yaw_deg=body_yaw,
        face_bucket=face_bucket,
        face_side=face_side,
        face_yaw_proxy=face_yaw_proxy,
        pose_bucket=pose_bucket,
        pose_profile_strength=pose_profile_strength,
        pose_frontal_strength=pose_frontal_strength,
        mask_symmetry=mask_symmetry,
        head_skin_ratio=head_skin_ratio,
        reasons=reasons,
        lane_scores=scores,
    )
