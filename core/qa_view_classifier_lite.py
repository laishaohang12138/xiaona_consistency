from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .qa_runtime import FaceFeat, PoseFeat
from .qa_utils import canonicalize_view_lane, clamp, estimate_view_bucket_and_side
from .qa_view_router import (
    _classify_lane_detail,
    _estimate_pose_lane_scores,
    _head_skin_ratio,
    _mask_symmetry_score,
    _subject_bbox,
)

_PROVIDER_NAME = "view_classifier_lite"
_PROVIDER_FAMILY = "view_classifier"
_PROVIDER_VERSION = "view_classifier_lite_shadow_v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array([float(scores[key]) for key in scores], dtype=np.float32)
    values -= float(np.max(values))
    exp_values = np.exp(values)
    denom = float(np.sum(exp_values))
    if denom <= 1e-8:
        return {key: 0.0 for key in scores}
    return {key: float(exp_values[idx] / denom) for idx, key in enumerate(scores)}


def _silhouette_aspect_ratio(mask_u8: Optional[np.ndarray]) -> Optional[float]:
    bbox = _subject_bbox(mask_u8)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w / h)


class ViewClassifierLiteProvider:
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": self.provider_version,
            "device": "cpu",
            "mode": "shadow_only",
            "reason": None,
        }

    def classify_view_lane(
        self,
        img_bgr: Optional[np.ndarray],
        *,
        face_feat: Optional[FaceFeat] = None,
        pose_feat: Optional[PoseFeat] = None,
        subject_mask: Optional[np.ndarray] = None,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        del img_bgr
        face_feat = face_feat or FaceFeat()
        pose_feat = pose_feat or PoseFeat()

        face_bucket = "unknown"
        face_side = "unknown"
        face_yaw_proxy = 0.0
        face_signal_available = bool(
            face_feat.ok
            and face_feat.kps5 is not None
            and _safe_float(getattr(face_feat, "confidence", 0.0), 0.0) >= 0.08
        )
        if face_signal_available:
            face_bucket, face_side, face_yaw_proxy = estimate_view_bucket_and_side(face_feat)
        face_lane = canonicalize_view_lane(face_feat, face_bucket) if face_signal_available else "unknown"

        pose_bucket, pose_profile_strength, pose_frontal_strength = _estimate_pose_lane_scores(pose_feat)
        mask_symmetry = _mask_symmetry_score(subject_mask)
        head_skin_ratio = _head_skin_ratio(subject_mask, skin_mask)
        silhouette_aspect_ratio = _silhouette_aspect_ratio(subject_mask)

        scores: Dict[str, float] = {
            "front": 0.0,
            "three_quarter": 0.0,
            "side_90": 0.0,
            "back_180": 0.0,
        }
        reasons = []

        face_conf = clamp(_safe_float(getattr(face_feat, "confidence", 0.0), 0.0), 0.0, 1.0)
        if face_lane == "front":
            scores["front"] += 0.62 * face_conf
            scores["three_quarter"] += 0.14 * face_conf
            reasons.append("VC_FACE_FRONT")
        elif face_lane == "three_quarter":
            scores["three_quarter"] += 0.64 * face_conf
            scores["front"] += 0.16 * face_conf
            scores["side_90"] += 0.14 * face_conf
            reasons.append("VC_FACE_THREE_QUARTER")
        elif face_lane == "side_90":
            scores["side_90"] += 0.70 * face_conf
            scores["three_quarter"] += 0.12 * face_conf
            reasons.append("VC_FACE_SIDE")
        elif face_signal_available:
            reasons.append("VC_FACE_SIGNAL_WEAK")
        else:
            reasons.append("VC_FACE_UNAVAILABLE")

        if pose_bucket == "side_like":
            scores["side_90"] += 0.36 * pose_profile_strength
            scores["three_quarter"] += 0.08 * pose_profile_strength
            reasons.append("VC_POSE_SIDE")
        elif pose_bucket == "frontal_like":
            scores["front"] += 0.22 * pose_frontal_strength
            scores["back_180"] += 0.18 * pose_frontal_strength
            scores["three_quarter"] += 0.06 * pose_frontal_strength
            reasons.append("VC_POSE_FRONTAL")
        else:
            reasons.append("VC_POSE_UNAVAILABLE")

        if mask_symmetry is not None:
            if mask_symmetry >= 0.74:
                symmetry_boost = clamp((mask_symmetry - 0.74) / 0.24, 0.0, 1.0)
                scores["back_180"] += 0.24 * symmetry_boost
                scores["front"] += 0.08 * symmetry_boost
                reasons.append("VC_MASK_HIGH_SYMMETRY")
            elif mask_symmetry <= 0.58:
                asymmetry_boost = clamp((0.58 - mask_symmetry) / 0.24, 0.0, 1.0)
                scores["side_90"] += 0.20 * asymmetry_boost
                reasons.append("VC_MASK_PROFILE_ASYMMETRY")

        if head_skin_ratio is not None:
            if head_skin_ratio >= 0.085:
                scores["front"] += 0.18
                scores["three_quarter"] += 0.16
                scores["side_90"] += 0.05
                reasons.append("VC_HEAD_SKIN_VISIBLE")
            elif head_skin_ratio <= 0.020:
                scores["back_180"] += 0.24
                reasons.append("VC_HEAD_SKIN_ABSENT")

        if silhouette_aspect_ratio is not None:
            if silhouette_aspect_ratio <= 0.34:
                scores["side_90"] += 0.06
                reasons.append("VC_SILHOUETTE_NARROW")
            elif silhouette_aspect_ratio >= 0.46 and (head_skin_ratio is None or head_skin_ratio <= 0.05):
                scores["back_180"] += 0.04
                reasons.append("VC_SILHOUETTE_WIDE")

        if (not face_signal_available) and pose_frontal_strength >= 0.55 and (head_skin_ratio is None or head_skin_ratio <= 0.020):
            scores["back_180"] += 0.20
            reasons.append("VC_NO_FACE_BACK_HINT")
        if (not face_signal_available) and pose_profile_strength >= 0.60:
            scores["side_90"] += 0.16
            reasons.append("VC_NO_FACE_SIDE_HINT")

        positive_scores = {key: value for key, value in scores.items() if value > 0.0}
        evidence_bits = 0
        evidence_bits += 1 if face_signal_available else 0
        evidence_bits += 1 if pose_bucket != "unknown" else 0
        evidence_bits += 1 if mask_symmetry is not None else 0
        evidence_bits += 1 if head_skin_ratio is not None else 0
        evidence_coverage = clamp(evidence_bits / 4.0, 0.0, 1.0)

        if not positive_scores:
            return {
                "enabled": True,
                "provider_name": self.provider_name,
                "provider_family": self.provider_family,
                "provider_version": self.provider_version,
                "model_id": self.provider_version,
                "device": "cpu",
                "mode": "shadow_only",
                "lane": "unknown",
                "lane_detail": "unknown",
                "confidence": 0.0,
                "lane_detail_confidence": 0.0,
                "lane_strictness_score": 0.0,
                "decision_margin": 0.0,
                "evidence_coverage": evidence_coverage,
                "body_yaw_deg": 0.0,
                "face_bucket": face_bucket,
                "face_side": face_side,
                "face_yaw_proxy": round(float(face_yaw_proxy), 6),
                "pose_bucket": pose_bucket,
                "pose_profile_strength": round(float(pose_profile_strength), 6),
                "pose_frontal_strength": round(float(pose_frontal_strength), 6),
                "mask_symmetry": None if mask_symmetry is None else round(float(mask_symmetry), 6),
                "head_skin_ratio": None if head_skin_ratio is None else round(float(head_skin_ratio), 6),
                "silhouette_aspect_ratio": None if silhouette_aspect_ratio is None else round(float(silhouette_aspect_ratio), 6),
                "lane_probs": {key: 0.0 for key in scores},
                "reasons": reasons + ["VC_NO_SIGNAL"],
            }

        ranked = sorted(positive_scores.items(), key=lambda item: item[1], reverse=True)
        best_lane, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        decision_margin = clamp(best_score - second_score, 0.0, 1.0)
        lane_probs = _softmax(scores)
        confidence = clamp(0.58 * lane_probs.get(best_lane, 0.0) + 0.24 * decision_margin + 0.18 * evidence_coverage, 0.0, 1.0)

        lane_to_yaw = {
            "front": 0.0,
            "three_quarter": 45.0,
            "side_90": 90.0,
            "back_180": 180.0,
        }
        score_sum = float(sum(positive_scores.values()))
        body_yaw = float(sum(lane_to_yaw[key] * value for key, value in positive_scores.items()) / max(1e-6, score_sum))
        lane_detail, lane_detail_confidence, lane_strictness_score = _classify_lane_detail(
            lane=best_lane,
            confidence=confidence,
            face_side=face_side,
            body_yaw_deg=body_yaw,
            pose_profile_strength=pose_profile_strength,
            pose_frontal_strength=pose_frontal_strength,
            mask_symmetry=mask_symmetry,
            head_skin_ratio=head_skin_ratio,
            face_signal_available=face_signal_available,
        )

        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": self.provider_version,
            "device": "cpu",
            "mode": "shadow_only",
            "lane": best_lane,
            "lane_detail": lane_detail,
            "confidence": round(float(confidence), 6),
            "lane_detail_confidence": round(float(lane_detail_confidence), 6),
            "lane_strictness_score": round(float(lane_strictness_score), 6),
            "decision_margin": round(float(decision_margin), 6),
            "evidence_coverage": round(float(evidence_coverage), 6),
            "body_yaw_deg": round(float(body_yaw), 3),
            "face_bucket": face_bucket,
            "face_side": face_side,
            "face_yaw_proxy": round(float(face_yaw_proxy), 6),
            "pose_bucket": pose_bucket,
            "pose_profile_strength": round(float(pose_profile_strength), 6),
            "pose_frontal_strength": round(float(pose_frontal_strength), 6),
            "mask_symmetry": None if mask_symmetry is None else round(float(mask_symmetry), 6),
            "head_skin_ratio": None if head_skin_ratio is None else round(float(head_skin_ratio), 6),
            "silhouette_aspect_ratio": None if silhouette_aspect_ratio is None else round(float(silhouette_aspect_ratio), 6),
            "lane_probs": {key: round(float(lane_probs.get(key, 0.0)), 6) for key in ["front", "three_quarter", "side_90", "back_180"]},
            "reasons": reasons[:12],
        }
