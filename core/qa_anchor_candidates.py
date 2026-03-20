from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .qa_features import extract_face_feat, extract_pose_feat
from .qa_pipeline import create_runtime
from .qa_utils import image_read_bgr
from .qa_view_router import route_view_lane


@dataclass
class AnchorCandidate:
    image: str
    lane: str
    lane_side: str
    status: str
    candidate_score: float
    scores: Dict[str, float]
    confidence: Dict[str, float]
    shadow_router: Dict[str, Any]
    reasons: List[str]
    summary: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "lane": self.lane,
            "lane_side": self.lane_side,
            "status": self.status,
            "candidate_score": round(float(self.candidate_score), 6),
            "scores": {key: round(float(value), 6) for key, value in sorted(self.scores.items())},
            "confidence": {
                key: round(float(value), 6) for key, value in sorted(self.confidence.items())
            },
            "shadow_router": self.shadow_router,
            "reasons": list(self.reasons),
            "summary": self.summary,
        }


def _read_report(report_path: Path) -> Dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"qa report must decode to an object: {report_path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _recompute_shadow_router(
    *,
    runtime: Any,
    image_name: str,
    input_dir: Path,
) -> Optional[Dict[str, Any]]:
    image_path = input_dir / image_name
    if not image_path.exists():
        return None
    img = image_read_bgr(image_path, runtime.config.standardization)
    if img is None:
        return None
    face = extract_face_feat(runtime, img, image_path)
    pose = extract_pose_feat(runtime, img)
    routed = route_view_lane(runtime, img, face, pose)
    return routed.to_json_dict()


def _select_lane_for_item(
    item: Dict[str, Any],
    *,
    prefer_shadow_router: bool,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    shadow_router = (
        debug.get("view_router_v2", {}) if isinstance(debug.get("view_router_v2", {}), dict) else {}
    )
    if prefer_shadow_router and shadow_router.get("lane"):
        lane = str(shadow_router.get("lane", "unknown"))
        lane_side = str(shadow_router.get("face_side", debug.get("view_side", "unknown")))
        return lane, lane_side, shadow_router
    lane = str(debug.get("view_lane", "unknown"))
    lane_side = str(debug.get("view_side", "unknown"))
    return lane, lane_side, shadow_router or None


def _score_side90_candidate(
    *,
    item: Dict[str, Any],
    lane: str,
    shadow_router: Optional[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    scores = item.get("scores", {}) if isinstance(item.get("scores", {}), dict) else {}
    confs = item.get("confidence", {}) if isinstance(item.get("confidence", {}), dict) else {}
    debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
    constitution = (
        debug.get("constitution_metrics", {})
        if isinstance(debug.get("constitution_metrics", {}), dict)
        else {}
    )
    depth = (
        debug.get("depth_3d_metrics", {})
        if isinstance(debug.get("depth_3d_metrics", {}), dict)
        else {}
    )
    framing = (
        debug.get("candidate_pose_framing", {})
        if isinstance(debug.get("candidate_pose_framing", {}), dict)
        else {}
    )
    reasons = set(item.get("reasons", []) or [])

    base_score = (
        0.32 * _safe_float(scores.get("full", 0.0))
        + 0.18 * _safe_float(scores.get("upper", 0.0))
        + 0.18 * _safe_float(scores.get("constitution", 0.0))
        + 0.16 * _safe_float(scores.get("depth_3d", 0.0))
        + 0.08 * _safe_float(scores.get("skin", 0.0))
        + 0.08 * _safe_float(confs.get("upper", 0.0))
    )

    penalty = 0.0
    penalty_flags = [
        ("FACE_NO_RELIABLE_SIGNAL", 0.70),
        ("SKIN_UNIFORMITY_STRONG_WARN", 0.55),
        ("SKIN_LIGHTING_RISK_HIGH", 0.34),
        ("BODY_CONSTITUTION_STRONG_WARN", 0.42),
        ("BODY_CONSTITUTION_WARN", 0.14),
        ("FACE_LOW_CONFIDENCE", 0.10),
        ("FACE_LOW_CONF_NEEDS_REVIEW", 0.08),
        ("FACE_TOO_SMALL", 0.10),
        ("FACE_DARKER_THAN_TONE_ANCHOR", 0.05),
        ("FACE_SOFTER_THAN_ANCHOR", 0.05),
        ("FACE_LOWER_TEXTURE_THAN_ANCHOR", 0.04),
        ("UPPER_FAIL", 0.08),
    ]
    for flag, weight in penalty_flags:
        if flag in reasons:
            penalty += weight

    bonus = 0.0
    router_lane = None
    router_conf = 0.0
    if shadow_router is not None:
        router_lane = str(shadow_router.get("lane", ""))
        router_conf = _safe_float(shadow_router.get("confidence", 0.0))
        if router_lane == lane:
            bonus += 0.10 + (0.10 * router_conf)

    if "FEET_IN_FRAME" in reasons:
        bonus += 0.04
    if "FRAMING_OK" in reasons:
        bonus += 0.04

    final_score = base_score + bonus - penalty
    summary = {
        "base_score": round(base_score, 6),
        "bonus": round(bonus, 6),
        "penalty": round(penalty, 6),
        "router_lane": router_lane,
        "router_confidence": round(router_conf, 6),
        "constitution_score": constitution.get("body_constitution_score"),
        "depth_score": depth.get("depth_3d_score"),
        "feet_in_frame": framing.get("feet_in_frame"),
    }
    return final_score, summary


def _score_generic_candidate(
    *,
    item: Dict[str, Any],
    lane: str,
    shadow_router: Optional[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    scores = item.get("scores", {}) if isinstance(item.get("scores", {}), dict) else {}
    confs = item.get("confidence", {}) if isinstance(item.get("confidence", {}), dict) else {}
    reasons = set(item.get("reasons", []) or [])

    base_score = (
        0.35 * _safe_float(scores.get("full", 0.0))
        + 0.20 * _safe_float(scores.get("upper", 0.0))
        + 0.20 * _safe_float(scores.get("depth_3d", 0.0))
        + 0.15 * _safe_float(scores.get("constitution", 0.0))
        + 0.10 * _safe_float(confs.get("full", 0.0))
    )

    penalty = 0.0
    if "FACE_NO_RELIABLE_SIGNAL" in reasons:
        penalty += 0.60

    bonus = 0.0
    router_lane = None
    router_conf = 0.0
    if shadow_router is not None:
        router_lane = str(shadow_router.get("lane", ""))
        router_conf = _safe_float(shadow_router.get("confidence", 0.0))
        if router_lane == lane:
            bonus += 0.08 + (0.10 * router_conf)

    final_score = base_score + bonus - penalty
    return final_score, {
        "base_score": round(base_score, 6),
        "bonus": round(bonus, 6),
        "penalty": round(penalty, 6),
        "router_lane": router_lane,
        "router_confidence": round(router_conf, 6),
    }


def rank_anchor_candidates(
    *,
    report_path: Path,
    lane: str,
    top_n: int = 10,
    prefer_shadow_router: bool = True,
    recompute_shadow_router: bool = False,
    base_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = _read_report(report_path)
    report_meta = payload.get("report_meta", {}) if isinstance(payload.get("report_meta", {}), dict) else {}
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("qa report items must be a list")

    resolved_base_dir = (base_dir or report_path.parent.parent).resolve()
    resolved_input_dir = (
        input_dir.resolve()
        if input_dir is not None
        else (resolved_base_dir / "input").resolve()
    )
    runtime_for_recompute = create_runtime(resolved_base_dir) if recompute_shadow_router else None

    candidates: List[AnchorCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_name = str(item.get("image", "")).strip()
        if not image_name:
            continue

        selected_lane, lane_side, shadow_router = _select_lane_for_item(
            item,
            prefer_shadow_router=prefer_shadow_router,
        )
        if recompute_shadow_router and (shadow_router is None or not shadow_router.get("lane")):
            shadow_router = _recompute_shadow_router(
                runtime=runtime_for_recompute,
                image_name=image_name,
                input_dir=resolved_input_dir,
            )
            if prefer_shadow_router and shadow_router is not None and shadow_router.get("lane"):
                selected_lane = str(shadow_router.get("lane", selected_lane))
                lane_side = str(shadow_router.get("face_side", lane_side))

        if selected_lane != lane:
            continue

        if lane == "side_90":
            candidate_score, summary = _score_side90_candidate(
                item=item,
                lane=lane,
                shadow_router=shadow_router,
            )
        else:
            candidate_score, summary = _score_generic_candidate(
                item=item,
                lane=lane,
                shadow_router=shadow_router,
            )

        scores = item.get("scores", {}) if isinstance(item.get("scores", {}), dict) else {}
        confs = item.get("confidence", {}) if isinstance(item.get("confidence", {}), dict) else {}
        debug = item.get("debug", {}) if isinstance(item.get("debug", {}), dict) else {}
        reasons = list(item.get("reasons", []) or [])
        candidates.append(
            AnchorCandidate(
                image=image_name,
                lane=selected_lane,
                lane_side=lane_side,
                status=str(item.get("status", "")),
                candidate_score=float(candidate_score),
                scores={key: _safe_float(value) for key, value in scores.items()},
                confidence={key: _safe_float(value) for key, value in confs.items()},
                shadow_router=shadow_router or {},
                reasons=reasons,
                summary={
                    **summary,
                    "view_side": debug.get("view_side"),
                    "legacy_view_lane": debug.get("view_lane"),
                },
            )
        )

    candidates.sort(key=lambda row: row.candidate_score, reverse=True)
    by_side: Dict[str, int] = {}
    for row in candidates:
        by_side[row.lane_side] = by_side.get(row.lane_side, 0) + 1

    return {
        "report_file": str(report_path.resolve()),
        "active_profile": report_meta.get("active_profile"),
        "lane": lane,
        "num_candidates": len(candidates),
        "side_distribution": by_side,
        "top_candidates": [row.to_json_dict() for row in candidates[: max(1, int(top_n))]],
    }


__all__ = ["AnchorCandidate", "rank_anchor_candidates"]
