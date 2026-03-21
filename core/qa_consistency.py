from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .qa_runtime import FaceFeat, PoseFeat, RuntimeContext
from .qa_utils import (
    clamp,
    dedupe_keep_order,
    linear_map_to_01,
    resolve_view_scoring_candidates,
    resolve_view_scoring_surface,
    safe_float,
)


def soft_range_score(x: Optional[float], lo: float, hi: float, margin: float) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return clamp(1.0 - (lo - x) / max(1e-6, margin), 0.0, 1.0)
    return clamp(1.0 - (x - hi) / max(1e-6, margin), 0.0, 1.0)


def weighted_mean_valid(items: List[Tuple[Optional[float], float]]) -> Optional[float]:
    vals = []
    weights = []
    for value, weight in items:
        if value is None:
            continue
        vals.append(float(value))
        weights.append(float(weight))
    if len(vals) == 0:
        return None
    return float(np.average(np.array(vals, dtype=np.float32), weights=np.array(weights, dtype=np.float32)))


def _norm_xy_to_px(xy: np.ndarray, h: int, w: int) -> Tuple[int, int]:
    x = int(round(float(xy[0]) * w))
    y = int(round(float(xy[1]) * h))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return x, y


def _clip_rect(x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _rect_from_center(cx: int, cy: int, rx: int, ry: int, h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    return _clip_rect(cx - rx, cy - ry, cx + rx, cy + ry, h, w)


def _median_lab_in_region(img_bgr: np.ndarray, region_mask_u8: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img_bgr is None or region_mask_u8 is None:
        return None
    ys, xs = np.where(region_mask_u8 > 0)
    if len(xs) < 30:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    vals = lab[ys, xs]
    if vals.shape[0] < 30:
        return None
    return np.median(vals, axis=0).astype(np.float32)


def _delta_e_lab(lab1: Optional[np.ndarray], lab2: Optional[np.ndarray]) -> Optional[float]:
    if lab1 is None or lab2 is None:
        return None
    delta = lab1.astype(np.float32) - lab2.astype(np.float32)
    return float(np.sqrt(np.sum(delta * delta)))


def _delta_l_lab(lab1: Optional[np.ndarray], lab2: Optional[np.ndarray]) -> Optional[float]:
    if lab1 is None or lab2 is None:
        return None
    return float(abs(float(lab1[0]) - float(lab2[0])))


def _delta_ab_lab(lab1: Optional[np.ndarray], lab2: Optional[np.ndarray]) -> Optional[float]:
    if lab1 is None or lab2 is None:
        return None
    da = float(lab1[1]) - float(lab2[1])
    db = float(lab1[2]) - float(lab2[2])
    return float(math.sqrt(da * da + db * db))


def _mean_l_in_region(img_bgr: np.ndarray, region_mask_u8: Optional[np.ndarray]) -> Optional[float]:
    if img_bgr is None or region_mask_u8 is None:
        return None
    ys, xs = np.where(region_mask_u8 > 0)
    if len(xs) < 30:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    vals = lab[ys, xs, 0].astype(np.float32)
    if vals.shape[0] < 30:
        return None
    return float(np.mean(vals))


def _highlight_ratio_in_region(
    img_bgr: np.ndarray,
    region_mask_u8: Optional[np.ndarray],
    l_threshold: float,
) -> Optional[float]:
    if img_bgr is None or region_mask_u8 is None:
        return None
    ys, xs = np.where(region_mask_u8 > 0)
    if len(xs) < 30:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    vals = lab[ys, xs, 0].astype(np.float32)
    if vals.shape[0] < 30:
        return None
    return float(np.mean(vals >= float(l_threshold)))


def _edge_margin_ratio(rect: Optional[Tuple[int, int, int, int]], h: int, w: int) -> Optional[float]:
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    margin = float(min(x1, y1, max(0, w - x2), max(0, h - y2)))
    return float(margin / max(1.0, min(h, w)))


@dataclass
class SkinPatchStats:
    name: str
    rect: Optional[Tuple[int, int, int, int]]
    mask: Optional[np.ndarray]
    lab: Optional[np.ndarray]
    purity: float
    pixel_count: int
    edge_margin_ratio: Optional[float]
    mean_l: Optional[float]
    highlight_ratio: Optional[float] = None


def _mean_valid(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None]
    if len(vals) == 0:
        return None
    return float(np.mean(np.array(vals, dtype=np.float32)))


def _median_lab_from_stats(stats: List[SkinPatchStats]) -> Optional[np.ndarray]:
    labs = [stat.lab for stat in stats if stat.lab is not None]
    if len(labs) == 0:
        return None
    return np.median(np.stack(labs, axis=0), axis=0).astype(np.float32)


def _build_skin_patch_stats(
    img_bgr: np.ndarray,
    valid_skin: np.ndarray,
    name: str,
    rect: Optional[Tuple[int, int, int, int]],
    min_pixels: int = 30,
    min_purity: float = 0.12,
    highlight_l: Optional[float] = None,
) -> SkinPatchStats:
    h, w = valid_skin.shape[:2]
    edge_margin_ratio = _edge_margin_ratio(rect, h, w)
    if rect is None:
        return SkinPatchStats(
            name=name,
            rect=None,
            mask=None,
            lab=None,
            purity=0.0,
            pixel_count=0,
            edge_margin_ratio=edge_margin_ratio,
            mean_l=None,
            highlight_ratio=None,
        )

    x1, y1, x2, y2 = rect
    area = max(1, (x2 - x1) * (y2 - y1))
    patch_mask = np.zeros((h, w), dtype=np.uint8)
    patch_mask[y1:y2, x1:x2] = 255
    region_mask = cv2.bitwise_and(patch_mask, valid_skin)
    pixel_count = int(np.sum(region_mask > 0))
    purity = float(pixel_count / area)

    if pixel_count < min_pixels or purity < min_purity:
        return SkinPatchStats(
            name=name,
            rect=rect,
            mask=None,
            lab=None,
            purity=purity,
            pixel_count=pixel_count,
            edge_margin_ratio=edge_margin_ratio,
            mean_l=None,
            highlight_ratio=None,
        )

    mean_l = _mean_l_in_region(img_bgr, region_mask)
    highlight_ratio = None
    if highlight_l is not None:
        highlight_ratio = _highlight_ratio_in_region(img_bgr, region_mask, highlight_l)

    return SkinPatchStats(
        name=name,
        rect=rect,
        mask=region_mask,
        lab=_median_lab_in_region(img_bgr, region_mask),
        purity=purity,
        pixel_count=pixel_count,
        edge_margin_ratio=edge_margin_ratio,
        mean_l=mean_l,
        highlight_ratio=highlight_ratio,
    )


def _exp_decay_score(value: Optional[float], decay: float) -> Optional[float]:
    if value is None:
        return None
    return float(math.exp(-float(value) / max(1e-6, float(decay))))


def _row_width(mask_u8: np.ndarray, y: int) -> Optional[int]:
    h, _ = mask_u8.shape[:2]
    y = max(0, min(h - 1, int(y)))
    xs = np.where(mask_u8[y, :] > 0)[0]
    if len(xs) < 2:
        return None
    return int(xs[-1] - xs[0] + 1)


def _mask_width_at_row(mask_u8: np.ndarray, y: int, band: int = 4) -> Optional[int]:
    h, _ = mask_u8.shape[:2]
    vals: List[int] = []
    for yy in range(max(0, y - band), min(h, y + band + 1)):
        width = _row_width(mask_u8, yy)
        if width is not None:
            vals.append(int(width))
    if len(vals) == 0:
        return None
    return int(np.median(np.array(vals, dtype=np.float32)))


def _mask_width_soft_min(
    mask_u8: np.ndarray,
    y1: int,
    y2: int,
    band: int = 4,
) -> Tuple[Optional[int], Optional[int]]:
    h, _ = mask_u8.shape[:2]
    widths: List[Tuple[int, int]] = []
    for yy in range(max(0, y1), min(h, y2 + 1)):
        width = _mask_width_at_row(mask_u8, yy, band=band)
        if width is not None:
            widths.append((yy, int(width)))
    if len(widths) == 0:
        return None, None
    widths_sorted = sorted(widths, key=lambda item: item[1])
    k = max(3, min(7, len(widths_sorted)))
    chosen = widths_sorted[:k]
    waist_y = int(round(float(np.median([yy for yy, _ in chosen]))))
    waist_w = int(round(float(np.median([ww for _, ww in chosen]))))
    return waist_y, waist_w


def _resolve_scoring_view_spec(
    scoring: Dict[str, Any],
    view_bucket: str,
    view_lane_detail: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    views = scoring.get("views", {})
    if not isinstance(views, dict):
        views = {}
    requested_surface = resolve_view_scoring_surface(view_bucket, view_lane_detail=view_lane_detail)
    for view_key in resolve_view_scoring_candidates(view_bucket, view_lane_detail=view_lane_detail):
        view_spec = views.get(view_key, {})
        if isinstance(view_spec, dict) and len(view_spec) > 0:
            return view_key, view_spec
    return requested_surface, {}


def _score_from_range_spec(value: Optional[float], spec: Dict[str, Any]) -> Optional[float]:
    return soft_range_score(
        value,
        float(spec.get("lo", 0.0)),
        float(spec.get("hi", 0.0)),
        float(spec.get("margin", 1.0)),
    )


def _score_metric_from_view_or_common(
    value: Optional[float],
    metric_key: str,
    view_spec: Dict[str, Any],
    common_spec: Dict[str, Any],
) -> Optional[float]:
    spec = view_spec.get(metric_key, None)
    if not isinstance(spec, dict) or len(spec) == 0:
        spec = common_spec.get(metric_key, {})
    return _score_from_range_spec(value, spec if isinstance(spec, dict) else {})


def score_body_constitution_measurements(
    measurements: Dict[str, Any],
    scoring: Dict[str, Any],
    view_bucket: str,
    view_lane_detail: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    requested_surface = resolve_view_scoring_surface(view_bucket, view_lane_detail=view_lane_detail)
    view_key, view_spec = _resolve_scoring_view_spec(
        scoring,
        view_bucket,
        view_lane_detail=view_lane_detail,
    )
    common_spec = scoring.get("common_ranges", {})
    weights = scoring.get("weights", {})
    profile_like_family = requested_surface not in {"front", "three_quarter"}

    waist_shoulder = _score_from_range_spec(
        measurements.get("waist_to_shoulder_ratio", None),
        view_spec.get("waist_to_shoulder", {}),
    )
    chest_waist = _score_from_range_spec(
        measurements.get("chest_to_waist_ratio", None),
        view_spec.get("chest_to_waist", {}),
    )
    hip_waist = _score_from_range_spec(
        measurements.get("hip_to_waist_ratio", None),
        view_spec.get("hip_to_waist", {}),
    )
    hip_shoulder = _score_from_range_spec(
        measurements.get("hip_to_shoulder_ratio", None),
        view_spec.get("hip_to_shoulder", {}),
    )
    waist_torso = _score_from_range_spec(
        measurements.get("waist_to_torso_ratio", None),
        view_spec.get("waist_to_torso", {}),
    )
    chest_torso = _score_from_range_spec(
        measurements.get("chest_to_torso_ratio", None),
        view_spec.get("chest_to_torso", {}),
    )
    hip_torso = _score_from_range_spec(
        measurements.get("hip_to_torso_ratio", None),
        view_spec.get("hip_to_torso", {}),
    )
    thigh_calf = _score_from_range_spec(
        measurements.get("thigh_to_calf_ratio", None),
        view_spec.get("thigh_to_calf", {}),
    )

    base_terms = [
        (
            _score_from_range_spec(
                measurements.get("leg_ratio", None),
                common_spec.get("leg_ratio", {}),
            ),
            float(weights.get("leg_ratio", 0.18)),
        ),
        (
            _score_from_range_spec(
                measurements.get("waist_height_ratio", None),
                common_spec.get("waist_height_ratio", {}),
            ),
            float(weights.get("waist_height_ratio", 0.18)),
        ),
        (chest_waist, float(weights.get("chest_to_waist", 0.12))),
        (hip_waist, float(weights.get("hip_to_waist", 0.14))),
        (thigh_calf, float(weights.get("thigh_to_calf", 0.08))),
    ]

    if profile_like_family:
        body_constitution_score = weighted_mean_valid(
            base_terms
            + [
                (waist_torso, float(weights.get("waist_to_torso", 0.22))),
                (chest_torso, float(weights.get("chest_to_torso", 0.10))),
                (hip_torso, float(weights.get("hip_to_torso", 0.10))),
            ]
        )
        pelvis_compactness = hip_torso if hip_torso is not None else hip_waist
        abdomen_flatness = waist_torso if waist_torso is not None else waist_shoulder
    else:
        body_constitution_score = weighted_mean_valid(
            base_terms
            + [
                (waist_shoulder, float(weights.get("waist_to_shoulder", 0.20))),
                (hip_shoulder, float(weights.get("hip_to_shoulder", 0.10))),
            ]
        )
        pelvis_compactness = hip_shoulder
        abdomen_flatness = waist_shoulder

    return {
        "pelvis_compactness_score": pelvis_compactness,
        "abdomen_flatness_score": abdomen_flatness,
        "lower_body_slenderness_score": thigh_calf,
        "body_constitution_score": body_constitution_score,
        "view_surface_requested": requested_surface,
        "view_surface_used": view_key,
    }


def compute_body_constitution_confidence(
    scoring: Dict[str, Any],
    view_bucket: str,
    width_ready: int,
    pose_visibility: float,
    torso_fill: Optional[float],
    view_lane_detail: Optional[str] = None,
) -> float:
    _, view_spec = _resolve_scoring_view_spec(
        scoring,
        view_bucket,
        view_lane_detail=view_lane_detail,
    )
    confidence_weights = scoring.get("confidence_weights", {})
    conf = weighted_mean_valid(
        [
            (float(width_ready) / 5.0, float(confidence_weights.get("width_ready", 0.34))),
            (pose_visibility, float(confidence_weights.get("pose_visibility", 0.34))),
            (torso_fill, float(confidence_weights.get("torso_fill", 0.22))),
            (
                float(view_spec.get("view_factor", 0.75)),
                float(confidence_weights.get("view_factor", 0.10)),
            ),
        ]
    )
    return 0.0 if conf is None else float(conf)


def score_depth_3d_lite_geometry(
    upper_geom: Dict[str, float],
    full_geom: Dict[str, float],
    view_bucket: str,
    yaw_proxy: float,
    body_yaw_deg: Optional[float],
    pose_frontal_strength: Optional[float],
    lane_strictness_score: Optional[float],
    mask_symmetry: Optional[float],
    head_skin_ratio: Optional[float],
    scoring: Dict[str, Any],
    view_lane_detail: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    requested_surface = resolve_view_scoring_surface(view_bucket, view_lane_detail=view_lane_detail)
    view_key, view_spec = _resolve_scoring_view_spec(
        scoring,
        view_bucket,
        view_lane_detail=view_lane_detail,
    )
    common_spec = scoring.get("common_ranges", {})
    weights = scoring.get("weights", {})
    profile_like_family = requested_surface not in {"front", "three_quarter"}
    back_family = requested_surface in {"strict_back_180", "back_like"}
    side_family = profile_like_family and not back_family

    sw = upper_geom.get("shoulder_width_norm", None)
    hw = upper_geom.get("hip_width_norm", None)
    spine_angle = upper_geom.get("spine_angle_deg", None)
    torso_len = upper_geom.get("torso_len_norm", None)
    torso_compactness = upper_geom.get("torso_compactness", None)
    center_offset = upper_geom.get("shoulder_hip_center_offset_norm", None)
    leg_straightness = full_geom.get("leg_straightness_min_deg", None)
    ankle_gap = full_geom.get("ankle_gap_norm", None)
    shoulder_level_delta = full_geom.get("shoulder_level_delta_norm", None)
    hip_level_delta = full_geom.get("hip_level_delta_norm", None)
    thigh_balance = full_geom.get("thigh_length_balance", None)
    calf_balance = full_geom.get("calf_length_balance", None)
    lower_limb_balance = full_geom.get("lower_limb_balance", None)
    foot_balance = full_geom.get("foot_length_balance", None)

    if sw is None or hw is None:
        return {
            "torso_volume_score": None,
            "pelvis_depth_score": None,
            "fake_turn_risk": None,
            "depth_3d_score": None,
            "side_profile_score": None,
            "posterior_score": None,
            "confidence": 0.0,
        }

    hip_shoulder_ratio = float(hw / max(1e-6, sw))
    turn_value = float(body_yaw_deg) if profile_like_family and body_yaw_deg is not None else float(yaw_proxy)
    turn_spec = view_spec.get("body_yaw_deg", {}) if profile_like_family else view_spec.get("yaw", {})
    if not isinstance(turn_spec, dict) or len(turn_spec) == 0:
        turn_spec = view_spec.get("yaw", {})
    turn_score = _score_from_range_spec(turn_value, turn_spec if isinstance(turn_spec, dict) else {})
    turn_signal_body_yaw_score = turn_score
    turn_signal_pose_frontal_score = None
    turn_signal_mask_symmetry_score = None
    turn_signal_head_skin_score = None
    turn_signal_lane_strictness_score = lane_strictness_score
    posterior_raw_score = None
    posterior_gate_score = None
    shoulder_score = _score_from_range_spec(sw, view_spec.get("shoulder_width", {}))
    hip_score = _score_from_range_spec(hw, view_spec.get("hip_width", {}))
    ratio_score = _score_metric_from_view_or_common(
        hip_shoulder_ratio,
        "hip_shoulder_ratio",
        view_spec,
        common_spec,
    )
    spine_score = _score_metric_from_view_or_common(spine_angle, "spine_angle", view_spec, common_spec)
    torso_score = _score_metric_from_view_or_common(torso_len, "torso_length", view_spec, common_spec)
    torso_compactness_score = None

    side_profile_score = None
    if side_family:
        side_weights = weights.get("side_profile", {})
        side_profile_score = weighted_mean_valid(
            [
                (
                    _score_metric_from_view_or_common(
                        center_offset,
                        "side_profile_center_offset",
                        view_spec,
                        common_spec,
                    ),
                    float(side_weights.get("center_offset", 0.34)),
                ),
                (
                    _score_metric_from_view_or_common(
                        leg_straightness,
                        "side_profile_leg_straightness",
                        view_spec,
                        common_spec,
                    ),
                    float(side_weights.get("leg_straightness", 0.40)),
                ),
                (
                    _score_metric_from_view_or_common(
                        torso_compactness,
                        "side_profile_torso_compactness",
                        view_spec,
                        common_spec,
                    ),
                    float(side_weights.get("torso_compactness", 0.18)),
                ),
                (
                    _score_metric_from_view_or_common(
                        ankle_gap,
                        "side_profile_ankle_gap",
                        view_spec,
                        common_spec,
                    ),
                    float(side_weights.get("ankle_gap", 0.08)),
                ),
            ]
        )

    posterior_score = None
    if back_family:
        back_turn_weights = weights.get("back_turn_signal", {})
        posterior_mask_symmetry_score = _score_metric_from_view_or_common(
            mask_symmetry,
            "posterior_mask_symmetry",
            view_spec,
            common_spec,
        )
        posterior_shoulder_level_score = _score_metric_from_view_or_common(
            shoulder_level_delta,
            "posterior_shoulder_level_delta",
            view_spec,
            common_spec,
        )
        posterior_hip_level_score = _score_metric_from_view_or_common(
            hip_level_delta,
            "posterior_hip_level_delta",
            view_spec,
            common_spec,
        )
        posterior_leg_balance_score = _score_metric_from_view_or_common(
            lower_limb_balance if lower_limb_balance is not None else thigh_balance,
            "posterior_leg_balance",
            view_spec,
            common_spec,
        )
        posterior_calf_balance_score = _score_metric_from_view_or_common(
            calf_balance,
            "posterior_calf_balance",
            view_spec,
            common_spec,
        )
        posterior_foot_balance_score = _score_metric_from_view_or_common(
            foot_balance,
            "posterior_foot_balance",
            view_spec,
            common_spec,
        )
        posterior_leg_straightness_score = _score_metric_from_view_or_common(
            leg_straightness,
            "posterior_leg_straightness",
            view_spec,
            common_spec,
        )
        posterior_head_skin_score = _score_metric_from_view_or_common(
            head_skin_ratio,
            "posterior_head_skin_ratio",
            view_spec,
            common_spec,
        )
        turn_signal_pose_frontal_score = _score_metric_from_view_or_common(
            pose_frontal_strength,
            "posterior_pose_frontal_strength",
            view_spec,
            common_spec,
        )
        turn_signal_mask_symmetry_score = _score_metric_from_view_or_common(
            mask_symmetry,
            "posterior_turn_mask_symmetry",
            view_spec,
            common_spec,
        )
        turn_signal_head_skin_score = _score_metric_from_view_or_common(
            head_skin_ratio,
            "posterior_head_skin_ratio",
            view_spec,
            common_spec,
        )
        turn_score = weighted_mean_valid(
            [
                (turn_signal_body_yaw_score, float(back_turn_weights.get("body_yaw", 0.34))),
                (turn_signal_pose_frontal_score, float(back_turn_weights.get("pose_frontal", 0.30))),
                (turn_signal_mask_symmetry_score, float(back_turn_weights.get("mask_symmetry", 0.22))),
                (turn_signal_head_skin_score, float(back_turn_weights.get("head_skin_ratio", 0.14))),
            ]
        )
        if isinstance(lane_strictness_score, (int, float)):
            turn_score = (
                min(float(turn_score), float(lane_strictness_score))
                if turn_score is not None
                else float(lane_strictness_score)
            )
        torso_compactness_score = _score_metric_from_view_or_common(
            torso_compactness,
            "posterior_torso_compactness",
            view_spec,
            common_spec,
        )
        posterior_weights = weights.get("posterior_profile", {})
        posterior_raw_score = weighted_mean_valid(
            [
                (
                    posterior_mask_symmetry_score,
                    float(posterior_weights.get("mask_symmetry", 0.28)),
                ),
                (
                    posterior_shoulder_level_score,
                    float(posterior_weights.get("shoulder_level", 0.18)),
                ),
                (
                    posterior_hip_level_score,
                    float(posterior_weights.get("hip_level", 0.18)),
                ),
                (
                    posterior_leg_balance_score,
                    float(posterior_weights.get("leg_balance", 0.14)),
                ),
                (
                    posterior_calf_balance_score,
                    float(posterior_weights.get("calf_balance", 0.10)),
                ),
                (
                    posterior_foot_balance_score,
                    float(posterior_weights.get("foot_balance", 0.06)),
                ),
                (
                    posterior_leg_straightness_score,
                    float(posterior_weights.get("leg_straightness", 0.06)),
                ),
                (
                    posterior_head_skin_score,
                    float(posterior_weights.get("head_skin_ratio", 0.04)),
                ),
                (
                    torso_compactness_score,
                    float(posterior_weights.get("torso_compactness", 0.06)),
                ),
            ]
        )
        posterior_gate_weights = weights.get("posterior_gate", {})
        posterior_gate_score = weighted_mean_valid(
            [
                (lane_strictness_score, float(posterior_gate_weights.get("lane_strictness", 0.52))),
                (posterior_mask_symmetry_score, float(posterior_gate_weights.get("mask_symmetry", 0.16))),
                (torso_compactness_score, float(posterior_gate_weights.get("torso_compactness", 0.14))),
                (posterior_calf_balance_score, float(posterior_gate_weights.get("calf_balance", 0.10))),
                (posterior_head_skin_score, float(posterior_gate_weights.get("head_skin_ratio", 0.08))),
            ]
        )
        if posterior_raw_score is None:
            posterior_score = posterior_gate_score
        elif posterior_gate_score is None:
            posterior_score = posterior_raw_score
        else:
            posterior_score = min(float(posterior_raw_score), float(posterior_gate_score))

    if back_family:
        torso_volume_weights = weights.get("torso_volume_back", {})
        torso_volume_score = weighted_mean_valid(
            [
                (posterior_score, float(torso_volume_weights.get("posterior", 0.46))),
                (torso_score, float(torso_volume_weights.get("torso_length", 0.16))),
                (spine_score, float(torso_volume_weights.get("spine_angle", 0.12))),
                (torso_compactness_score, float(torso_volume_weights.get("torso_compactness", 0.16))),
                (shoulder_score, float(torso_volume_weights.get("shoulder_width", 0.10))),
                (hip_score, float(torso_volume_weights.get("hip_width", 0.08))),
                (ratio_score, float(torso_volume_weights.get("hip_shoulder_ratio", 0.08))),
            ]
        )
        overall_weights = weights.get("overall_back", {})
        depth_3d_score = weighted_mean_valid(
            [
                (turn_score, float(overall_weights.get("turn_signal", 0.22))),
                (torso_volume_score, float(overall_weights.get("torso_volume", 0.44))),
                (posterior_score, float(overall_weights.get("posterior", 0.22))),
                (spine_score, float(overall_weights.get("spine_angle", 0.06))),
                (torso_score, float(overall_weights.get("torso_length", 0.06))),
            ]
        )
        pelvis_depth_score = posterior_score
    elif side_family:
        torso_volume_weights = weights.get("torso_volume_profile", {})
        torso_volume_score = weighted_mean_valid(
            [
                (side_profile_score, float(torso_volume_weights.get("side_profile", 0.46))),
                (torso_score, float(torso_volume_weights.get("torso_length", 0.18))),
                (spine_score, float(torso_volume_weights.get("spine_angle", 0.16))),
                (shoulder_score, float(torso_volume_weights.get("shoulder_width", 0.08))),
                (hip_score, float(torso_volume_weights.get("hip_width", 0.06))),
                (ratio_score, float(torso_volume_weights.get("hip_shoulder_ratio", 0.06))),
            ]
        )
        overall_weights = weights.get("overall_profile", {})
        depth_3d_score = weighted_mean_valid(
            [
                (turn_score, float(overall_weights.get("turn_signal", 0.18))),
                (torso_volume_score, float(overall_weights.get("torso_volume", 0.48))),
                (spine_score, float(overall_weights.get("spine_angle", 0.12))),
                (torso_score, float(overall_weights.get("torso_length", 0.08))),
                (side_profile_score, float(overall_weights.get("side_profile", 0.14))),
            ]
        )
        pelvis_depth_score = side_profile_score
    else:
        torso_volume_weights = weights.get("torso_volume", {})
        torso_volume_score = weighted_mean_valid(
            [
                (shoulder_score, float(torso_volume_weights.get("shoulder_width", 0.30))),
                (hip_score, float(torso_volume_weights.get("hip_width", 0.24))),
                (ratio_score, float(torso_volume_weights.get("hip_shoulder_ratio", 0.20))),
                (spine_score, float(torso_volume_weights.get("spine_angle", 0.14))),
                (torso_score, float(torso_volume_weights.get("torso_length", 0.12))),
                (
                    side_profile_score,
                    float(torso_volume_weights.get("side_profile", 0.18)) if profile_like_family else 0.0,
                ),
            ]
        )
        overall_weights = weights.get("overall", {})
        depth_3d_score = weighted_mean_valid(
            [
                (turn_score, float(overall_weights.get("yaw", 0.24))),
                (torso_volume_score, float(overall_weights.get("torso_volume", 0.52))),
                (spine_score, float(overall_weights.get("spine_angle", 0.14))),
                (torso_score, float(overall_weights.get("torso_length", 0.10))),
                (
                    side_profile_score,
                    float(overall_weights.get("side_profile", 0.16)) if profile_like_family else 0.0,
                ),
            ]
        )
        pelvis_depth_score = ratio_score

    if requested_surface == "front":
        confidence = float(view_spec.get("confidence_fixed", 0.20))
    elif requested_surface == "three_quarter":
        confidence_weights = weights.get("confidence_three_quarter", {})
        confidence = weighted_mean_valid(
            [
                (
                    _score_from_range_spec(
                        yaw_proxy,
                        view_spec.get("confidence_yaw", view_spec.get("yaw", {})),
                    ),
                    float(confidence_weights.get("yaw", 0.45)),
                ),
                (torso_volume_score, float(confidence_weights.get("torso_volume", 0.35))),
                (spine_score, float(confidence_weights.get("spine_angle", 0.20))),
            ]
        )
    elif back_family:
        confidence_weights = weights.get("confidence_back_family", {})
        confidence = weighted_mean_valid(
            [
                (turn_score, float(confidence_weights.get("turn_signal", 0.32))),
                (torso_volume_score, float(confidence_weights.get("torso_volume", 0.34))),
                (posterior_score, float(confidence_weights.get("posterior", 0.24))),
                (spine_score, float(confidence_weights.get("spine_angle", 0.10))),
            ]
        )
    else:
        confidence_weights = weights.get("confidence_profile_family", {})
        confidence_turn_spec = (
            view_spec.get("confidence_body_yaw", {})
            if body_yaw_deg is not None
            else view_spec.get("confidence_yaw", {})
        )
        if not isinstance(confidence_turn_spec, dict) or len(confidence_turn_spec) == 0:
            confidence_turn_spec = view_spec.get("confidence_yaw", turn_spec if isinstance(turn_spec, dict) else {})
        confidence = weighted_mean_valid(
            [
                (
                    _score_from_range_spec(
                        turn_value,
                        confidence_turn_spec if isinstance(confidence_turn_spec, dict) else {},
                    ),
                    float(confidence_weights.get("turn_signal", 0.36)),
                ),
                (torso_volume_score, float(confidence_weights.get("torso_volume", 0.38))),
                (side_profile_score, float(confidence_weights.get("side_profile", 0.18))),
                (spine_score, float(confidence_weights.get("spine_angle", 0.08))),
            ]
        )

    return {
        "torso_volume_score": torso_volume_score,
        "pelvis_depth_score": pelvis_depth_score,
        "fake_turn_risk": float(1.0 - torso_volume_score) if torso_volume_score is not None else None,
        "depth_3d_score": depth_3d_score,
        "side_profile_score": side_profile_score,
        "posterior_score": posterior_score,
        "posterior_raw_score": posterior_raw_score,
        "posterior_gate_score": posterior_gate_score,
        "turn_signal_value": turn_value,
        "turn_signal_score": turn_score,
        "turn_signal_body_yaw_score": turn_signal_body_yaw_score,
        "turn_signal_pose_frontal_score": turn_signal_pose_frontal_score,
        "turn_signal_mask_symmetry_score": turn_signal_mask_symmetry_score,
        "turn_signal_head_skin_score": turn_signal_head_skin_score,
        "turn_signal_lane_strictness_score": turn_signal_lane_strictness_score,
        "torso_compactness_score": torso_compactness_score,
        "view_surface_requested": requested_surface,
        "view_surface_used": view_key,
        "confidence": 0.0 if confidence is None else float(confidence),
    }


def _fg_fill_ratio(
    mask_u8: Optional[np.ndarray],
    rect: Optional[Tuple[int, int, int, int]],
) -> Optional[float]:
    if mask_u8 is None or rect is None:
        return None
    x1, y1, x2, y2 = rect
    area = max(1, (x2 - x1) * (y2 - y1))
    fill = int(np.sum(mask_u8[y1:y2, x1:x2] > 0))
    return float(fill / area)


def extract_body_constitution_metrics(
    runtime: RuntimeContext,
    img_bgr: np.ndarray,
    face_feat: FaceFeat,
    pose_feat: PoseFeat,
    view_bucket: str = "front",
    view_lane_detail: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "pose_visibility": None,
        "width_ready": 0,
        "torso_fill": None,
        "head_body_ratio_proxy": None,
        "leg_ratio": None,
        "waist_height_ratio": None,
        "waist_height_ratio_topdown": None,
        "shoulder_width_px": None,
        "shoulder_width_domain": "pose_landmark_span",
        "chest_width_px": None,
        "waist_width_px": None,
        "hip_width_px": None,
        "thigh_width_px": None,
        "calf_width_px": None,
        "torso_reference_width_px": None,
        "torso_reference_width_domain": "silhouette_mask_width",
        "waist_to_shoulder_ratio": None,
        "chest_to_waist_ratio": None,
        "hip_to_waist_ratio": None,
        "hip_to_shoulder_ratio": None,
        "waist_to_torso_ratio": None,
        "chest_to_torso_ratio": None,
        "hip_to_torso_ratio": None,
        "thigh_to_calf_ratio": None,
        "pelvis_compactness_score": None,
        "abdomen_flatness_score": None,
        "lower_body_slenderness_score": None,
        "body_constitution_score": None,
        "reasons": [],
    }

    if img_bgr is None or (not pose_feat.ok) or pose_feat.lm_xy is None or pose_feat.lm_vis is None:
        out["reasons"].append("BODY_CONSTITUTION_NOT_AVAILABLE")
        return out

    h, w = img_bgr.shape[:2]
    xy = pose_feat.lm_xy
    vis = pose_feat.lm_vis

    needed = [11, 12, 23, 24, 25, 26, 27, 28]
    pose_vis = float(np.mean([1.0 if float(vis[i]) > 0.30 else 0.0 for i in needed]))
    out["pose_visibility"] = pose_vis
    if pose_vis < 0.70:
        out["reasons"].append("BODY_CONSTITUTION_KEYPOINTS_INSUFFICIENT")
        out["confidence"] = clamp(0.25 + 0.50 * pose_vis, 0.0, 1.0)
        return out

    fg_mask = runtime.providers.get_subject_mask(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
    if fg_mask is None:
        out["reasons"].append("BODY_CONSTITUTION_FG_MASK_FAILED")
        out["confidence"] = 0.0
        return out

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28

    lsx, lsy = _norm_xy_to_px(xy[L_SH], h, w)
    rsx, rsy = _norm_xy_to_px(xy[R_SH], h, w)
    lhx, lhy = _norm_xy_to_px(xy[L_HIP], h, w)
    rhx, rhy = _norm_xy_to_px(xy[R_HIP], h, w)
    lkx, lky = _norm_xy_to_px(xy[L_KNEE], h, w)
    rkx, rky = _norm_xy_to_px(xy[R_KNEE], h, w)
    lax, lay = _norm_xy_to_px(xy[L_ANK], h, w)
    rax, ray = _norm_xy_to_px(xy[R_ANK], h, w)

    shoulder_mid_y = int(round((lsy + rsy) / 2.0))
    hip_mid_y = int(round((lhy + rhy) / 2.0))
    knee_mid_y = int(round((lky + rky) / 2.0))
    ankle_mid_y = int(round((lay + ray) / 2.0))

    ys_fg = np.where(np.any(fg_mask > 0, axis=1))[0]
    if len(ys_fg) < 10:
        out["reasons"].append("BODY_CONSTITUTION_FG_TOO_SMALL")
        out["confidence"] = 0.0
        return out

    subject_top = int(ys_fg[0])
    subject_bottom = int(ys_fg[-1])
    body_h = max(1, subject_bottom - subject_top + 1)

    shoulder_width_px = float(
        np.linalg.norm(np.array([lsx, lsy], dtype=np.float32) - np.array([rsx, rsy], dtype=np.float32))
    )
    out["shoulder_width_px"] = shoulder_width_px

    chest_y = int(round(shoulder_mid_y + 0.22 * (hip_mid_y - shoulder_mid_y)))
    chest_w = _mask_width_at_row(fg_mask, chest_y, band=5)

    search_y1 = int(round(shoulder_mid_y + 0.22 * (hip_mid_y - shoulder_mid_y)))
    search_y2 = int(round(shoulder_mid_y + 0.92 * (hip_mid_y - shoulder_mid_y)))
    search_y1 = max(subject_top, min(subject_bottom - 1, search_y1))
    search_y2 = max(search_y1 + 1, min(subject_bottom, search_y2))

    waist_y, waist_w = _mask_width_soft_min(fg_mask, search_y1, search_y2, band=5)
    hip_sample_y = int(round(hip_mid_y + 0.08 * (ankle_mid_y - hip_mid_y)))
    hip_w = _mask_width_at_row(fg_mask, hip_sample_y, band=5)
    thigh_w = _mask_width_at_row(fg_mask, int(round((hip_mid_y + knee_mid_y) / 2.0)), band=5)
    calf_w = _mask_width_at_row(fg_mask, int(round((knee_mid_y + ankle_mid_y) / 2.0)), band=5)

    out["chest_width_px"] = chest_w
    out["waist_width_px"] = waist_w
    out["hip_width_px"] = hip_w
    out["thigh_width_px"] = thigh_w
    out["calf_width_px"] = calf_w

    if face_feat.ok and face_feat.bbox_xyxy is not None:
        x1, y1, x2, y2 = face_feat.bbox_xyxy
        out["head_body_ratio_proxy"] = float(body_h / max(1, y2 - y1))

    out["leg_ratio"] = safe_float(pose_feat.full_geom.get("leg_ratio", 0.0), 0.0) or None
    if waist_y is not None:
        out["waist_height_ratio_topdown"] = float((waist_y - subject_top) / max(1, body_h))
        torso_span = max(1, hip_mid_y - shoulder_mid_y)
        # The frozen spec treats waist height as a torso-local placement cue, not a
        # full-image top-down framing ratio. Using shoulder->hip span avoids
        # headroom / hairstyle drift from collapsing the score.
        out["waist_height_ratio"] = float(clamp((waist_y - shoulder_mid_y) / torso_span, 0.0, 1.0))
    if waist_w is not None and shoulder_width_px > 1e-6:
        # Waist / hip widths are extracted from full silhouette mask widths, while the
        # shoulder denominator is a landmark bone span. Convert silhouette widths to a
        # half-width proxy before comparing them to the shoulder span.
        out["waist_to_shoulder_ratio"] = float((0.5 * waist_w) / shoulder_width_px)
    if chest_w is not None and waist_w is not None and waist_w > 1e-6:
        out["chest_to_waist_ratio"] = float(chest_w / waist_w)
    if hip_w is not None and waist_w is not None and waist_w > 1e-6:
        out["hip_to_waist_ratio"] = float(hip_w / waist_w)
    if hip_w is not None and shoulder_width_px > 1e-6:
        out["hip_to_shoulder_ratio"] = float((0.5 * hip_w) / shoulder_width_px)
    if thigh_w is not None and calf_w is not None and calf_w > 1e-6:
        out["thigh_to_calf_ratio"] = float(thigh_w / calf_w)

    torso_reference_candidates = [float(v) for v in [chest_w, waist_w, hip_w] if v is not None]
    torso_reference_w: Optional[float] = None
    if len(torso_reference_candidates) == 1:
        torso_reference_w = float(torso_reference_candidates[0])
    elif len(torso_reference_candidates) >= 2:
        # Side/back lanes should compare widths in the same silhouette domain
        # instead of mixing mask widths with landmark shoulder spans.
        torso_reference_w = float(np.median(np.array(torso_reference_candidates, dtype=np.float32)))
    out["torso_reference_width_px"] = torso_reference_w
    if torso_reference_w is not None and torso_reference_w > 1e-6:
        if chest_w is not None:
            out["chest_to_torso_ratio"] = float(chest_w / torso_reference_w)
        if waist_w is not None:
            out["waist_to_torso_ratio"] = float(waist_w / torso_reference_w)
        if hip_w is not None:
            out["hip_to_torso_ratio"] = float(hip_w / torso_reference_w)

    out.update(
        score_body_constitution_measurements(
            out,
            runtime.config.consistency.body_constitution_scoring,
            view_bucket=view_bucket,
            view_lane_detail=view_lane_detail,
        )
    )

    width_ready = sum(1 for value in [chest_w, waist_w, hip_w, thigh_w, calf_w] if value is not None)
    out["width_ready"] = width_ready
    torso_rect = _clip_rect(
        int(round(min(lsx, rsx, lhx, rhx) - 0.05 * w)),
        int(round(min(lsy, rsy) - 0.02 * h)),
        int(round(max(lsx, rsx, lhx, rhx) + 0.05 * w)),
        int(round(max(lhy, rhy) + 0.02 * h)),
        h,
        w,
    )
    torso_fill = _fg_fill_ratio(fg_mask, torso_rect)
    out["torso_fill"] = torso_fill
    out["confidence"] = compute_body_constitution_confidence(
        runtime.config.consistency.body_constitution_scoring,
        view_bucket=view_bucket,
        width_ready=width_ready,
        pose_visibility=pose_vis,
        torso_fill=torso_fill,
        view_lane_detail=view_lane_detail,
    )
    min_width_metrics = int(
        runtime.config.consistency.body_constitution_scoring.get("validity", {}).get("min_width_metrics", 3)
    )
    out["is_valid"] = (out["body_constitution_score"] is not None) and (width_ready >= min_width_metrics)
    out["reasons"].append("BODY_CONSTITUTION_READY" if out["is_valid"] else "BODY_CONSTITUTION_SCORE_EMPTY")
    return out


def extract_skin_consistency_metrics(
    runtime: RuntimeContext,
    img_bgr: np.ndarray,
    face_feat: FaceFeat,
    pose_feat: PoseFeat,
    tone_reference_lab: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "face_neck_deltaE": None,
        "face_neck_deltaL": None,
        "face_neck_deltaAB": None,
        "face_arm_deltaE": None,
        "face_arm_deltaL": None,
        "face_arm_deltaAB": None,
        "face_abdomen_deltaE": None,
        "face_abdomen_deltaL": None,
        "face_abdomen_deltaAB": None,
        "face_thigh_deltaE": None,
        "face_thigh_deltaL": None,
        "face_thigh_deltaAB": None,
        "face_calf_deltaE": None,
        "face_calf_deltaL": None,
        "face_calf_deltaAB": None,
        "face_side_deltaL": None,
        "leg_lr_deltaL": None,
        "leg_brightness_ratio": None,
        "face_highlight_ratio": None,
        "knee_dark_patch_score": None,
        "chroma_consistency_score": None,
        "luminance_consistency_score": None,
        "tone_baseline_face_deltaL": None,
        "tone_baseline_face_deltaAB": None,
        "tone_baseline_thigh_deltaL": None,
        "tone_baseline_thigh_deltaAB": None,
        "tone_baseline_calf_deltaL": None,
        "tone_baseline_calf_deltaAB": None,
        "tone_baseline_chroma_score": None,
        "tone_baseline_luminance_score": None,
        "tone_baseline_consistency_score": None,
        "tone_family_score": None,
        "sample_risk_score": None,
        "lighting_risk_score": None,
        "skin_score_mode": "strict",
        "skin_uniformity_score": None,
        "reasons": [],
    }

    if img_bgr is None or (not pose_feat.ok) or pose_feat.lm_xy is None or pose_feat.lm_vis is None:
        out["reasons"].append("SKIN_CONSISTENCY_NOT_AVAILABLE")
        return out

    h, w = img_bgr.shape[:2]
    xy = pose_feat.lm_xy
    vis = pose_feat.lm_vis

    fg_mask = runtime.providers.get_subject_mask(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
    if fg_mask is None:
        out["reasons"].append("SKIN_FG_MASK_FAILED")
        return out

    skin_region = runtime.providers.get_skin_region(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
    if skin_region is None:
        out["reasons"].append("SKIN_CONSISTENCY_NOT_AVAILABLE")
        return out
    valid_skin = cv2.bitwise_and(skin_region, fg_mask)

    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28
    consistency = runtime.config.consistency
    risk = consistency.skin_risk
    split = consistency.skin_split

    def limb_mid_rect(
        i1: int,
        i2: int,
        rx_ratio: float = 0.028,
        ry_ratio: float = 0.025,
    ) -> Optional[Tuple[int, int, int, int]]:
        if float(vis[i1]) <= 0.35 or float(vis[i2]) <= 0.35:
            return None
        x1, y1 = _norm_xy_to_px(xy[i1], h, w)
        x2, y2 = _norm_xy_to_px(xy[i2], h, w)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        return _rect_from_center(cx, cy, max(8, int(rx_ratio * w)), max(8, int(ry_ratio * h)), h, w)

    face_center = SkinPatchStats("face_center", None, None, None, 0.0, 0, None, None, None)
    face_left = SkinPatchStats("face_left", None, None, None, 0.0, 0, None, None, None)
    face_right = SkinPatchStats("face_right", None, None, None, 0.0, 0, None, None, None)
    face_lab: Optional[np.ndarray] = None
    face_ref_l: Optional[float] = None

    if face_feat.ok and face_feat.bbox_xyxy is not None:
        x1, y1, x2, y2 = face_feat.bbox_xyxy
        fw = x2 - x1
        fh = y2 - y1
        face_center = _build_skin_patch_stats(
            img_bgr,
            valid_skin,
            "face_center",
            _clip_rect(
                x1 + int(0.18 * fw),
                y1 + int(0.20 * fh),
                x2 - int(0.18 * fw),
                y2 - int(0.12 * fh),
                h,
                w,
            ),
            min_pixels=40,
            min_purity=0.10,
            highlight_l=risk.face_highlight_l,
        )
        face_left = _build_skin_patch_stats(
            img_bgr,
            valid_skin,
            "face_left",
            _clip_rect(
                x1 + int(0.10 * fw),
                y1 + int(0.24 * fh),
                x1 + int(0.40 * fw),
                y2 - int(0.18 * fh),
                h,
                w,
            ),
            min_pixels=28,
            min_purity=0.08,
            highlight_l=risk.face_highlight_l,
        )
        face_right = _build_skin_patch_stats(
            img_bgr,
            valid_skin,
            "face_right",
            _clip_rect(
                x2 - int(0.40 * fw),
                y1 + int(0.24 * fh),
                x2 - int(0.10 * fw),
                y2 - int(0.18 * fh),
                h,
                w,
            ),
            min_pixels=28,
            min_purity=0.08,
            highlight_l=risk.face_highlight_l,
        )
        face_lab = face_center.lab
        face_ref_l = face_center.mean_l

    if face_lab is None and face_feat.ok and face_feat.lab_mean is not None:
        face_lab = face_feat.lab_mean.astype(np.float32)
        face_ref_l = float(face_lab[0])

    if face_lab is None:
        out["reasons"].append("SKIN_FACE_REFERENCE_MISSING")
        return out

    if face_ref_l is None:
        face_ref_l = float(face_lab[0])

    lsx, lsy = _norm_xy_to_px(xy[L_SH], h, w)
    rsx, rsy = _norm_xy_to_px(xy[R_SH], h, w)
    shoulder_mid_x = int(round((lsx + rsx) / 2.0))
    shoulder_mid_y = int(round((lsy + rsy) / 2.0))

    neck_patch = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "neck",
        _rect_from_center(
            shoulder_mid_x,
            shoulder_mid_y - max(8, int(0.025 * h)),
            max(10, int(0.03 * w)),
            max(8, int(0.02 * h)),
            h,
            w,
        ),
        min_pixels=28,
        min_purity=0.10,
    )

    abdomen_patch = SkinPatchStats("abdomen", None, None, None, 0.0, 0, None, None, None)
    if all(float(vis[idx]) > 0.35 for idx in [L_SH, R_SH, L_HIP, R_HIP]):
        lhx, lhy = _norm_xy_to_px(xy[L_HIP], h, w)
        rhx, rhy = _norm_xy_to_px(xy[R_HIP], h, w)
        hip_mid_x = int(round((lhx + rhx) / 2.0))
        hip_mid_y = int(round((lhy + rhy) / 2.0))
        abdomen_patch = _build_skin_patch_stats(
            img_bgr,
            valid_skin,
            "abdomen",
            _rect_from_center(
                int(round((shoulder_mid_x + hip_mid_x) / 2.0)),
                int(round(shoulder_mid_y + 0.60 * (hip_mid_y - shoulder_mid_y))),
                max(12, int(0.035 * w)),
                max(10, int(0.025 * h)),
                h,
                w,
            ),
            min_pixels=28,
            min_purity=0.10,
        )

    left_arm = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "left_arm",
        limb_mid_rect(L_SH, L_EL),
        min_pixels=30,
        min_purity=0.12,
    )
    right_arm = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "right_arm",
        limb_mid_rect(R_SH, R_EL),
        min_pixels=30,
        min_purity=0.12,
    )
    left_thigh = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "left_thigh",
        limb_mid_rect(L_HIP, L_KNEE, 0.035, 0.030),
        min_pixels=36,
        min_purity=0.10,
    )
    right_thigh = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "right_thigh",
        limb_mid_rect(R_HIP, R_KNEE, 0.035, 0.030),
        min_pixels=36,
        min_purity=0.10,
    )
    left_calf = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "left_calf",
        limb_mid_rect(L_KNEE, L_ANK, 0.032, 0.028),
        min_pixels=34,
        min_purity=0.10,
    )
    right_calf = _build_skin_patch_stats(
        img_bgr,
        valid_skin,
        "right_calf",
        limb_mid_rect(R_KNEE, R_ANK, 0.032, 0.028),
        min_pixels=34,
        min_purity=0.10,
    )

    arm_lab = _median_lab_from_stats([left_arm, right_arm])
    thigh_lab = _median_lab_from_stats([left_thigh, right_thigh])
    calf_lab = _median_lab_from_stats([left_calf, right_calf])
    left_leg_mean_l = _mean_valid([left_thigh.mean_l, left_calf.mean_l])
    right_leg_mean_l = _mean_valid([right_thigh.mean_l, right_calf.mean_l])
    mean_leg_l = _mean_valid([left_leg_mean_l, right_leg_mean_l])

    for prefix, region_lab in [
        ("face_neck", neck_patch.lab),
        ("face_arm", arm_lab),
        ("face_abdomen", abdomen_patch.lab),
        ("face_thigh", thigh_lab),
        ("face_calf", calf_lab),
    ]:
        out[f"{prefix}_deltaE"] = _delta_e_lab(face_lab, region_lab)
        out[f"{prefix}_deltaL"] = _delta_l_lab(face_lab, region_lab)
        out[f"{prefix}_deltaAB"] = _delta_ab_lab(face_lab, region_lab)

    if tone_reference_lab is not None:
        tone_lab = tone_reference_lab.astype(np.float32)
        out["tone_baseline_face_deltaL"] = _delta_l_lab(face_lab, tone_lab)
        out["tone_baseline_face_deltaAB"] = _delta_ab_lab(face_lab, tone_lab)
        out["tone_baseline_thigh_deltaL"] = _delta_l_lab(thigh_lab, tone_lab)
        out["tone_baseline_thigh_deltaAB"] = _delta_ab_lab(thigh_lab, tone_lab)
        out["tone_baseline_calf_deltaL"] = _delta_l_lab(calf_lab, tone_lab)
        out["tone_baseline_calf_deltaAB"] = _delta_ab_lab(calf_lab, tone_lab)

    out["face_side_deltaL"] = None
    if face_left.mean_l is not None and face_right.mean_l is not None:
        out["face_side_deltaL"] = float(abs(face_left.mean_l - face_right.mean_l))

    out["leg_lr_deltaL"] = None
    if left_leg_mean_l is not None and right_leg_mean_l is not None:
        out["leg_lr_deltaL"] = float(abs(left_leg_mean_l - right_leg_mean_l))

    out["face_highlight_ratio"] = face_center.highlight_ratio
    if out["face_highlight_ratio"] is None:
        out["face_highlight_ratio"] = _mean_valid([face_left.highlight_ratio, face_right.highlight_ratio])

    if mean_leg_l is not None and face_ref_l is not None:
        out["leg_brightness_ratio"] = float(mean_leg_l / max(1e-6, face_ref_l))

    knee_scores: List[Optional[float]] = []
    for thigh_patch, calf_patch in [(left_thigh, left_calf), (right_thigh, right_calf)]:
        if thigh_patch.mean_l is None or calf_patch.mean_l is None or face_ref_l is None:
            continue
        knee_l_proxy = float((thigh_patch.mean_l + calf_patch.mean_l) / 2.0)
        knee_ratio = float(knee_l_proxy / max(1e-6, face_ref_l))
        knee_scores.append(
            soft_range_score(
                knee_ratio,
                split.knee_ratio_low,
                split.knee_ratio_high,
                split.knee_ratio_margin,
            )
        )
    out["knee_dark_patch_score"] = _mean_valid(knee_scores)

    out["chroma_consistency_score"] = weighted_mean_valid(
        [
            (_exp_decay_score(out["face_thigh_deltaAB"], split.delta_ab_decay_thigh), 0.56),
            (_exp_decay_score(out["face_calf_deltaAB"], split.delta_ab_decay_calf), 0.44),
        ]
    )
    out["luminance_consistency_score"] = weighted_mean_valid(
        [
            (_exp_decay_score(out["face_thigh_deltaL"], split.delta_l_decay_thigh), 0.30),
            (_exp_decay_score(out["face_calf_deltaL"], split.delta_l_decay_calf), 0.26),
            (
                soft_range_score(
                    out["leg_brightness_ratio"],
                    split.brightness_ratio_low,
                    split.brightness_ratio_high,
                    split.brightness_ratio_margin,
                ),
                0.44,
            ),
        ]
    )
    out["tone_baseline_chroma_score"] = weighted_mean_valid(
        [
            (_exp_decay_score(out["tone_baseline_face_deltaAB"], split.delta_ab_decay_thigh), 0.24),
            (_exp_decay_score(out["tone_baseline_thigh_deltaAB"], split.delta_ab_decay_thigh), 0.24),
            (_exp_decay_score(out["tone_baseline_calf_deltaAB"], split.delta_ab_decay_calf), 0.18),
        ]
    )
    out["tone_baseline_luminance_score"] = weighted_mean_valid(
        [
            (_exp_decay_score(out["tone_baseline_face_deltaL"], split.delta_l_decay_thigh), 0.16),
            (_exp_decay_score(out["tone_baseline_thigh_deltaL"], split.delta_l_decay_thigh), 0.10),
            (_exp_decay_score(out["tone_baseline_calf_deltaL"], split.delta_l_decay_calf), 0.08),
        ]
    )
    # Keep the existing field name as the lighting-invariant tone-baseline score so
    # downstream code can read the same key while decoupling it from luminance drift.
    out["tone_baseline_consistency_score"] = out["tone_baseline_chroma_score"]
    out["tone_family_score"] = weighted_mean_valid(
        [
            (out["chroma_consistency_score"], 0.60),
            (out["tone_baseline_chroma_score"], 0.40),
        ]
    )

    observed_stats = [
        face_center,
        face_left,
        face_right,
        neck_patch,
        abdomen_patch,
        left_arm,
        right_arm,
        left_thigh,
        right_thigh,
        left_calf,
        right_calf,
    ]
    purity_values = [stat.purity for stat in observed_stats if stat.rect is not None]
    purity_mean = _mean_valid([float(value) for value in purity_values]) or 0.0
    purity_std = None
    if len(purity_values) > 0:
        purity_std = float(np.std(np.array(purity_values, dtype=np.float32)))

    low_purity_risk = _mean_valid(
        [
            1.0 - linear_map_to_01(float(stat.purity), risk.low_purity_floor, 0.60)
            for stat in observed_stats
            if stat.rect is not None
        ]
    )
    edge_risk = _mean_valid(
        [
            1.0 - linear_map_to_01(
                float(stat.edge_margin_ratio),
                risk.edge_margin_ratio_floor,
                max(risk.edge_margin_ratio_floor * 4.0, risk.edge_margin_ratio_floor + 1e-6),
            )
            for stat in [left_thigh, right_thigh, left_calf, right_calf]
            if stat.edge_margin_ratio is not None
        ]
    )
    purity_variance_risk = None
    if purity_std is not None:
        purity_variance_risk = linear_map_to_01(
            purity_std,
            risk.purity_variance_warn,
            max(risk.purity_variance_warn * 2.0, risk.purity_variance_warn + 1e-6),
        )

    left_leg_valid = sum(1 for stat in [left_thigh, left_calf] if stat.lab is not None)
    right_leg_valid = sum(1 for stat in [right_thigh, right_calf] if stat.lab is not None)
    single_side_leg_risk = None
    if (left_leg_valid > 0) != (right_leg_valid > 0):
        single_side_leg_risk = 1.0
    elif left_leg_valid > 0 and right_leg_valid > 0:
        single_side_leg_risk = 0.0

    avail_main = sum(
        1 for value in [out["face_thigh_deltaE"], out["face_calf_deltaE"], out["leg_brightness_ratio"]] if value is not None
    )
    availability_risk = 1.0 - float(avail_main) / 3.0
    out["sample_risk_score"] = weighted_mean_valid(
        [
            (availability_risk, 0.28),
            (low_purity_risk, 0.30),
            (purity_variance_risk, 0.16),
            (edge_risk, 0.16),
            (single_side_leg_risk, 0.10),
        ]
    )

    out["lighting_risk_score"] = weighted_mean_valid(
        [
            (
                linear_map_to_01(
                    safe_float(out["face_side_deltaL"], 0.0),
                    risk.face_side_delta_l_warn,
                    max(risk.face_side_delta_l_warn * 2.0, risk.face_side_delta_l_warn + 1e-6),
                )
                if out["face_side_deltaL"] is not None
                else None,
                0.30,
            ),
            (
                linear_map_to_01(
                    safe_float(out["face_neck_deltaL"], 0.0),
                    risk.face_neck_delta_l_warn,
                    max(risk.face_neck_delta_l_warn * 2.0, risk.face_neck_delta_l_warn + 1e-6),
                )
                if out["face_neck_deltaL"] is not None
                else None,
                0.22,
            ),
            (
                linear_map_to_01(
                    safe_float(out["leg_lr_deltaL"], 0.0),
                    risk.leg_lr_delta_l_warn,
                    max(risk.leg_lr_delta_l_warn * 2.0, risk.leg_lr_delta_l_warn + 1e-6),
                )
                if out["leg_lr_deltaL"] is not None
                else None,
                0.28,
            ),
            (
                linear_map_to_01(
                    safe_float(out["face_highlight_ratio"], 0.0),
                    risk.face_highlight_ratio_warn,
                    max(risk.face_highlight_ratio_high, risk.face_highlight_ratio_warn + 1e-6),
                )
                if out["face_highlight_ratio"] is not None
                else None,
                0.20,
            ),
        ]
    )

    if out["lighting_risk_score"] is not None and float(out["lighting_risk_score"]) >= risk.lighting_high_th:
        out["skin_score_mode"] = "high_risk"
        weight_preset = consistency.skin_score_weights.high_risk
    elif out["lighting_risk_score"] is not None and float(out["lighting_risk_score"]) >= risk.lighting_warn_th:
        out["skin_score_mode"] = "chroma_dominant"
        weight_preset = consistency.skin_score_weights.chroma_dominant
    else:
        out["skin_score_mode"] = "strict"
        weight_preset = consistency.skin_score_weights.strict

    tone_weight = float(weight_preset.chroma) + float(weight_preset.baseline)
    out["skin_uniformity_score"] = weighted_mean_valid(
        [
            (out["tone_family_score"], tone_weight),
            (out["luminance_consistency_score"], weight_preset.luminance),
            (out["knee_dark_patch_score"], weight_preset.knee),
        ]
    )

    conf = weighted_mean_valid(
        [
            (float(avail_main) / 3.0, 0.45),
            (purity_mean, 0.35),
            (
                None
                if out["sample_risk_score"] is None
                else clamp(1.0 - float(out["sample_risk_score"]), 0.0, 1.0),
                0.20,
            ),
        ]
    )

    out["confidence"] = 0.0 if conf is None else float(conf)
    out["is_valid"] = (out["skin_uniformity_score"] is not None) and (avail_main >= 2)
    out["reasons"].append("SKIN_UNIFORMITY_READY" if out["is_valid"] else "SKIN_UNIFORMITY_EMPTY")
    return out


def extract_depth_3d_lite_metrics(
    face_feat: FaceFeat,
    pose_feat: PoseFeat,
    view_bucket: str,
    yaw_proxy: float,
    body_yaw_deg: Optional[float] = None,
    pose_frontal_strength: Optional[float] = None,
    lane_strictness_score: Optional[float] = None,
    mask_symmetry: Optional[float] = None,
    head_skin_ratio: Optional[float] = None,
    scoring: Optional[Dict[str, Any]] = None,
    view_lane_detail: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "torso_volume_score": None,
        "pelvis_depth_score": None,
        "fake_turn_risk": None,
        "depth_3d_score": None,
        "turn_signal_value": None,
        "turn_signal_score": None,
        "posterior_score": None,
        "posterior_raw_score": None,
        "posterior_gate_score": None,
        "turn_signal_body_yaw_score": None,
        "turn_signal_pose_frontal_score": None,
        "turn_signal_mask_symmetry_score": None,
        "turn_signal_head_skin_score": None,
        "turn_signal_lane_strictness_score": None,
        "torso_compactness_score": None,
        "reasons": [],
    }

    if not pose_feat.ok:
        out["reasons"].append("DEPTH_3D_LITE_NOT_AVAILABLE")
        return out

    del face_feat
    if pose_feat.upper_geom.get("shoulder_width_norm", None) is None or pose_feat.upper_geom.get("hip_width_norm", None) is None:
        out["reasons"].append("DEPTH_3D_LITE_GEOM_MISSING")
        return out

    out.update(
        score_depth_3d_lite_geometry(
            pose_feat.upper_geom,
            pose_feat.full_geom,
            view_bucket=view_bucket,
            yaw_proxy=yaw_proxy,
            body_yaw_deg=body_yaw_deg,
            pose_frontal_strength=pose_frontal_strength,
            lane_strictness_score=lane_strictness_score,
            mask_symmetry=mask_symmetry,
            head_skin_ratio=head_skin_ratio,
            scoring=scoring or {},
            view_lane_detail=view_lane_detail,
        )
    )
    if out["depth_3d_score"] is None:
        out["reasons"].append("DEPTH_3D_LITE_EMPTY")
        return out

    bottlenecks: List[Tuple[str, float]] = []

    def add_bottleneck(reason: str, value: Optional[float], threshold: float) -> None:
        if isinstance(value, (int, float)) and float(value) < threshold:
            bottlenecks.append((reason, float(value)))

    requested_surface = str(out.get("view_surface_requested", "") or "")
    if requested_surface in {"strict_back_180", "back_like"}:
        lane_strict = out.get("turn_signal_lane_strictness_score", lane_strictness_score)
        posterior_raw = out.get("posterior_raw_score", None)
        posterior_gate = out.get("posterior_gate_score", None)
        posterior_score = out.get("posterior_score", None)
        turn_score = out.get("turn_signal_score", None)
        torso_volume_score = out.get("torso_volume_score", None)
        turn_head = out.get("turn_signal_head_skin_score", None)
        turn_sym = out.get("turn_signal_mask_symmetry_score", None)
        turn_front = out.get("turn_signal_pose_frontal_score", None)
        turn_yaw = out.get("turn_signal_body_yaw_score", None)
        torso_compactness_score = out.get("torso_compactness_score", None)

        if view_lane_detail == "back_like" and isinstance(lane_strict, (int, float)) and float(lane_strict) < 0.72:
            out["reasons"].append("BACK_STRICTNESS_BELOW_STRICT")
        if (
            isinstance(posterior_raw, (int, float))
            and isinstance(posterior_gate, (int, float))
            and float(posterior_gate) + 0.04 < float(posterior_raw)
        ):
            out["reasons"].append("BACK_POSTERIOR_GATE_LIMITED")
        if isinstance(turn_score, (int, float)) and float(turn_score) < 0.70:
            out["reasons"].append("BACK_TURN_SIGNAL_WEAK")
        if isinstance(torso_volume_score, (int, float)) and float(torso_volume_score) < 0.74:
            out["reasons"].append("BACK_TORSO_VOLUME_WEAK")
        if isinstance(posterior_score, (int, float)) and float(posterior_score) < 0.80:
            out["reasons"].append("BACK_POSTERIOR_STRUCTURE_WEAK")

        add_bottleneck("BACK_STRICTNESS_LOW", lane_strict, 0.72)
        add_bottleneck("BACK_POSTERIOR_LOW", posterior_score, 0.80)
        add_bottleneck("BACK_TURN_LOW", turn_score, 0.70)
        add_bottleneck("BACK_TORSO_VOLUME_LOW", torso_volume_score, 0.74)
        add_bottleneck("BACK_HEAD_SKIN_OUT_OF_RANGE", turn_head, 0.55)
        add_bottleneck("BACK_MASK_SYMMETRY_LOW", turn_sym, 0.80)
        add_bottleneck("BACK_FRONTAL_POSE_SIGNAL_LOW", turn_front, 0.70)
        add_bottleneck("BACK_YAW_SIGNAL_LOW", turn_yaw, 0.70)
        add_bottleneck("BACK_TORSO_COMPACTNESS_LOW", torso_compactness_score, 0.70)

    bottlenecks = sorted(bottlenecks, key=lambda item: item[1])
    out["bottlenecks"] = [name for name, _ in bottlenecks[:4]]
    out["primary_bottleneck"] = bottlenecks[0][0] if bottlenecks else None

    out["is_valid"] = out["depth_3d_score"] is not None
    out["reasons"].append("DEPTH_3D_LITE_READY" if out["is_valid"] else "DEPTH_3D_LITE_EMPTY")
    return out


def apply_consistency_soft_gate(
    runtime: RuntimeContext,
    reasons_all: List[str],
    final_status: str,
    overall_state: str,
    constitution_metrics: Dict[str, Any],
    skin_metrics: Dict[str, Any],
    depth_3d_metrics: Dict[str, Any],
    view_bucket: str,
) -> Tuple[List[str], str, str, Dict[str, Any]]:
    gate_debug: Dict[str, Any] = {
        "constitution": {},
        "skin": {},
        "depth_3d": {},
        "triggered": [],
        "mode": runtime.config.consistency.mode,
    }
    consistency = runtime.config.consistency

    def downgrade_pass_to_warn(flag: str) -> None:
        nonlocal final_status, overall_state
        gate_debug["triggered"].append(flag)
        if final_status == "PASS":
            final_status = "WARN"
        if overall_state == "PASS":
            overall_state = "WARN"

    if consistency.mode == "observe":
        reasons_all.extend(constitution_metrics.get("reasons", []))
        reasons_all.extend(skin_metrics.get("reasons", []))
        reasons_all.extend(depth_3d_metrics.get("reasons", []))
        return dedupe_keep_order(reasons_all), final_status, overall_state, gate_debug

    c_score = constitution_metrics.get("body_constitution_score", None)
    c_conf = float(constitution_metrics.get("confidence", 0.0) or 0.0)
    c_valid = bool(constitution_metrics.get("is_valid", False))
    gate_debug["constitution"] = {"score": c_score, "confidence": c_conf, "valid": c_valid}
    if c_valid and c_score is not None:
        reasons_all.extend(constitution_metrics.get("reasons", []))
        if c_conf >= consistency.constitution_min_conf:
            if float(c_score) < consistency.constitution_strong_warn_th:
                reasons_all.append("BODY_CONSTITUTION_STRONG_WARN")
                downgrade_pass_to_warn("BODY_CONSTITUTION_STRONG_WARN")
            elif float(c_score) < consistency.constitution_soft_warn_th:
                reasons_all.append("BODY_CONSTITUTION_WARN")
                downgrade_pass_to_warn("BODY_CONSTITUTION_WARN")
        else:
            reasons_all.append("BODY_CONSTITUTION_LOW_CONF_SKIP")

    s_score = skin_metrics.get("skin_uniformity_score", None)
    s_conf = float(skin_metrics.get("confidence", 0.0) or 0.0)
    s_valid = bool(skin_metrics.get("is_valid", False))
    sample_risk = skin_metrics.get("sample_risk_score", None)
    lighting_risk = skin_metrics.get("lighting_risk_score", None)
    skin_mode = str(skin_metrics.get("skin_score_mode", "strict") or "strict")
    chroma_score = skin_metrics.get("chroma_consistency_score", None)
    luminance_score = skin_metrics.get("luminance_consistency_score", None)
    tone_baseline_score = skin_metrics.get("tone_baseline_consistency_score", None)
    tone_family_score = skin_metrics.get("tone_family_score", None)
    tone_baseline_luminance_score = skin_metrics.get("tone_baseline_luminance_score", None)
    gate_debug["skin"] = {
        "score": s_score,
        "confidence": s_conf,
        "valid": s_valid,
        "sample_risk": sample_risk,
        "lighting_risk": lighting_risk,
        "score_mode": skin_mode,
        "chroma_score": chroma_score,
        "luminance_score": luminance_score,
        "tone_family_score": tone_family_score,
        "tone_baseline_score": tone_baseline_score,
        "tone_baseline_luminance_score": tone_baseline_luminance_score,
    }
    reasons_all.extend(
        [reason for reason in skin_metrics.get("reasons", []) if reason != "SKIN_UNIFORMITY_EMPTY"]
    )

    sample_risk_high = sample_risk is not None and float(sample_risk) >= consistency.skin_risk.sample_high_th
    sample_risk_warn = sample_risk is not None and float(sample_risk) >= consistency.skin_risk.sample_warn_th
    lighting_risk_high = lighting_risk is not None and float(lighting_risk) >= consistency.skin_risk.lighting_high_th
    lighting_risk_warn = lighting_risk is not None and float(lighting_risk) >= consistency.skin_risk.lighting_warn_th

    if sample_risk_high:
        reasons_all.append("SKIN_SAMPLE_RISK_HIGH")
    elif sample_risk_warn:
        reasons_all.append("SKIN_SAMPLE_RISK_WARN")

    if lighting_risk_high:
        reasons_all.append("SKIN_LIGHTING_RISK_HIGH")
    elif lighting_risk_warn:
        reasons_all.append("SKIN_LIGHTING_RISK_WARN")

    risk_skip_skin_gate = sample_risk_high or lighting_risk_high
    if s_valid and s_score is not None:
        thigh_ab = skin_metrics.get("face_thigh_deltaAB", None)
        calf_ab = skin_metrics.get("face_calf_deltaAB", None)
        leg_ratio = skin_metrics.get("leg_brightness_ratio", None)

        severe_chroma = False
        if thigh_ab is not None and float(thigh_ab) >= consistency.skin_split.severe_delta_ab_thigh:
            severe_chroma = True
        if calf_ab is not None and float(calf_ab) >= consistency.skin_split.severe_delta_ab_calf:
            severe_chroma = True

        lighting_stable = (lighting_risk is None) or (float(lighting_risk) < consistency.skin_risk.lighting_warn_th)
        severe_luminance = False
        if lighting_stable:
            if (
                leg_ratio is not None
                and float(leg_ratio) < consistency.skin_split.severe_leg_brightness_ratio
            ):
                severe_luminance = True
            if (
                luminance_score is not None
                and float(luminance_score) < consistency.skin_split.severe_luminance_score
            ):
                severe_luminance = True
        severe_tone_baseline = (
            tone_family_score is not None
            and float(tone_family_score) < consistency.skin_strong_warn_th
        )
        warn_tone_baseline = (
            tone_family_score is not None
            and float(tone_family_score) < consistency.skin_soft_warn_th
        )

        if risk_skip_skin_gate:
            gate_debug["skin"]["risk_gate_skip"] = True
        elif s_conf >= consistency.skin_min_conf:
            severe_skin = severe_chroma or (skin_mode == "strict" and severe_luminance) or severe_tone_baseline
            if severe_skin or float(s_score) < consistency.skin_strong_warn_th:
                if severe_tone_baseline:
                    reasons_all.append("SKIN_TONE_BASELINE_STRONG_WARN")
                reasons_all.append("SKIN_UNIFORMITY_STRONG_WARN")
                downgrade_pass_to_warn("SKIN_UNIFORMITY_STRONG_WARN")
            elif warn_tone_baseline or float(s_score) < consistency.skin_soft_warn_th:
                if warn_tone_baseline:
                    reasons_all.append("SKIN_TONE_BASELINE_WARN")
                reasons_all.append("SKIN_UNIFORMITY_WARN")
                downgrade_pass_to_warn("SKIN_UNIFORMITY_WARN")
        else:
            reasons_all.append("SKIN_UNIFORMITY_LOW_CONF_SKIP")

    d_score = depth_3d_metrics.get("depth_3d_score", None)
    d_conf = float(depth_3d_metrics.get("confidence", 0.0) or 0.0)
    d_valid = bool(depth_3d_metrics.get("is_valid", False))
    gate_debug["depth_3d"] = {
        "score": d_score,
        "confidence": d_conf,
        "valid": d_valid,
        "view_bucket": view_bucket,
    }
    if d_valid and d_score is not None:
        reasons_all.extend(depth_3d_metrics.get("reasons", []))
        if view_bucket in {"three_quarter", "profile_like", "side_90", "back_180"}:
            if d_conf >= consistency.depth3d_min_conf:
                if float(d_score) < consistency.depth3d_strong_warn_th:
                    reasons_all.append("DEPTH_3D_LITE_STRONG_WARN")
                    downgrade_pass_to_warn("DEPTH_3D_LITE_STRONG_WARN")
                elif float(d_score) < consistency.depth3d_soft_warn_th:
                    reasons_all.append("DEPTH_3D_LITE_WARN")
                    downgrade_pass_to_warn("DEPTH_3D_LITE_WARN")
            else:
                reasons_all.append("DEPTH_3D_LITE_LOW_CONF_SKIP")

    return dedupe_keep_order(reasons_all), final_status, overall_state, gate_debug
