from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


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


def _normalize_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None or mask.ndim != 2:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _mask_bbox(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    normalized = _normalize_mask(mask)
    if normalized is None or np.count_nonzero(normalized) == 0:
        return None
    x, y, w, h = cv2.boundingRect(normalized)
    if w <= 1 or h <= 1:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def _clip_roi(
    roi: Optional[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    x1 = int(max(0, min(width - 1, round(float(x1)))))
    y1 = int(max(0, min(height - 1, round(float(y1)))))
    x2 = int(max(1, min(width, round(float(x2)))))
    y2 = int(max(1, min(height, round(float(y2)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _landmark_point(
    pose_feat: Any,
    index: int,
    width: int,
    height: int,
    vis_th: float = 0.35,
) -> Optional[Tuple[float, float]]:
    if not getattr(pose_feat, "ok", False):
        return None
    lm_xy = getattr(pose_feat, "lm_xy", None)
    lm_vis = getattr(pose_feat, "lm_vis", None)
    if lm_xy is None or lm_vis is None or len(lm_xy) <= index or len(lm_vis) <= index:
        return None
    if float(lm_vis[index]) < vis_th:
        return None
    x = float(lm_xy[index][0]) * float(width)
    y = float(lm_xy[index][1]) * float(height)
    return (x, y)


def _resolve_body_bbox(
    subject_mask: Optional[np.ndarray],
    pose_feat: Any,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    pose_bbox = getattr(pose_feat, "person_bbox_xywh", None)
    if pose_bbox is not None:
        x, y, w, h = pose_bbox
        return _clip_roi((x, y, x + w, y + h), width, height)
    return _mask_bbox(subject_mask)


def _torso_roi(
    pose_feat: Any,
    body_bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    l_shoulder = _landmark_point(pose_feat, 11, width, height)
    r_shoulder = _landmark_point(pose_feat, 12, width, height)
    l_hip = _landmark_point(pose_feat, 23, width, height)
    r_hip = _landmark_point(pose_feat, 24, width, height)
    if all(point is not None for point in (l_shoulder, r_shoulder, l_hip, r_hip)):
        xs = [point[0] for point in (l_shoulder, r_shoulder, l_hip, r_hip) if point is not None]
        ys = [point[1] for point in (l_shoulder, r_shoulder, l_hip, r_hip) if point is not None]
        bbox_w = max(8.0, max(xs) - min(xs))
        bbox_h = max(8.0, max(ys) - min(ys))
        return _clip_roi(
            (
                min(xs) - bbox_w * 0.16,
                min(ys) - bbox_h * 0.08,
                max(xs) + bbox_w * 0.16,
                max(ys) + bbox_h * 0.10,
            ),
            width,
            height,
        )
    x1, y1, x2, y2 = body_bbox
    bbox_w = float(x2 - x1)
    bbox_h = float(y2 - y1)
    return _clip_roi(
        (
            x1 + bbox_w * 0.22,
            y1 + bbox_h * 0.18,
            x2 - bbox_w * 0.22,
            y1 + bbox_h * 0.60,
        ),
        width,
        height,
    )


def _neckline_roi(
    pose_feat: Any,
    torso_roi: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    l_shoulder = _landmark_point(pose_feat, 11, width, height)
    r_shoulder = _landmark_point(pose_feat, 12, width, height)
    tx1, ty1, tx2, ty2 = torso_roi
    torso_w = float(tx2 - tx1)
    torso_h = float(ty2 - ty1)
    if l_shoulder is not None and r_shoulder is not None:
        sx1 = min(l_shoulder[0], r_shoulder[0])
        sx2 = max(l_shoulder[0], r_shoulder[0])
        sy = min(l_shoulder[1], r_shoulder[1])
        return _clip_roi(
            (
                sx1 + (sx2 - sx1) * 0.18,
                sy - torso_h * 0.02,
                sx2 - (sx2 - sx1) * 0.18,
                sy + torso_h * 0.28,
            ),
            width,
            height,
        )
    return _clip_roi(
        (
            tx1 + torso_w * 0.24,
            ty1,
            tx2 - torso_w * 0.24,
            ty1 + torso_h * 0.28,
        ),
        width,
        height,
    )


def _lower_body_roi(
    pose_feat: Any,
    body_bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    l_hip = _landmark_point(pose_feat, 23, width, height)
    r_hip = _landmark_point(pose_feat, 24, width, height)
    l_ankle = _landmark_point(pose_feat, 27, width, height)
    r_ankle = _landmark_point(pose_feat, 28, width, height)
    if all(point is not None for point in (l_hip, r_hip, l_ankle, r_ankle)):
        xs = [point[0] for point in (l_hip, r_hip, l_ankle, r_ankle) if point is not None]
        ys = [point[1] for point in (l_hip, r_hip, l_ankle, r_ankle) if point is not None]
        bbox_w = max(8.0, max(xs) - min(xs))
        return _clip_roi(
            (
                min(xs) - bbox_w * 0.10,
                min(ys) - (max(ys) - min(ys)) * 0.04,
                max(xs) + bbox_w * 0.10,
                max(ys) + (max(ys) - min(ys)) * 0.04,
            ),
            width,
            height,
        )
    x1, y1, x2, y2 = body_bbox
    bbox_w = float(x2 - x1)
    bbox_h = float(y2 - y1)
    return _clip_roi(
        (
            x1 + bbox_w * 0.24,
            y1 + bbox_h * 0.56,
            x2 - bbox_w * 0.24,
            y2 - bbox_h * 0.04,
        ),
        width,
        height,
    )


def _shoulder_rois(
    pose_feat: Any,
    torso_roi: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    l_shoulder = _landmark_point(pose_feat, 11, width, height)
    r_shoulder = _landmark_point(pose_feat, 12, width, height)
    tx1, ty1, tx2, ty2 = torso_roi
    torso_w = float(tx2 - tx1)
    torso_h = float(ty2 - ty1)
    if l_shoulder is not None and r_shoulder is not None:
        span = max(10.0, abs(r_shoulder[0] - l_shoulder[0]))
        half_w = max(8.0, span * 0.22)
        half_h = max(8.0, torso_h * 0.18)
        left_roi = _clip_roi(
            (l_shoulder[0] - half_w, l_shoulder[1] - half_h, l_shoulder[0] + half_w, l_shoulder[1] + half_h),
            width,
            height,
        )
        right_roi = _clip_roi(
            (r_shoulder[0] - half_w, r_shoulder[1] - half_h, r_shoulder[0] + half_w, r_shoulder[1] + half_h),
            width,
            height,
        )
        return left_roi, right_roi
    mid_x = (tx1 + tx2) / 2.0
    shoulder_y2 = ty1 + torso_h * 0.24
    left_roi = _clip_roi((tx1, ty1, mid_x, shoulder_y2), width, height)
    right_roi = _clip_roi((mid_x, ty1, tx2, shoulder_y2), width, height)
    return left_roi, right_roi


def _mask_ratio(
    mask: Optional[np.ndarray],
    subject_mask: Optional[np.ndarray],
    roi: Optional[Tuple[int, int, int, int]],
    min_subject_pixels: int = 48,
) -> Optional[float]:
    normalized_mask = _normalize_mask(mask)
    normalized_subject = _normalize_mask(subject_mask)
    if normalized_mask is None or normalized_subject is None or roi is None:
        return None
    x1, y1, x2, y2 = roi
    subject_crop = normalized_subject[y1:y2, x1:x2]
    if subject_crop.size == 0:
        return None
    subject_pixels = int(np.count_nonzero(subject_crop))
    if subject_pixels < int(min_subject_pixels):
        return None
    mask_crop = normalized_mask[y1:y2, x1:x2]
    return float(np.count_nonzero(mask_crop) / max(1, subject_pixels))


def extract_garment_metrics(
    runtime: Any,
    img_bgr: np.ndarray,
    face_feat: Any,
    pose_feat: Any,
    layer_tag: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "layer_tag": layer_tag,
        "subject_area_ratio": None,
        "body_area_ratio": None,
        "clothing_coverage_ratio": None,
        "skin_visible_ratio": None,
        "upper_cloth_coverage": None,
        "lower_cloth_coverage": None,
        "neckline_openness": None,
        "shoulder_exposure_balance": None,
        "torso_fill_ratio": None,
        "lower_fill_ratio": None,
        "confidence": 0.0,
        "reasons": [],
    }
    if img_bgr is None or img_bgr.size == 0:
        out["reasons"] = ["GARMENT_IMAGE_EMPTY"]
        return out

    subject_mask = _normalize_mask(runtime.providers.get_subject_mask(img_bgr, face_feat=face_feat, pose_feat=pose_feat))
    if subject_mask is None or np.count_nonzero(subject_mask) == 0:
        out["reasons"] = ["GARMENT_SUBJECT_MASK_MISSING"]
        return out

    skin_mask = _normalize_mask(runtime.providers.get_skin_region(img_bgr, face_feat=face_feat, pose_feat=pose_feat))
    if skin_mask is None:
        skin_mask = np.zeros(subject_mask.shape, dtype=np.uint8)
        out["reasons"].append("GARMENT_SKIN_REGION_MISSING")
    else:
        skin_mask = cv2.bitwise_and(skin_mask, subject_mask)

    height, width = img_bgr.shape[:2]
    body_bbox = _resolve_body_bbox(subject_mask, pose_feat, width, height)
    if body_bbox is None:
        out["reasons"].append("GARMENT_BODY_BOX_MISSING")
        return out

    torso_roi = _torso_roi(pose_feat, body_bbox, width, height)
    lower_roi = _lower_body_roi(pose_feat, body_bbox, width, height)
    neckline_roi = _neckline_roi(pose_feat, torso_roi, width, height) if torso_roi is not None else None
    left_shoulder_roi, right_shoulder_roi = _shoulder_rois(pose_feat, torso_roi, width, height) if torso_roi is not None else (None, None)

    body_x1, body_y1, body_x2, body_y2 = body_bbox
    body_roi = _clip_roi(
        (
            body_x1,
            torso_roi[1] if torso_roi is not None else body_y1,
            body_x2,
            lower_roi[3] if lower_roi is not None else body_y2,
        ),
        width,
        height,
    )

    subject_pixels = int(np.count_nonzero(subject_mask))
    out["subject_area_ratio"] = float(subject_pixels / max(1, width * height))
    if body_roi is not None:
        bx1, by1, bx2, by2 = body_roi
        body_area = max(1, (bx2 - bx1) * (by2 - by1))
        body_subject_pixels = int(np.count_nonzero(subject_mask[by1:by2, bx1:bx2]))
        body_skin_pixels = int(np.count_nonzero(skin_mask[by1:by2, bx1:bx2]))
        out["body_area_ratio"] = float(body_area / max(1, width * height))
        if body_subject_pixels >= 64:
            skin_visible_ratio = float(body_skin_pixels / max(1, body_subject_pixels))
            out["skin_visible_ratio"] = skin_visible_ratio
            out["clothing_coverage_ratio"] = float(1.0 - skin_visible_ratio)

    upper_skin_ratio = _mask_ratio(skin_mask, subject_mask, torso_roi, min_subject_pixels=72)
    lower_skin_ratio = _mask_ratio(skin_mask, subject_mask, lower_roi, min_subject_pixels=72)
    neckline_skin_ratio = _mask_ratio(skin_mask, subject_mask, neckline_roi, min_subject_pixels=32)
    left_shoulder_skin = _mask_ratio(skin_mask, subject_mask, left_shoulder_roi, min_subject_pixels=20)
    right_shoulder_skin = _mask_ratio(skin_mask, subject_mask, right_shoulder_roi, min_subject_pixels=20)

    if upper_skin_ratio is not None:
        out["upper_cloth_coverage"] = float(1.0 - upper_skin_ratio)
    if lower_skin_ratio is not None:
        out["lower_cloth_coverage"] = float(1.0 - lower_skin_ratio)
    if neckline_skin_ratio is not None:
        out["neckline_openness"] = float(neckline_skin_ratio)
    if left_shoulder_skin is not None and right_shoulder_skin is not None:
        out["shoulder_exposure_balance"] = float(1.0 - min(1.0, abs(left_shoulder_skin - right_shoulder_skin)))

    if torso_roi is not None:
        tx1, ty1, tx2, ty2 = torso_roi
        torso_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        out["torso_fill_ratio"] = float(np.count_nonzero(subject_mask[ty1:ty2, tx1:tx2]) / torso_area)
    if lower_roi is not None:
        lx1, ly1, lx2, ly2 = lower_roi
        lower_area = max(1, (lx2 - lx1) * (ly2 - ly1))
        out["lower_fill_ratio"] = float(np.count_nonzero(subject_mask[ly1:ly2, lx1:lx2]) / lower_area)

    pose_ready = 1.0 if getattr(pose_feat, "ok", False) else 0.45
    area_ready = None
    if out["subject_area_ratio"] is not None:
        area_ready = _clamp((float(out["subject_area_ratio"]) - 0.08) / 0.42, 0.0, 1.0)
    out["confidence"] = float(
        _weighted_mean(
            [
                (area_ready, 0.18),
                (out.get("torso_fill_ratio"), 0.26),
                (out.get("lower_fill_ratio"), 0.16),
                (1.0 if out.get("upper_cloth_coverage") is not None else 0.0, 0.14),
                (1.0 if out.get("neckline_openness") is not None else 0.0, 0.10),
                (pose_ready, 0.16),
            ]
        )
        or 0.0
    )

    if out.get("torso_fill_ratio") is None or float(out["torso_fill_ratio"]) < 0.22:
        out["reasons"].append("GARMENT_TORSO_SIGNAL_WEAK")
    if out.get("lower_fill_ratio") is None or float(out["lower_fill_ratio"]) < 0.18:
        out["reasons"].append("GARMENT_LOWER_SIGNAL_WEAK")
    if out.get("upper_cloth_coverage") is None:
        out["reasons"].append("GARMENT_UPPER_COVERAGE_UNAVAILABLE")
    if out.get("neckline_openness") is None:
        out["reasons"].append("GARMENT_NECKLINE_UNAVAILABLE")
    if out.get("shoulder_exposure_balance") is None:
        out["reasons"].append("GARMENT_SHOULDER_BALANCE_UNAVAILABLE")
    return out
