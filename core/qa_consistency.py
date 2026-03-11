from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .qa_runtime import FaceFeat, PoseFeat, RuntimeContext
from .qa_utils import clamp, dedupe_keep_order, safe_float


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
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "head_body_ratio_proxy": None,
        "leg_ratio": None,
        "waist_height_ratio": None,
        "shoulder_width_px": None,
        "chest_width_px": None,
        "waist_width_px": None,
        "hip_width_px": None,
        "thigh_width_px": None,
        "calf_width_px": None,
        "waist_to_shoulder_ratio": None,
        "chest_to_waist_ratio": None,
        "hip_to_waist_ratio": None,
        "hip_to_shoulder_ratio": None,
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
        out["waist_height_ratio"] = float((waist_y - subject_top) / max(1, body_h))
    if waist_w is not None and shoulder_width_px > 1e-6:
        out["waist_to_shoulder_ratio"] = float(waist_w / shoulder_width_px)
    if chest_w is not None and waist_w is not None and waist_w > 1e-6:
        out["chest_to_waist_ratio"] = float(chest_w / waist_w)
    if hip_w is not None and waist_w is not None and waist_w > 1e-6:
        out["hip_to_waist_ratio"] = float(hip_w / waist_w)
    if hip_w is not None and shoulder_width_px > 1e-6:
        out["hip_to_shoulder_ratio"] = float(hip_w / shoulder_width_px)
    if thigh_w is not None and calf_w is not None and calf_w > 1e-6:
        out["thigh_to_calf_ratio"] = float(thigh_w / calf_w)

    if view_bucket == "front":
        waist_shoulder = soft_range_score(out["waist_to_shoulder_ratio"], 0.42, 0.62, 0.18)
        chest_waist = soft_range_score(out["chest_to_waist_ratio"], 1.08, 1.48, 0.26)
        hip_waist = soft_range_score(out["hip_to_waist_ratio"], 1.12, 1.56, 0.28)
        hip_shoulder = soft_range_score(out["hip_to_shoulder_ratio"], 0.46, 0.72, 0.20)
        thigh_calf = soft_range_score(out["thigh_to_calf_ratio"], 1.10, 1.76, 0.40)
    elif view_bucket == "three_quarter":
        waist_shoulder = soft_range_score(out["waist_to_shoulder_ratio"], 0.44, 0.70, 0.22)
        chest_waist = soft_range_score(out["chest_to_waist_ratio"], 1.00, 1.42, 0.28)
        hip_waist = soft_range_score(out["hip_to_waist_ratio"], 1.04, 1.52, 0.32)
        hip_shoulder = soft_range_score(out["hip_to_shoulder_ratio"], 0.40, 0.76, 0.24)
        thigh_calf = soft_range_score(out["thigh_to_calf_ratio"], 1.08, 1.82, 0.44)
    else:
        waist_shoulder = soft_range_score(out["waist_to_shoulder_ratio"], 0.46, 0.78, 0.26)
        chest_waist = soft_range_score(out["chest_to_waist_ratio"], 0.96, 1.38, 0.32)
        hip_waist = soft_range_score(out["hip_to_waist_ratio"], 1.00, 1.48, 0.36)
        hip_shoulder = soft_range_score(out["hip_to_shoulder_ratio"], 0.36, 0.78, 0.28)
        thigh_calf = soft_range_score(out["thigh_to_calf_ratio"], 1.06, 1.88, 0.50)

    out["pelvis_compactness_score"] = hip_shoulder
    out["abdomen_flatness_score"] = waist_shoulder
    out["lower_body_slenderness_score"] = thigh_calf
    out["body_constitution_score"] = weighted_mean_valid(
        [
            (soft_range_score(out["leg_ratio"], 0.44, 0.58, 0.10), 0.18),
            (soft_range_score(out["waist_height_ratio"], 0.54, 0.67, 0.10), 0.18),
            (waist_shoulder, 0.20),
            (chest_waist, 0.12),
            (hip_waist, 0.14),
            (hip_shoulder, 0.10),
            (thigh_calf, 0.08),
        ]
    )

    width_ready = sum(1 for value in [chest_w, waist_w, hip_w, thigh_w, calf_w] if value is not None)
    torso_rect = _clip_rect(
        int(round(min(lsx, rsx, lhx, rhx) - 0.05 * w)),
        int(round(min(lsy, rsy) - 0.02 * h)),
        int(round(max(lsx, rsx, lhx, rhx) + 0.05 * w)),
        int(round(max(lhy, rhy) + 0.02 * h)),
        h,
        w,
    )
    torso_fill = _fg_fill_ratio(fg_mask, torso_rect)
    view_factor = 1.0 if view_bucket == "front" else (0.88 if view_bucket == "three_quarter" else 0.75)
    conf = weighted_mean_valid(
        [
            (float(width_ready) / 5.0, 0.34),
            (pose_vis, 0.34),
            (torso_fill, 0.22),
            (view_factor, 0.10),
        ]
    )
    out["confidence"] = 0.0 if conf is None else float(conf)
    out["is_valid"] = (out["body_constitution_score"] is not None) and (width_ready >= 3)
    out["reasons"].append("BODY_CONSTITUTION_READY" if out["is_valid"] else "BODY_CONSTITUTION_SCORE_EMPTY")
    return out


def extract_skin_consistency_metrics(
    runtime: RuntimeContext,
    img_bgr: np.ndarray,
    face_feat: FaceFeat,
    pose_feat: PoseFeat,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "face_neck_deltaE": None,
        "face_arm_deltaE": None,
        "face_abdomen_deltaE": None,
        "face_thigh_deltaE": None,
        "face_calf_deltaE": None,
        "leg_brightness_ratio": None,
        "knee_dark_patch_score": None,
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

    valid_skin = cv2.bitwise_and(
        runtime.providers.get_skin_region(img_bgr, face_feat=face_feat, pose_feat=pose_feat),
        fg_mask,
    )

    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28

    def patch_mask_from_rect(
        rect: Optional[Tuple[int, int, int, int]],
        min_pixels: int = 30,
        min_purity: float = 0.12,
    ) -> Tuple[Optional[np.ndarray], float]:
        if rect is None:
            return None, 0.0
        x1, y1, x2, y2 = rect
        area = max(1, (x2 - x1) * (y2 - y1))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, valid_skin)
        count = int(np.sum(mask > 0))
        purity = float(count / area)
        if count < min_pixels or purity < min_purity:
            return None, purity
        return mask, purity

    face_lab = None
    face_purity = 0.0
    if face_feat.ok and face_feat.bbox_xyxy is not None:
        x1, y1, x2, y2 = face_feat.bbox_xyxy
        fw = x2 - x1
        fh = y2 - y1
        rr = _clip_rect(
            x1 + int(0.18 * fw),
            y1 + int(0.20 * fh),
            x2 - int(0.18 * fw),
            y2 - int(0.12 * fh),
            h,
            w,
        )
        mask, face_purity = patch_mask_from_rect(rr, min_pixels=40, min_purity=0.10)
        face_lab = _median_lab_in_region(img_bgr, mask) if mask is not None else None

    if face_lab is None and face_feat.ok and face_feat.lab_mean is not None:
        face_lab = face_feat.lab_mean.astype(np.float32)
        face_purity = 0.20

    if face_lab is None:
        out["reasons"].append("SKIN_FACE_REFERENCE_MISSING")
        return out

    lsx, lsy = _norm_xy_to_px(xy[L_SH], h, w)
    rsx, rsy = _norm_xy_to_px(xy[R_SH], h, w)
    shoulder_mid_x = int(round((lsx + rsx) / 2.0))
    shoulder_mid_y = int(round((lsy + rsy) / 2.0))

    neck_rect = _rect_from_center(
        shoulder_mid_x,
        shoulder_mid_y - max(8, int(0.025 * h)),
        max(10, int(0.03 * w)),
        max(8, int(0.02 * h)),
        h,
        w,
    )

    abdomen_rect = None
    if all(float(vis[idx]) > 0.35 for idx in [L_SH, R_SH, L_HIP, R_HIP]):
        lhx, lhy = _norm_xy_to_px(xy[L_HIP], h, w)
        rhx, rhy = _norm_xy_to_px(xy[R_HIP], h, w)
        hip_mid_x = int(round((lhx + rhx) / 2.0))
        hip_mid_y = int(round((lhy + rhy) / 2.0))
        abdomen_rect = _rect_from_center(
            int(round((shoulder_mid_x + hip_mid_x) / 2.0)),
            int(round(shoulder_mid_y + 0.60 * (hip_mid_y - shoulder_mid_y))),
            max(12, int(0.035 * w)),
            max(10, int(0.025 * h)),
            h,
            w,
        )

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

    purities: List[float] = [face_purity]

    arm_labs = []
    for rect in [limb_mid_rect(L_SH, L_EL), limb_mid_rect(R_SH, R_EL)]:
        mask, purity = patch_mask_from_rect(rect)
        purities.append(purity)
        lab = _median_lab_in_region(img_bgr, mask) if mask is not None else None
        if lab is not None:
            arm_labs.append(lab)
    arm_lab = np.median(np.stack(arm_labs, axis=0), axis=0).astype(np.float32) if arm_labs else None

    thigh_labs = []
    for rect in [limb_mid_rect(L_HIP, L_KNEE, 0.035, 0.030), limb_mid_rect(R_HIP, R_KNEE, 0.035, 0.030)]:
        mask, purity = patch_mask_from_rect(rect, min_pixels=36, min_purity=0.10)
        purities.append(purity)
        lab = _median_lab_in_region(img_bgr, mask) if mask is not None else None
        if lab is not None:
            thigh_labs.append(lab)
    thigh_lab = np.median(np.stack(thigh_labs, axis=0), axis=0).astype(np.float32) if thigh_labs else None

    calf_labs = []
    for rect in [limb_mid_rect(L_KNEE, L_ANK, 0.032, 0.028), limb_mid_rect(R_KNEE, R_ANK, 0.032, 0.028)]:
        mask, purity = patch_mask_from_rect(rect, min_pixels=34, min_purity=0.10)
        purities.append(purity)
        lab = _median_lab_in_region(img_bgr, mask) if mask is not None else None
        if lab is not None:
            calf_labs.append(lab)
    calf_lab = np.median(np.stack(calf_labs, axis=0), axis=0).astype(np.float32) if calf_labs else None

    neck_mask, purity = patch_mask_from_rect(neck_rect)
    purities.append(purity)
    neck_lab = _median_lab_in_region(img_bgr, neck_mask) if neck_mask is not None else None

    abdomen_mask, purity = patch_mask_from_rect(abdomen_rect)
    purities.append(purity)
    abdomen_lab = _median_lab_in_region(img_bgr, abdomen_mask) if abdomen_mask is not None else None

    out["face_neck_deltaE"] = _delta_e_lab(face_lab, neck_lab)
    out["face_arm_deltaE"] = _delta_e_lab(face_lab, arm_lab)
    out["face_abdomen_deltaE"] = _delta_e_lab(face_lab, abdomen_lab)
    out["face_thigh_deltaE"] = _delta_e_lab(face_lab, thigh_lab)
    out["face_calf_deltaE"] = _delta_e_lab(face_lab, calf_lab)

    leg_l_vals = [float(lab[0]) for lab in [thigh_lab, calf_lab] if lab is not None]
    if leg_l_vals:
        out["leg_brightness_ratio"] = float(np.mean(leg_l_vals) / max(1e-6, float(face_lab[0])))

    knee_scores = []
    if thigh_lab is not None and calf_lab is not None:
        knee_l_proxy = float((thigh_lab[0] + calf_lab[0]) / 2.0)
        ratio = knee_l_proxy / max(1e-6, float(face_lab[0]))
        knee_scores.append(soft_range_score(ratio, 0.82, 1.08, 0.22))
    out["knee_dark_patch_score"] = float(np.mean(knee_scores)) if knee_scores else None

    out["skin_uniformity_score"] = weighted_mean_valid(
        [
            (
                None if out["face_thigh_deltaE"] is None else float(math.exp(-out["face_thigh_deltaE"] / 13.5)),
                0.38,
            ),
            (
                None if out["face_calf_deltaE"] is None else float(math.exp(-out["face_calf_deltaE"] / 14.5)),
                0.34,
            ),
            (soft_range_score(out["leg_brightness_ratio"], 0.84, 1.08, 0.22), 0.18),
            (out["knee_dark_patch_score"], 0.10),
        ]
    )

    avail_main = sum(
        1 for value in [out["face_thigh_deltaE"], out["face_calf_deltaE"], out["leg_brightness_ratio"]] if value is not None
    )
    purity_mean = float(np.mean(np.array(purities, dtype=np.float32))) if purities else 0.0
    conf = weighted_mean_valid(
        [
            (float(avail_main) / 3.0, 0.55),
            (purity_mean, 0.45),
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
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "is_valid": False,
        "confidence": 0.0,
        "torso_volume_score": None,
        "pelvis_depth_score": None,
        "fake_turn_risk": None,
        "depth_3d_score": None,
        "reasons": [],
    }

    if not pose_feat.ok:
        out["reasons"].append("DEPTH_3D_LITE_NOT_AVAILABLE")
        return out

    sw = pose_feat.upper_geom.get("shoulder_width_norm", None)
    hw = pose_feat.upper_geom.get("hip_width_norm", None)
    spine_angle = pose_feat.upper_geom.get("spine_angle_deg", None)
    torso_len = pose_feat.upper_geom.get("torso_len_norm", None)
    if sw is None or hw is None:
        out["reasons"].append("DEPTH_3D_LITE_GEOM_MISSING")
        return out

    hip_shoulder_ratio = float(hw / max(1e-6, sw))
    if view_bucket == "front":
        turn_score = soft_range_score(yaw_proxy, 0.00, 0.10, 0.06)
        shoulder_score = soft_range_score(sw, 0.22, 0.31, 0.08)
        hip_score = soft_range_score(hw, 0.11, 0.19, 0.06)
    elif view_bucket == "three_quarter":
        turn_score = soft_range_score(yaw_proxy, 0.10, 0.28, 0.08)
        shoulder_score = soft_range_score(sw, 0.21, 0.28, 0.07)
        hip_score = soft_range_score(hw, 0.10, 0.18, 0.06)
    else:
        turn_score = soft_range_score(yaw_proxy, 0.24, 0.42, 0.10)
        shoulder_score = soft_range_score(sw, 0.16, 0.24, 0.08)
        hip_score = soft_range_score(hw, 0.08, 0.16, 0.06)

    ratio_score = soft_range_score(hip_shoulder_ratio, 0.46, 0.72, 0.18)
    spine_score = soft_range_score(spine_angle, 0.0, 10.0, 6.0)
    torso_score = soft_range_score(torso_len, 0.20, 0.30, 0.08)

    torso_volume_score = weighted_mean_valid(
        [
            (shoulder_score, 0.30),
            (hip_score, 0.24),
            (ratio_score, 0.20),
            (spine_score, 0.14),
            (torso_score, 0.12),
        ]
    )

    out["fake_turn_risk"] = float(1.0 - torso_volume_score) if torso_volume_score is not None else None
    out["depth_3d_score"] = weighted_mean_valid(
        [
            (turn_score, 0.24),
            (torso_volume_score, 0.52),
            (spine_score, 0.14),
            (torso_score, 0.10),
        ]
    )

    if view_bucket == "front":
        conf = 0.20
    elif view_bucket == "three_quarter":
        conf = weighted_mean_valid(
            [
                (soft_range_score(yaw_proxy, 0.10, 0.28, 0.10), 0.45),
                (torso_volume_score, 0.35),
                (spine_score, 0.20),
            ]
        )
    else:
        conf = weighted_mean_valid(
            [
                (soft_range_score(yaw_proxy, 0.24, 0.42, 0.14), 0.45),
                (torso_volume_score, 0.35),
                (spine_score, 0.20),
            ]
        )

    out["confidence"] = 0.0 if conf is None else float(conf)
    out["torso_volume_score"] = torso_volume_score
    out["pelvis_depth_score"] = ratio_score
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
    gate_debug["skin"] = {"score": s_score, "confidence": s_conf, "valid": s_valid}
    if s_valid and s_score is not None:
        reasons_all.extend(skin_metrics.get("reasons", []))
        severe_skin = False
        thigh_de = skin_metrics.get("face_thigh_deltaE", None)
        calf_de = skin_metrics.get("face_calf_deltaE", None)
        leg_ratio = skin_metrics.get("leg_brightness_ratio", None)
        if (thigh_de is not None and float(thigh_de) > 24.0) or (calf_de is not None and float(calf_de) > 28.0):
            severe_skin = True
        if leg_ratio is not None and float(leg_ratio) < 0.78:
            severe_skin = True

        if s_conf >= consistency.skin_min_conf:
            if severe_skin and float(s_score) < consistency.skin_soft_warn_th:
                reasons_all.append("SKIN_UNIFORMITY_STRONG_WARN")
                downgrade_pass_to_warn("SKIN_UNIFORMITY_STRONG_WARN")
            elif float(s_score) < consistency.skin_soft_warn_th:
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
        if view_bucket in {"three_quarter", "profile_like"}:
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
