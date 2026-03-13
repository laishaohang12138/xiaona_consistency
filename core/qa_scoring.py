from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .qa_runtime import AnchorSet, FaceFeat, PoseFeat, RuntimeContext
from .qa_utils import (
    clamp,
    cosine_sim,
    get_face_size_bucket,
    hist_intersection,
    infer_anchor_view_from_path,
    linear_map_to_01,
    phash_similarity,
    safe_float,
    ssim_similarity,
    valid_face_feats,
)


def geom_similarity_face(g1: Dict[str, float], g2: Dict[str, float]) -> Optional[float]:
    keys = ["eye_dist_norm", "eye_y_norm", "nose_y_norm", "mouth_y_norm", "mouth_w_norm", "face_ar"]
    if not all(key in g1 for key in keys) or not all(key in g2 for key in keys):
        return None

    vals = []
    for key in keys:
        a = float(g1[key])
        b = float(g2[key])
        denom = max(1e-6, abs(a) + abs(b))
        vals.append(abs(a - b) / denom)

    if "eye_tilt_deg" in g1 and "eye_tilt_deg" in g2:
        tilt_diff = abs(float(g1["eye_tilt_deg"]) - float(g2["eye_tilt_deg"]))
        tilt_diff = min(tilt_diff, 360.0 - tilt_diff)
        vals.append(tilt_diff / 30.0)

    return clamp(1.0 - float(np.mean(vals)), 0.0, 1.0)


def calibrate_face_embedding_score(raw_cos: float, engine: str) -> float:
    if engine == "insightface":
        low, high = 0.45, 0.70
    else:
        low, high = 0.80, 0.93
    return linear_map_to_01(raw_cos, low, high)


def compare_face_feat(
    candidate: FaceFeat,
    anchor: FaceFeat,
    face_engine_mode: str,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "embedding": None,
        "geom": None,
        "hog": None,
        "lbp": None,
        "phash": None,
        "ssim": None,
        "luma": None,
        "chroma": None,
        "sharp": None,
        "texture": None,
    }

    emb_cos = cosine_sim(candidate.embedding, anchor.embedding)
    if emb_cos is not None:
        out["embedding"] = calibrate_face_embedding_score(emb_cos, face_engine_mode)

    if candidate.geom and anchor.geom:
        out["geom"] = geom_similarity_face(candidate.geom, anchor.geom)

    hog_sim = cosine_sim(candidate.hog_vec, anchor.hog_vec)
    if hog_sim is not None:
        out["hog"] = clamp((hog_sim + 1.0) / 2.0, 0.0, 1.0)

    lbp_sim = hist_intersection(candidate.lbp_hist, anchor.lbp_hist)
    if lbp_sim is not None:
        out["lbp"] = clamp(lbp_sim, 0.0, 1.0)

    ph_sim = phash_similarity(candidate.phash64, anchor.phash64)
    if ph_sim is not None:
        out["phash"] = clamp(ph_sim, 0.0, 1.0)

    ssim_sim = ssim_similarity(candidate.crop_gray_128, anchor.crop_gray_128)
    if ssim_sim is not None:
        out["ssim"] = clamp(ssim_sim, 0.0, 1.0)

    if candidate.lab_mean is not None and anchor.lab_mean is not None:
        cL, ca, cb = candidate.lab_mean
        aL, aa, ab = anchor.lab_mean
        out["luma"] = clamp(1.0 - abs(float(cL - aL)) / 18.0, 0.0, 1.0)
        dC = math.sqrt((float(ca - aa) ** 2) + (float(cb - ab) ** 2))
        out["chroma"] = clamp(1.0 - dC / 22.0, 0.0, 1.0)

    if candidate.lap_var > 0 and anchor.lap_var > 0:
        out["sharp"] = clamp(
            linear_map_to_01(candidate.lap_var / max(1e-6, anchor.lap_var), 0.55, 1.10),
            0.0,
            1.0,
        )

    if candidate.hf_energy > 0 and anchor.hf_energy > 0:
        out["texture"] = clamp(
            linear_map_to_01(candidate.hf_energy / max(1e-6, anchor.hf_energy), 0.55, 1.10),
            0.0,
            1.0,
        )

    return out


def fuse_face_identity_metrics(
    metrics: Dict[str, Optional[float]],
    view_bucket: str = "front",
) -> Tuple[float, Dict[str, float], List[str]]:
    if view_bucket == "front":
        base_weights = {
            "embedding": 0.42,
            "geom": 0.22,
            "hog": 0.14,
            "lbp": 0.10,
            "phash": 0.04,
            "ssim": 0.08,
        }
    elif view_bucket == "three_quarter":
        base_weights = {
            "embedding": 0.58,
            "geom": 0.10,
            "hog": 0.10,
            "lbp": 0.06,
            "phash": 0.02,
            "ssim": 0.14,
        }
    else:
        base_weights = {
            "embedding": 0.70,
            "geom": 0.05,
            "hog": 0.08,
            "lbp": 0.05,
            "phash": 0.00,
            "ssim": 0.12,
        }

    reasons: List[str] = []
    used: Dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for key, weight in base_weights.items():
        value = metrics.get(key, None)
        if value is None:
            reasons.append(f"FACE_ID_METRIC_MISSING_{key.upper()}")
            continue
        used[key] = float(value)
        weighted_sum += float(value) * weight
        weight_sum += weight

    if weight_sum < 1e-8:
        return 0.0, used, reasons

    return float(clamp(weighted_sum / weight_sum, 0.0, 1.0)), used, reasons


def score_face_against_anchor_set(
    runtime: RuntimeContext,
    candidate: FaceFeat,
    anchors: List[FaceFeat],
    view_bucket: str = "front",
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    debug: Dict[str, Any] = {"anchor_scores": []}

    if not candidate.ok:
        return 0.0, 0.0, ["FACE_NOT_AVAILABLE"] + candidate.reasons, debug
    if len(anchors) == 0:
        return 0.0, 0.0, ["NO_FACE_ANCHORS"], debug

    anchor_scores: List[Tuple[float, float]] = []
    anchor_used_metrics: List[Dict[str, Any]] = []

    for idx, anchor in enumerate(anchors):
        if not anchor.ok:
            continue
        metrics = compare_face_feat(candidate, anchor, runtime.engines.face_mode)
        fused, used, rs = fuse_face_identity_metrics(metrics, view_bucket=view_bucket)
        coverage = clamp(len(used) / 6.0, 0.0, 1.0)
        conf = clamp(candidate.confidence * anchor.confidence * (0.6 + 0.4 * coverage), 0.0, 1.0)

        anchor_scores.append((fused, conf))
        anchor_used_metrics.append(
            {
                "anchor_index": idx,
                "anchor_path": anchor.source_path,
                "score": fused,
                "conf": conf,
                "identity_metrics": used,
                "reasons": rs,
            }
        )

    if len(anchor_scores) == 0:
        return 0.0, 0.0, ["NO_VALID_FACE_ANCHORS"], debug

    anchor_scores_sorted = sorted(anchor_scores, key=lambda item: item[0], reverse=True)
    topk = anchor_scores_sorted[: min(3, len(anchor_scores_sorted))]
    scores = [score for score, _ in topk]
    confs = [conf for _, conf in topk]

    score = float(0.6 * np.mean(scores) + 0.4 * np.median(scores))
    conf = float(np.mean(confs))

    if candidate.bbox_area_ratio < 0.01:
        reasons.append("FACE_TOO_SMALL")
    if candidate.embedding is None:
        reasons.append("FACE_EMBEDDING_NOT_AVAILABLE")
    if not candidate.geom:
        reasons.append("FACE_GEOMETRY_NOT_AVAILABLE")

    debug["anchor_scores"] = anchor_used_metrics
    debug["bbox_area_ratio"] = candidate.bbox_area_ratio
    debug["face_size_bucket"] = get_face_size_bucket(candidate.bbox_area_ratio)
    debug["candidate_face_metrics_ready"] = {
        "embedding": candidate.embedding is not None,
        "geom": bool(candidate.geom),
        "hog": candidate.hog_vec is not None,
        "lbp": candidate.lbp_hist is not None,
        "phash": candidate.phash64 is not None,
        "ssim": candidate.crop_gray_128 is not None,
        "lab": candidate.lab_mean is not None,
        "lap_var": candidate.lap_var > 0,
        "hf_energy": candidate.hf_energy > 0,
    }
    return clamp(score, 0.0, 1.0), clamp(conf, 0.0, 1.0), reasons + candidate.reasons, debug


MP_IDS = {
    "NOSE": 0,
    "L_SH": 11,
    "R_SH": 12,
    "L_EL": 13,
    "R_EL": 14,
    "L_WR": 15,
    "R_WR": 16,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNEE": 25,
    "R_KNEE": 26,
    "L_ANK": 27,
    "R_ANK": 28,
}

UPPER_LM_IDS = [0, 11, 12, 13, 14, 23, 24]
FULL_LM_IDS = [0, 11, 12, 23, 24, 25, 26, 27, 28]


def normalize_pose_subset(
    xy: np.ndarray,
    vis: np.ndarray,
    ids: List[int],
    mode: str = "upper",
) -> Tuple[Optional[np.ndarray], float]:
    pts = np.array([xy[idx].copy() for idx in ids], dtype=np.float32)
    valid = np.array([1.0 if vis[idx] > 0.35 else 0.0 for idx in ids], dtype=np.float32)
    coverage = float(valid.mean())
    if coverage < 0.45:
        return None, coverage

    if mode == "upper":
        if vis[MP_IDS["L_SH"]] > 0.35 and vis[MP_IDS["R_SH"]] > 0.35:
            center = (xy[MP_IDS["L_SH"]] + xy[MP_IDS["R_SH"]]) / 2.0
            scale = float(np.linalg.norm(xy[MP_IDS["L_SH"]] - xy[MP_IDS["R_SH"]]))
        else:
            center = pts[0]
            scale = 0.0

        if (
            scale < 1e-5
            and vis[MP_IDS["L_HIP"]] > 0.35
            and vis[MP_IDS["R_HIP"]] > 0.35
            and vis[MP_IDS["L_SH"]] > 0.35
            and vis[MP_IDS["R_SH"]] > 0.35
        ):
            shoulder_mid = (xy[MP_IDS["L_SH"]] + xy[MP_IDS["R_SH"]]) / 2.0
            hip_mid = (xy[MP_IDS["L_HIP"]] + xy[MP_IDS["R_HIP"]]) / 2.0
            scale = float(np.linalg.norm(hip_mid - shoulder_mid))
    else:
        center = (xy[MP_IDS["L_HIP"]] + xy[MP_IDS["R_HIP"]]) / 2.0 if (
            vis[MP_IDS["L_HIP"]] > 0.35 and vis[MP_IDS["R_HIP"]] > 0.35
        ) else pts[0]

        candidates = []
        if vis[MP_IDS["L_SH"]] > 0.35 and vis[MP_IDS["R_SH"]] > 0.35:
            candidates.append(float(np.linalg.norm(xy[MP_IDS["L_SH"]] - xy[MP_IDS["R_SH"]])))
        if vis[MP_IDS["L_HIP"]] > 0.35 and vis[MP_IDS["R_HIP"]] > 0.35:
            candidates.append(float(np.linalg.norm(xy[MP_IDS["L_HIP"]] - xy[MP_IDS["R_HIP"]])))
        if vis[MP_IDS["NOSE"]] > 0.2 and vis[MP_IDS["L_ANK"]] > 0.2 and vis[MP_IDS["R_ANK"]] > 0.2:
            ankles_mid = (xy[MP_IDS["L_ANK"]] + xy[MP_IDS["R_ANK"]]) / 2.0
            candidates.append(float(np.linalg.norm(ankles_mid - xy[MP_IDS["NOSE"]])) * 0.35)
        scale = float(np.mean(candidates)) if candidates else 0.0

    if scale < 1e-5:
        return None, coverage

    norm_pts = (pts - center[None, :]) / scale
    for idx, flag in enumerate(valid):
        if flag < 0.5:
            norm_pts[idx] = 0.0
    return norm_pts.reshape(-1).astype(np.float32), coverage


def pose_vector_similarity(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> Optional[float]:
    if v1 is None or v2 is None or v1.shape != v2.shape or v1.ndim != 1 or (v1.size % 2 != 0):
        return None

    try:
        pts1 = v1.reshape(-1, 2).astype(np.float32)
        pts2 = v2.reshape(-1, 2).astype(np.float32)

        eps = 1e-8
        mask1 = ~(np.all(np.abs(pts1) < eps, axis=1))
        mask2 = ~(np.all(np.abs(pts2) < eps, axis=1))
        joint_mask = mask1 & mask2

        visible_count = int(np.sum(joint_mask))
        total_count = int(pts1.shape[0])
        if visible_count < 3:
            cos = cosine_sim(v1, v2)
            if cos is None:
                return None
            cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0)
            dist01 = float(math.exp(-2.5 * float(np.linalg.norm(v1 - v2))))
            return clamp(0.6 * cos01 + 0.4 * dist01, 0.0, 1.0)

        X = pts1[joint_mask].copy()
        Y = pts2[joint_mask].copy()
        Xc = X - np.mean(X, axis=0, keepdims=True)
        Yc = Y - np.mean(Y, axis=0, keepdims=True)

        X_norm = float(np.linalg.norm(Xc))
        Y_norm = float(np.linalg.norm(Yc))
        if X_norm < 1e-8 or Y_norm < 1e-8:
            cos = cosine_sim(v1, v2)
            return clamp((cos + 1.0) / 2.0, 0.0, 1.0) if cos is not None else None

        Xn = Xc / X_norm
        Yn = Yc / Y_norm
        H = Yn.T @ Xn
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = U @ Vt

        Yn_aligned = Yn @ R
        rmse = float(np.linalg.norm(Xn - Yn_aligned) / np.sqrt(Xn.shape[0]))
        shape_sim = float(math.exp(-4.0 * rmse))

        cos = cosine_sim(v1, v2)
        cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0) if cos is not None else 0.0
        support_ratio = visible_count / max(1, total_count)

        fused = (0.72 * shape_sim + 0.28 * cos01) * (0.90 + 0.10 * support_ratio)
        return clamp(fused, 0.0, 1.0)

    except Exception:
        cos = cosine_sim(v1, v2)
        if cos is None:
            return None
        cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        dist01 = float(math.exp(-2.5 * float(np.linalg.norm(v1 - v2))))
        return clamp(0.6 * cos01 + 0.4 * dist01, 0.0, 1.0)


def upper_geom_similarity(
    g1: Dict[str, float],
    g2: Dict[str, float],
    view_bucket: str = "front",
) -> Optional[float]:
    if view_bucket == "front":
        weight_map = {
            "shoulder_width_norm": 0.45,
            "torso_len_norm": 0.35,
            "hip_width_norm": 0.20,
        }
        tilt_weight = 0.10
        spine_weight = 0.04
    elif view_bucket == "three_quarter":
        weight_map = {
            "shoulder_width_norm": 0.10,
            "torso_len_norm": 0.50,
            "hip_width_norm": 0.40,
        }
        tilt_weight = 0.04
        spine_weight = 0.10
    else:
        weight_map = {
            "torso_len_norm": 0.34,
            "shoulder_hip_center_offset_norm": 0.24,
            "torso_compactness": 0.18,
            "hip_width_norm": 0.14,
            "hip_shoulder_ratio": 0.10,
        }
        tilt_weight = 0.00
        spine_weight = 0.18

    avail = [key for key in weight_map.keys() if key in g1 and key in g2]
    if len(avail) == 0:
        return None

    errs = []
    weights = []
    for key in avail:
        a = float(g1[key])
        b = float(g2[key])
        denom = max(1e-6, abs(a) + abs(b))
        errs.append(abs(a - b) / denom)
        weights.append(weight_map[key])

    if tilt_weight > 0 and "shoulder_tilt_deg" in g1 and "shoulder_tilt_deg" in g2:
        d = abs(float(g1["shoulder_tilt_deg"]) - float(g2["shoulder_tilt_deg"]))
        errs.append(min(d, 360.0 - d) / 25.0)
        weights.append(tilt_weight)

    if spine_weight > 0 and "spine_angle_deg" in g1 and "spine_angle_deg" in g2:
        d = abs(float(g1["spine_angle_deg"]) - float(g2["spine_angle_deg"]))
        errs.append(min(d, 360.0 - d) / 18.0)
        weights.append(spine_weight)

    return clamp(
        1.0 - float(np.average(np.array(errs, dtype=np.float32), weights=np.array(weights, dtype=np.float32))),
        0.0,
        1.0,
    )


def full_geom_similarity(
    g1: Dict[str, float],
    g2: Dict[str, float],
    view_bucket: str = "front",
) -> Optional[float]:
    if view_bucket == "front":
        weight_map = {
            "head_body_ratio": 0.58,
            "leg_ratio": 0.42,
        }
    elif view_bucket == "three_quarter":
        weight_map = {
            "head_body_ratio": 0.46,
            "leg_ratio": 0.36,
            "leg_straightness_mean_deg": 0.10,
            "ankle_gap_norm": 0.08,
        }
    else:
        weight_map = {
            "head_body_ratio": 0.22,
            "leg_ratio": 0.28,
            "leg_straightness_min_deg": 0.24,
            "leg_straightness_mean_deg": 0.12,
            "ankle_gap_norm": 0.08,
            "foot_length_proxy_norm": 0.06,
        }

    avail = [key for key in weight_map.keys() if key in g1 and key in g2]
    if len(avail) == 0:
        return None

    errs = []
    weights = []
    for key in avail:
        a = float(g1[key])
        b = float(g2[key])
        if key.endswith("_deg"):
            errs.append(abs(a - b) / 18.0)
        else:
            denom = max(1e-6, abs(a) + abs(b))
            errs.append(abs(a - b) / denom)
        weights.append(weight_map[key])
    return clamp(
        1.0 - float(np.average(np.array(errs, dtype=np.float32), weights=np.array(weights, dtype=np.float32))),
        0.0,
        1.0,
    )


def framing_score_from_pose_feat(pose_feat: PoseFeat) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if not pose_feat.ok:
        return 0.0, ["POSE_NOT_AVAILABLE"]

    if pose_feat.mode == "opencv":
        framing = pose_feat.framing
        subj = safe_float(framing.get("subject_height_ratio", 0.0))
        headroom = safe_float(framing.get("headroom_ratio", 1.0))
        feet = safe_float(framing.get("feet_in_frame", 0.0))
        score = 0.4 * linear_map_to_01(subj, 0.45, 0.85)
        score += 0.3 * (1.0 - clamp(abs(headroom - 0.08) / 0.20, 0.0, 1.0))
        score += 0.3 * feet
        reasons.append("FRAMING_APPROX_OPENCV")
        return clamp(score, 0.0, 1.0), reasons

    framing = pose_feat.framing
    feet = safe_float(framing.get("feet_in_frame", 0.0))
    subj = safe_float(framing.get("subject_height_ratio", 0.0))
    headroom = safe_float(framing.get("headroom_ratio", 1.0))

    score_feet = feet
    score_subj = 1.0 - clamp(abs(subj - 0.82) / 0.22, 0.0, 1.0)
    score_headroom = 1.0 - clamp(abs(headroom - 0.07) / 0.12, 0.0, 1.0)
    score = 0.40 * score_feet + 0.35 * score_subj + 0.25 * score_headroom

    if feet < 0.5:
        reasons.append("FEET_CROPPED_OR_NOT_IN_FRAME")
    if score_subj < 0.6:
        reasons.append("SUBJECT_SCALE_OFF")
    if score_headroom < 0.6:
        reasons.append("HEADROOM_OFF")
    if not reasons:
        reasons.append("FRAMING_OK")
    return clamp(score, 0.0, 1.0), reasons


def score_upper_against_anchor_set(
    candidate: PoseFeat,
    anchors: List[PoseFeat],
    view_bucket: str = "front",
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    debug: Dict[str, Any] = {"anchor_scores": [], "view_bucket_used": view_bucket}

    if not candidate.ok:
        return 0.0, 0.0, ["UPPER_POSE_NOT_AVAILABLE"] + candidate.reasons, debug
    if len(anchors) == 0:
        return 0.0, 0.0, ["NO_UPPER_ANCHORS"], debug

    if candidate.mode == "opencv":
        fr_score, fr_reasons = framing_score_from_pose_feat(candidate)
        conf = clamp(candidate.confidence_upper, 0.0, 1.0)
        return fr_score * 0.6 + 0.2, conf, ["UPPER_APPROX_ONLY_OPENCV"] + fr_reasons + candidate.reasons, debug

    vec_c, cov_c = normalize_pose_subset(candidate.lm_xy, candidate.lm_vis, UPPER_LM_IDS, mode="upper")

    scores: List[Tuple[float, float]] = []
    for idx, anchor in enumerate(anchors):
        if not anchor.ok or anchor.mode != "mediapipe" or anchor.lm_xy is None or anchor.lm_vis is None:
            continue

        vec_a, cov_a = normalize_pose_subset(anchor.lm_xy, anchor.lm_vis, UPPER_LM_IDS, mode="upper")
        s_pose = pose_vector_similarity(vec_c, vec_a)
        s_geom = upper_geom_similarity(candidate.upper_geom, anchor.upper_geom, view_bucket=view_bucket)

        parts = []
        if s_pose is not None:
            parts.append(("pose", s_pose, 0.35))
        if s_geom is not None:
            parts.append(("geom", s_geom, 0.65))
        if len(parts) == 0:
            continue

        ws = sum(weight for _, _, weight in parts)
        fused = sum(value * weight for _, value, weight in parts) / max(1e-8, ws)
        conf = clamp((cov_c * cov_a) * 0.9 + 0.1, 0.0, 1.0)

        scores.append((fused, conf))
        debug["anchor_scores"].append(
            {
                "anchor_index": idx,
                "pose_score": s_pose,
                "geom_score": s_geom,
                "fused": fused,
                "conf": conf,
            }
        )

    if len(scores) == 0:
        return 0.0, clamp(candidate.confidence_upper, 0.0, 1.0), ["NO_VALID_UPPER_ANCHORS"] + candidate.reasons, debug

    topk = sorted(scores, key=lambda item: item[0], reverse=True)[: min(3, len(scores))]
    svals = [score for score, _ in topk]
    cvals = [conf for _, conf in topk]
    score = float(0.6 * np.mean(svals) + 0.4 * np.median(svals))
    conf = float(np.mean(cvals))

    if candidate.confidence_upper < 0.5:
        reasons.append("UPPER_KEYPOINTS_LOW_CONFIDENCE")

    return clamp(score, 0.0, 1.0), clamp(conf, 0.0, 1.0), reasons + candidate.reasons, debug


def score_full_against_anchor_set(
    candidate: PoseFeat,
    anchors: List[PoseFeat],
    view_bucket: str = "front",
) -> Tuple[float, float, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    debug: Dict[str, Any] = {"anchor_scores": [], "view_bucket_used": view_bucket}

    if not candidate.ok:
        return 0.0, 0.0, ["FULL_POSE_NOT_AVAILABLE"] + candidate.reasons, debug
    if len(anchors) == 0:
        return 0.0, 0.0, ["NO_FULL_ANCHORS"], debug

    fr_score, fr_reasons = framing_score_from_pose_feat(candidate)
    reasons.extend(fr_reasons)

    if candidate.mode == "opencv":
        conf = clamp(candidate.confidence_full, 0.0, 1.0)
        score = 0.8 * fr_score + 0.2 * linear_map_to_01(candidate.person_bbox_area_ratio, 0.12, 0.45)
        reasons.append("FULL_APPROX_ONLY_OPENCV")
        return clamp(score, 0.0, 1.0), conf, reasons + candidate.reasons, debug

    vec_c, cov_c = normalize_pose_subset(candidate.lm_xy, candidate.lm_vis, FULL_LM_IDS, mode="full")

    scores: List[Tuple[float, float]] = []
    for idx, anchor in enumerate(anchors):
        if not anchor.ok or anchor.mode != "mediapipe" or anchor.lm_xy is None or anchor.lm_vis is None:
            continue

        vec_a, cov_a = normalize_pose_subset(anchor.lm_xy, anchor.lm_vis, FULL_LM_IDS, mode="full")
        s_pose = pose_vector_similarity(vec_c, vec_a)
        s_geom = full_geom_similarity(candidate.full_geom, anchor.full_geom, view_bucket=view_bucket)

        if view_bucket == "front":
            part_weights = {"framing": 0.45, "geom": 0.35, "pose": 0.20}
        elif view_bucket == "three_quarter":
            part_weights = {"framing": 0.40, "geom": 0.40, "pose": 0.20}
        else:
            part_weights = {"framing": 0.30, "geom": 0.50, "pose": 0.20}

        parts = [("framing", fr_score, part_weights["framing"])]
        if s_geom is not None:
            parts.append(("geom", s_geom, part_weights["geom"]))
        if s_pose is not None:
            parts.append(("pose", s_pose, part_weights["pose"]))

        ws = sum(weight for _, _, weight in parts)
        fused = sum(value * weight for _, value, weight in parts) / max(1e-8, ws)
        conf = clamp((cov_c * cov_a) * 0.9 + 0.1, 0.0, 1.0)
        scores.append((fused, conf))
        debug["anchor_scores"].append(
            {
                "anchor_index": idx,
                "framing_score": fr_score,
                "pose_score": s_pose,
                "geom_score": s_geom,
                "fused": fused,
                "conf": conf,
            }
        )

    if len(scores) == 0:
        return fr_score, clamp(candidate.confidence_full, 0.0, 1.0), ["NO_VALID_FULL_ANCHORS"] + reasons + candidate.reasons, debug

    topk = sorted(scores, key=lambda item: item[0], reverse=True)[: min(3, len(scores))]
    svals = [score for score, _ in topk]
    cvals = [conf for _, conf in topk]
    score = float(0.6 * np.mean(svals) + 0.4 * np.median(svals))
    conf = float(np.mean(cvals))

    if candidate.confidence_full < 0.5:
        reasons.append("FULL_KEYPOINTS_LOW_CONFIDENCE")

    return clamp(score, 0.0, 1.0), clamp(conf, 0.0, 1.0), reasons + candidate.reasons, debug


def classify_module(
    runtime: RuntimeContext,
    score: float,
    conf: float,
    pass_th: float,
    warn_th: float,
    module_name: str,
) -> Tuple[str, List[str]]:
    if conf < runtime.config.review.face_no_signal_conf_th:
        return "FAIL", [f"{module_name.upper()}_NO_RELIABLE_SIGNAL"]

    if conf < runtime.config.review.min_conf_for_strict_fail:
        reasons = [f"{module_name.upper()}_LOW_CONFIDENCE"]
        if score >= warn_th:
            return "WARN", reasons + [f"{module_name.upper()}_LOW_CONF_BUT_ACCEPTABLE"]
        return "WARN", reasons + [f"{module_name.upper()}_LOW_CONF_NEEDS_REVIEW"]

    if score >= pass_th:
        return "PASS", [f"{module_name.upper()}_PASS"]
    if score >= warn_th:
        return "WARN", [f"{module_name.upper()}_WARN"]
    return "FAIL", [f"{module_name.upper()}_FAIL"]


def fuse_overall(scores: Dict[str, float], confs: Dict[str, float], weights: Dict[str, float]) -> float:
    ws = 0.0
    acc = 0.0
    for key in ["face", "upper", "full"]:
        weight = float(weights.get(key, 0.0))
        conf = float(confs.get(key, 0.0))
        score = float(scores.get(key, 0.0))
        eff = weight * (0.25 + 0.75 * conf)
        acc += score * eff
        ws += eff
    if ws < 1e-8:
        return 0.0
    return float(acc / ws)


def make_recommendations(
    runtime: RuntimeContext,
    result: Dict[str, Any],
    profile_name: str,
) -> List[str]:
    del profile_name
    recs: List[str] = []
    scores = result.get("scores", {})
    confs = result.get("confidence", {})
    reasons = result.get("reasons", [])

    face_s = float(scores.get("face", 0.0))
    upper_s = float(scores.get("upper", 0.0))
    full_s = float(scores.get("full", 0.0))
    face_c = float(confs.get("face", 0.0))

    constitution_s = scores.get("constitution", None)
    skin_s = scores.get("skin", None)
    depth_s = scores.get("depth_3d", None)

    if any("FACE_TOO_SMALL" in reason for reason in reasons) or face_c < 0.5:
        recs.append("提高脸部有效像素（拉近构图或提升分辨率），否则身份分会抖动")

    if "FACE_UNDEREXPOSED_DARK" in reasons:
        recs.append("脸部实测欠曝：补正面填充光或锁曝光，不要让脸比身体更暗")
    elif "FACE_DARKER_THAN_ANCHOR" in reasons:
        recs.append("相对锚点略暗：这更像拍摄条件差异，先复查是否只是远景/补光变化")

    if any(
        flag in reasons
        for flag in [
            "FACE_TOO_SOFT_POSSIBLE_SMOOTHING",
            "FACE_LOW_MICROTEXTURE",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_LOWER_TEXTURE_THAN_ANCHOR",
        ]
    ):
        recs.append("皮肤细节偏软：提高脸部有效像素或减少磨皮/过强降噪")

    if face_s < 0.65:
        recs.append("身份相似度偏低：优先检查锚点一致性、脸部占比与 ref 冲突")
    if upper_s < 0.65:
        recs.append("半身比例/体态不稳：优先用 upper anchor 稳肩颈与锁骨")
    if full_s < 0.65:
        recs.append("全身构图/姿态不稳：先修全身入框（脚/头留白/主体占比）")

    consistency = runtime.config.consistency
    if constitution_s is not None and float(constitution_s) < consistency.constitution_soft_warn_th:
        recs.append("身材宪法偏移：复查腰线、骨盆紧凑度、腿型细长度与下半身轻盈感")
    if "SKIN_SAMPLE_RISK_HIGH" in reasons:
        recs.append("皮肤采样风险过高：腿部靠边、遮挡、purity 过低或区域太小，这类图不适合作为稳定肤色样本")
    elif "SKIN_SAMPLE_RISK_WARN" in reasons:
        recs.append("皮肤采样稳定性一般：复查腿部 patch 是否靠边、过小或被衣物/暗角污染")
    if "SKIN_LIGHTING_RISK_HIGH" in reasons:
        recs.append("光影风险过高：脸腿受光条件差异过大，这类图应降为观察样本，不要靠放宽肤色阈值放行")
    elif "SKIN_LIGHTING_RISK_WARN" in reasons:
        recs.append("光影差异偏大：优先复查面光、腿部阴影和局部高光，不要把受光差误当成肤色差")
    if skin_s is not None and float(skin_s) < consistency.skin_soft_warn_th:
        recs.append("肤色一致性不足：复查脸-脖子-腿部亮度与色偏，优先排查腿部偏暗/偏黄/膝盖脏影")
    if depth_s is not None and float(depth_s) < consistency.depth3d_soft_warn_th:
        recs.append("3/4 空间厚度不足：疑似假转体或 2.5D 贴脸，建议补肩胯透视与胸廓厚度")

    if any("FEET_CROPPED" in reason for reason in reasons):
        recs.append("构图返工：确保脚完整入框（full-body 任务为硬条件）")

    if False:
        recs.append("一致性表现良好，可进入人工终审/训练入库阶段")
    if "FACE_DARKER_THAN_TONE_ANCHOR" in reasons:
        recs.append("脸部相对 tone anchor 偏暗：优先复查补光和曝光，不要把受光差误判为身份偏移")
    elif "FACE_BRIGHTER_THAN_TONE_ANCHOR" in reasons:
        recs.append("脸部相对 tone anchor 偏亮：优先复查过曝或美白偏移，再判断是否是真实肤色漂移")

    if "VIEW_LANE_NOT_ALLOWED_FOR_PROFILE" in reasons:
        recs.append("当前视角不在该 profile 放行范围内：请改用匹配 profile，或转入 shadow lane 观察")
    if "PROFILE_PASS_CAPPED_TO_WARN" in reasons or "PROFILE_LIKE_NO_SIDE_ANCHOR_PASS_CAPPED" in reasons:
        recs.append("该图当前只适合作为 shadow / review 样本，不建议直接晋升到正式 frozen 配额")
    if len(recs) == 0:
        recs.append("该图未触发明显警报，可进入人工终审或训练入库下一步")
    return recs


def get_profile_policy(runtime: RuntimeContext, profile_name: str) -> Dict[str, Any]:
    return runtime.config.profile_policy.get(profile_name, runtime.config.profile_policy["lora_dataset"])


def get_identity_anchor_pool(
    runtime: RuntimeContext,
    profile_name: str,
    anchors: AnchorSet,
) -> List[FaceFeat]:
    policy = get_profile_policy(runtime, profile_name)
    mode = policy.get("identity_anchor_pool", "face")
    if mode == "face":
        return valid_face_feats(anchors.face_feats)
    if mode == "upper_first":
        pool = valid_face_feats(anchors.upper_face_feats)
        return pool if len(pool) > 0 else valid_face_feats(anchors.face_feats)
    return valid_face_feats(anchors.face_feats)


def filter_face_anchors_by_view(anchors: List[FaceFeat], view_bucket: str) -> List[FaceFeat]:
    valid = [anchor for anchor in anchors if anchor.ok]
    normalized_view = "profile_like" if view_bucket == "side_90" else view_bucket
    if view_bucket == "front":
        pool = [anchor for anchor in valid if infer_anchor_view_from_path(anchor.source_path) == "front"]
        return pool if len(pool) > 0 else valid
    if normalized_view == "three_quarter":
        pool = [anchor for anchor in valid if infer_anchor_view_from_path(anchor.source_path) == "three_quarter"]
        return pool if len(pool) > 0 else valid
    if normalized_view == "profile_like":
        pool = [anchor for anchor in valid if infer_anchor_view_from_path(anchor.source_path) == "profile_like"]
        if len(pool) > 0:
            return pool
        pool = [anchor for anchor in valid if infer_anchor_view_from_path(anchor.source_path) == "three_quarter"]
        return pool if len(pool) > 0 else valid
    return valid


def get_quality_anchor_pool(
    runtime: RuntimeContext,
    profile_name: str,
    anchors: AnchorSet,
) -> List[FaceFeat]:
    policy = get_profile_policy(runtime, profile_name)
    mode = policy.get("quality_anchor_pool", "face")

    if mode == "face":
        return valid_face_feats(anchors.face_feats)
    if mode == "upper_first":
        pool = valid_face_feats(anchors.upper_face_feats)
        return pool if len(pool) > 0 else valid_face_feats(anchors.face_feats)
    if mode == "upper_or_full":
        pool = valid_face_feats(anchors.upper_face_feats)
        if len(pool) > 0:
            return pool
        pool = valid_face_feats(anchors.full_face_feats)
        if len(pool) > 0:
            return pool
        return valid_face_feats(anchors.face_feats)
    return valid_face_feats(anchors.face_feats)


def get_tone_anchor_pool(
    runtime: RuntimeContext,
    profile_name: str,
    anchors: AnchorSet,
) -> List[FaceFeat]:
    policy = get_profile_policy(runtime, profile_name)
    mode = policy.get("tone_anchor_pool", policy.get("quality_anchor_pool", "face"))

    if mode == "face":
        return valid_face_feats(anchors.face_feats)
    if mode == "upper_first":
        pool = valid_face_feats(anchors.upper_face_feats)
        return pool if len(pool) > 0 else valid_face_feats(anchors.face_feats)
    if mode == "upper_or_full":
        pool = valid_face_feats(anchors.upper_face_feats)
        if len(pool) > 0:
            return pool
        pool = valid_face_feats(anchors.full_face_feats)
        if len(pool) > 0:
            return pool
        return valid_face_feats(anchors.face_feats)
    return valid_face_feats(anchors.face_feats)


def build_quality_reference_stats(face_feats: List[FaceFeat]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "all": {"L": [], "lap": [], "hf": [], "count": 0},
        "near": {"L": [], "lap": [], "hf": [], "count": 0},
        "mid": {"L": [], "lap": [], "hf": [], "count": 0},
        "full_far": {"L": [], "lap": [], "hf": [], "count": 0},
    }

    for anchor in face_feats:
        if not anchor.ok:
            continue
        bucket = get_face_size_bucket(anchor.bbox_area_ratio)
        for key in ["all", bucket]:
            if anchor.lab_mean is not None:
                stats[key]["L"].append(float(anchor.lab_mean[0]))
            if anchor.lap_var > 0:
                stats[key]["lap"].append(float(anchor.lap_var))
            if anchor.hf_energy > 0:
                stats[key]["hf"].append(float(anchor.hf_energy))
            stats[key]["count"] += 1
    return stats


def build_tone_reference_stats(face_feats: List[FaceFeat]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "all": {"L": [], "a": [], "b": [], "count": 0},
        "near": {"L": [], "a": [], "b": [], "count": 0},
        "mid": {"L": [], "a": [], "b": [], "count": 0},
        "full_far": {"L": [], "a": [], "b": [], "count": 0},
    }

    for anchor in face_feats:
        if not anchor.ok or anchor.lab_mean is None:
            continue
        bucket = get_face_size_bucket(anchor.bbox_area_ratio)
        L, a, b = [float(x) for x in anchor.lab_mean]
        for key in ["all", bucket]:
            stats[key]["L"].append(L)
            stats[key]["a"].append(a)
            stats[key]["b"].append(b)
            stats[key]["count"] += 1
    return stats


def get_stats_for_bucket(stats: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    selected = stats.get(bucket, {})
    if selected.get("count", 0) >= 1:
        return selected
    return stats.get("all", {"L": [], "lap": [], "hf": [], "count": 0})
