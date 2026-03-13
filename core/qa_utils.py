from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .qa_runtime import FaceFeat, QualityThresholds, StandardizationSettings


SKIMAGE_SSIM_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as skimage_ssim

    SKIMAGE_SSIM_AVAILABLE = True
except Exception:
    SKIMAGE_SSIM_AVAILABLE = False


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def l2norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def linear_map_to_01(x: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return float((x - low) / (high - low))


def standardize_input_bgr(
    img_bgr: Optional[np.ndarray],
    settings: StandardizationSettings,
) -> Optional[np.ndarray]:
    if img_bgr is None:
        return None
    if not settings.enabled:
        return img_bgr

    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= 0:
        return img_bgr

    target = int(settings.long_side)
    if max_side > target:
        scale = target / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if settings.upscale_small_input and max_side < target:
        scale = target / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return img_bgr


def image_read_bgr(path: Path, settings: StandardizationSettings) -> Optional[np.ndarray]:
    return standardize_input_bgr(cv2.imread(str(path)), settings)


def resize_gray(img_gray: np.ndarray, size: int = 128) -> np.ndarray:
    return cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA)


def compute_hog_vec(gray128: np.ndarray) -> np.ndarray:
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    vec = hog.compute(gray128)
    if vec is None:
        return np.zeros((1,), dtype=np.float32)
    return l2norm(vec.astype(np.float32).reshape(-1))


def compute_lbp_hist(gray128: np.ndarray) -> np.ndarray:
    g = gray128.astype(np.uint8)
    h, w = g.shape
    if h < 3 or w < 3:
        return np.zeros((256,), dtype=np.float32)

    center = g[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        g[:-2, :-2],
        g[:-2, 1:-1],
        g[:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, :-2],
        g[1:-1, :-2],
    ]
    for i, n in enumerate(neighbors):
        lbp |= (n >= center).astype(np.uint8) << i

    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape(-1).astype(np.float32)
    s = float(hist.sum())
    if s > 1e-8:
        hist /= s
    return hist


def hist_intersection(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> Optional[float]:
    if h1 is None or h2 is None:
        return None
    h1 = h1.reshape(-1).astype(np.float32)
    h2 = h2.reshape(-1).astype(np.float32)
    if h1.shape != h2.shape:
        return None
    return float(np.minimum(h1, h2).sum())


def compute_phash64(gray128: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray128, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    flat = dct[:8, :8].copy().flatten()
    med = float(np.median(flat[1:]))
    return (flat > med).astype(np.uint8).reshape(-1)


def phash_similarity(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    if p1.shape != p2.shape:
        return None
    dist = int(np.sum(p1 != p2))
    return float(1.0 - dist / len(p1))


def ssim_similarity(gray_a: Optional[np.ndarray], gray_b: Optional[np.ndarray]) -> Optional[float]:
    if gray_a is None or gray_b is None:
        return None
    if gray_a.shape != gray_b.shape:
        return None

    a = gray_a.astype(np.uint8)
    b = gray_b.astype(np.uint8)
    if SKIMAGE_SSIM_AVAILABLE:
        try:
            value = skimage_ssim(a, b)
            return float(clamp((value + 1.0) / 2.0, 0.0, 1.0))
        except Exception:
            pass

    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom < 1e-8:
        return None
    ncc = float(np.dot(aa, bb) / denom)
    return float(clamp((ncc + 1.0) / 2.0, 0.0, 1.0))


def crop_safe(img: np.ndarray, xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def bbox_xywh_to_xyxy(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return (x, y, x + w, y + h)


def bbox_area_ratio_xyxy(xyxy: Tuple[int, int, int, int], img_shape: Tuple[int, ...]) -> float:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = xyxy
    area = max(0, x2 - x1) * max(0, y2 - y1)
    total = max(1, h * w)
    return float(area / total)


def mean_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(axis=0)


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def high_freq_energy(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))


def robust_percentile(arr: List[float], q: float) -> float:
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(np.array(arr, dtype=np.float32), q))


def list_images_in_dir(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts]
    )


def list_images_recursive(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(
        [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    )


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def valid_face_feats(feats: List[FaceFeat]) -> List[FaceFeat]:
    return [feat for feat in feats if feat.ok]


def get_face_size_bucket(bbox_ratio: float) -> str:
    if bbox_ratio < 0.012:
        return "full_far"
    if bbox_ratio < 0.022:
        return "mid"
    return "near"


def get_quality_tolerances_by_face_size(
    bbox_ratio: float,
    quality_thresholds: QualityThresholds,
) -> Dict[str, float]:
    bucket = get_face_size_bucket(bbox_ratio)
    if bucket == "full_far":
        return {
            "bucket": bucket,
            "dark_delta_L": 16.0,
            "sharp_ratio_floor": 0.42,
            "texture_ratio_floor": 0.48,
            "abs_luma_warn": max(quality_thresholds.face_luma_dark_warn_l - 12.0, 92.0),
            "abs_lap_warn": max(quality_thresholds.face_lapvar_soft_warn * 0.70, 6.0),
            "abs_hf_warn": max(quality_thresholds.face_hfenergy_soft_warn * 0.75, 0.75),
        }
    if bucket == "mid":
        return {
            "bucket": bucket,
            "dark_delta_L": 13.0,
            "sharp_ratio_floor": 0.50,
            "texture_ratio_floor": 0.55,
            "abs_luma_warn": max(quality_thresholds.face_luma_dark_warn_l - 6.0, 98.0),
            "abs_lap_warn": max(quality_thresholds.face_lapvar_soft_warn * 0.85, 8.0),
            "abs_hf_warn": max(quality_thresholds.face_hfenergy_soft_warn * 0.85, 0.90),
        }
    return {
        "bucket": bucket,
        "dark_delta_L": 10.0,
        "sharp_ratio_floor": 0.55,
        "texture_ratio_floor": 0.60,
        "abs_luma_warn": quality_thresholds.face_luma_dark_warn_l,
        "abs_lap_warn": quality_thresholds.face_lapvar_soft_warn,
        "abs_hf_warn": quality_thresholds.face_hfenergy_soft_warn,
    }


def estimate_view_bucket_and_side(face_feat: FaceFeat) -> Tuple[str, str, float]:
    if (not face_feat.ok) or (face_feat.kps5 is None):
        return "front", "unknown", 0.0

    le, re, nose, ml, mr = [face_feat.kps5[i] for i in range(5)]

    eye_mid_x = float((le[0] + re[0]) / 2.0)
    eye_dist = max(1e-6, float(np.linalg.norm(le - re)))
    mouth_mid_x = float((ml[0] + mr[0]) / 2.0)

    nose_offset = float((nose[0] - eye_mid_x) / eye_dist)
    mouth_offset = float((mouth_mid_x - eye_mid_x) / eye_dist)
    yaw_proxy = 0.7 * abs(nose_offset) + 0.3 * abs(mouth_offset)

    if yaw_proxy < 0.10:
        bucket = "front"
    elif yaw_proxy < 0.28:
        bucket = "three_quarter"
    else:
        bucket = "profile_like"

    if nose_offset < -0.02:
        side = "left"
    elif nose_offset > 0.02:
        side = "right"
    else:
        side = "unknown"

    return bucket, side, float(yaw_proxy)


def canonicalize_view_lane(face_feat: FaceFeat, raw_view_bucket: str) -> str:
    if (not face_feat.ok) or (face_feat.kps5 is None):
        return "unknown"
    if raw_view_bucket == "profile_like":
        return "side_90"
    return raw_view_bucket


def infer_anchor_view_from_path(path_str: Optional[str]) -> str:
    if not path_str:
        return "front"
    s = str(path_str).replace("\\", "/").lower()
    if "/three_quarter/" in s or "/3q/" in s:
        return "three_quarter"
    if "/side_90/" in s or "/side/" in s:
        return "profile_like"
    if "/profile_like/" in s or "/profile/" in s:
        return "profile_like"
    if "/back_180/" in s or "/back/" in s:
        return "back_180"
    return "front"
