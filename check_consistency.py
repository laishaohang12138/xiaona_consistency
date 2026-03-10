# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

import cv2
import copy
import json
import math
import shutil
import traceback
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import importlib

# ============================================================
# ⚙️ 配置区（按你的目录改这里）
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
DIR_ANCHORS = BASE_DIR / "anchors"
DIR_INPUT = BASE_DIR / "input"
DIR_OUTPUT = BASE_DIR / "outputs"
DIR_CALIB = BASE_DIR / "calib_pass"   # 放你人工确认的 PASS 图做阈值校准

DIR_OUT_PASS = DIR_OUTPUT / "pass"
DIR_OUT_WARN = DIR_OUTPUT / "warn"
DIR_OUT_FAIL = DIR_OUTPUT / "fail"
REPORT_FILE = DIR_OUTPUT / "qa_report.json"
THRESH_FILE = DIR_OUTPUT / "quality_thresholds.json"

DIR_ANCHOR_FACE = DIR_ANCHORS / "face"
DIR_ANCHOR_UPPER = DIR_ANCHORS / "upper"
DIR_ANCHOR_FULL = DIR_ANCHORS / "full"

for d in [DIR_OUT_PASS, DIR_OUT_WARN, DIR_OUT_FAIL]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 🧠 运行模式
#   RUN_MODE = "qa"         -> 正常跑质检
#   RUN_MODE = "calibrate"  -> 用 calib_pass 自动校准阈值
# ============================================================
RUN_MODE = "qa"

# qa 模式下，是否自动加载 outputs/quality_thresholds.json
AUTO_LOAD_THRESHOLDS = False

# 默认建议：BODY GOLD 全身金标筛选
ACTIVE_PROFILE = "body_gold_fullbody"

# 低置信度门槛（低于这个，不直接判死，转 WARN）
MIN_CONF_FOR_STRICT_FAIL = 0.18

# face 模块“完全不可靠”的阈值（再低就直接 FAIL）
FACE_NO_SIGNAL_CONF_TH = 0.08

# ============================================================
# 🧪 输入标准化 / 一致性模块运行策略
# ============================================================
STANDARDIZE_INPUT = True
STANDARDIZE_LONG_SIDE = 1792  # 让 4K / 2K 输入尽量收敛到同一检测尺度
UPSCALE_SMALL_INPUT = False   # 小图不强行放大，避免伪细节与检测噪声

CONSISTENCY_MODE = "soft_gate"   # "observe" | "soft_gate"

CONSTITUTION_MIN_CONF = 0.68
SKIN_MIN_CONF = 0.60
DEPTH3D_MIN_CONF = 0.58

CONSTITUTION_SOFT_WARN_TH = 0.56
CONSTITUTION_STRONG_WARN_TH = 0.44

SKIN_SOFT_WARN_TH = 0.54
SKIN_STRONG_WARN_TH = 0.42

DEPTH3D_SOFT_WARN_TH = 0.54
DEPTH3D_STRONG_WARN_TH = 0.42


# ============================================================
# 🧠 任务模板（按任务类型切权重/门槛）
# ============================================================
TASK_PROFILES: Dict[str, Dict[str, Any]] = {
    "identity_lock": {
        "weights": {"face": 0.70, "upper": 0.20, "full": 0.10},
        "require": {"face": True, "upper": False, "full": False},
        "thresholds": {
            "face_pass": 0.82, "face_warn": 0.68,
            "upper_pass": 0.75, "upper_warn": 0.60,
            "full_pass": 0.72, "full_warn": 0.58,
            "overall_pass": 0.80, "overall_warn": 0.65,
        },
    },
    "upper_body_product": {
        "weights": {"face": 0.40, "upper": 0.45, "full": 0.15},
        "require": {"face": True, "upper": True, "full": False},
        "thresholds": {
            "face_pass": 0.78, "face_warn": 0.63,
            "upper_pass": 0.78, "upper_warn": 0.62,
            "full_pass": 0.65, "full_warn": 0.50,
            "overall_pass": 0.78, "overall_warn": 0.62,
        },
    },
    "full_body_outfit": {
        "weights": {"face": 0.25, "upper": 0.30, "full": 0.45},
        "require": {"face": False, "upper": True, "full": True},
        "thresholds": {
            "face_pass": 0.75, "face_warn": 0.58,
            "upper_pass": 0.75, "upper_warn": 0.58,
            "full_pass": 0.78, "full_warn": 0.62,
            "overall_pass": 0.78, "overall_warn": 0.62,
        },
    },
    # ✅ 泛 LoRA 入库：身份优先，姿态允许变化
    "lora_dataset": {
        "weights": {"face": 0.65, "upper": 0.10, "full": 0.25},
        "require": {"face": True, "upper": False, "full": False},
        "thresholds": {
            "face_pass": 0.78, "face_warn": 0.62,
            "upper_pass": 0.70, "upper_warn": 0.55,
            "full_pass": 0.70, "full_warn": 0.55,
            "overall_pass": 0.76, "overall_warn": 0.60,
        },
    },
    # ✅ BODY GOLD：脸不能飘 + 全身构图必须能过线
    "body_gold_fullbody": {
        "weights": {"face": 0.45, "upper": 0.15, "full": 0.40},
        "require": {"face": True, "upper": False, "full": True},
        "thresholds": {
            "face_pass": 0.76, "face_warn": 0.60,
            "upper_pass": 0.68, "upper_warn": 0.53,
            "full_pass": 0.74, "full_warn": 0.58,
            "overall_pass": 0.74, "overall_warn": 0.59,
        },
    },
}

PROFILE_POLICY: Dict[str, Dict[str, Any]] = {
    "identity_lock": {
        "identity_anchor_pool": "face",
        "quality_anchor_pool": "face",
        "soft_quality_hits_to_warn": 1,
        "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
    },
    "upper_body_product": {
        "identity_anchor_pool": "face",
        "quality_anchor_pool": "upper_first",
        "soft_quality_hits_to_warn": 1,
        "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
    },
    "full_body_outfit": {
        "identity_anchor_pool": "face",
        "quality_anchor_pool": "upper_or_full",
        "soft_quality_hits_to_warn": 2,
        "hard_quality_flags": {
            "FACE_UNDEREXPOSED_DARK",
            "FACE_NO_RELIABLE_SIGNAL",
            "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE",
        },
    },
    "lora_dataset": {
        "identity_anchor_pool": "face",
        "quality_anchor_pool": "face",
        "soft_quality_hits_to_warn": 2,
        "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
    },
    "body_gold_fullbody": {
        "identity_anchor_pool": "face",
        "quality_anchor_pool": "upper_or_full",
        "soft_quality_hits_to_warn": 2,
        "hard_quality_flags": {
            "FACE_UNDEREXPOSED_DARK",
            "FACE_NO_RELIABLE_SIGNAL",
            "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE",
        },
    },
}

EXTERNAL_CONFIG_STATUS: Dict[str, bool] = {
    "task_profiles": False,
    "consistency_thresholds": False,
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _yaml_scalar_from_text(text: str) -> Any:
    s = text.strip()
    if s == "":
        return ""

    lower = s.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except Exception:
        return s


def _load_simple_yaml(path: Path) -> Optional[Any]:
    if not path.exists():
        return None

    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[警告] 配置文件读取失败: {path} | {e}")
        return None

    entries: List[Tuple[int, str]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        entries.append((indent, line.strip()))

    if len(entries) == 0:
        return None

    def next_container_type(idx: int, cur_indent: int) -> str:
        if idx + 1 >= len(entries):
            return "dict"
        next_indent, next_content = entries[idx + 1]
        if next_indent <= cur_indent:
            return "dict"
        return "list" if next_content.startswith("- ") else "dict"

    root: Any = [] if entries[0][1].startswith("- ") else {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for idx, (indent, content) in enumerate(entries):
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML 解析失败，列表项缺少父列表: {path} :: {content}")
            item_text = content[2:].strip()
            parent.append(_yaml_scalar_from_text(item_text))
            continue

        if ":" not in content:
            raise ValueError(f"YAML 解析失败，非法映射行: {path} :: {content}")

        key, _, value_text = content.partition(":")
        key = key.strip()
        value_text = value_text.strip()

        if not isinstance(parent, dict):
            raise ValueError(f"YAML 解析失败，映射项缺少父字典: {path} :: {content}")

        if value_text == "":
            container_type = next_container_type(idx, indent)
            child: Any = [] if container_type == "list" else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _yaml_scalar_from_text(value_text)

    return root


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _normalize_profile_policy_map(data: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(data, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for profile_name, policy in data.items():
        if not isinstance(policy, dict):
            continue
        node = copy.deepcopy(policy)
        flags = node.get("hard_quality_flags", [])
        if isinstance(flags, list):
            node["hard_quality_flags"] = set(str(x) for x in flags)
        elif isinstance(flags, set):
            node["hard_quality_flags"] = set(str(x) for x in flags)
        elif flags is None:
            node["hard_quality_flags"] = set()
        else:
            node["hard_quality_flags"] = {str(flags)}
        out[str(profile_name)] = node
    return out


def _apply_external_project_configs() -> None:
    global ACTIVE_PROFILE
    global MIN_CONF_FOR_STRICT_FAIL, FACE_NO_SIGNAL_CONF_TH
    global CONSISTENCY_MODE
    global CONSTITUTION_MIN_CONF, SKIN_MIN_CONF, DEPTH3D_MIN_CONF
    global CONSTITUTION_SOFT_WARN_TH, CONSTITUTION_STRONG_WARN_TH
    global SKIN_SOFT_WARN_TH, SKIN_STRONG_WARN_TH
    global DEPTH3D_SOFT_WARN_TH, DEPTH3D_STRONG_WARN_TH
    global TASK_PROFILES, PROFILE_POLICY

    try:
        task_profile_cfg = _load_simple_yaml(CONFIG_DIR / "task_profiles.yaml")
    except Exception as e:
        print(f"[警告] task_profiles.yaml 解析失败，回退内置默认配置: {e}")
        task_profile_cfg = None
    if isinstance(task_profile_cfg, dict):
        loaded_profiles = task_profile_cfg.get("task_profiles", {})
        if isinstance(loaded_profiles, dict):
            TASK_PROFILES = _deep_merge_dict(TASK_PROFILES, loaded_profiles)

        loaded_profile_policy = _normalize_profile_policy_map(task_profile_cfg.get("profile_policy", {}))
        if len(loaded_profile_policy) > 0:
            PROFILE_POLICY = _deep_merge_dict(PROFILE_POLICY, loaded_profile_policy)

        review_policy = task_profile_cfg.get("review_policy", {})
        if isinstance(review_policy, dict):
            ACTIVE_PROFILE = str(task_profile_cfg.get("active_profile", review_policy.get("main_priority", ACTIVE_PROFILE)))
            MIN_CONF_FOR_STRICT_FAIL = _coerce_float(
                review_policy.get("strict_fail_min_conf", MIN_CONF_FOR_STRICT_FAIL),
                MIN_CONF_FOR_STRICT_FAIL,
            )
            FACE_NO_SIGNAL_CONF_TH = _coerce_float(
                review_policy.get("face_no_signal_conf", FACE_NO_SIGNAL_CONF_TH),
                FACE_NO_SIGNAL_CONF_TH,
            )

        EXTERNAL_CONFIG_STATUS["task_profiles"] = True

    try:
        consistency_cfg = _load_simple_yaml(CONFIG_DIR / "consistency_thresholds.yaml")
    except Exception as e:
        print(f"[警告] consistency_thresholds.yaml 解析失败，回退内置默认配置: {e}")
        consistency_cfg = None
    if isinstance(consistency_cfg, dict):
        consistency = consistency_cfg.get("consistency", {})
        if isinstance(consistency, dict):
            CONSISTENCY_MODE = str(consistency.get("mode", CONSISTENCY_MODE))

            min_conf = consistency.get("min_confidence", {})
            if isinstance(min_conf, dict):
                CONSTITUTION_MIN_CONF = _coerce_float(min_conf.get("constitution", CONSTITUTION_MIN_CONF), CONSTITUTION_MIN_CONF)
                SKIN_MIN_CONF = _coerce_float(min_conf.get("skin", SKIN_MIN_CONF), SKIN_MIN_CONF)
                DEPTH3D_MIN_CONF = _coerce_float(min_conf.get("depth3d", DEPTH3D_MIN_CONF), DEPTH3D_MIN_CONF)

            warn_th = consistency.get("warn_threshold", {})
            if isinstance(warn_th, dict):
                CONSTITUTION_SOFT_WARN_TH = _coerce_float(
                    warn_th.get("constitution_soft", CONSTITUTION_SOFT_WARN_TH),
                    CONSTITUTION_SOFT_WARN_TH,
                )
                CONSTITUTION_STRONG_WARN_TH = _coerce_float(
                    warn_th.get("constitution_strong", CONSTITUTION_STRONG_WARN_TH),
                    CONSTITUTION_STRONG_WARN_TH,
                )
                SKIN_SOFT_WARN_TH = _coerce_float(warn_th.get("skin_soft", SKIN_SOFT_WARN_TH), SKIN_SOFT_WARN_TH)
                SKIN_STRONG_WARN_TH = _coerce_float(warn_th.get("skin_strong", SKIN_STRONG_WARN_TH), SKIN_STRONG_WARN_TH)
                DEPTH3D_SOFT_WARN_TH = _coerce_float(warn_th.get("depth3d_soft", DEPTH3D_SOFT_WARN_TH), DEPTH3D_SOFT_WARN_TH)
                DEPTH3D_STRONG_WARN_TH = _coerce_float(
                    warn_th.get("depth3d_strong", DEPTH3D_STRONG_WARN_TH),
                    DEPTH3D_STRONG_WARN_TH,
                )

        EXTERNAL_CONFIG_STATUS["consistency_thresholds"] = True


_apply_external_project_configs()

# ============================================================
# 📦 可选依赖（有就上，没有就兜底）
# ============================================================
SKIMAGE_SSIM_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_SSIM_AVAILABLE = True
except Exception:
    SKIMAGE_SSIM_AVAILABLE = False

# ============================================================
# ✅ 质量门槛（可被自动校准覆盖）
# Lab 的 L 通道：0~255，越低越暗
# ============================================================
FACE_LUMA_DARK_WARN_L = 110.0
FACE_LAPVAR_SOFT_WARN = 12.0
FACE_HFENERGY_SOFT_WARN = 1.20

QUALITY_DEGRADE_FLAGS = {
    "FACE_UNDEREXPOSED_DARK",
    "FACE_TOO_SOFT_POSSIBLE_SMOOTHING",
    "FACE_LOW_MICROTEXTURE",
    "FACE_DARKER_THAN_ANCHOR",
    "FACE_SOFTER_THAN_ANCHOR",
    "FACE_LOWER_TEXTURE_THAN_ANCHOR",
}

# ============================================================
# 🧩 数据结构
# ============================================================
@dataclass
class EngineState:
    face_mode: str               # "insightface" | "opencv"
    pose_mode: str               # "mediapipe" | "opencv"
    face_app: Any = None
    pose_engine: Any = None
    mp_pose: Any = None
    hog_people: Any = None


@dataclass
class FaceFeat:
    ok: bool = False
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    bbox_area_ratio: float = 0.0
    embedding: Optional[np.ndarray] = None
    kps5: Optional[np.ndarray] = None
    crop_gray_128: Optional[np.ndarray] = None
    hog_vec: Optional[np.ndarray] = None
    lbp_hist: Optional[np.ndarray] = None
    phash64: Optional[np.ndarray] = None
    geom: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    lab_mean: Optional[np.ndarray] = None   # [L, a, b]
    lap_var: float = 0.0
    hf_energy: float = 0.0
    source_path: Optional[str] = None


@dataclass
class PoseFeat:
    ok: bool = False
    mode: str = "opencv"
    lm_xy: Optional[np.ndarray] = None
    lm_vis: Optional[np.ndarray] = None
    person_bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    person_bbox_area_ratio: float = 0.0
    framing: Dict[str, float] = field(default_factory=dict)
    upper_geom: Dict[str, float] = field(default_factory=dict)
    full_geom: Dict[str, float] = field(default_factory=dict)
    confidence_upper: float = 0.0
    confidence_full: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class AnchorSet:
    face_feats: List[FaceFeat] = field(default_factory=list)
    upper_face_feats: List[FaceFeat] = field(default_factory=list)
    full_face_feats: List[FaceFeat] = field(default_factory=list)
    upper_pose_feats: List[PoseFeat] = field(default_factory=list)
    full_pose_feats: List[PoseFeat] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 🛠️ 工具函数
# ============================================================
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


def standardize_input_bgr(img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img_bgr is None:
        return None
    if not STANDARDIZE_INPUT:
        return img_bgr

    h, w = img_bgr.shape[:2]
    max_side = max(h, w)
    if max_side <= 0:
        return img_bgr

    target = int(STANDARDIZE_LONG_SIDE)

    if max_side > target:
        scale = target / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if UPSCALE_SMALL_INPUT and max_side < target:
        scale = target / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return img_bgr


def image_read_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    return standardize_input_bgr(img)


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
    vec = vec.astype(np.float32).reshape(-1)
    return l2norm(vec)


def compute_lbp_hist(gray128: np.ndarray) -> np.ndarray:
    g = gray128.astype(np.uint8)
    h, w = g.shape
    if h < 3 or w < 3:
        return np.zeros((256,), dtype=np.float32)

    center = g[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
        g[2:, :-2], g[1:-1, :-2],
    ]
    for i, n in enumerate(neighbors):
        lbp |= ((n >= center).astype(np.uint8) << i)

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
    dct_low = dct[:8, :8].copy()
    flat = dct_low.flatten()
    vals = flat[1:]
    med = float(np.median(vals))
    bits = (flat > med).astype(np.uint8)
    return bits.reshape(-1)


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
            val = skimage_ssim(a, b)
            return float(clamp((val + 1.0) / 2.0, 0.0, 1.0))
        except Exception:
            pass

    aa = a.astype(np.float32).reshape(-1)
    bb = b.astype(np.float32).reshape(-1)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb))
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


def bbox_area_ratio_xyxy(xyxy: Tuple[int, int, int, int], img_shape: Tuple[int, int, int]) -> float:
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
    a = np.array(arr, dtype=np.float32)
    return float(np.percentile(a, q))


def list_images_in_dir(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

def list_images_recursive(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def valid_face_feats(feats: List[FaceFeat]) -> List[FaceFeat]:
    return [x for x in feats if x.ok]


def get_face_size_bucket(bbox_ratio: float) -> str:
    if bbox_ratio < 0.012:
        return "full_far"
    if bbox_ratio < 0.022:
        return "mid"
    return "near"


def get_quality_tolerances_by_face_size(bbox_ratio: float) -> Dict[str, float]:
    bucket = get_face_size_bucket(bbox_ratio)
    if bucket == "full_far":
        return {
            "bucket": bucket,
            "dark_delta_L": 16.0,
            "sharp_ratio_floor": 0.42,
            "texture_ratio_floor": 0.48,
            "abs_luma_warn": max(FACE_LUMA_DARK_WARN_L - 12.0, 92.0),
            "abs_lap_warn": max(FACE_LAPVAR_SOFT_WARN * 0.70, 6.0),
            "abs_hf_warn": max(FACE_HFENERGY_SOFT_WARN * 0.75, 0.75),
        }
    if bucket == "mid":
        return {
            "bucket": bucket,
            "dark_delta_L": 13.0,
            "sharp_ratio_floor": 0.50,
            "texture_ratio_floor": 0.55,
            "abs_luma_warn": max(FACE_LUMA_DARK_WARN_L - 6.0, 98.0),
            "abs_lap_warn": max(FACE_LAPVAR_SOFT_WARN * 0.85, 8.0),
            "abs_hf_warn": max(FACE_HFENERGY_SOFT_WARN * 0.85, 0.90),
        }
    return {
        "bucket": bucket,
        "dark_delta_L": 10.0,
        "sharp_ratio_floor": 0.55,
        "texture_ratio_floor": 0.60,
        "abs_luma_warn": FACE_LUMA_DARK_WARN_L,
        "abs_lap_warn": FACE_LAPVAR_SOFT_WARN,
        "abs_hf_warn": FACE_HFENERGY_SOFT_WARN,
    }

def estimate_view_bucket_and_side(face_feat: FaceFeat) -> Tuple[str, str, float]:
    """
    返回:
    - view_bucket: front | three_quarter | profile_like
    - view_side: left | right | unknown
    - yaw_proxy: 粗略侧转强度（不是物理角度，只是分桶代理值）
    """
    if (not face_feat.ok) or (face_feat.kps5 is None):
        return "front", "unknown", 0.0

    le, re, nose, ml, mr = [face_feat.kps5[i] for i in range(5)]

    eye_mid_x = float((le[0] + re[0]) / 2.0)
    eye_dist = max(1e-6, float(np.linalg.norm(le - re)))
    mouth_mid_x = float((ml[0] + mr[0]) / 2.0)

    nose_offset = float((nose[0] - eye_mid_x) / eye_dist)
    mouth_offset = float((mouth_mid_x - eye_mid_x) / eye_dist)

    # nose 更重要，mouth 作为辅助
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

def infer_anchor_view_from_path(path_str: Optional[str]) -> str:
    if not path_str:
        return "front"
    s = str(path_str).replace("\\", "/").lower()
    if "/three_quarter/" in s or "/3q/" in s:
        return "three_quarter"
    if "/profile_like/" in s or "/profile/" in s:
        return "profile_like"
    return "front"


# ============================================================
# 🧬 v0.5-lite: 身材一致性 / 肤色一致性 / 3D-lite 一致性
# 目标：
# - 新模块真实参与筛选
# - 但必须带置信度，避免误杀
# - 先作为 soft gate，逐步再升级
# ============================================================

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
    ws = []
    for v, w in items:
        if v is None:
            continue
        vals.append(float(v))
        ws.append(float(w))
    if len(vals) == 0:
        return None
    return float(np.average(np.array(vals, dtype=np.float32), weights=np.array(ws, dtype=np.float32)))


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_u8)
    out[labels == best] = 255
    return out


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
    d = lab1.astype(np.float32) - lab2.astype(np.float32)
    return float(np.sqrt(np.sum(d * d)))


def _skin_mask_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    mask = (
        (Y > 35) &
        (Cr >= 132) & (Cr <= 180) &
        (Cb >= 75) & (Cb <= 135)
    ).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def compute_foreground_mask(img_bgr: np.ndarray, face_feat: FaceFeat, pose_feat: PoseFeat) -> Optional[np.ndarray]:
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bd = max(4, int(round(min(h, w) * 0.04)))
    border_pixels = np.concatenate([
        lab[:bd, :, :].reshape(-1, 3),
        lab[-bd:, :, :].reshape(-1, 3),
        lab[:, :bd, :].reshape(-1, 3),
        lab[:, -bd:, :].reshape(-1, 3),
    ], axis=0)

    bg_med = np.median(border_pixels, axis=0).astype(np.float32)
    dist = np.sqrt(np.sum((lab - bg_med[None, None, :]) ** 2, axis=2))
    mask = (dist > 10.5).astype(np.uint8) * 255

    if pose_feat.ok and pose_feat.lm_xy is not None and pose_feat.lm_vis is not None:
        vis_ids = [i for i in range(len(pose_feat.lm_vis)) if float(pose_feat.lm_vis[i]) > 0.20]
        if len(vis_ids) > 0:
            xs = [float(pose_feat.lm_xy[i][0]) for i in vis_ids]
            ys = [float(pose_feat.lm_xy[i][1]) for i in vis_ids]
            x1 = int(max(0, (min(xs) - 0.08) * w))
            y1 = int(max(0, (min(ys) - 0.08) * h))
            x2 = int(min(w, (max(xs) + 0.08) * w))
            y2 = int(min(h, (max(ys) + 0.08) * h))
            roi = np.zeros((h, w), dtype=np.uint8)
            roi[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, roi)

    if face_feat.ok and face_feat.bbox_xyxy is not None:
        x1, y1, x2, y2 = face_feat.bbox_xyxy
        ex = int((x2 - x1) * 0.18)
        ey = int((y2 - y1) * 0.18)
        face_roi = np.zeros((h, w), dtype=np.uint8)
        rr = _clip_rect(x1 - ex, y1 - ey, x2 + ex, y2 + ey, h, w)
        if rr is not None:
            fx1, fy1, fx2, fy2 = rr
            face_roi[fy1:fy2, fx1:fx2] = 255
            mask = cv2.bitwise_or(mask, face_roi)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    mask = _largest_component(mask)
    return mask


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
        ww = _row_width(mask_u8, yy)
        if ww is not None:
            vals.append(int(ww))
    if len(vals) == 0:
        return None
    return int(np.median(np.array(vals, dtype=np.float32)))


def _mask_width_soft_min(mask_u8: np.ndarray, y1: int, y2: int, band: int = 4) -> Tuple[Optional[int], Optional[int]]:
    h, _ = mask_u8.shape[:2]
    widths: List[Tuple[int, int]] = []
    for yy in range(max(0, y1), min(h, y2 + 1)):
        ww = _mask_width_at_row(mask_u8, yy, band=band)
        if ww is not None:
            widths.append((yy, int(ww)))
    if len(widths) == 0:
        return None, None
    widths_sorted = sorted(widths, key=lambda x: x[1])
    k = max(3, min(7, len(widths_sorted)))
    chosen = widths_sorted[:k]
    waist_y = int(round(float(np.median([yy for yy, _ in chosen]))))
    waist_w = int(round(float(np.median([ww for _, ww in chosen]))))
    return waist_y, waist_w


def _fg_fill_ratio(mask_u8: Optional[np.ndarray], rect: Optional[Tuple[int, int, int, int]]) -> Optional[float]:
    if mask_u8 is None or rect is None:
        return None
    x1, y1, x2, y2 = rect
    area = max(1, (x2 - x1) * (y2 - y1))
    fill = int(np.sum(mask_u8[y1:y2, x1:x2] > 0))
    return float(fill / area)


def extract_body_constitution_metrics(
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

    fg_mask = compute_foreground_mask(img_bgr, face_feat, pose_feat)
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

    shoulder_width_px = float(np.linalg.norm(np.array([lsx, lsy], dtype=np.float32) - np.array([rsx, rsy], dtype=np.float32)))
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

    thigh_y = int(round((hip_mid_y + knee_mid_y) / 2.0))
    calf_y = int(round((knee_mid_y + ankle_mid_y) / 2.0))
    thigh_w = _mask_width_at_row(fg_mask, thigh_y, band=5)
    calf_w = _mask_width_at_row(fg_mask, calf_y, band=5)

    out["chest_width_px"] = chest_w
    out["waist_width_px"] = waist_w
    out["hip_width_px"] = hip_w
    out["thigh_width_px"] = thigh_w
    out["calf_width_px"] = calf_w

    # 仅作 debug，不参与总分硬约束
    if face_feat.ok and face_feat.bbox_xyxy is not None:
        x1, y1, x2, y2 = face_feat.bbox_xyxy
        face_h = max(1, y2 - y1)
        out["head_body_ratio_proxy"] = float(body_h / face_h)

    out["leg_ratio"] = safe_float(pose_feat.full_geom.get("leg_ratio", 0.0), 0.0) or None

    if waist_y is not None:
        out["waist_height_ratio"] = float((waist_y - subject_top) / max(1, body_h))

    if (waist_w is not None) and (shoulder_width_px > 1e-6):
        out["waist_to_shoulder_ratio"] = float(waist_w / shoulder_width_px)

    if (chest_w is not None) and (waist_w is not None) and waist_w > 1e-6:
        out["chest_to_waist_ratio"] = float(chest_w / waist_w)

    if (hip_w is not None) and (waist_w is not None) and waist_w > 1e-6:
        out["hip_to_waist_ratio"] = float(hip_w / waist_w)

    if (hip_w is not None) and (shoulder_width_px > 1e-6):
        out["hip_to_shoulder_ratio"] = float(hip_w / shoulder_width_px)

    if (thigh_w is not None) and (calf_w is not None) and calf_w > 1e-6:
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

    pelvis_compactness_score = hip_shoulder
    abdomen_flatness_score = waist_shoulder
    lower_body_slenderness_score = thigh_calf

    out["pelvis_compactness_score"] = pelvis_compactness_score
    out["abdomen_flatness_score"] = abdomen_flatness_score
    out["lower_body_slenderness_score"] = lower_body_slenderness_score

    body_constitution_score = weighted_mean_valid([
        (soft_range_score(out["leg_ratio"], 0.44, 0.58, 0.10), 0.18),
        (soft_range_score(out["waist_height_ratio"], 0.54, 0.67, 0.10), 0.18),
        (waist_shoulder, 0.20),
        (chest_waist, 0.12),
        (hip_waist, 0.14),
        (pelvis_compactness_score, 0.10),
        (lower_body_slenderness_score, 0.08),
    ])

    out["body_constitution_score"] = body_constitution_score

    width_ready = sum(1 for x in [chest_w, waist_w, hip_w, thigh_w, calf_w] if x is not None)
    torso_rect = _clip_rect(
        int(round(min(lsx, rsx, lhx, rhx) - 0.05 * w)),
        int(round(min(lsy, rsy) - 0.02 * h)),
        int(round(max(lsx, rsx, lhx, rhx) + 0.05 * w)),
        int(round(max(lhy, rhy) + 0.02 * h)),
        h, w
    )
    torso_fill = _fg_fill_ratio(fg_mask, torso_rect)
    view_factor = 1.0 if view_bucket == "front" else (0.88 if view_bucket == "three_quarter" else 0.75)

    conf = weighted_mean_valid([
        (float(width_ready) / 5.0, 0.34),
        (pose_vis, 0.34),
        (torso_fill, 0.22),
        (view_factor, 0.10),
    ])
    out["confidence"] = 0.0 if conf is None else float(conf)

    out["is_valid"] = (body_constitution_score is not None) and (width_ready >= 3)
    out["reasons"].append("BODY_CONSTITUTION_READY" if out["is_valid"] else "BODY_CONSTITUTION_SCORE_EMPTY")
    return out


def extract_skin_consistency_metrics(
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

    fg_mask = compute_foreground_mask(img_bgr, face_feat, pose_feat)
    if fg_mask is None:
        out["reasons"].append("SKIN_FG_MASK_FAILED")
        return out

    skin_mask = _skin_mask_ycrcb(img_bgr)
    valid_skin = cv2.bitwise_and(skin_mask, fg_mask)

    L_SH, R_SH = 11, 12
    L_EL, R_EL = 13, 14
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANK, R_ANK = 27, 28

    def patch_mask_from_rect(rect: Optional[Tuple[int, int, int, int]], min_pixels: int = 30, min_purity: float = 0.12) -> Tuple[Optional[np.ndarray], float]:
        if rect is None:
            return None, 0.0
        x1, y1, x2, y2 = rect
        area = max(1, (x2 - x1) * (y2 - y1))
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        m = cv2.bitwise_and(m, valid_skin)
        count = int(np.sum(m > 0))
        purity = float(count / area)
        if count < min_pixels or purity < min_purity:
            return None, purity
        return m, purity

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
            h, w
        )
        m, face_purity = patch_mask_from_rect(rr, min_pixels=40, min_purity=0.10)
        face_lab = _median_lab_in_region(img_bgr, m) if m is not None else None

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
        h, w
    )

    abdomen_rect = None
    if float(vis[L_SH]) > 0.35 and float(vis[R_SH]) > 0.35 and float(vis[L_HIP]) > 0.35 and float(vis[R_HIP]) > 0.35:
        lhx, lhy = _norm_xy_to_px(xy[L_HIP], h, w)
        rhx, rhy = _norm_xy_to_px(xy[R_HIP], h, w)
        hip_mid_x = int(round((lhx + rhx) / 2.0))
        hip_mid_y = int(round((lhy + rhy) / 2.0))
        abdomen_rect = _rect_from_center(
            int(round((shoulder_mid_x + hip_mid_x) / 2.0)),
            int(round(shoulder_mid_y + 0.60 * (hip_mid_y - shoulder_mid_y))),
            max(12, int(0.035 * w)),
            max(10, int(0.025 * h)),
            h, w
        )

    def limb_mid_rect(i1: int, i2: int, rx_ratio: float = 0.028, ry_ratio: float = 0.025) -> Optional[Tuple[int, int, int, int]]:
        if float(vis[i1]) <= 0.35 or float(vis[i2]) <= 0.35:
            return None
        x1, y1 = _norm_xy_to_px(xy[i1], h, w)
        x2, y2 = _norm_xy_to_px(xy[i2], h, w)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        return _rect_from_center(cx, cy, max(8, int(rx_ratio * w)), max(8, int(ry_ratio * h)), h, w)

    purities: List[float] = [face_purity]

    arm_labs = []
    for rr in [limb_mid_rect(L_SH, L_EL), limb_mid_rect(R_SH, R_EL)]:
        m, pur = patch_mask_from_rect(rr)
        purities.append(pur)
        lab = _median_lab_in_region(img_bgr, m) if m is not None else None
        if lab is not None:
            arm_labs.append(lab)
    arm_lab = np.median(np.stack(arm_labs, axis=0), axis=0).astype(np.float32) if len(arm_labs) > 0 else None

    thigh_labs = []
    for rr in [limb_mid_rect(L_HIP, L_KNEE, 0.035, 0.030), limb_mid_rect(R_HIP, R_KNEE, 0.035, 0.030)]:
        m, pur = patch_mask_from_rect(rr, min_pixels=36, min_purity=0.10)
        purities.append(pur)
        lab = _median_lab_in_region(img_bgr, m) if m is not None else None
        if lab is not None:
            thigh_labs.append(lab)
    thigh_lab = np.median(np.stack(thigh_labs, axis=0), axis=0).astype(np.float32) if len(thigh_labs) > 0 else None

    calf_labs = []
    for rr in [limb_mid_rect(L_KNEE, L_ANK, 0.032, 0.028), limb_mid_rect(R_KNEE, R_ANK, 0.032, 0.028)]:
        m, pur = patch_mask_from_rect(rr, min_pixels=34, min_purity=0.10)
        purities.append(pur)
        lab = _median_lab_in_region(img_bgr, m) if m is not None else None
        if lab is not None:
            calf_labs.append(lab)
    calf_lab = np.median(np.stack(calf_labs, axis=0), axis=0).astype(np.float32) if len(calf_labs) > 0 else None

    m_neck, pur = patch_mask_from_rect(neck_rect)
    purities.append(pur)
    neck_lab = _median_lab_in_region(img_bgr, m_neck) if m_neck is not None else None

    m_abd, pur = patch_mask_from_rect(abdomen_rect)
    purities.append(pur)
    abdomen_lab = _median_lab_in_region(img_bgr, m_abd) if m_abd is not None else None

    out["face_neck_deltaE"] = _delta_e_lab(face_lab, neck_lab)
    out["face_arm_deltaE"] = _delta_e_lab(face_lab, arm_lab)
    out["face_abdomen_deltaE"] = _delta_e_lab(face_lab, abdomen_lab)
    out["face_thigh_deltaE"] = _delta_e_lab(face_lab, thigh_lab)
    out["face_calf_deltaE"] = _delta_e_lab(face_lab, calf_lab)

    leg_L_vals = []
    for lab in [thigh_lab, calf_lab]:
        if lab is not None:
            leg_L_vals.append(float(lab[0]))

    if len(leg_L_vals) > 0:
        out["leg_brightness_ratio"] = float(np.mean(leg_L_vals) / max(1e-6, float(face_lab[0])))

    knee_scores = []
    if thigh_lab is not None and calf_lab is not None:
        knee_L_proxy = float((thigh_lab[0] + calf_lab[0]) / 2.0)
        face_L = float(face_lab[0])
        ratio = knee_L_proxy / max(1e-6, face_L)
        knee_scores.append(soft_range_score(ratio, 0.82, 1.08, 0.22))
    out["knee_dark_patch_score"] = float(np.mean(knee_scores)) if len(knee_scores) > 0 else None

    # 主分先聚焦最稳的三项：腿色差 + 腿亮度
    skin_uniformity_score = weighted_mean_valid([
        (None if out["face_thigh_deltaE"] is None else float(math.exp(-out["face_thigh_deltaE"] / 13.5)), 0.38),
        (None if out["face_calf_deltaE"] is None else float(math.exp(-out["face_calf_deltaE"] / 14.5)), 0.34),
        (soft_range_score(out["leg_brightness_ratio"], 0.84, 1.08, 0.22), 0.18),
        (out["knee_dark_patch_score"], 0.10),
    ])

    avail_main = sum(1 for x in [out["face_thigh_deltaE"], out["face_calf_deltaE"], out["leg_brightness_ratio"]] if x is not None)
    purity_mean = float(np.mean(np.array(purities, dtype=np.float32))) if len(purities) > 0 else 0.0
    conf = weighted_mean_valid([
        (float(avail_main) / 3.0, 0.55),
        (purity_mean, 0.45),
    ])

    out["confidence"] = 0.0 if conf is None else float(conf)
    out["skin_uniformity_score"] = skin_uniformity_score
    out["is_valid"] = (skin_uniformity_score is not None) and (avail_main >= 2)
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

    torso_volume_score = weighted_mean_valid([
        (shoulder_score, 0.30),
        (hip_score, 0.24),
        (ratio_score, 0.20),
        (spine_score, 0.14),
        (torso_score, 0.12),
    ])

    fake_turn_risk = None
    if torso_volume_score is not None:
        fake_turn_risk = float(1.0 - torso_volume_score)

    depth_3d_score = weighted_mean_valid([
        (turn_score, 0.24),
        (torso_volume_score, 0.52),
        (spine_score, 0.14),
        (torso_score, 0.10),
    ])

    if view_bucket == "front":
        conf = 0.20
    elif view_bucket == "three_quarter":
        conf = weighted_mean_valid([
            (soft_range_score(yaw_proxy, 0.10, 0.28, 0.10), 0.45),
            (torso_volume_score, 0.35),
            (spine_score, 0.20),
        ])
    else:
        conf = weighted_mean_valid([
            (soft_range_score(yaw_proxy, 0.24, 0.42, 0.14), 0.45),
            (torso_volume_score, 0.35),
            (spine_score, 0.20),
        ])

    out["confidence"] = 0.0 if conf is None else float(conf)
    out["torso_volume_score"] = torso_volume_score
    out["pelvis_depth_score"] = ratio_score
    out["fake_turn_risk"] = fake_turn_risk
    out["depth_3d_score"] = depth_3d_score
    out["is_valid"] = depth_3d_score is not None
    out["reasons"].append("DEPTH_3D_LITE_READY" if out["is_valid"] else "DEPTH_3D_LITE_EMPTY")
    return out


def apply_consistency_soft_gate(
    reasons_all: List[str],
    final_status: str,
    overall_state: str,
    constitution_metrics: Dict[str, Any],
    skin_metrics: Dict[str, Any],
    depth_3d_metrics: Dict[str, Any],
    view_bucket: str,
) -> Tuple[List[str], str, str, Dict[str, Any]]:
    """
    一致性模块真实参与筛选，但以“高置信 soft gate”为主：
    - 低置信：只记录，不改状态
    - 中高置信 + 异常：PASS -> WARN
    - 不直接把图打成 FAIL
    """
    gate_debug: Dict[str, Any] = {
        "constitution": {},
        "skin": {},
        "depth_3d": {},
        "triggered": [],
        "mode": CONSISTENCY_MODE,
    }

    def downgrade_pass_to_warn(flag: str) -> None:
        nonlocal final_status, overall_state
        gate_debug["triggered"].append(flag)
        if final_status == "PASS":
            final_status = "WARN"
        if overall_state == "PASS":
            overall_state = "WARN"

    # 只观察，不改状态
    if CONSISTENCY_MODE == "observe":
        reasons_all.extend(constitution_metrics.get("reasons", []))
        reasons_all.extend(skin_metrics.get("reasons", []))
        reasons_all.extend(depth_3d_metrics.get("reasons", []))
        reasons_all = dedupe_keep_order(reasons_all)
        return reasons_all, final_status, overall_state, gate_debug

    # 1) 身材一致性
    c_score = constitution_metrics.get("body_constitution_score", None)
    c_conf = float(constitution_metrics.get("confidence", 0.0) or 0.0)
    c_valid = bool(constitution_metrics.get("is_valid", False))
    gate_debug["constitution"] = {
        "score": c_score,
        "confidence": c_conf,
        "valid": c_valid,
    }
    if c_valid and c_score is not None:
        reasons_all.extend(constitution_metrics.get("reasons", []))
        if c_conf >= CONSTITUTION_MIN_CONF:
            if float(c_score) < CONSTITUTION_STRONG_WARN_TH:
                reasons_all.append("BODY_CONSTITUTION_STRONG_WARN")
                downgrade_pass_to_warn("BODY_CONSTITUTION_STRONG_WARN")
            elif float(c_score) < CONSTITUTION_SOFT_WARN_TH:
                reasons_all.append("BODY_CONSTITUTION_WARN")
                downgrade_pass_to_warn("BODY_CONSTITUTION_WARN")
        else:
            reasons_all.append("BODY_CONSTITUTION_LOW_CONF_SKIP")

    # 2) 肤色一致性
    s_score = skin_metrics.get("skin_uniformity_score", None)
    s_conf = float(skin_metrics.get("confidence", 0.0) or 0.0)
    s_valid = bool(skin_metrics.get("is_valid", False))
    gate_debug["skin"] = {
        "score": s_score,
        "confidence": s_conf,
        "valid": s_valid,
    }
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

        if s_conf >= SKIN_MIN_CONF:
            if severe_skin and float(s_score) < SKIN_SOFT_WARN_TH:
                reasons_all.append("SKIN_UNIFORMITY_STRONG_WARN")
                downgrade_pass_to_warn("SKIN_UNIFORMITY_STRONG_WARN")
            elif float(s_score) < SKIN_SOFT_WARN_TH:
                reasons_all.append("SKIN_UNIFORMITY_WARN")
                downgrade_pass_to_warn("SKIN_UNIFORMITY_WARN")
        else:
            reasons_all.append("SKIN_UNIFORMITY_LOW_CONF_SKIP")

    # 3) 3D-lite（只在 3/4 及以上视角生效）
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
            if d_conf >= DEPTH3D_MIN_CONF:
                if float(d_score) < DEPTH3D_STRONG_WARN_TH:
                    reasons_all.append("DEPTH_3D_LITE_STRONG_WARN")
                    downgrade_pass_to_warn("DEPTH_3D_LITE_STRONG_WARN")
                elif float(d_score) < DEPTH3D_SOFT_WARN_TH:
                    reasons_all.append("DEPTH_3D_LITE_WARN")
                    downgrade_pass_to_warn("DEPTH_3D_LITE_WARN")
            else:
                reasons_all.append("DEPTH_3D_LITE_LOW_CONF_SKIP")

    reasons_all = dedupe_keep_order(reasons_all)
    return reasons_all, final_status, overall_state, gate_debug

# ============================================================
# 🧠 引擎初始化
# ============================================================
def _try_init_insightface() -> Tuple[str, object]:
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = ort.get_available_providers()
        print(f"[系统] ONNXRuntime providers: {providers}")

        app = FaceAnalysis(name="buffalo_l")

        if "CUDAExecutionProvider" in providers:
            try:
                app.prepare(ctx_id=0, det_size=(640, 640))
                print("[系统] InsightFace 已启用：GPU (CUDAExecutionProvider)")
            except Exception as e:
                print(f"[警告] InsightFace GPU 初始化失败，回退 CPU。原因: {e}")
                app.prepare(ctx_id=-1, det_size=(640, 640))
                print("[系统] InsightFace 已启用：CPU")
        else:
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[系统] InsightFace 已启用：CPU（未检测到 CUDAExecutionProvider）")

        return "insightface", app

    except ModuleNotFoundError as e:
        print(f"[警告] InsightFace 不可用（缺依赖: {e.name}），回退 OpenCV。")
        return "opencv", None
    except Exception as e:
        print(f"[警告] InsightFace 初始化失败（{e}），回退 OpenCV。")
        return "opencv", None


def _try_init_mediapipe_pose() -> Tuple[str, object, object]:
    try:
        import mediapipe as mp
        print("[系统] 启动 MediaPipe Pose (骨骼引擎)...")
        print(f"[系统] mediapipe module: {getattr(mp, '__file__', None)}")
        print(f"[系统] mediapipe has solutions: {hasattr(mp, 'solutions')}")

        mp_pose = None
        tried = []

        if hasattr(mp, "solutions"):
            try:
                mp_pose = mp.solutions.pose
                print("[系统] MediaPipe 使用经典入口: mp.solutions.pose")
            except Exception as e:
                tried.append(f"mp.solutions.pose -> {e}")

        if mp_pose is None:
            candidates = [
                "mediapipe.python.solutions.pose",
                "mediapipe.modules.pose_landmark",
            ]
            for mod_name in candidates:
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, "Pose"):
                        mp_pose = mod
                        print(f"[系统] MediaPipe 使用兼容入口: {mod_name}")
                        break
                    else:
                        tried.append(f"{mod_name} imported but no Pose class")
                except Exception as e:
                    tried.append(f"{mod_name} -> {e}")

        if mp_pose is None:
            raise RuntimeError(" ; ".join(tried) if tried else "No valid MediaPipe Pose entry found")

        pose_engine = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=1,
        )
        return "mediapipe", pose_engine, mp_pose

    except ModuleNotFoundError as e:
        print(f"[警告] MediaPipe 不可用（缺依赖: {e.name}），回退 OpenCV HOG。")
        return "opencv", None, None
    except Exception as e:
        print(f"[警告] MediaPipe 初始化失败（{e}），回退 OpenCV HOG。")
        return "opencv", None, None


def init_engines() -> EngineState:
    face_mode, face_app = _try_init_insightface()
    pose_mode, pose_engine, mp_pose = _try_init_mediapipe_pose()

    hog_people = None
    if pose_mode == "opencv":
        hog_people = cv2.HOGDescriptor()
        detector = cv2.HOGDescriptor_getDefaultPeopleDetector()  # type: ignore[attr-defined]
        hog_people.setSVMDetector(detector)
        print("[系统] OpenCV HOG 人体检测器已启用（兜底）")

    return EngineState(
        face_mode=face_mode,
        pose_mode=pose_mode,
        face_app=face_app,
        pose_engine=pose_engine,
        mp_pose=mp_pose,
        hog_people=hog_people,
    )


ENGINES = init_engines()

# ============================================================
# 😶 人脸检测与特征提取
# ============================================================
def opencv_largest_face_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = str(Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return bbox_xywh_to_xyxy(int(x), int(y), int(w), int(h))


def detect_face_insightface(img_bgr: np.ndarray) -> Optional[Any]:
    if ENGINES.face_mode != "insightface" or ENGINES.face_app is None:
        return None
    faces = ENGINES.face_app.get(img_bgr)
    if not faces:
        return None

    def area(f: Any) -> float:
        x1, y1, x2, y2 = f.bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    return max(faces, key=area)


def compute_face_geom_from_kps5(kps5: np.ndarray, face_xyxy: Tuple[int, int, int, int]) -> Dict[str, float]:
    x1, y1, x2, y2 = face_xyxy
    fw = max(1.0, x2 - x1)
    fh = max(1.0, y2 - y1)

    le, re, nose, ml, mr = [kps5[i] for i in range(5)]
    eye_dist = float(np.linalg.norm(le - re))
    eye_center = (le + re) / 2.0
    mouth_center = (ml + mr) / 2.0
    mouth_w = float(np.linalg.norm(ml - mr))
    eye_tilt = math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))

    geom = {
        "eye_dist_norm": eye_dist / fw,
        "eye_y_norm": (float(eye_center[1]) - y1) / fh,
        "nose_y_norm": (float(nose[1]) - y1) / fh,
        "mouth_y_norm": (float(mouth_center[1]) - y1) / fh,
        "mouth_w_norm": mouth_w / fw,
        "face_ar": fw / fh,
        "eye_tilt_deg": eye_tilt,
    }
    return geom


def geom_similarity_face(g1: Dict[str, float], g2: Dict[str, float]) -> Optional[float]:
    keys = ["eye_dist_norm", "eye_y_norm", "nose_y_norm", "mouth_y_norm", "mouth_w_norm", "face_ar"]
    if not all(k in g1 for k in keys) or not all(k in g2 for k in keys):
        return None

    vals = []
    for k in keys:
        a = float(g1[k]); b = float(g2[k])
        denom = max(1e-6, abs(a) + abs(b))
        rel_err = abs(a - b) / denom
        vals.append(rel_err)

    if "eye_tilt_deg" in g1 and "eye_tilt_deg" in g2:
        tilt_diff = abs(float(g1["eye_tilt_deg"]) - float(g2["eye_tilt_deg"]))
        tilt_diff = min(tilt_diff, 360.0 - tilt_diff)
        vals.append(tilt_diff / 30.0)

    mean_err = float(np.mean(vals))
    sim = 1.0 - mean_err
    return clamp(sim, 0.0, 1.0)


def extract_face_feat(img_bgr: np.ndarray, source_path: Optional[Path] = None) -> FaceFeat:
    feat = FaceFeat(ok=False, source_path=str(source_path) if source_path else None)

    if img_bgr is None:
        feat.reasons.append("IMAGE_READ_ERROR")
        return feat

    face_xyxy = None
    embedding = None
    kps5 = None

    if ENGINES.face_mode == "insightface":
        f = detect_face_insightface(img_bgr)
        if f is not None:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            face_xyxy = (x1, y1, x2, y2)
            emb = getattr(f, "embedding", None)
            if emb is not None:
                embedding = np.array(emb, dtype=np.float32)
            kps = getattr(f, "kps", None)
            if kps is not None:
                kps5 = np.array(kps, dtype=np.float32)

    if face_xyxy is None:
        face_xyxy = opencv_largest_face_bbox(img_bgr)

    if face_xyxy is None:
        feat.reasons.append("FACE_NOT_FOUND")
        return feat

    crop = crop_safe(img_bgr, face_xyxy)
    if crop is None:
        feat.reasons.append("FACE_CROP_FAILED")
        return feat

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray128 = resize_gray(gray, 128)
    hog_vec = compute_hog_vec(gray128)
    lbp_hist = compute_lbp_hist(gray128)
    phash = compute_phash64(gray128)

    bbox_ratio = bbox_area_ratio_xyxy(face_xyxy, img_bgr.shape)
    conf = linear_map_to_01(bbox_ratio, 0.006, 0.035)

    geom: Dict[str, float] = {}
    if kps5 is not None:
        geom = compute_face_geom_from_kps5(kps5, face_xyxy)

    feat.ok = True
    feat.bbox_xyxy = face_xyxy
    feat.bbox_area_ratio = bbox_ratio
    feat.embedding = embedding
    feat.kps5 = kps5
    feat.crop_gray_128 = gray128
    feat.hog_vec = hog_vec
    feat.lbp_hist = lbp_hist
    feat.phash64 = phash
    feat.geom = geom
    feat.confidence = conf
    feat.lab_mean = mean_lab(crop)
    feat.lap_var = laplacian_var(gray128)
    feat.hf_energy = high_freq_energy(gray128)

    if embedding is not None:
        feat.reasons.append("FACE_EMBEDDING_READY")
    else:
        feat.reasons.append("FACE_EMBEDDING_MISSING_USING_TEXTURE_GEOM")

    if kps5 is not None:
        feat.reasons.append("FACE_LANDMARKS_READY")
    else:
        feat.reasons.append("FACE_LANDMARKS_MISSING")

    if bbox_ratio < 0.01:
        feat.reasons.append("FACE_TOO_SMALL")

    return feat


# ============================================================
# 🧍 Pose / Framing 特征提取
# ============================================================
def extract_pose_feat(img_bgr: np.ndarray) -> PoseFeat:
    feat = PoseFeat(ok=False, mode=ENGINES.pose_mode)

    if img_bgr is None:
        feat.reasons.append("IMAGE_READ_ERROR")
        return feat

    if ENGINES.pose_mode == "mediapipe":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = ENGINES.pose_engine.process(img_rgb)
        if not results.pose_landmarks:
            feat.reasons.append("POSE_NOT_DETECTED")
            return feat

        lms = results.pose_landmarks.landmark
        xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
        vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)

        feat.ok = True
        feat.lm_xy = xy
        feat.lm_vis = vis

        reasons: List[str] = []

        NOSE = 0
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_ANKLE, R_ANKLE = 27, 28

        ankles = [xy[L_ANKLE], xy[R_ANKLE]]
        ankles_vis = [vis[L_ANKLE], vis[R_ANKLE]]
        visible_ankles = [a for a, v in zip(ankles, ankles_vis) if v > 0.35]
        if len(visible_ankles) > 0:
            max_ankle_y = float(max(a[1] for a in visible_ankles))
            feet_in_frame = 1.0 if 0.80 <= max_ankle_y <= 0.995 else 0.0
            if feet_in_frame > 0:
                reasons.append("FEET_IN_FRAME")
            else:
                reasons.append("FEET_CROPPED_OR_TOO_HIGH")
        else:
            max_ankle_y = 1.0
            feet_in_frame = 0.0
            reasons.append("ANKLES_NOT_VISIBLE")

        nose_y = float(xy[NOSE][1]) if vis[NOSE] > 0.2 else 0.08
        top_y_est = max(0.0, nose_y - 0.09)
        subject_height = max(0.0, max_ankle_y - top_y_est)
        headroom = top_y_est

        feat.framing = {
            "nose_y": nose_y,
            "top_y_est": top_y_est,
            "max_ankle_y": max_ankle_y,
            "subject_height_ratio": subject_height,
            "headroom_ratio": headroom,
            "feet_in_frame": feet_in_frame,
        }

        upper_geom: Dict[str, float] = {}
        if vis[L_SHOULDER] > 0.35 and vis[R_SHOULDER] > 0.35:
            shoulder_w = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
            upper_geom["shoulder_width_norm"] = shoulder_w
            upper_geom["shoulder_tilt_deg"] = math.degrees(
                math.atan2(
                    float(xy[R_SHOULDER][1] - xy[L_SHOULDER][1]),
                    float(xy[R_SHOULDER][0] - xy[L_SHOULDER][0]),
                )
            )

        if vis[L_HIP] > 0.35 and vis[R_HIP] > 0.35:
            hip_w = float(np.linalg.norm(xy[L_HIP] - xy[R_HIP]))
            upper_geom["hip_width_norm"] = hip_w

        if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]):
            shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
            hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
            torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
            upper_geom["torso_len_norm"] = torso_len

            # ✅ 新增：脊柱偏移角，拦截 pose 味 / 扭胯
            spine_dx = float(shoulder_mid[0] - hip_mid[0])
            spine_dy = float(hip_mid[1] - shoulder_mid[1])

            if abs(spine_dy) > 1e-6:
                spine_angle_deg = float(math.degrees(math.atan2(abs(spine_dx), abs(spine_dy))))
            else:
                spine_angle_deg = 90.0

            upper_geom["spine_angle_deg"] = spine_angle_deg

            if spine_angle_deg > 12.0:
                reasons.append("HIP_POP_DETECTED_POSSIBLE_MODEL_POSE")

        feat.upper_geom = upper_geom

        full_geom: Dict[str, float] = {}
        if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, NOSE]):
            shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
            head_proxy = float(np.linalg.norm(xy[NOSE] - shoulder_mid)) * 1.6
            if head_proxy > 1e-5 and subject_height > 1e-5:
                full_geom["head_body_ratio"] = subject_height / head_proxy

        if all(vis[idx] > 0.35 for idx in [L_HIP, R_HIP, L_ANKLE, R_ANKLE]):
            hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
            ankles_mid = (xy[L_ANKLE] + xy[R_ANKLE]) / 2
            leg_len = float(np.linalg.norm(hip_mid - ankles_mid))
            if subject_height > 1e-5:
                full_geom["leg_ratio"] = leg_len / subject_height

        feat.full_geom = full_geom

        upper_ids = [0, 11, 12, 13, 14, 23, 24]
        full_ids = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        upper_vis = float(np.mean([1.0 if vis[i] > 0.35 else 0.0 for i in upper_ids]))
        full_vis = float(np.mean([1.0 if vis[i] > 0.35 else 0.0 for i in full_ids]))

        feat.confidence_upper = clamp(upper_vis, 0.0, 1.0)
        feat.confidence_full = clamp(full_vis, 0.0, 1.0)

        if feat.confidence_upper < 0.5:
            reasons.append("UPPER_KEYPOINTS_LOW_CONFIDENCE")
        if feat.confidence_full < 0.5:
            reasons.append("FULL_KEYPOINTS_LOW_CONFIDENCE")

        feat.reasons.extend(reasons)
        return feat

    h0, w0 = img_bgr.shape[:2]
    max_side = max(h0, w0)
    scale = 1.0
    img_small = img_bgr
    if max_side > 1000:
        scale = 1000.0 / max_side
        img_small = cv2.resize(
            img_bgr,
            (int(w0 * scale), int(h0 * scale)),
            interpolation=cv2.INTER_AREA,
        )

    rects, weights = ENGINES.hog_people.detectMultiScale(
        img_small,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    if len(rects) == 0:
        feat.reasons.append("PERSON_NOT_DETECTED_OPENCV_HOG")
        return feat

    best_i = int(np.argmax(weights)) if len(weights) > 0 else 0
    x, y, bw, bh = rects[best_i]
    x, y, bw, bh = int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)

    feat.ok = True
    feat.person_bbox_xywh = (x, y, bw, bh)
    feat.person_bbox_area_ratio = float((bw * bh) / max(1, h0 * w0))
    feat.confidence_upper = 0.35
    feat.confidence_full = 0.35

    top = y
    bottom = y + bh
    subject_height = bh / float(h0)
    feet_in_frame = 1.0 if (bottom < int(h0 * 0.995)) else 0.0
    feat.framing = {
        "subject_height_ratio": subject_height,
        "headroom_ratio": top / float(h0),
        "feet_in_frame": feet_in_frame,
    }
    feat.reasons.extend(["POSE_ENGINE_FALLBACK_OPENCV_HOG", "FRAMING_APPROXIMATE"])
    return feat


# ============================================================
# 🧮 面部相似度
# ============================================================
def calibrate_face_embedding_score(raw_cos: float, engine: str) -> float:
    if engine == "insightface":
        low, high = 0.45, 0.70
    else:
        low, high = 0.80, 0.93
    return linear_map_to_01(raw_cos, low, high)


def compare_face_feat(candidate: FaceFeat, anchor: FaceFeat, face_engine_mode: str) -> Dict[str, Optional[float]]:
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
        dL = abs(float(cL - aL))
        out["luma"] = clamp(1.0 - dL / 18.0, 0.0, 1.0)
        dC = math.sqrt((float(ca - aa) ** 2) + (float(cb - ab) ** 2))
        out["chroma"] = clamp(1.0 - dC / 22.0, 0.0, 1.0)

    if candidate.lap_var > 0 and anchor.lap_var > 0:
        ratio = candidate.lap_var / max(1e-6, anchor.lap_var)
        out["sharp"] = clamp(linear_map_to_01(ratio, 0.55, 1.10), 0.0, 1.0)

    if candidate.hf_energy > 0 and anchor.hf_energy > 0:
        ratio = candidate.hf_energy / max(1e-6, anchor.hf_energy)
        out["texture"] = clamp(linear_map_to_01(ratio, 0.55, 1.10), 0.0, 1.0)

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
        # 3/4 时更信 embedding，少信 2D 投影敏感特征
        base_weights = {
            "embedding": 0.58,
            "geom": 0.10,
            "hog": 0.10,
            "lbp": 0.06,
            "phash": 0.02,
            "ssim": 0.14,
        }
    else:  # profile_like
        # 没有侧身锚点时，profile_like 先尽量减少误杀
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

    for k, w in base_weights.items():
        v = metrics.get(k, None)
        if v is None:
            reasons.append(f"FACE_ID_METRIC_MISSING_{k.upper()}")
            continue
        used[k] = float(v)
        weighted_sum += float(v) * w
        weight_sum += w

    if weight_sum < 1e-8:
        return 0.0, used, reasons

    score = weighted_sum / weight_sum
    return float(clamp(score, 0.0, 1.0)), used, reasons


def score_face_against_anchor_set(
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

    for i, a in enumerate(anchors):
        if not a.ok:
            continue
        metrics = compare_face_feat(candidate, a, ENGINES.face_mode)
        fused, used, rs = fuse_face_identity_metrics(metrics, view_bucket=view_bucket)

        coverage = clamp(len(used) / 6.0, 0.0, 1.0)
        conf = clamp(candidate.confidence * a.confidence * (0.6 + 0.4 * coverage), 0.0, 1.0)

        anchor_scores.append((fused, conf))
        anchor_used_metrics.append({
            "anchor_index": i,
            "anchor_path": a.source_path,
            "score": fused,
            "conf": conf,
            "identity_metrics": used,
            "reasons": rs,
        })

    if len(anchor_scores) == 0:
        return 0.0, 0.0, ["NO_VALID_FACE_ANCHORS"], debug

    anchor_scores_sorted = sorted(anchor_scores, key=lambda x: x[0], reverse=True)
    topk = anchor_scores_sorted[:min(3, len(anchor_scores_sorted))]
    scores = [s for s, _ in topk]
    confs = [c for _, c in topk]

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


# ============================================================
# 🦴 Pose / Upper / Full 相似度
# ============================================================
MP_IDS = {
    "NOSE": 0,
    "L_SH": 11, "R_SH": 12,
    "L_EL": 13, "R_EL": 14,
    "L_WR": 15, "R_WR": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANK": 27, "R_ANK": 28,
}

UPPER_LM_IDS = [0, 11, 12, 13, 14, 23, 24]
FULL_LM_IDS = [0, 11, 12, 23, 24, 25, 26, 27, 28]


def normalize_pose_subset(
    xy: np.ndarray,
    vis: np.ndarray,
    ids: List[int],
    mode: str = "upper",
) -> Tuple[Optional[np.ndarray], float]:
    pts = []
    valid_flags = []
    for idx in ids:
        pts.append(xy[idx].copy())
        valid_flags.append(1.0 if vis[idx] > 0.35 else 0.0)

    pts = np.array(pts, dtype=np.float32)
    valid = np.array(valid_flags, dtype=np.float32)
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
            and vis[MP_IDS["L_HIP"]] > 0.35 and vis[MP_IDS["R_HIP"]] > 0.35
            and vis[MP_IDS["L_SH"]] > 0.35 and vis[MP_IDS["R_SH"]] > 0.35
        ):
            shoulder_mid = (xy[MP_IDS["L_SH"]] + xy[MP_IDS["R_SH"]]) / 2.0
            hip_mid = (xy[MP_IDS["L_HIP"]] + xy[MP_IDS["R_HIP"]]) / 2.0
            scale = float(np.linalg.norm(hip_mid - shoulder_mid))
    else:
        if vis[MP_IDS["L_HIP"]] > 0.35 and vis[MP_IDS["R_HIP"]] > 0.35:
            center = (xy[MP_IDS["L_HIP"]] + xy[MP_IDS["R_HIP"]]) / 2.0
        else:
            center = pts[0]

        s_candidates = []
        if vis[MP_IDS["L_SH"]] > 0.35 and vis[MP_IDS["R_SH"]] > 0.35:
            s_candidates.append(float(np.linalg.norm(xy[MP_IDS["L_SH"]] - xy[MP_IDS["R_SH"]])))
        if vis[MP_IDS["L_HIP"]] > 0.35 and vis[MP_IDS["R_HIP"]] > 0.35:
            s_candidates.append(float(np.linalg.norm(xy[MP_IDS["L_HIP"]] - xy[MP_IDS["R_HIP"]])))
        if vis[MP_IDS["NOSE"]] > 0.2 and vis[MP_IDS["L_ANK"]] > 0.2 and vis[MP_IDS["R_ANK"]] > 0.2:
            ankles_mid = (xy[MP_IDS["L_ANK"]] + xy[MP_IDS["R_ANK"]]) / 2.0
            s_candidates.append(float(np.linalg.norm(ankles_mid - xy[MP_IDS["NOSE"]])) * 0.35)

        scale = float(np.mean(s_candidates)) if len(s_candidates) > 0 else 0.0

    if scale < 1e-5:
        return None, coverage

    norm_pts = (pts - center[None, :]) / scale
    for i, flag in enumerate(valid):
        if flag < 0.5:
            norm_pts[i] = 0.0

    vec = norm_pts.reshape(-1).astype(np.float32)
    return vec, coverage


def pose_vector_similarity(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> Optional[float]:

    if v1 is None or v2 is None:
        return None
    if v1.shape != v2.shape:
        return None
    if v1.ndim != 1 or (v1.size % 2 != 0):
        return None

    try:
        pts1 = v1.reshape(-1, 2).astype(np.float32)
        pts2 = v2.reshape(-1, 2).astype(np.float32)

        # ============================================================
        # 1) 从当前 normalize_pose_subset 的“零向量占位”里恢复可见性 mask
        #    被置为 [0, 0] 的点视为不可见 / 不参与对齐
        # ============================================================
        eps = 1e-8
        mask1 = ~(np.all(np.abs(pts1) < eps, axis=1))
        mask2 = ~(np.all(np.abs(pts2) < eps, axis=1))
        joint_mask = mask1 & mask2

        visible_count = int(np.sum(joint_mask))
        total_count = int(pts1.shape[0])

        # 共同可见点太少，直接退回旧逻辑兜底
        if visible_count < 3:
            cos = cosine_sim(v1, v2)
            if cos is None:
                return None
            cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0)

            dist = float(np.linalg.norm(v1 - v2))
            dist01 = float(math.exp(-2.5 * dist))
            return clamp(0.6 * cos01 + 0.4 * dist01, 0.0, 1.0)

        X = pts1[joint_mask].copy()
        Y = pts2[joint_mask].copy()

        # ============================================================
        # 2) 去中心化
        # ============================================================
        X_mean = np.mean(X, axis=0, keepdims=True)
        Y_mean = np.mean(Y, axis=0, keepdims=True)
        Xc = X - X_mean
        Yc = Y - Y_mean

        # ============================================================
        # 3) 尺度归一化
        # ============================================================
        X_norm = float(np.linalg.norm(Xc))
        Y_norm = float(np.linalg.norm(Yc))

        if X_norm < 1e-8 or Y_norm < 1e-8:
            cos = cosine_sim(v1, v2)
            if cos is None:
                return None
            return clamp((cos + 1.0) / 2.0, 0.0, 1.0)

        Xn = Xc / X_norm
        Yn = Yc / Y_norm

        # ============================================================
        # 4) Orthogonal Procrustes / SVD
        #    求最优旋转矩阵 R，使 Yn @ R 最贴合 Xn
        # ============================================================
        H = Yn.T @ Xn
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt

        # 防止镜像翻转
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = U @ Vt

        Yn_aligned = Yn @ R

        # ============================================================
        # 5) 对齐后误差 -> 相似度
        #    RMSE 越小越好
        # ============================================================
        rmse = float(np.linalg.norm(Xn - Yn_aligned) / np.sqrt(Xn.shape[0]))

        # 这个系数 4.0 是偏保守的，适合你“宁缺毋滥”
        shape_sim = float(math.exp(-4.0 * rmse))

        # ============================================================
        # 6) 轻量 cosine 兜底
        # ============================================================
        cos = cosine_sim(v1, v2)
        cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0) if cos is not None else 0.0

        # ============================================================
        # 7) 可见点覆盖率微调
        #    可见共同点越多，越敢信 shape_sim
        # ============================================================
        support_ratio = visible_count / max(1, total_count)

        # 主体靠 Procrustes，对工程噪声保留一点 cosine 底边
        fused = 0.72 * shape_sim + 0.28 * cos01

        # 覆盖率轻微调制，不搞太狠，避免把分数又压死
        fused *= (0.90 + 0.10 * support_ratio)

        return clamp(fused, 0.0, 1.0)

    except Exception:
        # 任何异常都回退旧逻辑，保证工程稳
        cos = cosine_sim(v1, v2)
        if cos is None:
            return None

        cos01 = clamp((cos + 1.0) / 2.0, 0.0, 1.0)
        dist = float(np.linalg.norm(v1 - v2))
        dist01 = float(math.exp(-2.5 * dist))
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
    elif view_bucket == "three_quarter":
        # 3/4 时 shoulder 容易因投影缩窄，降权
        weight_map = {
            "shoulder_width_norm": 0.10,
            "torso_len_norm": 0.50,
            "hip_width_norm": 0.40,
        }
        tilt_weight = 0.04
    else:  # profile_like
        # profile_like 先别信 shoulder width
        weight_map = {
            "torso_len_norm": 0.60,
            "hip_width_norm": 0.40,
        }
        tilt_weight = 0.00

    avail = [k for k in weight_map.keys() if k in g1 and k in g2]
    if len(avail) == 0:
        return None

    errs = []
    ws = []

    for k in avail:
        a, b = float(g1[k]), float(g2[k])
        denom = max(1e-6, abs(a) + abs(b))
        errs.append(abs(a - b) / denom)
        ws.append(weight_map[k])

    if tilt_weight > 0 and "shoulder_tilt_deg" in g1 and "shoulder_tilt_deg" in g2:
        d = abs(float(g1["shoulder_tilt_deg"]) - float(g2["shoulder_tilt_deg"]))
        d = min(d, 360.0 - d)
        errs.append(d / 25.0)
        ws.append(tilt_weight)

    if len(errs) == 0:
        return None

    mean_err = float(np.average(np.array(errs, dtype=np.float32), weights=np.array(ws, dtype=np.float32)))
    sim = 1.0 - mean_err
    return clamp(sim, 0.0, 1.0)


def full_geom_similarity(g1: Dict[str, float], g2: Dict[str, float]) -> Optional[float]:
    keys = ["head_body_ratio", "leg_ratio"]
    avail = [k for k in keys if k in g1 and k in g2]
    if len(avail) == 0:
        return None

    errs = []
    for k in avail:
        a, b = float(g1[k]), float(g2[k])
        denom = max(1e-6, abs(a) + abs(b))
        errs.append(abs(a - b) / denom)

    sim = 1.0 - float(np.mean(errs))
    return clamp(sim, 0.0, 1.0)


def framing_score_from_pose_feat(p: PoseFeat) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if not p.ok:
        return 0.0, ["POSE_NOT_AVAILABLE"]

    if p.mode == "opencv":
        fr = p.framing
        subj = safe_float(fr.get("subject_height_ratio", 0.0))
        headroom = safe_float(fr.get("headroom_ratio", 1.0))
        feet = safe_float(fr.get("feet_in_frame", 0.0))

        score = 0.0
        score += 0.4 * linear_map_to_01(subj, 0.45, 0.85)
        score += 0.3 * (1.0 - clamp(abs(headroom - 0.08) / 0.20, 0.0, 1.0))
        score += 0.3 * feet
        reasons.append("FRAMING_APPROX_OPENCV")
        return clamp(score, 0.0, 1.0), reasons

    fr = p.framing
    feet = safe_float(fr.get("feet_in_frame", 0.0))
    subj = safe_float(fr.get("subject_height_ratio", 0.0))
    headroom = safe_float(fr.get("headroom_ratio", 1.0))

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
    for i, a in enumerate(anchors):
        if not a.ok or a.mode != "mediapipe" or a.lm_xy is None or a.lm_vis is None:
            continue

        vec_a, cov_a = normalize_pose_subset(a.lm_xy, a.lm_vis, UPPER_LM_IDS, mode="upper")
        s_pose = pose_vector_similarity(vec_c, vec_a)
        s_geom = upper_geom_similarity(candidate.upper_geom, a.upper_geom, view_bucket=view_bucket)

        parts = []
        if s_pose is not None:
            parts.append(("pose", s_pose, 0.35))
        if s_geom is not None:
            parts.append(("geom", s_geom, 0.65))
        if len(parts) == 0:
            continue

        ws = sum(w for _, _, w in parts)
        fused = sum(v * w for _, v, w in parts) / max(1e-8, ws)
        conf = clamp((cov_c * cov_a) * 0.9 + 0.1, 0.0, 1.0)

        scores.append((fused, conf))
        debug["anchor_scores"].append({
            "anchor_index": i,
            "pose_score": s_pose,
            "geom_score": s_geom,
            "fused": fused,
            "conf": conf,
        })

    if len(scores) == 0:
        return 0.0, clamp(candidate.confidence_upper, 0.0, 1.0), ["NO_VALID_UPPER_ANCHORS"] + candidate.reasons, debug

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    topk = scores_sorted[:min(3, len(scores_sorted))]
    svals = [s for s, _ in topk]
    cvals = [c for _, c in topk]
    score = float(0.6 * np.mean(svals) + 0.4 * np.median(svals))
    conf = float(np.mean(cvals))

    if candidate.confidence_upper < 0.5:
        reasons.append("UPPER_KEYPOINTS_LOW_CONFIDENCE")

    return clamp(score, 0.0, 1.0), clamp(conf, 0.0, 1.0), reasons + candidate.reasons, debug


def score_full_against_anchor_set(candidate: PoseFeat, anchors: List[PoseFeat]) -> Tuple[float, float, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    debug: Dict[str, Any] = {"anchor_scores": []}

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
    for i, a in enumerate(anchors):
        if not a.ok or a.mode != "mediapipe" or a.lm_xy is None or a.lm_vis is None:
            continue

        vec_a, cov_a = normalize_pose_subset(a.lm_xy, a.lm_vis, FULL_LM_IDS, mode="full")
        s_pose = pose_vector_similarity(vec_c, vec_a)
        s_geom = full_geom_similarity(candidate.full_geom, a.full_geom)

        parts = [("framing", fr_score, 0.45)]
        if s_geom is not None:
            parts.append(("geom", s_geom, 0.35))
        if s_pose is not None:
            parts.append(("pose", s_pose, 0.20))

        ws = sum(w for _, _, w in parts)
        fused = sum(v * w for _, v, w in parts) / max(1e-8, ws)
        conf = clamp((cov_c * cov_a) * 0.9 + 0.1, 0.0, 1.0)

        scores.append((fused, conf))
        debug["anchor_scores"].append({
            "anchor_index": i,
            "framing_score": fr_score,
            "pose_score": s_pose,
            "geom_score": s_geom,
            "fused": fused,
            "conf": conf,
        })

    if len(scores) == 0:
        return fr_score, clamp(candidate.confidence_full, 0.0, 1.0), ["NO_VALID_FULL_ANCHORS"] + reasons + candidate.reasons, debug

    scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
    topk = scores_sorted[:min(3, len(scores_sorted))]
    svals = [s for s, _ in topk]
    cvals = [c for _, c in topk]
    score = float(0.6 * np.mean(svals) + 0.4 * np.median(svals))
    conf = float(np.mean(cvals))

    if candidate.confidence_full < 0.5:
        reasons.append("FULL_KEYPOINTS_LOW_CONFIDENCE")

    return clamp(score, 0.0, 1.0), clamp(conf, 0.0, 1.0), reasons + candidate.reasons, debug


# ============================================================
# 🏷️ 状态判定 / 融合 / 建议
# ============================================================
def classify_module(score: float, conf: float, pass_th: float, warn_th: float, module_name: str) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if conf < FACE_NO_SIGNAL_CONF_TH:
        return "FAIL", [f"{module_name.upper()}_NO_RELIABLE_SIGNAL"]

    if conf < MIN_CONF_FOR_STRICT_FAIL:
        reasons.append(f"{module_name.upper()}_LOW_CONFIDENCE")
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
    for k in ["face", "upper", "full"]:
        w = float(weights.get(k, 0.0))
        c = float(confs.get(k, 0.0))
        s = float(scores.get(k, 0.0))
        eff = w * (0.25 + 0.75 * c)
        acc += s * eff
        ws += eff
    if ws < 1e-8:
        return 0.0
    return float(acc / ws)



def make_recommendations(result: Dict[str, Any], profile_name: str) -> List[str]:
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

    if any("FACE_TOO_SMALL" in r for r in reasons) or face_c < 0.5:
        recs.append("提高脸部有效像素（拉近构图或提升分辨率），否则身份分会抖动")

    if "FACE_UNDEREXPOSED_DARK" in reasons:
        recs.append("脸部实测欠曝：补正面填充光或锁曝光，不要让脸比身体更暗")
    elif "FACE_DARKER_THAN_ANCHOR" in reasons:
        recs.append("相对锚点略暗：这更像拍摄条件差异，先复查是否只是远景/补光变化")

    if any(x in reasons for x in [
        "FACE_TOO_SOFT_POSSIBLE_SMOOTHING",
        "FACE_LOW_MICROTEXTURE",
        "FACE_SOFTER_THAN_ANCHOR",
        "FACE_LOWER_TEXTURE_THAN_ANCHOR",
    ]):
        recs.append("皮肤细节偏软：提高脸部有效像素或减少磨皮/过强降噪")

    if face_s < 0.65:
        recs.append("身份相似度偏低：优先检查锚点一致性、脸部占比与 ref 冲突")
    if upper_s < 0.65:
        recs.append("半身比例/体态不稳：优先用 upper anchor 稳肩颈与锁骨")
    if full_s < 0.65:
        recs.append("全身构图/姿态不稳：先修全身入框（脚/头留白/主体占比）")

    if constitution_s is not None and float(constitution_s) < CONSTITUTION_SOFT_WARN_TH:
        recs.append("身材宪法偏移：复查腰线、骨盆紧凑度、腿型细长度与下半身轻盈感")

    if skin_s is not None and float(skin_s) < SKIN_SOFT_WARN_TH:
        recs.append("肤色一致性不足：复查脸-脖子-腿部亮度与色偏，优先排查腿部偏暗/偏黄/膝盖脏影")

    if depth_s is not None and float(depth_s) < DEPTH3D_SOFT_WARN_TH:
        recs.append("3/4 空间厚度不足：疑似假转体或 2.5D 贴脸，建议补肩胯透视与胸廓厚度")

    if any("FEET_CROPPED" in r for r in reasons):
        recs.append("构图返工：确保脚完整入框（full-body 任务为硬条件）")

    if len(recs) == 0:
        recs.append("一致性表现良好，可进入人工终审/训练入库阶段")

    return recs

def get_profile_policy(profile_name: str) -> Dict[str, Any]:
    return PROFILE_POLICY.get(profile_name, PROFILE_POLICY["lora_dataset"])


def get_identity_anchor_pool(profile_name: str, anchors: AnchorSet) -> List[FaceFeat]:
    policy = get_profile_policy(profile_name)
    mode = policy.get("identity_anchor_pool", "face")
    if mode == "face":
        return valid_face_feats(anchors.face_feats)
    if mode == "upper_first":
        pool = valid_face_feats(anchors.upper_face_feats)
        return pool if len(pool) > 0 else valid_face_feats(anchors.face_feats)
    return valid_face_feats(anchors.face_feats)

def filter_face_anchors_by_view(
    anchors: List[FaceFeat],
    view_bucket: str,
) -> List[FaceFeat]:
    valid = [a for a in anchors if a.ok]

    if view_bucket == "front":
        pool = [a for a in valid if infer_anchor_view_from_path(a.source_path) == "front"]
        return pool if len(pool) > 0 else valid

    if view_bucket == "three_quarter":
        pool = [a for a in valid if infer_anchor_view_from_path(a.source_path) == "three_quarter"]
        return pool if len(pool) > 0 else valid

    if view_bucket == "profile_like":
        pool = [a for a in valid if infer_anchor_view_from_path(a.source_path) == "profile_like"]
        if len(pool) > 0:
            return pool
        pool = [a for a in valid if infer_anchor_view_from_path(a.source_path) == "three_quarter"]
        return pool if len(pool) > 0 else valid

    return valid

def get_quality_anchor_pool(profile_name: str, anchors: AnchorSet) -> List[FaceFeat]:
    policy = get_profile_policy(profile_name)
    mode = policy.get("quality_anchor_pool", "face")

    if mode == "face":
        return valid_face_feats(anchors.face_feats)

    if mode == "upper_first":
        pool = valid_face_feats(anchors.upper_face_feats)
        if len(pool) > 0:
            return pool
        pool = valid_face_feats(anchors.face_feats)
        return pool

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

    for a in face_feats:
        if not a.ok:
            continue
        bucket = get_face_size_bucket(a.bbox_area_ratio)
        target_keys = ["all", bucket]
        for k in target_keys:
            if a.lab_mean is not None:
                stats[k]["L"].append(float(a.lab_mean[0]))
            if a.lap_var > 0:
                stats[k]["lap"].append(float(a.lap_var))
            if a.hf_energy > 0:
                stats[k]["hf"].append(float(a.hf_energy))
            stats[k]["count"] += 1

    return stats


def get_stats_for_bucket(stats: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    selected = stats.get(bucket, {})
    if selected.get("count", 0) >= 1:
        return selected
    return stats.get("all", {"L": [], "lap": [], "hf": [], "count": 0})


# ============================================================
# 🗂️ Anchor Set 加载
# ============================================================
def load_anchor_set() -> AnchorSet:
    anchors = AnchorSet()

    # ✅ face 改为递归读取，支持 front / three_quarter / profile_like 子目录
    anchors.meta["face_paths"] = [str(p) for p in list_images_recursive(DIR_ANCHOR_FACE)]

    # upper / full 继续维持原本一层读取就行
    anchors.meta["upper_paths"] = [str(p) for p in list_images_in_dir(DIR_ANCHOR_UPPER)]
    anchors.meta["full_paths"] = [str(p) for p in list_images_in_dir(DIR_ANCHOR_FULL)]

    print("\n[初始化] 加载 Anchor Set...")
    print(f"  Face Anchors : {len(anchors.meta['face_paths'])}")
    print(f"  Upper Anchors: {len(anchors.meta['upper_paths'])}")
    print(f"  Full Anchors : {len(anchors.meta['full_paths'])}")

    for path_str in anchors.meta["face_paths"]:
        p = Path(path_str)
        img = image_read_bgr(p)
        ff = extract_face_feat(img, p) if img is not None else FaceFeat(
            ok=False,
            reasons=["IMAGE_READ_ERROR"],
            source_path=str(p),
        )
        anchors.face_feats.append(ff)

    for path_str in anchors.meta["upper_paths"]:
        p = Path(path_str)
        img = image_read_bgr(p)
        if img is None:
            anchors.upper_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.upper_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(p)))
            continue
        anchors.upper_pose_feats.append(extract_pose_feat(img))
        anchors.upper_face_feats.append(extract_face_feat(img, p))

    for path_str in anchors.meta["full_paths"]:
        p = Path(path_str)
        img = image_read_bgr(p)
        if img is None:
            anchors.full_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.full_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(p)))
            continue
        anchors.full_pose_feats.append(extract_pose_feat(img))
        anchors.full_face_feats.append(extract_face_feat(img, p))

    print(f"  Upper Face-like Quality Refs: {len(valid_face_feats(anchors.upper_face_feats))}")
    print(f"  Full  Face-like Quality Refs: {len(valid_face_feats(anchors.full_face_feats))}")

    # 调试时你会很爽：能看到 face 锚点分桶情况
    face_front = sum(1 for x in anchors.face_feats if infer_anchor_view_from_path(x.source_path) == "front")
    face_3q = sum(1 for x in anchors.face_feats if infer_anchor_view_from_path(x.source_path) == "three_quarter")
    face_profile = sum(1 for x in anchors.face_feats if infer_anchor_view_from_path(x.source_path) == "profile_like")
    print(f"  Face Buckets => front={face_front} | three_quarter={face_3q} | profile_like={face_profile}")

    return anchors


# ============================================================
# 🧪 自动阈值校准
# ============================================================
def calibrate_quality_thresholds(calib_dir: Path) -> Dict[str, float]:
    imgs = list_images_in_dir(calib_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"校准目录为空: {calib_dir}")

    lumas: List[float] = []
    lap_vars: List[float] = []
    hf_energies: List[float] = []
    used = 0

    for p in imgs:
        img = image_read_bgr(p)
        if img is None:
            continue
        ff = extract_face_feat(img, p)
        if not ff.ok or ff.lab_mean is None:
            continue

        lumas.append(float(ff.lab_mean[0]))
        if ff.lap_var > 0:
            lap_vars.append(float(ff.lap_var))
        if ff.hf_energy > 0:
            hf_energies.append(float(ff.hf_energy))
        used += 1

    if used < 8:
        raise RuntimeError(f"有效校准样本太少，仅 {used} 张，建议至少 8 张，最好 20–40 张")

    out = {
        "FACE_LUMA_DARK_WARN_L": robust_percentile(lumas, 10),
        "FACE_LAPVAR_SOFT_WARN": robust_percentile(lap_vars, 15),
        "FACE_HFENERGY_SOFT_WARN": robust_percentile(hf_energies, 15),
        "num_used": used,
        "luma_mean": float(np.mean(lumas)) if lumas else 0.0,
        "lap_var_mean": float(np.mean(lap_vars)) if lap_vars else 0.0,
        "hf_energy_mean": float(np.mean(hf_energies)) if hf_energies else 0.0,
    }
    return out


def save_thresholds_to_file(thresholds: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)


def load_thresholds_from_file(path: Path) -> None:
    global FACE_LUMA_DARK_WARN_L, FACE_LAPVAR_SOFT_WARN, FACE_HFENERGY_SOFT_WARN
    if not path.exists():
        print(f"[提示] 阈值文件不存在，继续使用默认阈值: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    FACE_LUMA_DARK_WARN_L = float(data.get("FACE_LUMA_DARK_WARN_L", FACE_LUMA_DARK_WARN_L))
    FACE_LAPVAR_SOFT_WARN = float(data.get("FACE_LAPVAR_SOFT_WARN", FACE_LAPVAR_SOFT_WARN))
    FACE_HFENERGY_SOFT_WARN = float(data.get("FACE_HFENERGY_SOFT_WARN", FACE_HFENERGY_SOFT_WARN))

    print("\n[已加载自动校准阈值]")
    print(f"  FACE_LUMA_DARK_WARN_L = {FACE_LUMA_DARK_WARN_L:.4f}")
    print(f"  FACE_LAPVAR_SOFT_WARN = {FACE_LAPVAR_SOFT_WARN:.4f}")
    print(f"  FACE_HFENERGY_SOFT_WARN = {FACE_HFENERGY_SOFT_WARN:.4f}")


# ============================================================
# 🏭 主流程：批处理质检
# ============================================================

def run_pipeline(profile_name: str = ACTIVE_PROFILE) -> None:
    if profile_name not in TASK_PROFILES:
        raise ValueError(f"未知任务模板: {profile_name}. 可选: {list(TASK_PROFILES.keys())}")

    profile = TASK_PROFILES[profile_name]
    weights = profile["weights"]
    reqs = profile["require"]
    th = profile["thresholds"]
    policy = get_profile_policy(profile_name)

    if not DIR_INPUT.exists():
        print(f"[致命错误] 输入目录不存在: {DIR_INPUT}")
        return

    images = list_images_in_dir(DIR_INPUT)
    if len(images) == 0:
        print(f"[提示] 输入目录为空: {DIR_INPUT}")
        return

    anchors = load_anchor_set()
    face_identity_anchors = get_identity_anchor_pool(profile_name, anchors)
    face_quality_anchors = get_quality_anchor_pool(profile_name, anchors)
    quality_ref_stats = build_quality_reference_stats(face_quality_anchors)

    if len(face_identity_anchors) == 0:
        print("[警告] 没有可用面部身份锚点，将导致 face 模块不可用")
    if len(anchors.upper_pose_feats) == 0:
        print("[警告] 没有半身锚点，将导致 upper 模块不可用")
    if len(anchors.full_pose_feats) == 0:
        print("[警告] 没有全身锚点，将导致 full 模块不可用")

    report: List[Dict[str, Any]] = []
    print(f"\n[运行中] 任务模板: {profile_name}")
    print(f"[运行中] 身份锚池(face): {len(face_identity_anchors)}")
    print(f"[运行中] 质量锚池(face-like): {len(face_quality_anchors)}")
    print(f"[运行中] STANDARDIZE_INPUT={STANDARDIZE_INPUT} long_side={STANDARDIZE_LONG_SIDE}")
    print(f"[运行中] CONSISTENCY_MODE={CONSISTENCY_MODE}")
    print("[运行中] 开始批处理质检...\n")

    for img_path in images:
        print(f"-> 检测: {img_path.name}")
        try:
            img = image_read_bgr(img_path)
            if img is None:
                raise RuntimeError("IMAGE_READ_ERROR")

            cand_face = extract_face_feat(img, img_path)
            cand_pose = extract_pose_feat(img)
            view_bucket, view_side, yaw_proxy = estimate_view_bucket_and_side(cand_face)

            # 新增：一致性模块（仍由 face/upper/full 主导，新模块做高置信 soft gate）
            constitution_metrics = extract_body_constitution_metrics(
                img,
                cand_face,
                cand_pose,
                view_bucket=view_bucket,
            )
            skin_metrics = extract_skin_consistency_metrics(
                img,
                cand_face,
                cand_pose,
            )
            depth_3d_metrics = extract_depth_3d_lite_metrics(
                cand_face,
                cand_pose,
                view_bucket=view_bucket,
                yaw_proxy=yaw_proxy,
            )

            constitution_score = constitution_metrics.get("body_constitution_score", None)
            skin_score = skin_metrics.get("skin_uniformity_score", None)
            depth_3d_score = depth_3d_metrics.get("depth_3d_score", None)

            # face score（按视角过滤 + 3/4 镜像规范化）
            face_identity_anchors_view = filter_face_anchors_by_view(face_identity_anchors, view_bucket)

            face_score_o, face_conf_o, face_reasons_o, face_debug_o = score_face_against_anchor_set(
                cand_face,
                face_identity_anchors_view,
                view_bucket=view_bucket,
            )

            face_score = face_score_o
            face_conf = face_conf_o
            face_reasons = face_reasons_o
            face_debug = {
                "view_bucket": view_bucket,
                "view_side": view_side,
                "yaw_proxy": yaw_proxy,
                "flip_canonicalized": False,
                "identity_anchor_count_view": len(face_identity_anchors_view),
                "original": face_debug_o,
            }

            if view_bucket != "front":
                img_flipped = cv2.flip(img, 1)
                cand_face_flip = extract_face_feat(img_flipped, None)
                face_score_f, face_conf_f, face_reasons_f, face_debug_f = score_face_against_anchor_set(
                    cand_face_flip,
                    face_identity_anchors_view,
                    view_bucket=view_bucket,
                )
                face_debug["flipped"] = face_debug_f
                if face_score_f > face_score_o:
                    face_score = face_score_f
                    face_conf = face_conf_f
                    face_reasons = ["FACE_FLIP_CANONICALIZED"] + face_reasons_f
                    face_debug["flip_canonicalized"] = True

            upper_score, upper_conf, upper_reasons, upper_debug = score_upper_against_anchor_set(
                cand_pose,
                anchors.upper_pose_feats,
                view_bucket=view_bucket,
            )
            full_score, full_conf, full_reasons, full_debug = score_full_against_anchor_set(
                cand_pose,
                anchors.full_pose_feats,
            )

            face_state, face_state_reasons = classify_module(face_score, face_conf, th["face_pass"], th["face_warn"], "face")
            upper_state, upper_state_reasons = classify_module(upper_score, upper_conf, th["upper_pass"], th["upper_warn"], "upper")
            full_state, full_state_reasons = classify_module(full_score, full_conf, th["full_pass"], th["full_warn"], "full")

            scores = {"face": face_score, "upper": upper_score, "full": full_score}
            confs = {"face": face_conf, "upper": upper_conf, "full": full_conf}
            overall_score = fuse_overall(scores, confs, weights)

            if overall_score >= th["overall_pass"]:
                overall_state = "PASS"
            elif overall_score >= th["overall_warn"]:
                overall_state = "WARN"
            else:
                overall_state = "FAIL"

            hard_fail = False
            hard_warn = False

            if reqs.get("face", False):
                if face_state == "FAIL" and face_conf >= MIN_CONF_FOR_STRICT_FAIL:
                    hard_fail = True
                elif face_state != "PASS":
                    hard_warn = True

            if reqs.get("upper", False):
                if upper_state == "FAIL" and upper_conf >= MIN_CONF_FOR_STRICT_FAIL:
                    hard_fail = True
                elif upper_state != "PASS":
                    hard_warn = True

            if reqs.get("full", False):
                if full_state == "FAIL" and full_conf >= MIN_CONF_FOR_STRICT_FAIL:
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
                face_state_reasons + upper_state_reasons + full_state_reasons +
                face_reasons + upper_reasons + full_reasons
            )

            extra_flags: List[str] = []
            quality_debug: Dict[str, Any] = {}

            if not cand_face.ok or face_conf < FACE_NO_SIGNAL_CONF_TH:
                extra_flags.append("FACE_NO_RELIABLE_SIGNAL")
            else:
                qtol = get_quality_tolerances_by_face_size(cand_face.bbox_area_ratio)
                bucket = qtol["bucket"]
                bucket_stats = get_stats_for_bucket(quality_ref_stats, bucket)

                quality_debug["face_size_bucket"] = bucket
                quality_debug["bucket_quality_ref_count"] = bucket_stats.get("count", 0)
                quality_debug["bucket_quality_tolerances"] = qtol

                if cand_face.lab_mean is not None:
                    cand_L = float(cand_face.lab_mean[0])
                    quality_debug["candidate_face_L"] = cand_L
                    if cand_L < qtol["abs_luma_warn"]:
                        extra_flags.append("FACE_UNDEREXPOSED_DARK")

                    if len(bucket_stats.get("L", [])) > 0:
                        anchor_L_mean = float(np.mean(bucket_stats["L"]))
                        quality_debug["anchor_face_L_mean_bucket"] = anchor_L_mean
                        if cand_L < (anchor_L_mean - qtol["dark_delta_L"]):
                            extra_flags.append("FACE_DARKER_THAN_ANCHOR")

                if cand_face.lap_var > 0:
                    quality_debug["candidate_face_lap_var"] = cand_face.lap_var
                    if cand_face.lap_var < qtol["abs_lap_warn"]:
                        extra_flags.append("FACE_TOO_SOFT_POSSIBLE_SMOOTHING")

                    if len(bucket_stats.get("lap", [])) > 0:
                        anchor_lap_mean = float(np.mean(bucket_stats["lap"]))
                        quality_debug["anchor_face_lap_mean_bucket"] = anchor_lap_mean
                        if cand_face.lap_var < (anchor_lap_mean * qtol["sharp_ratio_floor"]):
                            extra_flags.append("FACE_SOFTER_THAN_ANCHOR")

                if cand_face.hf_energy > 0:
                    quality_debug["candidate_face_hf_energy"] = cand_face.hf_energy
                    if cand_face.hf_energy < qtol["abs_hf_warn"]:
                        extra_flags.append("FACE_LOW_MICROTEXTURE")

                    if len(bucket_stats.get("hf", [])) > 0:
                        anchor_hf_mean = float(np.mean(bucket_stats["hf"]))
                        quality_debug["anchor_face_hf_mean_bucket"] = anchor_hf_mean
                        if cand_face.hf_energy < (anchor_hf_mean * qtol["texture_ratio_floor"]):
                            extra_flags.append("FACE_LOWER_TEXTURE_THAN_ANCHOR")

            reasons_all.extend(extra_flags)
            reasons_all = dedupe_keep_order(reasons_all)

            # 新模块以高置信 soft gate 方式参与
            reasons_all, final_status, overall_state, consistency_gate_debug = apply_consistency_soft_gate(
                reasons_all=reasons_all,
                final_status=final_status,
                overall_state=overall_state,
                constitution_metrics=constitution_metrics,
                skin_metrics=skin_metrics,
                depth_3d_metrics=depth_3d_metrics,
                view_bucket=view_bucket,
            )

            hard_quality_flags = set(policy.get("hard_quality_flags", set()))
            soft_quality_flags = QUALITY_DEGRADE_FLAGS - hard_quality_flags

            hard_hits = sum(1 for x in reasons_all if x in hard_quality_flags)
            soft_hits = sum(1 for x in reasons_all if x in soft_quality_flags)
            soft_hit_limit = int(policy.get("soft_quality_hits_to_warn", 2))

            if final_status == "PASS":
                if hard_hits >= 1:
                    final_status = "WARN"
                    overall_state = "WARN"
                elif soft_hits >= soft_hit_limit:
                    final_status = "WARN"
                    overall_state = "WARN"

            if "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE" in reasons_all and final_status == "PASS":
                final_status = "WARN"
                overall_state = "WARN"

            if profile_name == "body_gold_fullbody":
                if view_bucket == "profile_like" and final_status == "PASS":
                    final_status = "WARN"
                    overall_state = "WARN"
                    reasons_all.append("PROFILE_LIKE_NO_SIDE_ANCHOR_PASS_CAPPED")
                elif view_bucket == "three_quarter":
                    reasons_all.append("THREE_QUARTER_SOFT_REVIEW")

            reasons_all = dedupe_keep_order(reasons_all)

            result_node = {
                "image": img_path.name,
                "task_profile": profile_name,
                "status": final_status,
                "scores": {
                    "face": round(face_score, 4),
                    "upper": round(upper_score, 4),
                    "full": round(full_score, 4),
                    "overall": round(overall_score, 4),
                    "constitution": round(float(constitution_score), 4) if constitution_score is not None else None,
                    "skin": round(float(skin_score), 4) if skin_score is not None else None,
                    "depth_3d": round(float(depth_3d_score), 4) if depth_3d_score is not None else None,
                },
                "confidence": {
                    "face": round(face_conf, 4),
                    "upper": round(upper_conf, 4),
                    "full": round(full_conf, 4),
                    "constitution": round(float(constitution_metrics.get("confidence", 0.0)), 4),
                    "skin": round(float(skin_metrics.get("confidence", 0.0)), 4),
                    "depth_3d": round(float(depth_3d_metrics.get("confidence", 0.0)), 4),
                },
                "module_state": {
                    "face": face_state,
                    "upper": upper_state,
                    "full": full_state,
                    "overall": overall_state,
                },
                "reasons": reasons_all,
                "reasons_face": face_reasons,
                "reasons_upper": upper_reasons,
                "reasons_full": full_reasons,
                "recommendations": [],
                "engine": {
                    "face": ENGINES.face_mode,
                    "pose": ENGINES.pose_mode,
                    "ssim_backend": "skimage" if SKIMAGE_SSIM_AVAILABLE else "ncc_fallback",
                },
                "debug": {
                    "face": face_debug,
                    "upper": upper_debug,
                    "full": full_debug,
                    "constitution_metrics": constitution_metrics,
                    "skin_metrics": skin_metrics,
                    "depth_3d_metrics": depth_3d_metrics,
                    "consistency_gate": consistency_gate_debug,
                    "candidate_pose_framing": cand_pose.framing,
                    "candidate_upper_geom": cand_pose.upper_geom,
                    "candidate_full_geom": cand_pose.full_geom,
                    "candidate_face_bbox_area_ratio": cand_face.bbox_area_ratio if cand_face.ok else 0.0,
                    "candidate_face_lab_mean": cand_face.lab_mean.tolist() if (cand_face.ok and cand_face.lab_mean is not None) else None,
                    "candidate_face_lap_var": cand_face.lap_var if cand_face.ok else 0.0,
                    "candidate_face_hf_energy": cand_face.hf_energy if cand_face.ok else 0.0,
                    "quality_gate_flags": extra_flags,
                    "quality_gate_soft_hits": soft_hits,
                    "quality_gate_hard_hits": hard_hits,
                    "quality_anchor_pool_mode": policy.get("quality_anchor_pool"),
                    "quality_ref_stats": quality_debug,
                    "view_bucket": view_bucket,
                    "view_side": view_side,
                    "yaw_proxy": yaw_proxy,
                    "identity_anchor_count_view": len(face_identity_anchors_view),
                    "input_shape": list(img.shape[:2]),
                },
            }

            result_node["recommendations"] = make_recommendations(result_node, profile_name)
            report.append(result_node)

            c_show = "NA" if constitution_score is None else f"{constitution_score:.3f}"
            s_show = "NA" if skin_score is None else f"{skin_score:.3f}"
            d_show = "NA" if depth_3d_score is None else f"{depth_3d_score:.3f}"

            if final_status == "PASS":
                shutil.copy2(img_path, DIR_OUT_PASS / img_path.name)
                print(
                    f"   PASS | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={c_show} skin={s_show} depth3d={d_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            elif final_status == "WARN":
                shutil.copy2(img_path, DIR_OUT_WARN / img_path.name)
                print(
                    f"   WARN | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={c_show} skin={s_show} depth3d={d_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            else:
                shutil.copy2(img_path, DIR_OUT_FAIL / img_path.name)
                print(
                    f"   FAIL | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={c_show} skin={s_show} depth3d={d_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )

        except Exception as e:
            print(f"   FAIL (异常): {e}")
            traceback.print_exc()

            fail_node = {
                "image": img_path.name,
                "task_profile": profile_name,
                "status": "FAIL",
                "scores": {
                    "face": 0.0,
                    "upper": 0.0,
                    "full": 0.0,
                    "overall": 0.0,
                    "constitution": None,
                    "skin": None,
                    "depth_3d": None,
                },
                "confidence": {
                    "face": 0.0,
                    "upper": 0.0,
                    "full": 0.0,
                    "constitution": 0.0,
                    "skin": 0.0,
                    "depth_3d": 0.0,
                },
                "module_state": {"face": "FAIL", "upper": "FAIL", "full": "FAIL", "overall": "FAIL"},
                "reasons": ["RUNTIME_EXCEPTION", str(e)],
                "reasons_face": [],
                "reasons_upper": [],
                "reasons_full": [],
                "recommendations": ["检查依赖、图片可读性、模型环境与日志"],
                "engine": {"face": ENGINES.face_mode, "pose": ENGINES.pose_mode},
                "debug": {
                    "constitution_metrics": None,
                    "skin_metrics": None,
                    "depth_3d_metrics": None,
                    "consistency_gate": None,
                },
            }
            report.append(fail_node)

            try:
                shutil.copy2(img_path, DIR_OUT_FAIL / img_path.name)
            except Exception:
                pass

    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[完工] 质检完成 ✅")
    print(f"[报告] {REPORT_FILE}")
    print(f"[输出目录] PASS={DIR_OUT_PASS} | WARN={DIR_OUT_WARN} | FAIL={DIR_OUT_FAIL}")

# ============================================================
# 🚀 主入口
# ============================================================
if __name__ == "__main__":
    print(f"[CONFIG] RUN_MODE={RUN_MODE}")
    print(f"[CONFIG] ACTIVE_PROFILE={ACTIVE_PROFILE}")
    print(f"[CONFIG] CONFIG_DIR={CONFIG_DIR}")
    print(f"[CONFIG] EXTERNAL_CONFIG_STATUS={EXTERNAL_CONFIG_STATUS}")
    print("[CONFIG] FACE_CONF_MAP = linear_map_to_01(bbox_ratio, 0.006, 0.035)")
    print(f"[CONFIG] FACE_NO_RELIABLE_SIGNAL_TH = {FACE_NO_SIGNAL_CONF_TH}")
    print(f"[CONFIG] MIN_CONF_FOR_STRICT_FAIL = {MIN_CONF_FOR_STRICT_FAIL}")
    print(f"[CONFIG] CONSISTENCY_MODE={CONSISTENCY_MODE}")

    if RUN_MODE not in {"qa", "calibrate"}:
        raise ValueError("RUN_MODE 只能是 'qa' 或 'calibrate'")

    if RUN_MODE == "calibrate":
        print(f"[校准模式] 从目录读取样本: {DIR_CALIB}")
        thresholds = calibrate_quality_thresholds(DIR_CALIB)
        save_thresholds_to_file(thresholds, THRESH_FILE)
        print("\n[自动校准完成 ✅]")
        print(json.dumps(thresholds, indent=2, ensure_ascii=False))
        print(f"[阈值文件已保存] {THRESH_FILE}")
    else:
        if AUTO_LOAD_THRESHOLDS:
            load_thresholds_from_file(THRESH_FILE)
        run_pipeline(profile_name=ACTIVE_PROFILE)
