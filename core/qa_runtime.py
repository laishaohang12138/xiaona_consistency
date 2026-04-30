from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml as pyyaml
except Exception:  # pragma: no cover - optional dependency fallback
    pyyaml = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _default_body_constitution_scoring() -> Dict[str, Any]:
    return {
        "views": {
            "front": {
                "waist_to_shoulder": {"lo": 0.42, "hi": 0.62, "margin": 0.18},
                "chest_to_waist": {"lo": 1.08, "hi": 1.48, "margin": 0.26},
                "hip_to_waist": {"lo": 1.12, "hi": 1.56, "margin": 0.28},
                "hip_to_shoulder": {"lo": 0.46, "hi": 0.72, "margin": 0.20},
                "thigh_to_calf": {"lo": 1.10, "hi": 1.76, "margin": 0.40},
                "view_factor": 1.0,
            },
            "three_quarter": {
                "waist_to_shoulder": {"lo": 0.44, "hi": 0.70, "margin": 0.22},
                "chest_to_waist": {"lo": 1.00, "hi": 1.42, "margin": 0.28},
                "hip_to_waist": {"lo": 1.04, "hi": 1.52, "margin": 0.32},
                "hip_to_shoulder": {"lo": 0.40, "hi": 0.76, "margin": 0.24},
                "thigh_to_calf": {"lo": 1.08, "hi": 1.82, "margin": 0.44},
                "view_factor": 0.88,
            },
            "profile_like": {
                "waist_to_shoulder": {"lo": 0.46, "hi": 0.78, "margin": 0.26},
                "chest_to_waist": {"lo": 0.96, "hi": 1.38, "margin": 0.32},
                "hip_to_waist": {"lo": 1.00, "hi": 1.48, "margin": 0.36},
                "hip_to_shoulder": {"lo": 0.36, "hi": 0.78, "margin": 0.28},
                "thigh_to_calf": {"lo": 1.06, "hi": 1.88, "margin": 0.50},
                "view_factor": 0.75,
            },
        },
        "common_ranges": {
            "leg_ratio": {"lo": 0.44, "hi": 0.58, "margin": 0.10},
            "waist_height_ratio": {"lo": 0.54, "hi": 0.67, "margin": 0.10},
        },
        "weights": {
            "leg_ratio": 0.18,
            "waist_height_ratio": 0.18,
            "waist_to_shoulder": 0.20,
            "chest_to_waist": 0.12,
            "hip_to_waist": 0.14,
            "hip_to_shoulder": 0.10,
            "thigh_to_calf": 0.08,
        },
        "confidence_weights": {
            "width_ready": 0.34,
            "pose_visibility": 0.34,
            "torso_fill": 0.22,
            "view_factor": 0.10,
        },
        "validity": {
            "min_width_metrics": 3,
        },
    }


def _default_depth3d_scoring() -> Dict[str, Any]:
    return {
        "views": {
            "front": {
                "yaw": {"lo": 0.00, "hi": 0.10, "margin": 0.06},
                "shoulder_width": {"lo": 0.22, "hi": 0.31, "margin": 0.08},
                "hip_width": {"lo": 0.11, "hi": 0.19, "margin": 0.06},
                "confidence_fixed": 0.20,
            },
            "three_quarter": {
                "yaw": {"lo": 0.10, "hi": 0.28, "margin": 0.08},
                "shoulder_width": {"lo": 0.21, "hi": 0.28, "margin": 0.07},
                "hip_width": {"lo": 0.10, "hi": 0.18, "margin": 0.06},
                "confidence_yaw": {"lo": 0.10, "hi": 0.28, "margin": 0.10},
            },
            "profile_like": {
                "yaw": {"lo": 0.24, "hi": 0.42, "margin": 0.10},
                "shoulder_width": {"lo": 0.16, "hi": 0.24, "margin": 0.08},
                "hip_width": {"lo": 0.08, "hi": 0.16, "margin": 0.06},
                "confidence_yaw": {"lo": 0.24, "hi": 0.42, "margin": 0.14},
            },
        },
        "common_ranges": {
            "hip_shoulder_ratio": {"lo": 0.46, "hi": 0.72, "margin": 0.18},
            "spine_angle": {"lo": 0.0, "hi": 10.0, "margin": 6.0},
            "torso_length": {"lo": 0.20, "hi": 0.30, "margin": 0.08},
            "side_profile_center_offset": {"lo": 0.00, "hi": 0.10, "margin": 0.10},
            "side_profile_leg_straightness": {"lo": 166.0, "hi": 180.0, "margin": 10.0},
            "side_profile_torso_compactness": {"lo": 0.52, "hi": 1.08, "margin": 0.34},
            "side_profile_ankle_gap": {"lo": 0.02, "hi": 0.16, "margin": 0.10},
        },
        "weights": {
            "side_profile": {
                "center_offset": 0.34,
                "leg_straightness": 0.40,
                "torso_compactness": 0.18,
                "ankle_gap": 0.08,
            },
            "torso_volume": {
                "shoulder_width": 0.30,
                "hip_width": 0.24,
                "hip_shoulder_ratio": 0.20,
                "spine_angle": 0.14,
                "torso_length": 0.12,
                "side_profile": 0.18,
            },
            "overall": {
                "yaw": 0.24,
                "torso_volume": 0.52,
                "spine_angle": 0.14,
                "torso_length": 0.10,
                "side_profile": 0.16,
            },
            "confidence_three_quarter": {
                "yaw": 0.45,
                "torso_volume": 0.35,
                "spine_angle": 0.20,
            },
            "confidence_profile_like": {
                "yaw": 0.45,
                "torso_volume": 0.35,
                "spine_angle": 0.10,
                "side_profile": 0.10,
            },
        },
    }


def _default_score_fusion() -> Dict[str, Any]:
    return {
        "face_identity": {
            "views": {
                "front": {
                    "weights": {
                        "embedding": 0.42,
                        "geom": 0.22,
                        "hog": 0.14,
                        "lbp": 0.10,
                        "phash": 0.04,
                        "ssim": 0.08,
                    }
                },
                "three_quarter": {
                    "weights": {
                        "embedding": 0.58,
                        "geom": 0.10,
                        "hog": 0.10,
                        "lbp": 0.06,
                        "phash": 0.02,
                        "ssim": 0.14,
                    }
                },
                "profile_like": {
                    "weights": {
                        "embedding": 0.70,
                        "geom": 0.05,
                        "hog": 0.08,
                        "lbp": 0.05,
                        "phash": 0.00,
                        "ssim": 0.12,
                    }
                },
            },
            "confidence": {
                "base": 0.60,
                "coverage": 0.40,
            },
            "topk": {
                "limit": 3,
                "mean": 0.60,
                "median": 0.40,
            },
        },
        "upper_geom": {
            "views": {
                "front": {
                    "weights": {
                        "shoulder_width_norm": 0.45,
                        "torso_len_norm": 0.35,
                        "hip_width_norm": 0.20,
                    },
                    "tilt_weight": 0.10,
                    "spine_weight": 0.04,
                },
                "three_quarter": {
                    "weights": {
                        "shoulder_width_norm": 0.10,
                        "torso_len_norm": 0.50,
                        "hip_width_norm": 0.40,
                    },
                    "tilt_weight": 0.04,
                    "spine_weight": 0.10,
                },
                "profile_like": {
                    "weights": {
                        "torso_len_norm": 0.34,
                        "shoulder_hip_center_offset_norm": 0.24,
                        "torso_compactness": 0.18,
                        "hip_width_norm": 0.14,
                        "hip_shoulder_ratio": 0.10,
                    },
                    "tilt_weight": 0.00,
                    "spine_weight": 0.18,
                },
            }
        },
        "full_geom": {
            "views": {
                "front": {
                    "weights": {
                        "head_body_ratio": 0.58,
                        "leg_ratio": 0.42,
                    }
                },
                "three_quarter": {
                    "weights": {
                        "head_body_ratio": 0.46,
                        "leg_ratio": 0.36,
                        "leg_straightness_mean_deg": 0.10,
                        "ankle_gap_norm": 0.08,
                    }
                },
                "profile_like": {
                    "weights": {
                        "head_body_ratio": 0.22,
                        "leg_ratio": 0.28,
                        "leg_straightness_min_deg": 0.24,
                        "leg_straightness_mean_deg": 0.12,
                        "ankle_gap_norm": 0.08,
                        "foot_length_proxy_norm": 0.06,
                    }
                },
            }
        },
        "framing": {
            "opencv": {
                "subject_height_weight": 0.40,
                "subject_height_low": 0.45,
                "subject_height_high": 0.85,
                "headroom_weight": 0.30,
                "headroom_target": 0.08,
                "headroom_margin": 0.20,
                "feet_weight": 0.30,
            },
            "mediapipe": {
                "feet_weight": 0.40,
                "subject_height_weight": 0.35,
                "subject_height_target": 0.82,
                "subject_height_margin": 0.22,
                "headroom_weight": 0.25,
                "headroom_target": 0.07,
                "headroom_margin": 0.12,
            },
        },
        "upper_anchor": {
            "parts": {
                "pose": 0.35,
                "geom": 0.65,
            },
            "opencv": {
                "framing_scale": 0.60,
                "framing_bias": 0.20,
            },
            "confidence": {
                "mul": 0.90,
                "bias": 0.10,
            },
            "topk": {
                "limit": 3,
                "mean": 0.60,
                "median": 0.40,
            },
        },
        "full_anchor": {
            "views": {
                "front": {
                    "framing": 0.45,
                    "geom": 0.35,
                    "pose": 0.20,
                },
                "three_quarter": {
                    "framing": 0.40,
                    "geom": 0.40,
                    "pose": 0.20,
                },
                "profile_like": {
                    "framing": 0.30,
                    "geom": 0.50,
                    "pose": 0.20,
                },
            },
            "opencv": {
                "framing": 0.80,
                "bbox": 0.20,
                "bbox_low": 0.12,
                "bbox_high": 0.45,
            },
            "confidence": {
                "mul": 0.90,
                "bias": 0.10,
            },
            "topk": {
                "limit": 3,
                "mean": 0.60,
                "median": 0.40,
            },
        },
        "overall": {
            "confidence_floor": 0.25,
            "confidence_scale": 0.75,
        },
    }


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    config_dir: Path
    dir_anchors: Path
    dir_input: Path
    dir_output: Path
    dir_master_truth: Path
    dir_heavy_cache: Path
    dir_calib: Path
    dir_out_pass: Path
    dir_out_warn: Path
    dir_out_fail: Path
    report_file: Path
    thresh_file: Path
    dir_anchor_face: Path
    dir_anchor_upper: Path
    dir_anchor_full: Path
    dir_anchor_tone: Path


@dataclass
class ReviewPolicy:
    active_profile: str = "body_gold_fullbody"
    min_conf_for_strict_fail: float = 0.18
    face_no_signal_conf_th: float = 0.08
    allow_classic_cv_fallback: bool = False
    fatal_on_engine_unavailable: bool = True


@dataclass
class StandardizationSettings:
    enabled: bool = True
    long_side: int = 1792
    upscale_small_input: bool = False


@dataclass
class SkinRiskSettings:
    lighting_warn_th: float = 0.30
    lighting_high_th: float = 0.55
    sample_warn_th: float = 0.32
    sample_high_th: float = 0.55
    face_side_delta_l_warn: float = 16.0
    face_neck_delta_l_warn: float = 18.0
    leg_lr_delta_l_warn: float = 20.0
    face_highlight_l: float = 242.0
    face_highlight_ratio_warn: float = 0.08
    face_highlight_ratio_high: float = 0.16
    edge_margin_ratio_floor: float = 0.018
    low_purity_floor: float = 0.18
    purity_variance_warn: float = 0.18


@dataclass
class SkinScoreWeightPreset:
    chroma: float = 0.62
    luminance: float = 0.24
    knee: float = 0.14
    baseline: float = 0.12


@dataclass
class SkinScoreWeightSettings:
    strict: SkinScoreWeightPreset = field(default_factory=SkinScoreWeightPreset)
    chroma_dominant: SkinScoreWeightPreset = field(
        default_factory=lambda: SkinScoreWeightPreset(chroma=0.70, luminance=0.20, knee=0.10, baseline=0.10)
    )
    high_risk: SkinScoreWeightPreset = field(
        default_factory=lambda: SkinScoreWeightPreset(chroma=0.82, luminance=0.08, knee=0.10, baseline=0.10)
    )


@dataclass
class SkinSplitSettings:
    delta_ab_decay_thigh: float = 11.5
    delta_ab_decay_calf: float = 12.5
    delta_l_decay_thigh: float = 18.0
    delta_l_decay_calf: float = 20.0
    brightness_ratio_low: float = 0.84
    brightness_ratio_high: float = 1.08
    brightness_ratio_margin: float = 0.22
    knee_ratio_low: float = 0.82
    knee_ratio_high: float = 1.08
    knee_ratio_margin: float = 0.22
    severe_delta_ab_thigh: float = 11.0
    severe_delta_ab_calf: float = 12.5
    severe_leg_brightness_ratio: float = 0.78
    severe_luminance_score: float = 0.40


@dataclass
class ConsistencySettings:
    mode: str = "soft_gate"
    constitution_min_conf: float = 0.68
    skin_min_conf: float = 0.60
    depth3d_min_conf: float = 0.58
    constitution_soft_warn_th: float = 0.56
    constitution_strong_warn_th: float = 0.44
    skin_soft_warn_th: float = 0.54
    skin_strong_warn_th: float = 0.42
    depth3d_soft_warn_th: float = 0.54
    depth3d_strong_warn_th: float = 0.42
    skin_risk: SkinRiskSettings = field(default_factory=SkinRiskSettings)
    skin_split: SkinSplitSettings = field(default_factory=SkinSplitSettings)
    skin_score_weights: SkinScoreWeightSettings = field(default_factory=SkinScoreWeightSettings)
    body_constitution_scoring: Dict[str, Any] = field(default_factory=_default_body_constitution_scoring)
    depth3d_scoring: Dict[str, Any] = field(default_factory=_default_depth3d_scoring)
    score_fusion: Dict[str, Any] = field(default_factory=_default_score_fusion)


@dataclass
class QualityThresholds:
    face_luma_dark_warn_l: float = 110.0
    face_lapvar_soft_warn: float = 12.0
    face_hfenergy_soft_warn: float = 1.20
    degrade_flags: set[str] = field(
        default_factory=lambda: {
            "FACE_UNDEREXPOSED_DARK",
            "FACE_TOO_SOFT_POSSIBLE_SMOOTHING",
            "FACE_LOW_MICROTEXTURE",
            "FACE_DARKER_THAN_ANCHOR",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_LOWER_TEXTURE_THAN_ANCHOR",
        }
    )

    def to_json_dict(self) -> Dict[str, float]:
        return {
            "FACE_LUMA_DARK_WARN_L": float(self.face_luma_dark_warn_l),
            "FACE_LAPVAR_SOFT_WARN": float(self.face_lapvar_soft_warn),
            "FACE_HFENERGY_SOFT_WARN": float(self.face_hfenergy_soft_warn),
        }

    def apply_json_dict(self, data: Dict[str, Any]) -> None:
        self.face_luma_dark_warn_l = float(
            data.get("FACE_LUMA_DARK_WARN_L", self.face_luma_dark_warn_l)
        )
        self.face_lapvar_soft_warn = float(
            data.get("FACE_LAPVAR_SOFT_WARN", self.face_lapvar_soft_warn)
        )
        self.face_hfenergy_soft_warn = float(
            data.get("FACE_HFENERGY_SOFT_WARN", self.face_hfenergy_soft_warn)
        )


@dataclass
class EngineState:
    face_mode: str
    pose_mode: str
    face_app: Any = None
    pose_engine: Any = None
    mp_pose: Any = None
    hog_people: Any = None
    face_reason: Optional[str] = None
    pose_reason: Optional[str] = None
    classic_cv_fallback_active: bool = False
    fatal: bool = False
    fatal_reasons: List[str] = field(default_factory=list)
    policy_allow_classic_cv_fallback: bool = False
    policy_fatal_on_engine_unavailable: bool = True


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
    lab_mean: Optional[np.ndarray] = None
    lap_var: float = 0.0
    hf_energy: float = 0.0
    source_path: Optional[str] = None


@dataclass
class PoseFeat:
    ok: bool = False
    mode: str = "opencv"
    lm_xy: Optional[np.ndarray] = None
    lm_vis: Optional[np.ndarray] = None
    lm_world: Optional[np.ndarray] = None
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
    tone_face_feats: List[FaceFeat] = field(default_factory=list)
    upper_pose_feats: List[PoseFeat] = field(default_factory=list)
    full_pose_feats: List[PoseFeat] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    paths: ProjectPaths
    run_mode: str = "qa"
    auto_load_thresholds: bool = False
    review: ReviewPolicy = field(default_factory=ReviewPolicy)
    standardization: StandardizationSettings = field(default_factory=StandardizationSettings)
    consistency: ConsistencySettings = field(default_factory=ConsistencySettings)
    task_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    profile_policy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    external_config_status: Dict[str, bool] = field(default_factory=dict)
    anchor_registry: Dict[str, Any] = field(default_factory=dict)
    layer_quotas: Dict[str, Any] = field(default_factory=dict)
    release_gates: Dict[str, Any] = field(default_factory=dict)
    provider_policy: Dict[str, str] = field(default_factory=dict)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)


@dataclass
class RuntimeContext:
    config: RuntimeConfig
    providers: Any
    engines: EngineState


def _default_task_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "identity_lock": {
            "weights": {"face": 0.70, "upper": 0.20, "full": 0.10},
            "require": {"face": True, "upper": False, "full": False},
            "thresholds": {
                "face_pass": 0.82,
                "face_warn": 0.68,
                "upper_pass": 0.75,
                "upper_warn": 0.60,
                "full_pass": 0.72,
                "full_warn": 0.58,
                "overall_pass": 0.80,
                "overall_warn": 0.65,
            },
        },
        "upper_body_product": {
            "weights": {"face": 0.40, "upper": 0.45, "full": 0.15},
            "require": {"face": True, "upper": True, "full": False},
            "thresholds": {
                "face_pass": 0.78,
                "face_warn": 0.63,
                "upper_pass": 0.78,
                "upper_warn": 0.62,
                "full_pass": 0.65,
                "full_warn": 0.50,
                "overall_pass": 0.78,
                "overall_warn": 0.62,
            },
        },
        "full_body_outfit": {
            "weights": {"face": 0.25, "upper": 0.30, "full": 0.45},
            "require": {"face": False, "upper": True, "full": True},
            "thresholds": {
                "face_pass": 0.75,
                "face_warn": 0.58,
                "upper_pass": 0.75,
                "upper_warn": 0.58,
                "full_pass": 0.78,
                "full_warn": 0.62,
                "overall_pass": 0.78,
                "overall_warn": 0.62,
            },
        },
        "lora_dataset": {
            "weights": {"face": 0.65, "upper": 0.10, "full": 0.25},
            "require": {"face": True, "upper": False, "full": False},
            "thresholds": {
                "face_pass": 0.78,
                "face_warn": 0.62,
                "upper_pass": 0.70,
                "upper_warn": 0.55,
                "full_pass": 0.70,
                "full_warn": 0.55,
                "overall_pass": 0.76,
                "overall_warn": 0.60,
            },
        },
        "body_gold_fullbody": {
            "weights": {"face": 0.45, "upper": 0.15, "full": 0.40},
            "require": {"face": True, "upper": False, "full": True},
            "thresholds": {
                "face_pass": 0.76,
                "face_warn": 0.60,
                "upper_pass": 0.68,
                "upper_warn": 0.53,
                "full_pass": 0.74,
                "full_warn": 0.58,
                "overall_pass": 0.74,
                "overall_warn": 0.59,
            },
        },
        "body_gold_threequarter_review": {
            "weights": {"face": 0.45, "upper": 0.15, "full": 0.40},
            "require": {"face": True, "upper": False, "full": True},
            "thresholds": {
                "face_pass": 0.76,
                "face_warn": 0.60,
                "upper_pass": 0.68,
                "upper_warn": 0.53,
                "full_pass": 0.74,
                "full_warn": 0.58,
                "overall_pass": 0.74,
                "overall_warn": 0.59,
            },
        },
    }


def _default_profile_policy() -> Dict[str, Dict[str, Any]]:
    return {
        "identity_lock": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "face",
            "tone_anchor_pool": "face",
            "soft_quality_hits_to_warn": 1,
            "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
            "skin_lighting_high_caps_pass": False,
            "skin_sample_high_caps_pass": False,
            "allowed_view_buckets": ["front", "three_quarter", "side_90"],
            "soft_review_buckets": [],
            "pass_cap_mode": "none",
            "quota_bucket": "IDENTITY_LOCK",
        },
        "upper_body_product": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "upper_first",
            "tone_anchor_pool": "upper_first",
            "soft_quality_hits_to_warn": 1,
            "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
            "skin_lighting_high_caps_pass": False,
            "skin_sample_high_caps_pass": False,
            "allowed_view_buckets": ["front", "three_quarter", "side_90"],
            "soft_review_buckets": [],
            "pass_cap_mode": "none",
            "quota_bucket": "UPPER_BODY_PRODUCT",
        },
        "full_body_outfit": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "upper_or_full",
            "tone_anchor_pool": "upper_or_full",
            "soft_quality_hits_to_warn": 2,
            "hard_quality_flags": {
                "FACE_UNDEREXPOSED_DARK",
                "FACE_NO_RELIABLE_SIGNAL",
                "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE",
            },
            "skin_lighting_high_caps_pass": False,
            "skin_sample_high_caps_pass": False,
            "allowed_view_buckets": ["front", "three_quarter", "side_90"],
            "soft_review_buckets": [],
            "pass_cap_mode": "none",
            "quota_bucket": "FULL_BODY_OUTFIT",
        },
        "lora_dataset": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "face",
            "tone_anchor_pool": "face",
            "soft_quality_hits_to_warn": 2,
            "hard_quality_flags": {"FACE_UNDEREXPOSED_DARK", "FACE_NO_RELIABLE_SIGNAL"},
            "skin_lighting_high_caps_pass": False,
            "skin_sample_high_caps_pass": False,
            "allowed_view_buckets": ["front", "three_quarter", "side_90"],
            "soft_review_buckets": [],
            "pass_cap_mode": "none",
            "quota_bucket": "LORA_DATASET",
        },
        "body_gold_fullbody": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "upper_or_full",
            "tone_anchor_pool": "upper_or_full",
            "soft_quality_hits_to_warn": 2,
            "hard_quality_flags": {
                "FACE_UNDEREXPOSED_DARK",
                "FACE_NO_RELIABLE_SIGNAL",
                "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE",
            },
            "skin_lighting_high_caps_pass": True,
            "skin_sample_high_caps_pass": True,
            "allowed_view_buckets": ["front", "three_quarter", "side_90"],
            "soft_review_buckets": ["three_quarter"],
            "pass_cap_mode": "body_gold_front_core",
            "quota_bucket": "BODY_GOLD.front_core",
        },
        "body_gold_threequarter_review": {
            "identity_anchor_pool": "face",
            "quality_anchor_pool": "upper_or_full",
            "tone_anchor_pool": "upper_or_full",
            "soft_quality_hits_to_warn": 2,
            "hard_quality_flags": {
                "FACE_UNDEREXPOSED_DARK",
                "FACE_NO_RELIABLE_SIGNAL",
                "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE",
            },
            "skin_lighting_high_caps_pass": True,
            "skin_sample_high_caps_pass": True,
            "allowed_view_buckets": ["three_quarter"],
            "soft_review_buckets": [],
            "pass_cap_mode": "always_warn",
            "quota_bucket": "BODY_GOLD.three_quarter_review",
            "preferred_heavy_evidence": "segformer_body_truth_fusion",
        },
    }


def _default_external_config_status() -> Dict[str, bool]:
    return {
        "anchor_registry": False,
        "task_profiles": False,
        "consistency_thresholds": False,
        "layer_quotas": False,
        "release_gates": False,
    }


def _default_anchor_registry() -> Dict[str, Any]:
    return {
        "anchors": {},
        "rules": {},
    }


def _default_layer_quotas() -> Dict[str, Any]:
    return {
        "training_layers": {},
        "frozen_total_pass_target": 0,
    }


def _default_release_gates() -> Dict[str, Any]:
    return {
        "schema_version": "qa_release_gates_v1",
        "release_gates": {
            "BODY_GOLD.front_core": {
                "release_state": "primary",
                "machine_status_ceiling": "PASS",
                "training_admission_allowed": True,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": True,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front"],
                "notes": "Primary front-core lane. Training admission still requires explicit human seal.",
            },
            "BODY_GOLD.three_quarter_review": {
                "release_state": "review",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": True,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["three_quarter"],
                "notes": "Three-quarter lane remains review-only until canonical truth and frozen benchmark coverage are stronger.",
            },
            "BODY_GOLD.side90_shadow": {
                "release_state": "shadow",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": True,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["side"],
                "notes": "Side-90 lane is shadow-only and should not be treated as final image-set membership or training admission.",
            },
            "BODY_GOLD.back180_shadow": {
                "release_state": "shadow",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": True,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["back"],
                "notes": "Back-180 lane is shadow-only and should not be treated as final image-set membership or training admission.",
            },
            "BRIDGE.simple_outfit": {
                "release_state": "primary",
                "machine_status_ceiling": "PASS",
                "training_admission_allowed": True,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front", "three_quarter"],
                "notes": "Bridge simple-outfit can be sealed manually when identity continuity stays stable.",
            },
            "BRIDGE.review": {
                "release_state": "review",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front", "three_quarter", "side", "back"],
                "notes": "Generic full-body outfit review is advisory and not a sealed training lane.",
            },
            "UPPER_BODY_PRODUCT": {
                "release_state": "review",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front", "three_quarter"],
                "notes": "Upper-body product profile is not a final XiaoNa training admission lane.",
            },
            "IDENTITY_LOCK": {
                "release_state": "review",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front", "three_quarter", "side"],
                "notes": "Identity-lock profile is for review and filtering, not direct training admission.",
            },
            "IDENTITY_FILTER_ONLY": {
                "release_state": "filter_only",
                "machine_status_ceiling": "WARN",
                "training_admission_allowed": False,
                "manual_training_admission_required": True,
                "optuna_fit_allowed": False,
                "requires_frozen_benchmark": False,
                "requires_curated_winner_bank": False,
                "required_lane_families": ["front", "three_quarter", "side"],
                "notes": "Dataset filter profile is not a sealed release lane.",
            },
        },
    }


def _default_provider_policy() -> Dict[str, str]:
    return {
        "subject_mask": "human_parsing",
        "skin_region": "human_parsing",
        "heavy_evidence": "segformer_body_fusion",
        "view_classifier": "view_classifier_lite",
        "face_canonical": "face_pose_canonical_3ddfa",
        "anchor_source": "registry_then_directory_fallback",
    }


def create_project_paths(base_dir: Optional[Path] = None) -> ProjectPaths:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    dir_anchors = root / "anchors"
    dir_output = root / "outputs"
    return ProjectPaths(
        base_dir=root,
        config_dir=root / "configs",
        dir_anchors=dir_anchors,
        dir_input=root / "input",
        dir_output=dir_output,
        dir_master_truth=dir_output / "master_truth",
        dir_heavy_cache=dir_output / "heavy_evidence_cache",
        dir_calib=root / "calib_pass",
        dir_out_pass=dir_output / "pass",
        dir_out_warn=dir_output / "warn",
        dir_out_fail=dir_output / "fail",
        report_file=dir_output / "qa_report.json",
        thresh_file=dir_output / "quality_thresholds.json",
        dir_anchor_face=dir_anchors / "face",
        dir_anchor_upper=dir_anchors / "upper",
        dir_anchor_full=dir_anchors / "full",
        dir_anchor_tone=dir_anchors / "tone",
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    for target in [
        paths.dir_output,
        paths.dir_master_truth,
        paths.dir_heavy_cache,
        paths.dir_out_pass,
        paths.dir_out_warn,
        paths.dir_out_fail,
    ]:
        target.mkdir(parents=True, exist_ok=True)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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
    except Exception as exc:
        print(f"[警告] 配置文件读取失败: {path} | {exc}")
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
            parent.append(_yaml_scalar_from_text(content[2:].strip()))
            continue

        if ":" not in content:
            raise ValueError(f"YAML 解析失败，非法映射行: {path} :: {content}")

        key, _, value_text = content.partition(":")
        key = key.strip()
        value_text = value_text.strip()

        if not isinstance(parent, dict):
            raise ValueError(f"YAML 解析失败，映射项缺少父字典: {path} :: {content}")

        if value_text == "":
            child: Any = [] if next_container_type(idx, indent) == "list" else {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _yaml_scalar_from_text(value_text)

    return root


def _load_yaml_config(path: Path) -> Optional[Any]:
    if not path.exists():
        return None

    if pyyaml is not None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[警告] 配置文件读取失败: {path} | {exc}")
            return None
        try:
            return pyyaml.safe_load(raw_text)
        except Exception as exc:
            print(f"[警告] PyYAML 解析失败，回退到简单 YAML 解析器: {path} | {exc}")
            return _load_simple_yaml(path)

    print(f"[警告] PyYAML 不可用，回退到简单 YAML 解析器: {path}")
    return _load_simple_yaml(path)


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
        for key in ["allowed_view_buckets", "soft_review_buckets"]:
            values = node.get(key, [])
            if isinstance(values, list):
                node[key] = [str(x) for x in values]
            elif values is None:
                node[key] = []
            else:
                node[key] = [str(values)]
        out[str(profile_name)] = node
    return out


def get_preferred_heavy_evidence_for_profile(
    config: RuntimeConfig,
    profile_name: Optional[str],
) -> Optional[str]:
    profile_key = str(profile_name or "").strip()
    if not profile_key:
        return None
    policy = config.profile_policy.get(profile_key, {})
    if not isinstance(policy, dict):
        return None
    provider_name = str(policy.get("preferred_heavy_evidence") or "").strip()
    return provider_name or None


def _normalize_anchor_registry(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"anchors": {}, "rules": {}}

    anchors_node = data.get("anchors", {})
    rules_node = data.get("rules", {})
    out_anchors: Dict[str, Dict[str, Any]] = {}

    if isinstance(anchors_node, dict):
        for anchor_id, node in anchors_node.items():
            if not isinstance(node, dict):
                continue
            owns = node.get("owns", [])
            supports = node.get("supports", [])
            out_anchors[str(anchor_id)] = {
                "role": str(node.get("role", "")),
                "anchor_tier": str(node.get("anchor_tier", "")).strip().lower(),
                "priority": _coerce_float(node.get("priority", 0), 0.0),
                "required_default": bool(node.get("required_default", False)),
                "path": str(node.get("path", "")),
                "owns": [str(x) for x in owns] if isinstance(owns, list) else [],
                "view_bucket": str(node.get("view_bucket", "")),
                "view_side": str(node.get("view_side", "unknown")),
                "body_plane": str(node.get("body_plane", "")),
                "supports": [str(x) for x in supports] if isinstance(supports, list) else [],
            }

    out_rules = copy.deepcopy(rules_node) if isinstance(rules_node, dict) else {}
    return {"anchors": out_anchors, "rules": out_rules}


def _normalize_layer_quotas(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _default_layer_quotas()

    training_layers = data.get("training_layers", {})
    out_layers: Dict[str, Dict[str, Any]] = {}
    if isinstance(training_layers, dict):
        for layer_name, node in training_layers.items():
            if not isinstance(node, dict):
                continue
            child = copy.deepcopy(node)
            if child.get("quota") is not None:
                child["quota"] = _coerce_int(child["quota"], 0)
            if child.get("priority") is not None:
                child["priority"] = _coerce_int(child["priority"], 0)
            sub_buckets = child.get("sub_buckets", None)
            if isinstance(sub_buckets, dict):
                normalized_buckets: Dict[str, Dict[str, Any]] = {}
                for bucket_name, bucket_node in sub_buckets.items():
                    if not isinstance(bucket_node, dict):
                        continue
                    bucket_child = copy.deepcopy(bucket_node)
                    if bucket_child.get("quota") is not None:
                        bucket_child["quota"] = _coerce_int(bucket_child["quota"], 0)
                    if bucket_child.get("priority") is not None:
                        bucket_child["priority"] = _coerce_int(bucket_child["priority"], 0)
                    normalized_buckets[str(bucket_name)] = bucket_child
                child["sub_buckets"] = normalized_buckets
            out_layers[str(layer_name)] = child

    return {
        "training_layers": out_layers,
        "frozen_total_pass_target": _coerce_int(data.get("frozen_total_pass_target", 0), 0),
    }


def _normalize_release_gates(data: Any) -> Dict[str, Any]:
    base = _default_release_gates()
    if not isinstance(data, dict):
        return base

    gates_node = data.get("release_gates", data)
    if not isinstance(gates_node, dict):
        return base

    out = copy.deepcopy(base)
    raw_schema_version = str(data.get("schema_version", "")).strip()
    if raw_schema_version:
        out["schema_version"] = raw_schema_version

    normalized_gates: Dict[str, Dict[str, Any]] = copy.deepcopy(out.get("release_gates", {}))
    valid_ceiling = {"FAIL", "WARN", "PASS"}
    for bucket_name, raw_node in gates_node.items():
        if not isinstance(raw_node, dict):
            continue
        bucket_key = str(bucket_name).strip()
        if not bucket_key:
            continue
        node = copy.deepcopy(normalized_gates.get(bucket_key, {}))
        node["release_state"] = str(raw_node.get("release_state", node.get("release_state", "review"))).strip() or "review"
        ceiling = str(raw_node.get("machine_status_ceiling", node.get("machine_status_ceiling", "WARN"))).strip().upper()
        node["machine_status_ceiling"] = ceiling if ceiling in valid_ceiling else str(node.get("machine_status_ceiling", "WARN"))
        node["training_admission_allowed"] = bool(
            raw_node.get("training_admission_allowed", node.get("training_admission_allowed", False))
        )
        node["manual_training_admission_required"] = bool(
            raw_node.get(
                "manual_training_admission_required",
                node.get("manual_training_admission_required", True),
            )
        )
        node["optuna_fit_allowed"] = bool(raw_node.get("optuna_fit_allowed", node.get("optuna_fit_allowed", False)))
        node["requires_frozen_benchmark"] = bool(
            raw_node.get("requires_frozen_benchmark", node.get("requires_frozen_benchmark", False))
        )
        node["requires_curated_winner_bank"] = bool(
            raw_node.get("requires_curated_winner_bank", node.get("requires_curated_winner_bank", False))
        )
        required_lane_families = raw_node.get("required_lane_families", node.get("required_lane_families", []))
        if isinstance(required_lane_families, list):
            node["required_lane_families"] = [str(item).strip() for item in required_lane_families if str(item).strip()]
        elif required_lane_families is None:
            node["required_lane_families"] = []
        else:
            text = str(required_lane_families).strip()
            node["required_lane_families"] = [text] if text else []
        node["notes"] = str(raw_node.get("notes", node.get("notes", ""))).strip()
        normalized_gates[bucket_key] = node
    out["release_gates"] = normalized_gates
    return out


def _resolve_registry_path(config: RuntimeConfig, raw_path: str) -> Path:
    expanded = str(raw_path).strip()
    expanded = expanded.replace("${PROJECT_ROOT}", str(config.paths.base_dir))
    expanded = expanded.replace("${CONFIG_DIR}", str(config.paths.config_dir))
    path = Path(expanded)
    if not path.is_absolute():
        path = (config.paths.base_dir / path).resolve()
    return path


def _list_image_files_in_dir(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def _list_image_files_recursive(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def apply_external_project_configs(config: RuntimeConfig) -> None:
    task_profiles_path = config.paths.config_dir / "task_profiles.yaml"
    task_data = _load_yaml_config(task_profiles_path)
    if isinstance(task_data, dict):
        task_profiles = task_data.get("task_profiles", None)
        if isinstance(task_profiles, dict):
            config.task_profiles = _deep_merge_dict(config.task_profiles, task_profiles)

        review_policy = task_data.get("review_policy", None)
        if isinstance(review_policy, dict):
            if isinstance(review_policy.get("main_priority"), str):
                config.review.active_profile = str(review_policy["main_priority"])
            if review_policy.get("strict_fail_min_conf") is not None:
                config.review.min_conf_for_strict_fail = float(
                    review_policy["strict_fail_min_conf"]
                )
            if review_policy.get("face_no_signal_conf") is not None:
                config.review.face_no_signal_conf_th = float(
                    review_policy["face_no_signal_conf"]
                )
            if review_policy.get("allow_classic_cv_fallback") is not None:
                config.review.allow_classic_cv_fallback = bool(
                    review_policy["allow_classic_cv_fallback"]
                )
            if review_policy.get("fatal_on_engine_unavailable") is not None:
                config.review.fatal_on_engine_unavailable = bool(
                    review_policy["fatal_on_engine_unavailable"]
                )
            if isinstance(review_policy.get("consistency_mode"), str):
                config.consistency.mode = str(review_policy["consistency_mode"])

        profile_policy = task_data.get("profile_policy", None)
        if isinstance(profile_policy, dict):
            config.profile_policy = _deep_merge_dict(
                config.profile_policy,
                _normalize_profile_policy_map(profile_policy),
            )

        config.external_config_status["task_profiles"] = True

    anchor_registry_path = config.paths.config_dir / "anchor_registry.yaml"
    anchor_data = _load_yaml_config(anchor_registry_path)
    if anchor_data is not None:
        config.anchor_registry = _normalize_anchor_registry(anchor_data)
        config.external_config_status["anchor_registry"] = True

    layer_quota_path = config.paths.config_dir / "layer_quotas.yaml"
    quota_data = _load_yaml_config(layer_quota_path)
    if quota_data is not None:
        config.layer_quotas = _normalize_layer_quotas(quota_data)
        config.external_config_status["layer_quotas"] = True

    release_gate_path = config.paths.config_dir / "release_gates.yaml"
    release_gate_data = _load_yaml_config(release_gate_path)
    if release_gate_data is not None:
        config.release_gates = _normalize_release_gates(release_gate_data)
        config.external_config_status["release_gates"] = True

    consistency_path = config.paths.config_dir / "consistency_thresholds.yaml"
    consistency_data = _load_yaml_config(consistency_path)
    if isinstance(consistency_data, dict):
        consistency_node = consistency_data.get("consistency", None)
        if isinstance(consistency_node, dict):
            if isinstance(consistency_node.get("mode"), str):
                config.consistency.mode = str(consistency_node["mode"])

            min_conf_node = consistency_node.get("min_confidence", None)
            if isinstance(min_conf_node, dict):
                if min_conf_node.get("constitution") is not None:
                    config.consistency.constitution_min_conf = float(min_conf_node["constitution"])
                if min_conf_node.get("skin") is not None:
                    config.consistency.skin_min_conf = float(min_conf_node["skin"])
                if min_conf_node.get("depth3d") is not None:
                    config.consistency.depth3d_min_conf = float(min_conf_node["depth3d"])

            warn_node = consistency_node.get("warn_threshold", None)
            if isinstance(warn_node, dict):
                if warn_node.get("constitution_soft") is not None:
                    config.consistency.constitution_soft_warn_th = float(
                        warn_node["constitution_soft"]
                    )
                if warn_node.get("constitution_strong") is not None:
                    config.consistency.constitution_strong_warn_th = float(
                        warn_node["constitution_strong"]
                    )
                if warn_node.get("skin_soft") is not None:
                    config.consistency.skin_soft_warn_th = float(warn_node["skin_soft"])
                if warn_node.get("skin_strong") is not None:
                    config.consistency.skin_strong_warn_th = float(warn_node["skin_strong"])
                if warn_node.get("depth3d_soft") is not None:
                    config.consistency.depth3d_soft_warn_th = float(warn_node["depth3d_soft"])
                if warn_node.get("depth3d_strong") is not None:
                    config.consistency.depth3d_strong_warn_th = float(
                        warn_node["depth3d_strong"]
                    )

            skin_risk_node = consistency_node.get("skin_risk", None)
            if isinstance(skin_risk_node, dict):
                if skin_risk_node.get("lighting_warn") is not None:
                    config.consistency.skin_risk.lighting_warn_th = float(
                        skin_risk_node["lighting_warn"]
                    )
                if skin_risk_node.get("lighting_high") is not None:
                    config.consistency.skin_risk.lighting_high_th = float(
                        skin_risk_node["lighting_high"]
                    )
                if skin_risk_node.get("sample_warn") is not None:
                    config.consistency.skin_risk.sample_warn_th = float(
                        skin_risk_node["sample_warn"]
                    )
                if skin_risk_node.get("sample_high") is not None:
                    config.consistency.skin_risk.sample_high_th = float(
                        skin_risk_node["sample_high"]
                    )
                if skin_risk_node.get("face_side_delta_l_warn") is not None:
                    config.consistency.skin_risk.face_side_delta_l_warn = float(
                        skin_risk_node["face_side_delta_l_warn"]
                    )
                if skin_risk_node.get("face_neck_delta_l_warn") is not None:
                    config.consistency.skin_risk.face_neck_delta_l_warn = float(
                        skin_risk_node["face_neck_delta_l_warn"]
                    )
                if skin_risk_node.get("leg_lr_delta_l_warn") is not None:
                    config.consistency.skin_risk.leg_lr_delta_l_warn = float(
                        skin_risk_node["leg_lr_delta_l_warn"]
                    )
                if skin_risk_node.get("face_highlight_l") is not None:
                    config.consistency.skin_risk.face_highlight_l = float(
                        skin_risk_node["face_highlight_l"]
                    )
                if skin_risk_node.get("face_highlight_ratio_warn") is not None:
                    config.consistency.skin_risk.face_highlight_ratio_warn = float(
                        skin_risk_node["face_highlight_ratio_warn"]
                    )
                if skin_risk_node.get("face_highlight_ratio_high") is not None:
                    config.consistency.skin_risk.face_highlight_ratio_high = float(
                        skin_risk_node["face_highlight_ratio_high"]
                    )
                if skin_risk_node.get("edge_margin_ratio_floor") is not None:
                    config.consistency.skin_risk.edge_margin_ratio_floor = float(
                        skin_risk_node["edge_margin_ratio_floor"]
                    )
                if skin_risk_node.get("low_purity_floor") is not None:
                    config.consistency.skin_risk.low_purity_floor = float(
                        skin_risk_node["low_purity_floor"]
                    )
                if skin_risk_node.get("purity_variance_warn") is not None:
                    config.consistency.skin_risk.purity_variance_warn = float(
                        skin_risk_node["purity_variance_warn"]
                    )

            skin_split_node = consistency_node.get("skin_split", None)
            if isinstance(skin_split_node, dict):
                if skin_split_node.get("delta_ab_decay_thigh") is not None:
                    config.consistency.skin_split.delta_ab_decay_thigh = float(
                        skin_split_node["delta_ab_decay_thigh"]
                    )
                if skin_split_node.get("delta_ab_decay_calf") is not None:
                    config.consistency.skin_split.delta_ab_decay_calf = float(
                        skin_split_node["delta_ab_decay_calf"]
                    )
                if skin_split_node.get("delta_l_decay_thigh") is not None:
                    config.consistency.skin_split.delta_l_decay_thigh = float(
                        skin_split_node["delta_l_decay_thigh"]
                    )
                if skin_split_node.get("delta_l_decay_calf") is not None:
                    config.consistency.skin_split.delta_l_decay_calf = float(
                        skin_split_node["delta_l_decay_calf"]
                    )
                if skin_split_node.get("brightness_ratio_low") is not None:
                    config.consistency.skin_split.brightness_ratio_low = float(
                        skin_split_node["brightness_ratio_low"]
                    )
                if skin_split_node.get("brightness_ratio_high") is not None:
                    config.consistency.skin_split.brightness_ratio_high = float(
                        skin_split_node["brightness_ratio_high"]
                    )
                if skin_split_node.get("brightness_ratio_margin") is not None:
                    config.consistency.skin_split.brightness_ratio_margin = float(
                        skin_split_node["brightness_ratio_margin"]
                    )
                if skin_split_node.get("knee_ratio_low") is not None:
                    config.consistency.skin_split.knee_ratio_low = float(
                        skin_split_node["knee_ratio_low"]
                    )
                if skin_split_node.get("knee_ratio_high") is not None:
                    config.consistency.skin_split.knee_ratio_high = float(
                        skin_split_node["knee_ratio_high"]
                    )
                if skin_split_node.get("knee_ratio_margin") is not None:
                    config.consistency.skin_split.knee_ratio_margin = float(
                        skin_split_node["knee_ratio_margin"]
                    )
                if skin_split_node.get("severe_delta_ab_thigh") is not None:
                    config.consistency.skin_split.severe_delta_ab_thigh = float(
                        skin_split_node["severe_delta_ab_thigh"]
                    )
                if skin_split_node.get("severe_delta_ab_calf") is not None:
                    config.consistency.skin_split.severe_delta_ab_calf = float(
                        skin_split_node["severe_delta_ab_calf"]
                    )
                if skin_split_node.get("severe_leg_brightness_ratio") is not None:
                    config.consistency.skin_split.severe_leg_brightness_ratio = float(
                        skin_split_node["severe_leg_brightness_ratio"]
                    )
                if skin_split_node.get("severe_luminance_score") is not None:
                    config.consistency.skin_split.severe_luminance_score = float(
                        skin_split_node["severe_luminance_score"]
                    )

            skin_weights_node = consistency_node.get("skin_score_weights", None)
            if isinstance(skin_weights_node, dict):
                for key, target in [
                    ("strict", config.consistency.skin_score_weights.strict),
                    ("chroma_dominant", config.consistency.skin_score_weights.chroma_dominant),
                    ("high_risk", config.consistency.skin_score_weights.high_risk),
                ]:
                    node = skin_weights_node.get(key, None)
                    if not isinstance(node, dict):
                        continue
                    if node.get("chroma") is not None:
                        target.chroma = float(node["chroma"])
                    if node.get("luminance") is not None:
                        target.luminance = float(node["luminance"])
                    if node.get("knee") is not None:
                        target.knee = float(node["knee"])
                    if node.get("baseline") is not None:
                        target.baseline = float(node["baseline"])

            body_constitution_node = consistency_node.get("body_constitution_scoring", None)
            if isinstance(body_constitution_node, dict):
                config.consistency.body_constitution_scoring = _deep_merge_dict(
                    config.consistency.body_constitution_scoring,
                    body_constitution_node,
                )

            depth3d_node = consistency_node.get("depth3d_scoring", None)
            if isinstance(depth3d_node, dict):
                config.consistency.depth3d_scoring = _deep_merge_dict(
                    config.consistency.depth3d_scoring,
                    depth3d_node,
                )

            score_fusion_node = consistency_node.get("score_fusion", None)
            if isinstance(score_fusion_node, dict):
                config.consistency.score_fusion = _deep_merge_dict(
                    config.consistency.score_fusion,
                    score_fusion_node,
                )

        algorithm_policy = consistency_data.get("algorithm_policy", None)
        if isinstance(algorithm_policy, dict):
            provider_defaults = algorithm_policy.get("provider_defaults", None)
            if isinstance(provider_defaults, dict):
                subject_mask = provider_defaults.get(
                    "subject_mask", config.provider_policy["subject_mask"]
                )
                skin_region = provider_defaults.get(
                    "skin_region", config.provider_policy["skin_region"]
                )
                heavy_evidence = provider_defaults.get(
                    "heavy_evidence", config.provider_policy["heavy_evidence"]
                )
                view_classifier = provider_defaults.get(
                    "view_classifier", config.provider_policy["view_classifier"]
                )
                face_canonical = provider_defaults.get(
                    "face_canonical", config.provider_policy["face_canonical"]
                )
                anchor_source = provider_defaults.get(
                    "anchor_source", config.provider_policy["anchor_source"]
                )
                config.provider_policy["subject_mask"] = str(subject_mask)
                config.provider_policy["skin_region"] = str(skin_region)
                config.provider_policy["heavy_evidence"] = str(heavy_evidence)
                config.provider_policy["view_classifier"] = str(view_classifier)
                config.provider_policy["face_canonical"] = str(face_canonical)
                config.provider_policy["anchor_source"] = str(anchor_source)

        config.external_config_status["consistency_thresholds"] = True


def create_runtime_config(base_dir: Optional[Path] = None) -> RuntimeConfig:
    paths = create_project_paths(base_dir)
    ensure_output_dirs(paths)
    config = RuntimeConfig(
        paths=paths,
        task_profiles=_default_task_profiles(),
        profile_policy=_default_profile_policy(),
        external_config_status=_default_external_config_status(),
        anchor_registry=_default_anchor_registry(),
        layer_quotas=_default_layer_quotas(),
        release_gates=_default_release_gates(),
        provider_policy=_default_provider_policy(),
    )
    apply_external_project_configs(config)
    return config


def _anchor_paths_from_registry(config: RuntimeConfig) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {
        "face_paths": [],
        "upper_paths": [],
        "full_paths": [],
        "tone_paths": [],
    }

    anchors_node = config.anchor_registry.get("anchors", {})
    if not isinstance(anchors_node, dict):
        return grouped

    missing_required: List[str] = []
    for anchor_id, node in anchors_node.items():
        if not isinstance(node, dict):
            continue
        role = str(node.get("role", "")).upper()
        path_str = str(node.get("path", "")).strip()
        if not path_str:
            continue

        path = _resolve_registry_path(config, path_str)
        target_key: Optional[str] = None

        if role == "FACE_MASTER":
            target_key = "face_paths"
        elif role == "UPPER_SUPPORT":
            target_key = "upper_paths"
        elif role == "FULL_BODY_MASTER":
            target_key = "full_paths"
        elif role in {"TONE_BASELINE", "TONE_MASTER"}:
            target_key = "tone_paths"
        else:
            normalized = str(path).replace("\\", "/").lower()
            if "/anchors/face/" in normalized:
                target_key = "face_paths"
            elif "/anchors/upper/" in normalized:
                target_key = "upper_paths"
            elif "/anchors/full/" in normalized:
                target_key = "full_paths"
            elif "/anchors/tone/" in normalized:
                target_key = "tone_paths"

        if target_key is None:
            continue

        if path.exists():
            grouped[target_key].append(str(path))
        else:
            print(f"[警告] Anchor registry 路径不存在，已跳过: {anchor_id} -> {path}")

    for anchor_id, node in anchors_node.items():
        if not isinstance(node, dict):
            continue
        if not bool(node.get("required_default", False)):
            continue
        path_str = str(node.get("path", "")).strip()
        if not path_str:
            continue
        path = _resolve_registry_path(config, path_str)
        if not path.exists():
            missing_required.append(f"{anchor_id} -> {path}")

    if missing_required:
        raise FileNotFoundError("Required anchors missing from registry: " + "; ".join(sorted(set(missing_required))))
    return grouped


def _default_anchor_paths_from_dirs(config: RuntimeConfig) -> Dict[str, List[str]]:
    return {
        "face_paths": [str(path) for path in _list_image_files_recursive(config.paths.dir_anchor_face)],
        "upper_paths": [str(path) for path in _list_image_files_in_dir(config.paths.dir_anchor_upper)],
        "full_paths": [str(path) for path in _list_image_files_in_dir(config.paths.dir_anchor_full)],
        "tone_paths": [str(path) for path in _list_image_files_recursive(config.paths.dir_anchor_tone)],
    }


def resolve_anchor_paths(config: RuntimeConfig) -> Dict[str, List[str]]:
    default_paths = _default_anchor_paths_from_dirs(config)
    source_mode = str(config.provider_policy.get("anchor_source", "registry_then_directory_fallback"))
    if source_mode == "directory_only":
        return default_paths

    registry_paths = _anchor_paths_from_registry(config)
    if source_mode in {"registry_only", "registry_required"}:
        return registry_paths
    if source_mode != "registry_then_directory_fallback":
        raise ValueError(f"Unsupported anchor_source mode: {source_mode}")

    out: Dict[str, List[str]] = {}
    for key in ["face_paths", "upper_paths", "full_paths", "tone_paths"]:
        vals = registry_paths.get(key, [])
        out[key] = vals if len(vals) > 0 else default_paths.get(key, [])
    return out


def anchor_registry_summary(config: RuntimeConfig) -> Dict[str, int]:
    resolved = resolve_anchor_paths(config)
    return {
        "registered_anchors": len(config.anchor_registry.get("anchors", {})),
        "face_paths": len(resolved.get("face_paths", [])),
        "upper_paths": len(resolved.get("upper_paths", [])),
        "full_paths": len(resolved.get("full_paths", [])),
        "tone_paths": len(resolved.get("tone_paths", [])),
    }


def anchor_registry_snapshot(config: RuntimeConfig) -> Dict[str, Any]:
    anchors_node = config.anchor_registry.get("anchors", {})
    rules_node = config.anchor_registry.get("rules", {})
    snapshot: Dict[str, Any] = {
        "anchor_source": str(config.provider_policy.get("anchor_source", "registry_then_directory_fallback")),
        "rules": copy.deepcopy(rules_node) if isinstance(rules_node, dict) else {},
        "entries": {},
    }
    if not isinstance(anchors_node, dict):
        return snapshot

    for anchor_id, node in anchors_node.items():
        if not isinstance(node, dict):
            continue
        raw_path = str(node.get("path", "")).strip()
        resolved_path = _resolve_registry_path(config, raw_path) if raw_path else None
        snapshot["entries"][str(anchor_id)] = {
            "role": str(node.get("role", "")),
            "anchor_tier": str(node.get("anchor_tier", "")).strip().lower(),
            "required_default": bool(node.get("required_default", False)),
            "view_bucket": str(node.get("view_bucket", "")),
            "view_side": str(node.get("view_side", "unknown")),
            "body_plane": str(node.get("body_plane", "")),
            "owns": list(node.get("owns", [])) if isinstance(node.get("owns", []), list) else [],
            "supports": list(node.get("supports", [])) if isinstance(node.get("supports", []), list) else [],
            "path": raw_path,
            "resolved_path": str(resolved_path) if resolved_path is not None else "",
            "exists": bool(resolved_path and resolved_path.exists()),
        }
    if isinstance(snapshot.get("rules"), dict):
        truth_map = {
            "face_truth_anchor": "face_identity",
            "body_truth_anchor": "body_master",
            "upper_support_anchor": "upper_support",
        }
        truth_snapshot: Dict[str, Any] = {}
        for rule_key, label in truth_map.items():
            anchor_id = str(snapshot["rules"].get(rule_key, "")).strip()
            anchor_node = snapshot["entries"].get(anchor_id, {}) if anchor_id else {}
            truth_snapshot[label] = {
                "anchor_id": anchor_id,
                "resolved_path": str(anchor_node.get("resolved_path", "")),
                "exists": bool(anchor_node.get("exists", False)),
                "role": str(anchor_node.get("role", "")),
            }
        snapshot["truth_anchors"] = truth_snapshot
    return snapshot


def save_thresholds_to_file(thresholds: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")


def load_thresholds_from_file(config: RuntimeConfig, path: Path) -> None:
    if not path.exists():
        print(f"[提示] 阈值文件不存在，继续使用默认阈值: {path}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    config.quality_thresholds.apply_json_dict(data)

    print("\n[已加载自动校准阈值]")
    print(f"  FACE_LUMA_DARK_WARN_L = {config.quality_thresholds.face_luma_dark_warn_l:.4f}")
    print(f"  FACE_LAPVAR_SOFT_WARN = {config.quality_thresholds.face_lapvar_soft_warn:.4f}")
    print(f"  FACE_HFENERGY_SOFT_WARN = {config.quality_thresholds.face_hfenergy_soft_warn:.4f}")
