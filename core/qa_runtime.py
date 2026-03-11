from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    config_dir: Path
    dir_anchors: Path
    dir_input: Path
    dir_output: Path
    dir_calib: Path
    dir_out_pass: Path
    dir_out_warn: Path
    dir_out_fail: Path
    report_file: Path
    thresh_file: Path
    dir_anchor_face: Path
    dir_anchor_upper: Path
    dir_anchor_full: Path


@dataclass
class ReviewPolicy:
    active_profile: str = "body_gold_fullbody"
    min_conf_for_strict_fail: float = 0.18
    face_no_signal_conf_th: float = 0.08


@dataclass
class StandardizationSettings:
    enabled: bool = True
    long_side: int = 1792
    upscale_small_input: bool = False


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
    }


def _default_profile_policy() -> Dict[str, Dict[str, Any]]:
    return {
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


def _default_external_config_status() -> Dict[str, bool]:
    return {
        "anchor_registry": False,
        "task_profiles": False,
        "consistency_thresholds": False,
    }


def _default_anchor_registry() -> Dict[str, Any]:
    return {
        "anchors": {},
        "rules": {},
    }


def _default_provider_policy() -> Dict[str, str]:
    return {
        "subject_mask": "human_parsing",
        "skin_region": "human_parsing",
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
        dir_calib=root / "calib_pass",
        dir_out_pass=dir_output / "pass",
        dir_out_warn=dir_output / "warn",
        dir_out_fail=dir_output / "fail",
        report_file=dir_output / "qa_report.json",
        thresh_file=dir_output / "quality_thresholds.json",
        dir_anchor_face=dir_anchors / "face",
        dir_anchor_upper=dir_anchors / "upper",
        dir_anchor_full=dir_anchors / "full",
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    for target in [paths.dir_out_pass, paths.dir_out_warn, paths.dir_out_fail]:
        target.mkdir(parents=True, exist_ok=True)


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
            out_anchors[str(anchor_id)] = {
                "role": str(node.get("role", "")),
                "priority": _coerce_float(node.get("priority", 0), 0.0),
                "required_default": bool(node.get("required_default", False)),
                "path": str(node.get("path", "")),
                "owns": [str(x) for x in owns] if isinstance(owns, list) else [],
            }

    out_rules = copy.deepcopy(rules_node) if isinstance(rules_node, dict) else {}
    return {"anchors": out_anchors, "rules": out_rules}


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
    task_data = _load_simple_yaml(task_profiles_path)
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
    anchor_data = _load_simple_yaml(anchor_registry_path)
    if anchor_data is not None:
        config.anchor_registry = _normalize_anchor_registry(anchor_data)
        config.external_config_status["anchor_registry"] = True

    consistency_path = config.paths.config_dir / "consistency_thresholds.yaml"
    consistency_data = _load_simple_yaml(consistency_path)
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
                anchor_source = provider_defaults.get(
                    "anchor_source", config.provider_policy["anchor_source"]
                )
                config.provider_policy["subject_mask"] = str(subject_mask)
                config.provider_policy["skin_region"] = str(skin_region)
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
        provider_policy=_default_provider_policy(),
    )
    apply_external_project_configs(config)
    return config


def _anchor_paths_from_registry(config: RuntimeConfig) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {
        "face_paths": [],
        "upper_paths": [],
        "full_paths": [],
    }

    anchors_node = config.anchor_registry.get("anchors", {})
    if not isinstance(anchors_node, dict):
        return grouped

    for anchor_id, node in anchors_node.items():
        if not isinstance(node, dict):
            continue
        role = str(node.get("role", "")).upper()
        path_str = str(node.get("path", "")).strip()
        if not path_str:
            continue

        path = Path(path_str)
        target_key: Optional[str] = None

        if role == "FACE_MASTER":
            target_key = "face_paths"
        elif role == "UPPER_SUPPORT":
            target_key = "upper_paths"
        elif role == "FULL_BODY_MASTER":
            target_key = "full_paths"
        else:
            normalized = str(path).replace("\\", "/").lower()
            if "/anchors/face/" in normalized:
                target_key = "face_paths"
            elif "/anchors/upper/" in normalized:
                target_key = "upper_paths"
            elif "/anchors/full/" in normalized:
                target_key = "full_paths"

        if target_key is None:
            continue

        if path.exists():
            grouped[target_key].append(str(path))
        else:
            print(f"[警告] Anchor registry 路径不存在，已跳过: {anchor_id} -> {path}")

    return grouped


def _default_anchor_paths_from_dirs(config: RuntimeConfig) -> Dict[str, List[str]]:
    return {
        "face_paths": [str(path) for path in _list_image_files_recursive(config.paths.dir_anchor_face)],
        "upper_paths": [str(path) for path in _list_image_files_in_dir(config.paths.dir_anchor_upper)],
        "full_paths": [str(path) for path in _list_image_files_in_dir(config.paths.dir_anchor_full)],
    }


def resolve_anchor_paths(config: RuntimeConfig) -> Dict[str, List[str]]:
    default_paths = _default_anchor_paths_from_dirs(config)
    if config.provider_policy.get("anchor_source", "registry_then_directory_fallback") != "registry_then_directory_fallback":
        return default_paths

    registry_paths = _anchor_paths_from_registry(config)
    out: Dict[str, List[str]] = {}
    for key in ["face_paths", "upper_paths", "full_paths"]:
        vals = registry_paths.get(key, [])
        out[key] = vals if len(vals) > 0 else default_paths.get(key, [])
    return out


def anchor_registry_summary(config: RuntimeConfig) -> Dict[str, int]:
    resolved = resolve_anchor_paths(config)
    return {
        "face_paths": len(resolved.get("face_paths", [])),
        "upper_paths": len(resolved.get("upper_paths", [])),
        "full_paths": len(resolved.get("full_paths", [])),
    }


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
