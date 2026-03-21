from __future__ import annotations

import copy
import hashlib
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .providers import build_provider_bundle
from .qa_consistency import (
    apply_consistency_soft_gate,
    extract_body_constitution_metrics,
    extract_depth_3d_lite_metrics,
    extract_skin_consistency_metrics,
)
from .qa_features import extract_face_feat, extract_pose_feat, init_engines
from .qa_garment import extract_garment_metrics
from .qa_outfit import (
    apply_collection_diagnostics,
    build_collection_aggregates,
    build_shot_selection_report,
    evaluate_active_batch_gate,
    infer_layer_tag_from_profile,
    parse_collection_metadata,
)
from .qa_runtime import (
    AnchorSet,
    EngineState,
    FaceFeat,
    PoseFeat,
    RuntimeContext,
    _deep_merge_dict,
    anchor_registry_snapshot,
    anchor_registry_summary,
    create_runtime_config,
    load_thresholds_from_file as load_runtime_thresholds_from_file,
    resolve_anchor_paths,
    save_thresholds_to_file,
)
from .qa_scoring import (
    build_tone_reference_stats,
    build_quality_reference_stats,
    classify_module,
    filter_face_anchors_by_view,
    fuse_overall,
    get_identity_anchor_pool,
    get_profile_policy,
    get_quality_anchor_pool,
    get_tone_anchor_pool,
    get_stats_for_bucket,
    make_recommendations,
    score_face_against_anchor_set,
    score_full_against_anchor_set,
    score_upper_against_anchor_set,
)
from .qa_view_router import route_view_lane
from .qa_utils import (
    SKIMAGE_SSIM_AVAILABLE,
    canonicalize_view_lane,
    dedupe_keep_order,
    estimate_view_bucket_and_side,
    get_face_size_bucket,
    get_quality_tolerances_by_face_size,
    image_read_bgr,
    list_images_in_dir,
    robust_percentile,
    valid_face_feats,
)


def create_runtime(base_dir: Optional[Path] = None) -> RuntimeContext:
    config = create_runtime_config(base_dir)
    providers = build_provider_bundle(config.provider_policy)
    engines = init_engines()
    return RuntimeContext(config=config, providers=providers, engines=engines)


def print_runtime_config(runtime: RuntimeContext) -> None:
    config = runtime.config
    providers_desc = runtime.providers.describe() if hasattr(runtime.providers, "describe") else {"mode": "config_only"}
    print(f"[CONFIG] RUN_MODE={config.run_mode}")
    print(f"[CONFIG] ACTIVE_PROFILE={config.review.active_profile}")
    print(f"[CONFIG] CONFIG_DIR={config.paths.config_dir}")
    print(f"[CONFIG] EXTERNAL_CONFIG_STATUS={config.external_config_status}")
    print(f"[CONFIG] PROVIDER_POLICY={config.provider_policy}")
    print(f"[CONFIG] PROVIDERS={providers_desc}")
    print(f"[CONFIG] ANCHOR_REGISTRY_SUMMARY={anchor_registry_summary(config)}")
    print(f"[CONFIG] LAYER_QUOTAS_LOADED={bool(config.layer_quotas.get('training_layers'))}")
    print("[CONFIG] FACE_CONF_MAP = linear_map_to_01(bbox_ratio, 0.006, 0.035)")
    print(f"[CONFIG] FACE_NO_RELIABLE_SIGNAL_TH = {config.review.face_no_signal_conf_th}")
    print(f"[CONFIG] MIN_CONF_FOR_STRICT_FAIL = {config.review.min_conf_for_strict_fail}")
    print(f"[CONFIG] CONSISTENCY_MODE={config.consistency.mode}")


def load_anchor_set(runtime: RuntimeContext) -> AnchorSet:
    anchors = AnchorSet()
    resolved_anchor_paths = resolve_anchor_paths(runtime.config)

    anchors.meta["anchor_source_mode"] = runtime.config.provider_policy.get(
        "anchor_source", "registry_then_directory_fallback"
    )
    anchors.meta["face_paths"] = list(resolved_anchor_paths.get("face_paths", []))
    anchors.meta["upper_paths"] = list(resolved_anchor_paths.get("upper_paths", []))
    anchors.meta["full_paths"] = list(resolved_anchor_paths.get("full_paths", []))
    anchors.meta["tone_paths"] = list(resolved_anchor_paths.get("tone_paths", []))

    print("\n[初始化] 加载 Anchor Set...")
    print(f"  Anchor Source Mode: {anchors.meta['anchor_source_mode']}")
    print(f"  Face Anchors : {len(anchors.meta['face_paths'])}")
    print(f"  Upper Anchors: {len(anchors.meta['upper_paths'])}")
    print(f"  Full Anchors : {len(anchors.meta['full_paths'])}")
    print(f"  Tone Anchors : {len(anchors.meta['tone_paths'])}")

    for path_str in anchors.meta["face_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        face_feat = extract_face_feat(runtime, img, path) if img is not None else FaceFeat(
            ok=False,
            reasons=["IMAGE_READ_ERROR"],
            source_path=str(path),
        )
        anchors.face_feats.append(face_feat)

    for path_str in anchors.meta["upper_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            anchors.upper_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.upper_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(path)))
            continue
        anchors.upper_pose_feats.append(extract_pose_feat(runtime, img))
        anchors.upper_face_feats.append(extract_face_feat(runtime, img, path))

    for path_str in anchors.meta["full_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            anchors.full_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.full_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(path)))
            continue
        anchors.full_pose_feats.append(extract_pose_feat(runtime, img))
        anchors.full_face_feats.append(extract_face_feat(runtime, img, path))

    for path_str in anchors.meta["tone_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        face_feat = extract_face_feat(runtime, img, path) if img is not None else FaceFeat(
            ok=False,
            reasons=["IMAGE_READ_ERROR"],
            source_path=str(path),
        )
        anchors.tone_face_feats.append(face_feat)

    print(f"  Upper Face-like Quality Refs: {len(valid_face_feats(anchors.upper_face_feats))}")
    print(f"  Full  Face-like Quality Refs: {len(valid_face_feats(anchors.full_face_feats))}")
    print(f"  Tone Face-like Refs       : {len(valid_face_feats(anchors.tone_face_feats))}")

    face_front = sum(1 for feat in anchors.face_feats if feat.ok and "/front/" in str(feat.source_path).replace("\\", "/").lower())
    face_3q = sum(1 for feat in anchors.face_feats if feat.ok and "/three_quarter/" in str(feat.source_path).replace("\\", "/").lower())
    face_profile = sum(
        1
        for feat in anchors.face_feats
        if feat.ok and (
            "/profile_like/" in str(feat.source_path).replace("\\", "/").lower()
            or "/profile/" in str(feat.source_path).replace("\\", "/").lower()
        )
    )
    print(f"  Face Buckets => front={face_front} | three_quarter={face_3q} | profile_like={face_profile}")
    return anchors


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _build_threshold_snapshot(runtime: RuntimeContext, profile_name: str) -> Dict[str, Any]:
    profile_thresholds = runtime.config.task_profiles.get(profile_name, {}).get("thresholds", {})
    profile_weights = runtime.config.task_profiles.get(profile_name, {}).get("weights", {})
    snapshot = {
        "consistency": {
            "mode": runtime.config.consistency.mode,
            "constitution_min_conf": runtime.config.consistency.constitution_min_conf,
            "skin_min_conf": runtime.config.consistency.skin_min_conf,
            "depth3d_min_conf": runtime.config.consistency.depth3d_min_conf,
            "constitution_soft_warn_th": runtime.config.consistency.constitution_soft_warn_th,
            "constitution_strong_warn_th": runtime.config.consistency.constitution_strong_warn_th,
            "skin_soft_warn_th": runtime.config.consistency.skin_soft_warn_th,
            "skin_strong_warn_th": runtime.config.consistency.skin_strong_warn_th,
            "depth3d_soft_warn_th": runtime.config.consistency.depth3d_soft_warn_th,
            "depth3d_strong_warn_th": runtime.config.consistency.depth3d_strong_warn_th,
            "skin_risk": _json_ready(runtime.config.consistency.skin_risk.__dict__),
            "skin_split": _json_ready(runtime.config.consistency.skin_split.__dict__),
            "skin_score_weights": {
                "strict": _json_ready(runtime.config.consistency.skin_score_weights.strict.__dict__),
                "chroma_dominant": _json_ready(runtime.config.consistency.skin_score_weights.chroma_dominant.__dict__),
                "high_risk": _json_ready(runtime.config.consistency.skin_score_weights.high_risk.__dict__),
            },
            "body_constitution_scoring": _json_ready(runtime.config.consistency.body_constitution_scoring),
            "depth3d_scoring": _json_ready(runtime.config.consistency.depth3d_scoring),
            "score_fusion": _json_ready(runtime.config.consistency.score_fusion),
        },
        "quality_thresholds": runtime.config.quality_thresholds.to_json_dict(),
        "task_profile_thresholds": _json_ready(profile_thresholds),
        "task_profile_weights": _json_ready(profile_weights),
    }
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")
    snapshot["hash"] = hashlib.sha1(payload).hexdigest()[:12]
    return snapshot


def _build_report_meta(
    runtime: RuntimeContext,
    target_profile: str,
    anchors: AnchorSet,
    input_count: int,
) -> Dict[str, Any]:
    profile_policy = get_profile_policy(runtime, target_profile)
    return {
        "schema_version": "qa_report_v2_3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_profile": target_profile,
        "review_policy": {
            "active_profile_default": runtime.config.review.active_profile,
            "strict_fail_min_conf": runtime.config.review.min_conf_for_strict_fail,
            "face_no_signal_conf_th": runtime.config.review.face_no_signal_conf_th,
        },
        "profile_policy": _json_ready(profile_policy),
        "provider_policy": _json_ready(runtime.config.provider_policy),
        "providers": _json_ready(runtime.providers.describe()),
        "anchor_registry_summary": anchor_registry_summary(runtime.config),
        "anchor_registry_snapshot": anchor_registry_snapshot(runtime.config),
        "anchor_governance": {
            "face_identity_policy": "absolute_only",
            "body_master_policy": "absolute_only",
            "support_anchor_policy": "assist_only",
        },
        "anchor_paths_resolved": _json_ready(anchors.meta),
        "layer_quotas": _json_ready(runtime.config.layer_quotas),
        "threshold_snapshot": _build_threshold_snapshot(runtime, target_profile),
        "engine": {
            "face": runtime.engines.face_mode,
            "pose": runtime.engines.pose_mode,
            "ssim_backend": "skimage" if SKIMAGE_SSIM_AVAILABLE else "ncc_fallback",
        },
        "view_detector": {
            "raw_buckets": ["front", "three_quarter", "profile_like"],
            "canonical_lanes": ["front", "three_quarter", "side_90"],
            "lane_details": [
                "front",
                "three_quarter",
                "strict_side_90_left",
                "strict_side_90_right",
                "side_like_left",
                "side_like_right",
                "strict_back_180",
                "back_like",
            ],
            "scoring_surfaces": [
                "front",
                "three_quarter",
                "strict_side_90",
                "side_like",
                "strict_back_180",
                "back_like",
                "profile_like",
            ],
            "back_180_native_detection": False,
            "active_source": "legacy_face_router",
            "shadow_router_v2": {
                "enabled": True,
                "mode": "shadow_only",
                "lanes": ["front", "three_quarter", "side_90", "back_180"],
                "native_back_180_detection": True,
                "source_priority": ["face", "pose", "subject_mask", "skin_region"],
            },
        },
        "collection_parser": {
            "enabled": True,
            "mode": "path_derived",
            "layer_tags": ["BODY_GOLD", "BRIDGE", "NECKLINE", "OUTER", "FACE_LOCK"],
            "group_keys": ["layer_tag", "look_key", "outfit_key", "slot_key", "view_expected"],
            "fallback_grouping": "active_profile_batch",
            "requires_explicit_look_key_for_set_qa": False,
        },
        "outfit_measurement": {
            "enabled": True,
            "mode": "lightweight_mask_pose",
            "uses_new_models": False,
            "single_image_outputs": [
                "clothing_coverage_ratio",
                "upper_cloth_coverage",
                "lower_cloth_coverage",
                "neckline_openness",
                "shoulder_exposure_balance",
            ],
            "batch_outputs": [
                "garment_profile_stability",
                "garment_boundary_stability",
                "body_under_clothes_continuity",
                "routing_consistency",
                "batch_clothfree_identity_cohesion",
                "batch_hybrid_identity_cohesion",
                "batch_3d_cohesion",
            ],
            "batch_gate_supported": True,
        },
        "shot_batch_selection": {
            "enabled": True,
            "mode": "advisory_rank_only",
            "final_decision_owner": "custom_gpt_plus_human",
            "goal": "provide ranking and explanations for batch review, not final auto-selection",
            "outputs": ["selection_score", "top_ranked_image", "shortlist", "component_scores", "penalties"],
            "components": [
                "absolute_face_identity",
                "batch_face_alignment",
                "clothfree_body_alignment",
                "depth_alignment",
                "structure_stability",
            ],
        },
        "input_count": int(input_count),
    }


def _apply_profile_view_policy(
    target_profile: str,
    policy: Dict[str, Any],
    view_lane: str,
    final_status: str,
    overall_state: str,
    reasons_all: List[str],
) -> tuple[str, str]:
    def downgrade(reason: str) -> None:
        nonlocal final_status, overall_state
        reasons_all.append(reason)
        if final_status == "PASS":
            final_status = "WARN"
            overall_state = "WARN"

    allowed_view_buckets = policy.get("allowed_view_buckets", [])
    if isinstance(allowed_view_buckets, list) and len(allowed_view_buckets) > 0:
        allowed = {str(item) for item in allowed_view_buckets}
        if view_lane not in allowed:
            downgrade("VIEW_LANE_NOT_ALLOWED_FOR_PROFILE")

    for bucket in policy.get("soft_review_buckets", []) or []:
        if view_lane == str(bucket):
            if view_lane == "three_quarter":
                reasons_all.append("THREE_QUARTER_SOFT_REVIEW")
            else:
                reasons_all.append("VIEW_LANE_SOFT_REVIEW")

    pass_cap_mode = str(policy.get("pass_cap_mode", "none"))
    if pass_cap_mode == "always_warn" and final_status == "PASS":
        downgrade("PROFILE_PASS_CAPPED_TO_WARN")
    elif pass_cap_mode == "warn_non_front" and view_lane != "front" and final_status == "PASS":
        downgrade("NON_FRONT_PASS_CAPPED_TO_WARN")
    elif pass_cap_mode == "body_gold_front_core":
        if view_lane == "side_90" and final_status == "PASS":
            downgrade("PROFILE_LIKE_NO_SIDE_ANCHOR_PASS_CAPPED")
        elif target_profile == "body_gold_fullbody" and view_lane == "unknown":
            reasons_all.append("BODY_GOLD_VIEW_LANE_UNKNOWN")

    return final_status, overall_state


def _body_identity_signature(
    cand_pose: PoseFeat,
    constitution_metrics: Dict[str, Any],
    depth_3d_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    upper = cand_pose.upper_geom or {}
    full = cand_pose.full_geom or {}

    feature_specs = [
        ("head_body_ratio", full.get("head_body_ratio"), 5.0, 2.0),
        ("leg_ratio", full.get("leg_ratio"), 0.47, 0.14),
        ("torso_len_norm", upper.get("torso_len_norm"), 0.24, 0.08),
        ("hip_shoulder_ratio", upper.get("hip_shoulder_ratio"), 0.58, 0.22),
        ("torso_compactness", upper.get("torso_compactness"), 0.92, 0.26),
        ("shoulder_hip_center_offset_norm", upper.get("shoulder_hip_center_offset_norm"), 0.02, 0.08),
        ("lower_limb_balance", full.get("lower_limb_balance"), 0.96, 0.12),
        ("foot_length_balance", full.get("foot_length_balance"), 0.93, 0.14),
        ("waist_to_torso_ratio", constitution_metrics.get("waist_to_torso_ratio"), 0.80, 0.22),
        ("hip_to_torso_ratio", constitution_metrics.get("hip_to_torso_ratio"), 1.00, 0.24),
    ]

    values: List[float] = []
    ready = 0
    for _, raw_value, center, scale in feature_specs:
        if raw_value is None:
            values.append(0.0)
            continue
        try:
            numeric = float(raw_value)
        except Exception:
            values.append(0.0)
            continue
        values.append((numeric - float(center)) / max(1e-6, float(scale)))
        ready += 1

    signature = None
    if ready >= 4:
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            signature = (vector / norm).astype(np.float32)

    pose_weight = float(getattr(cand_pose, "confidence_full", 0.0) or 0.0)
    constitution_conf = float(constitution_metrics.get("confidence", 0.0) or 0.0)
    depth_conf = float(depth_3d_metrics.get("confidence", 0.0) or 0.0)
    weight = float(
        0.40 * pose_weight
        + 0.35 * constitution_conf
        + 0.25 * depth_conf
    )
    return {
        "signature": signature,
        "ready_features": ready,
        "feature_count": len(feature_specs),
        "weight": weight if signature is not None else 0.0,
    }


def _depth_identity_signature(
    cand_pose: PoseFeat,
    depth_3d_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    upper = cand_pose.upper_geom or {}
    full = cand_pose.full_geom or {}

    feature_specs = [
        ("shoulder_width_norm", upper.get("shoulder_width_norm"), 0.17, 0.05),
        ("hip_width_norm", upper.get("hip_width_norm"), 0.10, 0.04),
        ("hip_shoulder_ratio", upper.get("hip_shoulder_ratio"), 0.58, 0.22),
        ("torso_len_norm", upper.get("torso_len_norm"), 0.24, 0.08),
        ("torso_compactness", upper.get("torso_compactness"), 0.92, 0.26),
        ("shoulder_hip_center_offset_norm", upper.get("shoulder_hip_center_offset_norm"), 0.02, 0.08),
        ("spine_angle_deg", upper.get("spine_angle_deg"), 0.0, 8.0),
        ("lower_limb_balance", full.get("lower_limb_balance"), 0.96, 0.12),
        ("thigh_length_balance", full.get("thigh_length_balance"), 0.95, 0.10),
        ("calf_length_balance", full.get("calf_length_balance"), 0.95, 0.10),
        ("foot_length_balance", full.get("foot_length_balance"), 0.93, 0.14),
        ("ankle_gap_norm", full.get("ankle_gap_norm"), 0.09, 0.08),
        ("depth_3d_score", depth_3d_metrics.get("depth_3d_score"), 0.82, 0.16),
        ("turn_signal_score", depth_3d_metrics.get("turn_signal_score"), 0.75, 0.16),
        ("torso_volume_score", depth_3d_metrics.get("torso_volume_score"), 0.80, 0.16),
        ("side_profile_score", depth_3d_metrics.get("side_profile_score"), 0.84, 0.14),
        ("posterior_score", depth_3d_metrics.get("posterior_score"), 0.84, 0.14),
        ("torso_compactness_score", depth_3d_metrics.get("torso_compactness_score"), 0.80, 0.16),
    ]

    values: List[float] = []
    ready = 0
    for _, raw_value, center, scale in feature_specs:
        if raw_value is None:
            values.append(0.0)
            continue
        try:
            numeric = float(raw_value)
        except Exception:
            values.append(0.0)
            continue
        values.append((numeric - float(center)) / max(1e-6, float(scale)))
        ready += 1

    signature = None
    if ready >= 6:
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            signature = (vector / norm).astype(np.float32)

    pose_weight = float(getattr(cand_pose, "confidence_full", 0.0) or 0.0)
    depth_conf = float(depth_3d_metrics.get("confidence", 0.0) or 0.0)
    depth_score = float(depth_3d_metrics.get("depth_3d_score", 0.0) or 0.0)
    weight = float(
        0.45 * depth_conf
        + 0.35 * pose_weight
        + 0.20 * depth_score
    )
    return {
        "signature": signature,
        "ready_features": ready,
        "feature_count": len(feature_specs),
        "weight": weight if signature is not None else 0.0,
    }


def calibrate_quality_thresholds(
    runtime: RuntimeContext,
    calib_dir: Optional[Path] = None,
) -> Dict[str, float]:
    target_dir = calib_dir or runtime.config.paths.dir_calib
    images = list_images_in_dir(target_dir)
    if len(images) == 0:
        raise RuntimeError(f"校准目录为空: {target_dir}")

    lumas: List[float] = []
    lap_vars: List[float] = []
    hf_energies: List[float] = []
    used = 0

    for path in images:
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            continue
        face_feat = extract_face_feat(runtime, img, path)
        if not face_feat.ok or face_feat.lab_mean is None:
            continue

        lumas.append(float(face_feat.lab_mean[0]))
        if face_feat.lap_var > 0:
            lap_vars.append(float(face_feat.lap_var))
        if face_feat.hf_energy > 0:
            hf_energies.append(float(face_feat.hf_energy))
        used += 1

    if used < 8:
        raise RuntimeError(f"有效校准样本太少，仅 {used} 张，建议至少 8 张，最好 20–40 张")

    return {
        "FACE_LUMA_DARK_WARN_L": robust_percentile(lumas, 10),
        "FACE_LAPVAR_SOFT_WARN": robust_percentile(lap_vars, 15),
        "FACE_HFENERGY_SOFT_WARN": robust_percentile(hf_energies, 15),
        "num_used": used,
        "luma_mean": float(np.mean(lumas)) if lumas else 0.0,
        "lap_var_mean": float(np.mean(lap_vars)) if lap_vars else 0.0,
        "hf_energy_mean": float(np.mean(hf_energies)) if hf_energies else 0.0,
    }


def load_thresholds_from_file(
    runtime: RuntimeContext,
    path: Optional[Path] = None,
) -> None:
    load_runtime_thresholds_from_file(runtime.config, path or runtime.config.paths.thresh_file)


def _set_override_attr(
    target: Any,
    attr_name: str,
    raw_value: Any,
    cast_type: Any,
    path: str,
    applied: Dict[str, Any],
) -> None:
    try:
        value = cast_type(raw_value)
    except Exception as exc:
        raise ValueError(f"Invalid threshold_override value at {path}: {raw_value!r}") from exc
    setattr(target, attr_name, value)
    applied[path] = value


def _apply_mapped_overrides(
    target: Any,
    node: Dict[str, Any],
    mapping: Dict[str, tuple[str, Any]],
    path_prefix: str,
    applied: Dict[str, Any],
) -> None:
    unknown_keys = sorted(set(node.keys()) - set(mapping.keys()))
    if unknown_keys:
        raise ValueError(f"Unsupported threshold_override keys under {path_prefix}: {unknown_keys}")
    for key, raw_value in node.items():
        attr_name, cast_type = mapping[key]
        _set_override_attr(target, attr_name, raw_value, cast_type, f"{path_prefix}.{attr_name}", applied)


def _apply_threshold_override(
    runtime: RuntimeContext,
    target_profile: str,
    threshold_override: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(threshold_override, dict):
        raise TypeError("threshold_override must be a dict when provided")

    applied: Dict[str, Any] = {}
    config = runtime.config

    consistency_direct_map = {
        "mode": ("mode", str),
        "constitution_min_conf": ("constitution_min_conf", float),
        "skin_min_conf": ("skin_min_conf", float),
        "depth3d_min_conf": ("depth3d_min_conf", float),
        "constitution_soft_warn_th": ("constitution_soft_warn_th", float),
        "constitution_strong_warn_th": ("constitution_strong_warn_th", float),
        "skin_soft_warn_th": ("skin_soft_warn_th", float),
        "skin_strong_warn_th": ("skin_strong_warn_th", float),
        "depth3d_soft_warn_th": ("depth3d_soft_warn_th", float),
        "depth3d_strong_warn_th": ("depth3d_strong_warn_th", float),
    }
    consistency_min_conf_map = {
        "constitution": ("constitution_min_conf", float),
        "skin": ("skin_min_conf", float),
        "depth3d": ("depth3d_min_conf", float),
    }
    consistency_warn_map = {
        "constitution_soft": ("constitution_soft_warn_th", float),
        "constitution_strong": ("constitution_strong_warn_th", float),
        "skin_soft": ("skin_soft_warn_th", float),
        "skin_strong": ("skin_strong_warn_th", float),
        "depth3d_soft": ("depth3d_soft_warn_th", float),
        "depth3d_strong": ("depth3d_strong_warn_th", float),
    }
    skin_risk_map = {
        "lighting_warn": ("lighting_warn_th", float),
        "lighting_warn_th": ("lighting_warn_th", float),
        "lighting_high": ("lighting_high_th", float),
        "lighting_high_th": ("lighting_high_th", float),
        "sample_warn": ("sample_warn_th", float),
        "sample_warn_th": ("sample_warn_th", float),
        "sample_high": ("sample_high_th", float),
        "sample_high_th": ("sample_high_th", float),
        "face_side_delta_l_warn": ("face_side_delta_l_warn", float),
        "face_neck_delta_l_warn": ("face_neck_delta_l_warn", float),
        "leg_lr_delta_l_warn": ("leg_lr_delta_l_warn", float),
        "face_highlight_l": ("face_highlight_l", float),
        "face_highlight_ratio_warn": ("face_highlight_ratio_warn", float),
        "face_highlight_ratio_high": ("face_highlight_ratio_high", float),
        "edge_margin_ratio_floor": ("edge_margin_ratio_floor", float),
        "low_purity_floor": ("low_purity_floor", float),
        "purity_variance_warn": ("purity_variance_warn", float),
    }
    skin_split_map = {
        "delta_ab_decay_thigh": ("delta_ab_decay_thigh", float),
        "delta_ab_decay_calf": ("delta_ab_decay_calf", float),
        "delta_l_decay_thigh": ("delta_l_decay_thigh", float),
        "delta_l_decay_calf": ("delta_l_decay_calf", float),
        "brightness_ratio_low": ("brightness_ratio_low", float),
        "brightness_ratio_high": ("brightness_ratio_high", float),
        "brightness_ratio_margin": ("brightness_ratio_margin", float),
        "knee_ratio_low": ("knee_ratio_low", float),
        "knee_ratio_high": ("knee_ratio_high", float),
        "knee_ratio_margin": ("knee_ratio_margin", float),
        "severe_delta_ab_thigh": ("severe_delta_ab_thigh", float),
        "severe_delta_ab_calf": ("severe_delta_ab_calf", float),
        "severe_leg_brightness_ratio": ("severe_leg_brightness_ratio", float),
        "severe_luminance_score": ("severe_luminance_score", float),
    }
    quality_threshold_map = {
        "face_luma_dark_warn_l": ("face_luma_dark_warn_l", float),
        "FACE_LUMA_DARK_WARN_L": ("face_luma_dark_warn_l", float),
        "face_lapvar_soft_warn": ("face_lapvar_soft_warn", float),
        "FACE_LAPVAR_SOFT_WARN": ("face_lapvar_soft_warn", float),
        "face_hfenergy_soft_warn": ("face_hfenergy_soft_warn", float),
        "FACE_HFENERGY_SOFT_WARN": ("face_hfenergy_soft_warn", float),
    }
    profile_threshold_map = {
        "face_pass": float,
        "face_warn": float,
        "upper_pass": float,
        "upper_warn": float,
        "full_pass": float,
        "full_warn": float,
        "overall_pass": float,
        "overall_warn": float,
    }
    profile_weight_map = {
        "face": float,
        "upper": float,
        "full": float,
    }

    allowed_top_level = {
        "consistency",
        "quality_thresholds",
        "profile_thresholds",
        "profile_weights",
        "task_profiles",
    }
    unknown_top_level = sorted(set(threshold_override.keys()) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"Unsupported threshold_override sections: {unknown_top_level}")

    consistency_node = threshold_override.get("consistency", None)
    if consistency_node is not None:
        if not isinstance(consistency_node, dict):
            raise ValueError("threshold_override.consistency must be a dict")
        nested_consistency_keys = {
            "min_confidence",
            "warn_threshold",
            "skin_risk",
            "skin_split",
            "skin_score_weights",
            "body_constitution_scoring",
            "depth3d_scoring",
            "score_fusion",
        }
        direct_consistency_node = {
            key: value for key, value in consistency_node.items() if key not in nested_consistency_keys
        }
        if direct_consistency_node:
            _apply_mapped_overrides(
                config.consistency,
                direct_consistency_node,
                consistency_direct_map,
                "consistency",
                applied,
            )
        min_conf_node = consistency_node.get("min_confidence", None)
        if min_conf_node is not None:
            if not isinstance(min_conf_node, dict):
                raise ValueError("threshold_override.consistency.min_confidence must be a dict")
            _apply_mapped_overrides(
                config.consistency,
                min_conf_node,
                consistency_min_conf_map,
                "consistency.min_confidence",
                applied,
            )
        warn_node = consistency_node.get("warn_threshold", None)
        if warn_node is not None:
            if not isinstance(warn_node, dict):
                raise ValueError("threshold_override.consistency.warn_threshold must be a dict")
            _apply_mapped_overrides(
                config.consistency,
                warn_node,
                consistency_warn_map,
                "consistency.warn_threshold",
                applied,
            )
        skin_risk_node = consistency_node.get("skin_risk", None)
        if skin_risk_node is not None:
            if not isinstance(skin_risk_node, dict):
                raise ValueError("threshold_override.consistency.skin_risk must be a dict")
            _apply_mapped_overrides(
                config.consistency.skin_risk,
                skin_risk_node,
                skin_risk_map,
                "consistency.skin_risk",
                applied,
            )
        skin_split_node = consistency_node.get("skin_split", None)
        if skin_split_node is not None:
            if not isinstance(skin_split_node, dict):
                raise ValueError("threshold_override.consistency.skin_split must be a dict")
            _apply_mapped_overrides(
                config.consistency.skin_split,
                skin_split_node,
                skin_split_map,
                "consistency.skin_split",
                applied,
            )
        skin_score_weights_node = consistency_node.get("skin_score_weights", None)
        if skin_score_weights_node is not None:
            if not isinstance(skin_score_weights_node, dict):
                raise ValueError("threshold_override.consistency.skin_score_weights must be a dict")
            allowed_presets = {
                "strict": config.consistency.skin_score_weights.strict,
                "chroma_dominant": config.consistency.skin_score_weights.chroma_dominant,
                "high_risk": config.consistency.skin_score_weights.high_risk,
            }
            unknown_presets = sorted(set(skin_score_weights_node.keys()) - set(allowed_presets.keys()))
            if unknown_presets:
                raise ValueError(
                    f"Unsupported threshold_override.consistency.skin_score_weights presets: {unknown_presets}"
                )
            weight_map = {
                "chroma": ("chroma", float),
                "luminance": ("luminance", float),
                "knee": ("knee", float),
                "baseline": ("baseline", float),
            }
            for preset_name, preset_node in skin_score_weights_node.items():
                if not isinstance(preset_node, dict):
                    raise ValueError(
                        f"threshold_override.consistency.skin_score_weights.{preset_name} must be a dict"
                    )
                _apply_mapped_overrides(
                    allowed_presets[preset_name],
                    preset_node,
                    weight_map,
                    f"consistency.skin_score_weights.{preset_name}",
                    applied,
                )

        body_constitution_scoring_node = consistency_node.get("body_constitution_scoring", None)
        if body_constitution_scoring_node is not None:
            if not isinstance(body_constitution_scoring_node, dict):
                raise ValueError("threshold_override.consistency.body_constitution_scoring must be a dict")
            config.consistency.body_constitution_scoring = _deep_merge_dict(
                config.consistency.body_constitution_scoring,
                body_constitution_scoring_node,
            )
            applied["consistency.body_constitution_scoring"] = copy.deepcopy(body_constitution_scoring_node)

        depth3d_scoring_node = consistency_node.get("depth3d_scoring", None)
        if depth3d_scoring_node is not None:
            if not isinstance(depth3d_scoring_node, dict):
                raise ValueError("threshold_override.consistency.depth3d_scoring must be a dict")
            config.consistency.depth3d_scoring = _deep_merge_dict(
                config.consistency.depth3d_scoring,
                depth3d_scoring_node,
            )
            applied["consistency.depth3d_scoring"] = copy.deepcopy(depth3d_scoring_node)

        score_fusion_node = consistency_node.get("score_fusion", None)
        if score_fusion_node is not None:
            if not isinstance(score_fusion_node, dict):
                raise ValueError("threshold_override.consistency.score_fusion must be a dict")
            config.consistency.score_fusion = _deep_merge_dict(
                config.consistency.score_fusion,
                score_fusion_node,
            )
            applied["consistency.score_fusion"] = copy.deepcopy(score_fusion_node)

    quality_threshold_node = threshold_override.get("quality_thresholds", None)
    if quality_threshold_node is not None:
        if not isinstance(quality_threshold_node, dict):
            raise ValueError("threshold_override.quality_thresholds must be a dict")
        _apply_mapped_overrides(
            config.quality_thresholds,
            quality_threshold_node,
            quality_threshold_map,
            "quality_thresholds",
            applied,
        )

    profile_weight_node = threshold_override.get("profile_weights", None)
    if profile_weight_node is not None:
        if not isinstance(profile_weight_node, dict):
            raise ValueError("threshold_override.profile_weights must be a dict")
        if target_profile not in config.task_profiles:
            raise ValueError(f"Unknown target profile for threshold_override.profile_weights: {target_profile}")
        unknown_profile_weights = sorted(set(profile_weight_node.keys()) - set(profile_weight_map.keys()))
        if unknown_profile_weights:
            raise ValueError(
                f"Unsupported threshold_override.profile_weights keys: {unknown_profile_weights}"
            )
        for key, raw_value in profile_weight_node.items():
            cast_type = profile_weight_map[key]
            try:
                value = cast_type(raw_value)
            except Exception as exc:
                raise ValueError(f"Invalid threshold_override.profile_weights.{key}: {raw_value!r}") from exc
            config.task_profiles[target_profile]["weights"][key] = value
            applied[f"task_profiles.{target_profile}.weights.{key}"] = value

    profile_threshold_node = threshold_override.get("profile_thresholds", None)
    if profile_threshold_node is not None:
        if not isinstance(profile_threshold_node, dict):
            raise ValueError("threshold_override.profile_thresholds must be a dict")
        if target_profile not in config.task_profiles:
            raise ValueError(f"Unknown target profile for threshold_override.profile_thresholds: {target_profile}")
        unknown_profile_thresholds = sorted(set(profile_threshold_node.keys()) - set(profile_threshold_map.keys()))
        if unknown_profile_thresholds:
            raise ValueError(
                f"Unsupported threshold_override.profile_thresholds keys: {unknown_profile_thresholds}"
            )
        for key, raw_value in profile_threshold_node.items():
            cast_type = profile_threshold_map[key]
            try:
                value = cast_type(raw_value)
            except Exception as exc:
                raise ValueError(f"Invalid threshold_override.profile_thresholds.{key}: {raw_value!r}") from exc
            config.task_profiles[target_profile]["thresholds"][key] = value
            applied[f"task_profiles.{target_profile}.thresholds.{key}"] = value

    task_profiles_node = threshold_override.get("task_profiles", None)
    if task_profiles_node is not None:
        if not isinstance(task_profiles_node, dict):
            raise ValueError("threshold_override.task_profiles must be a dict")
        for profile_name, profile_node in task_profiles_node.items():
            if profile_name not in config.task_profiles:
                raise ValueError(f"Unknown threshold_override task profile: {profile_name}")
            if not isinstance(profile_node, dict):
                raise ValueError(f"threshold_override.task_profiles.{profile_name} must be a dict")
            unknown_profile_sections = sorted(set(profile_node.keys()) - {"thresholds", "weights"})
            if unknown_profile_sections:
                raise ValueError(
                    f"Unsupported threshold_override.task_profiles.{profile_name} sections: {unknown_profile_sections}"
                )

            threshold_node = profile_node.get("thresholds", None)
            if threshold_node is None and any(key in profile_threshold_map for key in profile_node.keys()):
                threshold_node = {key: value for key, value in profile_node.items() if key in profile_threshold_map}
            if threshold_node is not None:
                if not isinstance(threshold_node, dict):
                    raise ValueError(f"threshold_override.task_profiles.{profile_name}.thresholds must be a dict")
                unknown_threshold_keys = sorted(set(threshold_node.keys()) - set(profile_threshold_map.keys()))
                if unknown_threshold_keys:
                    raise ValueError(
                        f"Unsupported threshold_override.task_profiles.{profile_name}.thresholds keys: "
                        f"{unknown_threshold_keys}"
                    )
                for key, raw_value in threshold_node.items():
                    cast_type = profile_threshold_map[key]
                    try:
                        value = cast_type(raw_value)
                    except Exception as exc:
                        raise ValueError(
                            f"Invalid threshold_override.task_profiles.{profile_name}.thresholds.{key}: {raw_value!r}"
                        ) from exc
                    config.task_profiles[profile_name]["thresholds"][key] = value
                    applied[f"task_profiles.{profile_name}.thresholds.{key}"] = value

            weight_node = profile_node.get("weights", None)
            if weight_node is not None:
                if not isinstance(weight_node, dict):
                    raise ValueError(f"threshold_override.task_profiles.{profile_name}.weights must be a dict")
                unknown_weight_keys = sorted(set(weight_node.keys()) - set(profile_weight_map.keys()))
                if unknown_weight_keys:
                    raise ValueError(
                        f"Unsupported threshold_override.task_profiles.{profile_name}.weights keys: "
                        f"{unknown_weight_keys}"
                    )
                for key, raw_value in weight_node.items():
                    cast_type = profile_weight_map[key]
                    try:
                        value = cast_type(raw_value)
                    except Exception as exc:
                        raise ValueError(
                            f"Invalid threshold_override.task_profiles.{profile_name}.weights.{key}: {raw_value!r}"
                        ) from exc
                    config.task_profiles[profile_name]["weights"][key] = value
                    applied[f"task_profiles.{profile_name}.weights.{key}"] = value

    return applied


def run_pipeline(
    runtime: RuntimeContext,
    profile_name: Optional[str] = None,
    threshold_override: Optional[Dict[str, Any]] = None,
) -> None:
    original_config = copy.deepcopy(runtime.config) if threshold_override is not None else None
    try:
        if threshold_override is not None:
            target_profile = profile_name or runtime.config.review.active_profile
            applied = _apply_threshold_override(runtime, target_profile, threshold_override)
            if applied:
                print(f"[CONFIG] Applied in-memory threshold_override ({len(applied)} items)")
                for key in sorted(applied.keys()):
                    print(f"  {key} = {applied[key]}")
        _run_pipeline_impl(runtime, profile_name=profile_name)
    finally:
        if original_config is not None:
            runtime.config = original_config


def _run_pipeline_impl(
    runtime: RuntimeContext,
    profile_name: Optional[str] = None,
) -> None:
    config = runtime.config
    target_profile = profile_name or config.review.active_profile
    if target_profile not in config.task_profiles:
        raise ValueError(f"未知任务模板: {target_profile}. 可选: {list(config.task_profiles.keys())}")

    profile = config.task_profiles[target_profile]
    weights = profile["weights"]
    reqs = profile["require"]
    th = profile["thresholds"]
    policy = get_profile_policy(runtime, target_profile)

    if not config.paths.dir_input.exists():
        print(f"[致命错误] 输入目录不存在: {config.paths.dir_input}")
        return

    images = list_images_in_dir(config.paths.dir_input)
    if len(images) == 0:
        print(f"[提示] 输入目录为空: {config.paths.dir_input}")
        return

    anchors = load_anchor_set(runtime)
    face_identity_anchors = get_identity_anchor_pool(runtime, target_profile, anchors)
    face_quality_anchors = get_quality_anchor_pool(runtime, target_profile, anchors)
    face_tone_anchors = get_tone_anchor_pool(runtime, target_profile, anchors)
    quality_ref_stats = build_quality_reference_stats(face_quality_anchors)
    tone_ref_stats = build_tone_reference_stats(face_tone_anchors)

    if len(face_identity_anchors) == 0:
        print("[警告] 没有可用面部身份锚点，将导致 face 模块不可用")
    if len(anchors.upper_pose_feats) == 0:
        print("[警告] 没有半身锚点，将导致 upper 模块不可用")
    if len(anchors.full_pose_feats) == 0:
        print("[警告] 没有全身锚点，将导致 full 模块不可用")

    report_items: List[Dict[str, Any]] = []
    batch_identity_samples: List[Dict[str, Any]] = []
    print(f"[RUN] TONE_FACE_REFS={len(face_tone_anchors)}")
    report_meta = _build_report_meta(runtime, target_profile, anchors, len(images))
    print(f"\n[运行中] 任务模板: {target_profile}")
    print(f"[运行中] 身份锚池(face): {len(face_identity_anchors)}")
    print(f"[运行中] 质量锚池(face-like): {len(face_quality_anchors)}")
    print(
        f"[运行中] STANDARDIZE_INPUT={config.standardization.enabled} "
        f"long_side={config.standardization.long_side}"
    )
    print(f"[运行中] CONSISTENCY_MODE={config.consistency.mode}")
    print("[运行中] 开始批处理质检...\n")

    for img_path in images:
        print(f"-> 检测: {img_path.name}")
        collection_meta = parse_collection_metadata(img_path, config.paths.dir_input)
        if not collection_meta.get("layer_tag"):
            inferred_layer = infer_layer_tag_from_profile(target_profile)
            if inferred_layer:
                collection_meta["layer_tag"] = inferred_layer
                if str(collection_meta.get("naming_source") or "none") == "none":
                    collection_meta["naming_source"] = "profile_fallback"
        collection_meta["batch_key"] = f"batch::{target_profile}"
        collection_meta["aggregate_mode"] = (
            "path_group" if bool(collection_meta.get("groupable", False)) else "active_profile_batch"
        )
        try:
            img = image_read_bgr(img_path, config.standardization)
            if img is None:
                raise RuntimeError("IMAGE_READ_ERROR")

            cand_face = extract_face_feat(runtime, img, img_path)
            cand_pose = extract_pose_feat(runtime, img)
            legacy_view_bucket, legacy_view_side, yaw_proxy = estimate_view_bucket_and_side(cand_face)
            legacy_view_lane = canonicalize_view_lane(cand_face, legacy_view_bucket)
            shadow_view_route = route_view_lane(runtime, img, cand_face, cand_pose)
            view_bucket = legacy_view_bucket
            view_side = legacy_view_side
            view_lane = legacy_view_lane
            if view_lane == "unknown" and shadow_view_route.lane != "unknown":
                view_lane = shadow_view_route.lane
            if view_bucket == "unknown" and shadow_view_route.face_bucket != "unknown":
                view_bucket = shadow_view_route.face_bucket
            if view_side == "unknown" and shadow_view_route.face_side in {"left", "right"}:
                view_side = shadow_view_route.face_side
            face_size_bucket = get_face_size_bucket(cand_face.bbox_area_ratio if cand_face.ok else 0.0)
            tone_bucket_stats = get_stats_for_bucket(tone_ref_stats, face_size_bucket)
            tone_reference_lab = None
            if (
                len(tone_bucket_stats.get("L", [])) > 0
                and len(tone_bucket_stats.get("a", [])) > 0
                and len(tone_bucket_stats.get("b", [])) > 0
            ):
                tone_reference_lab = np.array(
                    [
                        float(np.mean(tone_bucket_stats["L"])),
                        float(np.mean(tone_bucket_stats["a"])),
                        float(np.mean(tone_bucket_stats["b"])),
                    ],
                    dtype=np.float32,
                )

            constitution_metrics = extract_body_constitution_metrics(
                runtime,
                img,
                cand_face,
                cand_pose,
                view_bucket=view_lane,
                view_lane_detail=shadow_view_route.lane_detail,
            )
            skin_metrics = extract_skin_consistency_metrics(
                runtime,
                img,
                cand_face,
                cand_pose,
                tone_reference_lab=tone_reference_lab,
            )
            garment_metrics = extract_garment_metrics(
                runtime,
                img,
                cand_face,
                cand_pose,
                layer_tag=collection_meta.get("layer_tag") or infer_layer_tag_from_profile(target_profile),
            )
            depth_3d_metrics = extract_depth_3d_lite_metrics(
                cand_face,
                cand_pose,
                view_bucket=view_lane,
                yaw_proxy=yaw_proxy,
                body_yaw_deg=shadow_view_route.body_yaw_deg,
                pose_frontal_strength=shadow_view_route.pose_frontal_strength,
                lane_strictness_score=shadow_view_route.lane_strictness_score,
                mask_symmetry=shadow_view_route.mask_symmetry,
                head_skin_ratio=shadow_view_route.head_skin_ratio,
                scoring=config.consistency.depth3d_scoring,
                view_lane_detail=shadow_view_route.lane_detail,
            )
            body_identity = _body_identity_signature(
                cand_pose,
                constitution_metrics,
                depth_3d_metrics,
            )
            depth_identity = _depth_identity_signature(
                cand_pose,
                depth_3d_metrics,
            )

            constitution_score = constitution_metrics.get("body_constitution_score", None)
            skin_score = skin_metrics.get("skin_uniformity_score", None)
            depth_3d_score = depth_3d_metrics.get("depth_3d_score", None)

            face_identity_anchors_view = filter_face_anchors_by_view(
                face_identity_anchors,
                view_lane,
                view_side=view_side,
            )
            face_score_o, face_conf_o, face_reasons_o, face_debug_o = score_face_against_anchor_set(
                runtime,
                cand_face,
                face_identity_anchors_view,
                view_bucket=view_lane,
                view_lane_detail=shadow_view_route.lane_detail,
            )

            face_score = face_score_o
            face_conf = face_conf_o
            face_reasons = face_reasons_o
            face_debug = {
                "view_bucket": view_bucket,
                "view_lane": view_lane,
                "view_side": view_side,
                "yaw_proxy": yaw_proxy,
                "flip_canonicalized": False,
                "identity_anchor_count_view": len(face_identity_anchors_view),
                "view_surface_requested": face_debug_o.get("view_surface_requested"),
                "view_surface_used": face_debug_o.get("view_surface_used"),
                "original": face_debug_o,
            }

            if view_bucket != "front":
                img_flipped = cv2.flip(img, 1)
                cand_face_flip = extract_face_feat(runtime, img_flipped, None)
                face_score_f, face_conf_f, face_reasons_f, face_debug_f = score_face_against_anchor_set(
                    runtime,
                    cand_face_flip,
                    face_identity_anchors_view,
                    view_bucket=view_lane,
                    view_lane_detail=shadow_view_route.lane_detail,
                )
                face_debug["flipped"] = face_debug_f
                if face_score_f > face_score_o:
                    face_score = face_score_f
                    face_conf = face_conf_f
                    face_reasons = ["FACE_FLIP_CANONICALIZED"] + face_reasons_f
                    face_debug["flip_canonicalized"] = True
                    face_debug["view_surface_requested"] = face_debug_f.get("view_surface_requested")
                    face_debug["view_surface_used"] = face_debug_f.get("view_surface_used")

            batch_identity_samples.append(
                {
                    "record_key": str(collection_meta.get("input_relative_path") or img_path.name),
                    "embedding": cand_face.embedding.copy() if getattr(cand_face, "embedding", None) is not None else None,
                    "body_signature": body_identity.get("signature"),
                    "body_weight": float(body_identity.get("weight", 0.0) or 0.0),
                    "body_ready_features": int(body_identity.get("ready_features", 0) or 0),
                    "depth_signature": depth_identity.get("signature"),
                    "depth_weight": float(depth_identity.get("weight", 0.0) or 0.0),
                    "depth_ready_features": int(depth_identity.get("ready_features", 0) or 0),
                    "face_conf": float(face_conf),
                    "face_score": float(face_score),
                    "bbox_area_ratio": float(cand_face.bbox_area_ratio if cand_face.ok else 0.0),
                    "view_lane": view_lane,
                    "view_lane_detail": shadow_view_route.lane_detail,
                }
            )

            upper_score, upper_conf, upper_reasons, upper_debug = score_upper_against_anchor_set(
                runtime,
                cand_pose,
                anchors.upper_pose_feats,
                view_bucket=view_lane,
                view_lane_detail=shadow_view_route.lane_detail,
            )
            full_score, full_conf, full_reasons, full_debug = score_full_against_anchor_set(
                runtime,
                cand_pose,
                anchors.full_pose_feats,
                view_bucket=view_lane,
                view_lane_detail=shadow_view_route.lane_detail,
            )

            face_state, face_state_reasons = classify_module(
                runtime, face_score, face_conf, th["face_pass"], th["face_warn"], "face"
            )
            upper_state, upper_state_reasons = classify_module(
                runtime, upper_score, upper_conf, th["upper_pass"], th["upper_warn"], "upper"
            )
            full_state, full_state_reasons = classify_module(
                runtime, full_score, full_conf, th["full_pass"], th["full_warn"], "full"
            )

            scores = {"face": face_score, "upper": upper_score, "full": full_score}
            confs = {"face": face_conf, "upper": upper_conf, "full": full_conf}
            overall_score = fuse_overall(scores, confs, weights, scoring=config.consistency.score_fusion)

            if overall_score >= th["overall_pass"]:
                overall_state = "PASS"
            elif overall_score >= th["overall_warn"]:
                overall_state = "WARN"
            else:
                overall_state = "FAIL"

            hard_fail = False
            hard_warn = False
            strict_fail_min_conf = config.review.min_conf_for_strict_fail

            if reqs.get("face", False):
                if face_state == "FAIL" and face_conf >= strict_fail_min_conf:
                    hard_fail = True
                elif face_state != "PASS":
                    hard_warn = True

            if reqs.get("upper", False):
                if upper_state == "FAIL" and upper_conf >= strict_fail_min_conf:
                    hard_fail = True
                elif upper_state != "PASS":
                    hard_warn = True

            if reqs.get("full", False):
                if full_state == "FAIL" and full_conf >= strict_fail_min_conf:
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
                face_state_reasons
                + upper_state_reasons
                + full_state_reasons
                + face_reasons
                + upper_reasons
                + full_reasons
            )

            extra_flags: List[str] = []
            quality_debug: Dict[str, Any] = {}
            quality_thresholds = config.quality_thresholds

            if not cand_face.ok or face_conf < config.review.face_no_signal_conf_th:
                extra_flags.append("FACE_NO_RELIABLE_SIGNAL")
            else:
                qtol = get_quality_tolerances_by_face_size(cand_face.bbox_area_ratio, quality_thresholds)
                bucket = qtol["bucket"]
                bucket_stats = get_stats_for_bucket(quality_ref_stats, bucket)
                tone_bucket_stats = get_stats_for_bucket(tone_ref_stats, bucket)

                quality_debug["face_size_bucket"] = bucket
                quality_debug["bucket_quality_ref_count"] = bucket_stats.get("count", 0)
                quality_debug["bucket_tone_ref_count"] = tone_bucket_stats.get("count", 0)
                quality_debug["bucket_quality_tolerances"] = qtol

                if cand_face.lab_mean is not None:
                    cand_L = float(cand_face.lab_mean[0])
                    cand_a = float(cand_face.lab_mean[1])
                    cand_b = float(cand_face.lab_mean[2])
                    quality_debug["candidate_face_L"] = cand_L
                    quality_debug["candidate_face_ab"] = [cand_a, cand_b]
                    if cand_L < qtol["abs_luma_warn"]:
                        extra_flags.append("FACE_UNDEREXPOSED_DARK")

                    if len(tone_bucket_stats.get("L", [])) > 0:
                        tone_L_mean = float(np.mean(tone_bucket_stats["L"]))
                        quality_debug["tone_face_L_mean_bucket"] = tone_L_mean
                        if cand_L < (tone_L_mean - qtol["dark_delta_L"]):
                            extra_flags.append("FACE_DARKER_THAN_TONE_ANCHOR")
                        elif cand_L > (tone_L_mean + qtol["dark_delta_L"]):
                            extra_flags.append("FACE_BRIGHTER_THAN_TONE_ANCHOR")

                    if len(tone_bucket_stats.get("a", [])) > 0 and len(tone_bucket_stats.get("b", [])) > 0:
                        tone_a_mean = float(np.mean(tone_bucket_stats["a"]))
                        tone_b_mean = float(np.mean(tone_bucket_stats["b"]))
                        quality_debug["tone_face_ab_mean_bucket"] = [tone_a_mean, tone_b_mean]
                        quality_debug["tone_face_delta_ab_bucket"] = float(
                            np.linalg.norm(np.array([cand_a - tone_a_mean, cand_b - tone_b_mean], dtype=np.float32))
                        )

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

            reasons_all, final_status, overall_state, consistency_gate_debug = apply_consistency_soft_gate(
                runtime=runtime,
                reasons_all=reasons_all,
                final_status=final_status,
                overall_state=overall_state,
                constitution_metrics=constitution_metrics,
                skin_metrics=skin_metrics,
                depth_3d_metrics=depth_3d_metrics,
                view_bucket=view_lane,
            )

            hard_quality_flags = set(policy.get("hard_quality_flags", set()))
            soft_quality_flags = quality_thresholds.degrade_flags - hard_quality_flags
            hard_hits = sum(1 for reason in reasons_all if reason in hard_quality_flags)
            soft_hits = sum(1 for reason in reasons_all if reason in soft_quality_flags)
            soft_hit_limit = int(policy.get("soft_quality_hits_to_warn", 2))

            if final_status == "PASS":
                if hard_hits >= 1 or soft_hits >= soft_hit_limit:
                    final_status = "WARN"
                    overall_state = "WARN"

            if "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE" in reasons_all and final_status == "PASS":
                final_status = "WARN"
                overall_state = "WARN"

            skin_sample_risk = float(skin_metrics.get("sample_risk_score", 0.0) or 0.0)
            skin_lighting_risk = float(skin_metrics.get("lighting_risk_score", 0.0) or 0.0)
            skin_risk_policy = config.consistency.skin_risk
            if final_status == "PASS":
                if (
                    bool(policy.get("skin_sample_high_caps_pass", False))
                    and skin_sample_risk >= skin_risk_policy.sample_high_th
                ):
                    final_status = "WARN"
                    overall_state = "WARN"
                    reasons_all.append("BODY_GOLD_SKIN_SAMPLE_RISK_PASS_CAPPED")
                if (
                    bool(policy.get("skin_lighting_high_caps_pass", False))
                    and skin_lighting_risk >= skin_risk_policy.lighting_high_th
                ):
                    final_status = "WARN"
                    overall_state = "WARN"
                    reasons_all.append("BODY_GOLD_SKIN_LIGHTING_RISK_PASS_CAPPED")

            final_status, overall_state = _apply_profile_view_policy(
                target_profile=target_profile,
                policy=policy,
                view_lane=view_lane,
                final_status=final_status,
                overall_state=overall_state,
                reasons_all=reasons_all,
            )

            reasons_all = dedupe_keep_order(reasons_all)

            result_node = {
                "image": img_path.name,
                "task_profile": target_profile,
                "quota_bucket": policy.get("quota_bucket"),
                "collection": collection_meta,
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
                    "face": runtime.engines.face_mode,
                    "pose": runtime.engines.pose_mode,
                    "ssim_backend": "skimage" if SKIMAGE_SSIM_AVAILABLE else "ncc_fallback",
                },
                "debug": {
                    "face": face_debug,
                    "upper": upper_debug,
                    "full": full_debug,
                    "constitution_metrics": constitution_metrics,
                    "skin_metrics": skin_metrics,
                    "garment_metrics": garment_metrics,
                    "body_identity_signature": {
                        "ready_features": body_identity.get("ready_features"),
                        "feature_count": body_identity.get("feature_count"),
                        "weight": body_identity.get("weight"),
                    },
                    "depth_identity_signature": {
                        "ready_features": depth_identity.get("ready_features"),
                        "feature_count": depth_identity.get("feature_count"),
                        "weight": depth_identity.get("weight"),
                    },
                    "depth_3d_metrics": depth_3d_metrics,
                    "consistency_gate": consistency_gate_debug,
                    "collection_metadata": collection_meta,
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
                    "tone_anchor_pool_mode": policy.get("tone_anchor_pool"),
                    "quality_ref_stats": quality_debug,
                    "view_bucket": view_bucket,
                    "view_lane": view_lane,
                    "view_lane_detail": shadow_view_route.lane_detail,
                    "view_lane_detail_source": "shadow_router_v2",
                    "view_lane_detail_confidence": shadow_view_route.lane_detail_confidence,
                    "view_lane_strictness_score": shadow_view_route.lane_strictness_score,
                    "legacy_view_bucket": legacy_view_bucket,
                    "legacy_view_lane": legacy_view_lane,
                    "view_lane_source": "legacy_face_router" if legacy_view_lane != "unknown" else "shadow_router_v2_fallback",
                    "view_side": view_side,
                    "yaw_proxy": yaw_proxy,
                    "view_scoring_surface_requested": face_debug.get("view_surface_requested"),
                    "view_router_v2": shadow_view_route.to_json_dict(),
                    "view_router_v2_disagrees": shadow_view_route.lane != view_lane,
                    "identity_anchor_count_view": len(face_identity_anchors_view),
                    "input_shape": list(img.shape[:2]),
                },
            }

            result_node["recommendations"] = make_recommendations(runtime, result_node, target_profile)
            report_items.append(result_node)

            constitution_show = "NA" if constitution_score is None else f"{constitution_score:.3f}"
            skin_show = "NA" if skin_score is None else f"{skin_score:.3f}"
            depth_show = "NA" if depth_3d_score is None else f"{depth_3d_score:.3f}"

            if final_status == "PASS":
                shutil.copy2(img_path, config.paths.dir_out_pass / img_path.name)
                print(
                    f"   PASS | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}->{view_lane}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            elif final_status == "WARN":
                shutil.copy2(img_path, config.paths.dir_out_warn / img_path.name)
                print(
                    f"   WARN | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}->{view_lane}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            else:
                shutil.copy2(img_path, config.paths.dir_out_fail / img_path.name)
                print(
                    f"   FAIL | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}->{view_lane}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )

        except Exception as exc:
            print(f"   FAIL (异常): {exc}")
            traceback.print_exc()

            report_items.append(
                {
                    "image": img_path.name,
                    "task_profile": target_profile,
                    "quota_bucket": policy.get("quota_bucket"),
                    "collection": collection_meta,
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
                    "reasons": ["RUNTIME_EXCEPTION", str(exc)],
                    "reasons_face": [],
                    "reasons_upper": [],
                    "reasons_full": [],
                    "recommendations": ["检查依赖、图片可读性、模型环境与日志"],
                    "engine": {"face": runtime.engines.face_mode, "pose": runtime.engines.pose_mode},
                    "debug": {
                        "collection_metadata": collection_meta,
                        "constitution_metrics": None,
                        "skin_metrics": None,
                        "garment_metrics": None,
                        "depth_3d_metrics": None,
                        "consistency_gate": None,
                    },
                }
            )

            try:
                shutil.copy2(img_path, config.paths.dir_out_fail / img_path.name)
            except Exception:
                pass

    config.paths.dir_output.mkdir(parents=True, exist_ok=True)
    collection_aggregates = build_collection_aggregates(report_items, target_profile=target_profile)
    collection_aggregates = apply_collection_diagnostics(
        report_items,
        collection_aggregates,
        target_profile=target_profile,
        identity_samples=batch_identity_samples,
    )
    shot_selection = build_shot_selection_report(
        report_items,
        collection_aggregates,
        target_profile=target_profile,
    )
    batch_gate = evaluate_active_batch_gate(
        collection_aggregates,
        policy,
        target_profile=target_profile,
    )
    if bool(batch_gate.get("applied")) and str(batch_gate.get("status")) != "pass":
        gate_reasons = list(batch_gate.get("reasons") or [])
        gate_mode = str(batch_gate.get("mode") or "warn_all_pass")
        for item in report_items:
            item_debug = item.setdefault("debug", {})
            item_debug["batch_gate"] = batch_gate
            item["reasons"] = dedupe_keep_order(list(item.get("reasons") or []) + gate_reasons)
            if gate_mode == "warn_all_pass" and item.get("status") == "PASS":
                item["status"] = "WARN"
                module_state = item.setdefault("module_state", {})
                module_state["overall"] = "WARN"
                item["reasons"] = dedupe_keep_order(list(item.get("reasons") or []) + ["PROFILE_BATCH_GATE_CAPPED_TO_WARN"])
    else:
        for item in report_items:
            item.setdefault("debug", {})["batch_gate"] = batch_gate
    for item in report_items:
        item["recommendations"] = make_recommendations(runtime, item, target_profile)
    report_payload = {
        "report_meta": report_meta,
        "collection_aggregates": collection_aggregates,
        "shot_selection": shot_selection,
        "items": report_items,
    }
    with open(config.paths.report_file, "w", encoding="utf-8") as file:
        json.dump(report_payload, file, indent=2, ensure_ascii=False)
    ranked_candidates_file = config.paths.dir_output / "ranked_candidates.json"
    with open(ranked_candidates_file, "w", encoding="utf-8") as file:
        json.dump(shot_selection, file, indent=2, ensure_ascii=False)

    print("\n[完工] 质检完成 [OK]")
    print(f"[报告] {config.paths.report_file}")
    print(f"[排序] {ranked_candidates_file}")
    collection_summary = collection_aggregates.get("summary", {})
    print(
        "[集合聚合] "
        f"groupable_items={collection_summary.get('groupable_items', 0)} "
        f"look_groups={collection_summary.get('look_group_count', 0)} "
        f"layer_groups={collection_summary.get('layer_group_count', 0)}"
    )
    print(
        f"[输出目录] PASS={config.paths.dir_out_pass} | WARN={config.paths.dir_out_warn} | FAIL={config.paths.dir_out_fail}"
    )


def main(
    base_dir: Optional[Path] = None,
    profile_name: Optional[str] = None,
    run_mode: Optional[str] = None,
    auto_load_thresholds: Optional[bool] = None,
    threshold_override: Optional[Dict[str, Any]] = None,
    benchmark_report_path: Optional[Path] = None,
    benchmark_labels_path: Optional[Path] = None,
    benchmark_output_path: Optional[Path] = None,
    benchmark_template_out: Optional[Path] = None,
    benchmark_dataset_role: Optional[str] = None,
    benchmark_optuna_ready: Optional[bool] = None,
    benchmark_id: Optional[str] = None,
    benchmark_freeze_tag: Optional[str] = None,
    benchmark_update_labels: bool = False,
) -> None:
    effective_run_mode = str(run_mode) if run_mode is not None else "qa"
    if effective_run_mode == "benchmark":
        config = create_runtime_config(base_dir)
        runtime = RuntimeContext(
            config=config,
            providers=None,
            engines=EngineState(face_mode="disabled", pose_mode="disabled"),
        )
        runtime.config.run_mode = "benchmark"
    else:
        runtime = create_runtime(base_dir)
        if run_mode is not None:
            runtime.config.run_mode = str(run_mode)
    if profile_name is not None:
        runtime.config.review.active_profile = str(profile_name)
    if auto_load_thresholds is not None:
        runtime.config.auto_load_thresholds = bool(auto_load_thresholds)
    print_runtime_config(runtime)

    if runtime.config.run_mode not in {"qa", "calibrate", "benchmark"}:
        raise ValueError("RUN_MODE 只能是 'qa'、'calibrate' 或 'benchmark'")

    if runtime.config.run_mode == "calibrate":
        print(f"[校准模式] 从目录读取样本: {runtime.config.paths.dir_calib}")
        thresholds = calibrate_quality_thresholds(runtime, runtime.config.paths.dir_calib)
        save_thresholds_to_file(thresholds, runtime.config.paths.thresh_file)
        print("\n[自动校准完成 ✅]")
        print(json.dumps(thresholds, indent=2, ensure_ascii=False))
        print(f"[阈值文件已保存] {runtime.config.paths.thresh_file}")
        return

    if runtime.config.run_mode == "benchmark":
        from .qa_benchmark import benchmark_report, export_benchmark_template, update_benchmark_label_metadata

        report_path = (benchmark_report_path or runtime.config.paths.report_file).resolve()
        if benchmark_template_out is not None:
            template = export_benchmark_template(
                report_path,
                benchmark_template_out.resolve(),
                dataset_role=benchmark_dataset_role or "candidate_review",
                optuna_ready=bool(benchmark_optuna_ready) if benchmark_optuna_ready is not None else False,
                benchmark_id=benchmark_id or "",
                freeze_tag=benchmark_freeze_tag or "",
            )
            print(f"[Benchmark 模板] {benchmark_template_out.resolve()}")
            print(json.dumps(template, indent=2, ensure_ascii=False))
            return
        if benchmark_update_labels:
            if benchmark_labels_path is None:
                raise ValueError("benchmark 标签更新需要 --benchmark-labels")
            payload = update_benchmark_label_metadata(
                benchmark_labels_path.resolve(),
                dataset_role=benchmark_dataset_role,
                optuna_ready=benchmark_optuna_ready,
                benchmark_id=benchmark_id,
                freeze_tag=benchmark_freeze_tag,
            )
            print(f"[Benchmark 标签已更新] {benchmark_labels_path.resolve()}")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        if benchmark_labels_path is None:
            raise ValueError("benchmark 模式需要 --benchmark-labels，或使用 --benchmark-template-out 导出模板")

        result = benchmark_report(
            runtime=runtime,
            report_path=report_path,
            labels_path=benchmark_labels_path.resolve(),
            threshold_override=threshold_override,
        )
        if benchmark_output_path is not None:
            benchmark_output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            benchmark_output_path.resolve().write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[Benchmark 输出] {benchmark_output_path.resolve()}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if runtime.config.auto_load_thresholds:
        load_thresholds_from_file(runtime, runtime.config.paths.thresh_file)
    run_pipeline(
        runtime,
        profile_name=runtime.config.review.active_profile,
        threshold_override=threshold_override,
    )
