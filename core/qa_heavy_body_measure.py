from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .providers import HeavyEvidenceProvider
from .qa_consistency import extract_body_constitution_metrics, extract_depth_3d_lite_metrics
from .qa_features import extract_face_feat, extract_pose_feat
from .qa_io import atomic_write_json
from .qa_master_consistency import (
    build_body_identity_signature,
    build_depth_identity_signature,
    build_world3d_identity_signature,
)
from .qa_utils import dedupe_keep_order, image_read_bgr
from .qa_view_router import route_view_lane

_PROVIDER_NAME = "body_measure_lite"
_PROVIDER_FAMILY = "body_measurement"
_PROVIDER_VERSION = "body_measure_lite_v1"
_MODEL_ID = "mediapipe_pose+body_measure_lite_v1"
_CACHE_SCHEMA = "body_measure_cache_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _weighted_mean(items: List[tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        if weight <= 0.0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 1e-8:
        return None
    return float(numerator / denominator)


def _signature_ref(name: str, signature: Any) -> Dict[str, Any]:
    vector = None
    try:
        if signature is not None:
            vector = np.asarray(signature, dtype=np.float32).reshape(-1)
    except Exception:
        vector = None
    return {
        "kind": "signature",
        "name": str(name),
        "available": vector is not None and vector.size > 0,
        "dimension": int(vector.shape[0]) if vector is not None and vector.size > 0 else 0,
    }


def _metric_spec(
    metric_name: str,
    metric_value: Any,
    *,
    confidence: Optional[float],
    coverage: Optional[float],
    signature_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "metric_name": str(metric_name),
        "metric_value": metric_value,
        "confidence": confidence,
        "coverage": coverage,
    }
    if isinstance(signature_ref, dict):
        spec["signature_ref"] = signature_ref
    return spec


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_view_bucket(route: Any) -> str:
    lane = str(getattr(route, "lane", "unknown") or "unknown")
    face_bucket = str(getattr(route, "face_bucket", "unknown") or "unknown")
    if face_bucket in {"front", "three_quarter", "side", "back"}:
        return face_bucket
    if lane == "front":
        return "front"
    if lane == "three_quarter":
        return "three_quarter"
    if lane == "side_90":
        return "side"
    if lane == "back_180":
        return "back"
    return "front"


def _cache_dir(runtime: Any) -> Path:
    cache_root = getattr(getattr(runtime.config, "paths", None), "dir_heavy_cache", Path("outputs") / "heavy_evidence_cache")
    cache_dir = Path(cache_root) / _PROVIDER_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _build_cache_key(runtime: Any, image_path: Path) -> tuple[str, Dict[str, Any]]:
    resolved = image_path.resolve()
    stat = resolved.stat()
    standardization = getattr(runtime.config, "standardization", None)
    payload = {
        "provider_name": _PROVIDER_NAME,
        "provider_version": _PROVIDER_VERSION,
        "source_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "face_mode": str(getattr(runtime.engines, "face_mode", "")),
        "pose_mode": str(getattr(runtime.engines, "pose_mode", "")),
        "standardization": {
            "enabled": bool(getattr(standardization, "enabled", True)),
            "long_side": int(getattr(standardization, "long_side", 0) or 0),
            "upscale_small_input": bool(getattr(standardization, "upscale_small_input", False)),
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest, payload


def _load_cached_metrics(runtime: Any, image_path: Path) -> tuple[Optional[Dict[str, Any]], str, Path]:
    cache_key, _ = _build_cache_key(runtime, image_path)
    cache_file = _cache_dir(runtime) / f"{cache_key}.json"
    if not cache_file.exists():
        return None, cache_key, cache_file
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None, cache_key, cache_file
    if str(payload.get("schema_version") or "") != _CACHE_SCHEMA:
        return None, cache_key, cache_file
    if str(payload.get("provider_version") or "") != _PROVIDER_VERSION:
        return None, cache_key, cache_file
    metrics = dict(payload.get("metrics") or {})
    metrics["cache_key"] = cache_key
    metrics["cache_file"] = str(cache_file)
    metrics["cache_state"] = "hit"
    summary = dict(metrics.get("summary") or {})
    summary["cache_hit_count"] = 1
    summary["cache_miss_count"] = 0
    summary["cache_write_count"] = 0
    metrics["summary"] = summary
    return metrics, cache_key, cache_file


def _write_cached_metrics(
    runtime: Any,
    image_path: Path,
    cache_key: str,
    cache_file: Path,
    metrics: Dict[str, Any],
) -> bool:
    del runtime, image_path
    payload = {
        "schema_version": _CACHE_SCHEMA,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "cache_key": cache_key,
        "metrics": _json_ready(metrics),
    }
    try:
        atomic_write_json(cache_file, payload)
        return True
    except Exception:
        return False


class BodyMeasureHeavyEvidenceProvider(HeavyEvidenceProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "cpu",
            "reason": None,
            "evidence_schema_version": "heavy_evidence_v1",
        }

    def get_heavy_evidence_metrics(
        self,
        runtime: Any,
        image_path: Path,
    ) -> Dict[str, Any]:
        resolved_path = Path(image_path).resolve()
        cached, cache_key, cache_file = _load_cached_metrics(runtime, resolved_path)
        if cached is not None:
            return cached
        img_bgr = image_read_bgr(resolved_path, runtime.config.standardization)
        if img_bgr is None:
            return {
                "ok": False,
                "provider_name": self.provider_name,
                "provider_family": self.provider_family,
                "provider_version": self.provider_version,
                "model_id": _MODEL_ID,
                "device": "cpu",
                "source_path": str(resolved_path),
                "cache_key": cache_key,
                "cache_file": str(cache_file),
                "cache_state": "miss",
                "reasons": ["BODY_MEASURE_IMAGE_READ_ERROR"],
                "metric_specs": [],
                "summary": {
                    "cache_hit_count": 0,
                    "cache_miss_count": 1,
                    "cache_write_count": 0,
                    "guidance": ["原图读取失败，body measure 证据不可用。"],
                },
            }

        face_feat = extract_face_feat(runtime, img_bgr, source_path=resolved_path)
        pose_feat = extract_pose_feat(runtime, img_bgr)
        route = route_view_lane(runtime, img_bgr, face_feat, pose_feat)
        view_bucket = _resolve_view_bucket(route)
        body_metrics = extract_body_constitution_metrics(
            runtime,
            img_bgr,
            face_feat,
            pose_feat,
            view_bucket=view_bucket,
            view_lane_detail=getattr(route, "lane_detail", None),
        )
        depth_metrics = extract_depth_3d_lite_metrics(
            face_feat,
            pose_feat,
            view_bucket=view_bucket,
            yaw_proxy=float(getattr(route, "face_yaw_proxy", 0.0) or 0.0),
            body_yaw_deg=getattr(route, "body_yaw_deg", None),
            pose_frontal_strength=getattr(route, "pose_frontal_strength", None),
            lane_strictness_score=getattr(route, "lane_strictness_score", None),
            mask_symmetry=getattr(route, "mask_symmetry", None),
            head_skin_ratio=getattr(route, "head_skin_ratio", None),
            scoring=getattr(runtime.config.consistency, "depth3d_scoring", None),
            view_lane_detail=getattr(route, "lane_detail", None),
        )

        body_identity = build_body_identity_signature(pose_feat, body_metrics, depth_metrics)
        depth_identity = build_depth_identity_signature(pose_feat, depth_metrics)
        world3d_identity = build_world3d_identity_signature(pose_feat)

        body_cov = _safe_float(body_identity.get("coverage"), 0.0) or 0.0
        depth_cov = _safe_float(depth_identity.get("coverage"), 0.0) or 0.0
        world3d_cov = _safe_float(world3d_identity.get("coverage"), 0.0) or 0.0
        body_conf = _safe_float(body_metrics.get("confidence"), 0.0) or 0.0
        depth_conf = _safe_float(depth_metrics.get("confidence"), 0.0) or 0.0
        route_conf = _safe_float(getattr(route, "confidence", 0.0), 0.0) or 0.0

        confidence = _weighted_mean(
            [
                (body_conf, 0.40),
                (depth_conf, 0.35),
                (route_conf, 0.15),
                (_safe_float(getattr(face_feat, "confidence", 0.0), 0.0) or 0.0, 0.10),
            ]
        )
        coverage = _weighted_mean(
            [
                (body_cov, 0.45),
                (depth_cov, 0.35),
                (world3d_cov, 0.20),
            ]
        )

        metric_specs = [
            _metric_spec(
                "body_constitution_score",
                body_metrics.get("body_constitution_score"),
                confidence=body_conf,
                coverage=body_cov,
                signature_ref=_signature_ref("body_identity_signature", body_identity.get("signature")),
            ),
            _metric_spec(
                "pelvis_compactness_score",
                body_metrics.get("pelvis_compactness_score"),
                confidence=body_conf,
                coverage=body_cov,
            ),
            _metric_spec(
                "abdomen_flatness_score",
                body_metrics.get("abdomen_flatness_score"),
                confidence=body_conf,
                coverage=body_cov,
            ),
            _metric_spec(
                "lower_body_slenderness_score",
                body_metrics.get("lower_body_slenderness_score"),
                confidence=body_conf,
                coverage=body_cov,
            ),
            _metric_spec(
                "depth_3d_score",
                depth_metrics.get("depth_3d_score"),
                confidence=depth_conf,
                coverage=depth_cov,
                signature_ref=_signature_ref("depth_identity_signature", depth_identity.get("signature")),
            ),
            _metric_spec(
                "torso_volume_score",
                depth_metrics.get("torso_volume_score"),
                confidence=depth_conf,
                coverage=depth_cov,
            ),
            _metric_spec(
                "turn_signal_score",
                depth_metrics.get("turn_signal_score"),
                confidence=depth_conf,
                coverage=depth_cov,
            ),
            _metric_spec(
                "posterior_score",
                depth_metrics.get("posterior_score"),
                confidence=depth_conf,
                coverage=depth_cov,
            ),
            _metric_spec(
                "torso_compactness_score",
                depth_metrics.get("torso_compactness_score"),
                confidence=depth_conf,
                coverage=depth_cov,
            ),
            _metric_spec(
                "world3d_feature_coverage",
                world3d_cov,
                confidence=_safe_float(world3d_identity.get("weight"), 0.0),
                coverage=world3d_cov,
                signature_ref=_signature_ref("world3d_identity_signature", world3d_identity.get("signature")),
            ),
            _metric_spec(
                "body_identity_weight",
                body_identity.get("weight"),
                confidence=body_conf,
                coverage=body_cov,
            ),
            _metric_spec(
                "depth_identity_weight",
                depth_identity.get("weight"),
                confidence=depth_conf,
                coverage=depth_cov,
            ),
            _metric_spec(
                "world3d_identity_weight",
                world3d_identity.get("weight"),
                confidence=_safe_float(world3d_identity.get("weight"), 0.0),
                coverage=world3d_cov,
            ),
        ]

        guidance: List[str] = []
        body_score = _safe_float(body_metrics.get("body_constitution_score"))
        depth_score = _safe_float(depth_metrics.get("depth_3d_score"))
        if body_score is not None and body_score < 0.78:
            guidance.append("体态比例和腰臀结构与完美小娜基准存在漂移，人工复核要重点看腰线、骨盆和腿部比例。")
        if depth_score is not None and depth_score < 0.76:
            guidance.append("3D 体积或转向信号偏弱，人工复核要重点看侧后轮廓、躯干厚度和站姿稳定性。")
        if world3d_cov < 0.45:
            guidance.append("world3d 骨架覆盖不足，这张图的空间结构证据较弱。")
        if len(guidance) == 0:
            guidance.append("body measure 证据稳定，可将其作为人工复核时的体态支撑读数。")

        reasons = dedupe_keep_order(
            list(face_feat.reasons)
            + list(pose_feat.reasons)
            + list(body_metrics.get("reasons") or [])
            + list(depth_metrics.get("reasons") or [])
        )

        metrics = {
            "ok": bool(body_metrics.get("is_valid")) or bool(depth_metrics.get("is_valid")) or world3d_cov >= 0.35,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "cpu",
            "source_path": str(resolved_path),
            "cache_key": cache_key,
            "cache_file": str(cache_file),
            "cache_state": "miss",
            "confidence": confidence,
            "coverage": coverage,
            "metric_specs": metric_specs,
            "summary": {
                "view_bucket": view_bucket,
                "view_lane": getattr(route, "lane", "unknown"),
                "view_lane_detail": getattr(route, "lane_detail", "unknown"),
                "body_constitution_score": body_metrics.get("body_constitution_score"),
                "depth_3d_score": depth_metrics.get("depth_3d_score"),
                "body_identity_coverage": body_cov,
                "depth_identity_coverage": depth_cov,
                "world3d_identity_coverage": world3d_cov,
                "body_identity_weight": body_identity.get("weight"),
                "depth_identity_weight": depth_identity.get("weight"),
                "world3d_identity_weight": world3d_identity.get("weight"),
                "primary_bottleneck": depth_metrics.get("primary_bottleneck"),
                "candidate_count": 1,
                "cache_hit_count": 0,
                "cache_miss_count": 1,
                "cache_write_count": 0,
                "guidance": guidance[:4],
            },
            "reasons": reasons[:12],
            "signature_refs": {
                "body_identity_signature": _signature_ref("body_identity_signature", body_identity.get("signature")),
                "depth_identity_signature": _signature_ref("depth_identity_signature", depth_identity.get("signature")),
                "world3d_identity_signature": _signature_ref("world3d_identity_signature", world3d_identity.get("signature")),
            },
        }
        if _write_cached_metrics(runtime, resolved_path, cache_key, cache_file, metrics):
            metrics["cache_state"] = "write"
            metrics["summary"]["cache_write_count"] = 1
        return metrics
