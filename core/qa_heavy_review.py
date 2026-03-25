from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .providers import HeavyEvidenceProvider
from .qa_utils import image_read_bgr

try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover - optional heavy dependency
    torch = None
    torch_f = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional heavy dependency
    Image = None

try:
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
except Exception:  # pragma: no cover - optional heavy dependency
    AutoModelForSemanticSegmentation = None
    SegformerImageProcessor = None


_HEAVY_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
_HEAVY_BUNDLE: Optional[Dict[str, Any]] = None
HEAVY_EVIDENCE_SCHEMA = "heavy_evidence_v1"
HEAVY_CACHE_SCHEMA = "heavy_parser_metrics_v1"
_HEAVY_PROVIDER_NAME = "segformer_parser"
_HEAVY_PROVIDER_FAMILY = "semantic_segmentation"
_HEAVY_PROVIDER_VERSION = _HEAVY_MODEL_ID

_GARMENT_LABELS = {
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Scarf",
}
_UPPER_LABELS = {
    "Upper-clothes",
    "Dress",
    "Belt",
    "Scarf",
}
_LOWER_LABELS = {
    "Skirt",
    "Pants",
    "Dress",
}
_VISIBLE_BODY_LABELS = {
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
}
_FACE_LABELS = {"Face"}
_ARM_LABELS = {"Left-arm", "Right-arm"}
_LEG_LABELS = {"Left-leg", "Right-leg"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _json_ready_heavy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready_heavy_value(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_heavy_value(node) for node in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        emb = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return None
    return emb / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


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


def _weighted_sum(items: Sequence[tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
    numerator: Optional[np.ndarray] = None
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        if numerator is None:
            numerator = np.zeros_like(value, dtype=np.float32)
        numerator += value.astype(np.float32) * float(weight)
        denominator += float(weight)
    if numerator is None or denominator <= 0.0:
        return None
    return numerator / float(denominator)


def _weighted_geometric_mean(items: Sequence[tuple[Optional[float], float]], floor: float = 1e-4) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        clipped = _clamp(float(value), floor, 1.0)
        numerator += float(weight) * float(np.log(clipped))
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(np.exp(numerator / denominator))


def _signature_summary(signature: Any, *, name: str) -> Dict[str, Any]:
    vector = _normalize_embedding(signature)
    return {
        "kind": "signature",
        "name": name,
        "available": vector is not None,
        "dimension": int(vector.shape[0]) if vector is not None else 0,
    }


def _metric_record(
    metric_name: str,
    metric_value: Any,
    *,
    confidence: Optional[float],
    coverage: Optional[float],
    lane_scope: str,
    failure_reason: Optional[str],
    provider_name: str,
    provider_family: str,
    provider_version: str,
    signature_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metric_numeric = _float_or_none(metric_value)
    return {
        "metric_name": str(metric_name),
        "metric_value": _round_or_none(metric_numeric),
        "confidence": _round_or_none(confidence),
        "coverage": _round_or_none(coverage),
        "provider_name": str(provider_name),
        "provider_family": str(provider_family),
        "provider_version": str(provider_version),
        "lane_scope": str(lane_scope),
        "failure_reason": str(failure_reason or "").strip() or None,
        "feature_vector_or_signature": signature_ref or {
            "kind": "scalar",
            "name": str(metric_name),
            "available": metric_numeric is not None,
        },
    }


def build_heavy_evidence_bundle(
    metrics: Dict[str, Any],
    *,
    lane_scope: str,
    advisory_only: bool = True,
    mode: str = "shortlist_only_advisory",
    record_key: str = "",
    image: str = "",
) -> Dict[str, Any]:
    raw = metrics if isinstance(metrics, dict) else {}
    provider_name = str(raw.get("provider_name") or _HEAVY_PROVIDER_NAME)
    provider_family = str(raw.get("provider_family") or _HEAVY_PROVIDER_FAMILY)
    provider_version = str(raw.get("provider_version") or _HEAVY_PROVIDER_VERSION)
    reasons = [str(reason) for reason in raw.get("reasons", []) if str(reason).strip()]
    failure_reason = reasons[0] if len(reasons) > 0 else None
    confidence = _float_or_none(raw.get("confidence"))
    coverage = _float_or_none(raw.get("coverage"))
    metric_specs = raw.get("metric_specs")
    boundary_signature_ref = _signature_summary(raw.get("boundary_signature"), name="boundary_signature")
    visible_signature_ref = _signature_summary(raw.get("visible_body_signature"), name="visible_body_signature")
    model_id = raw.get("model_id")
    if model_id is None and provider_name == _HEAVY_PROVIDER_NAME:
        model_id = _HEAVY_MODEL_ID

    signature_refs: Dict[str, Any]
    summary_node: Dict[str, Any]
    if isinstance(metric_specs, list) and len(metric_specs) > 0:
        metric_rows = []
        signature_refs = dict(raw.get("signature_refs") or {}) if isinstance(raw.get("signature_refs"), dict) else {}
        coverage_values: List[float] = []
        for spec in metric_specs:
            if not isinstance(spec, dict):
                continue
            metric_name = str(spec.get("metric_name") or "").strip()
            if not metric_name:
                continue
            metric_value = spec.get("metric_value")
            metric_conf = _float_or_none(spec.get("confidence"))
            if metric_conf is None:
                metric_conf = confidence
            metric_cov = _float_or_none(spec.get("coverage"))
            if metric_cov is None:
                metric_cov = 1.0 if _float_or_none(metric_value) is not None else 0.0
            signature_ref = spec.get("signature_ref") if isinstance(spec.get("signature_ref"), dict) else None
            metric_rows.append(
                _metric_record(
                    metric_name,
                    metric_value,
                    confidence=metric_conf,
                    coverage=metric_cov,
                    lane_scope=str(spec.get("lane_scope") or lane_scope),
                    failure_reason=str(spec.get("failure_reason") or failure_reason or "").strip() or None,
                    provider_name=str(spec.get("provider_name") or provider_name),
                    provider_family=str(spec.get("provider_family") or provider_family),
                    provider_version=str(spec.get("provider_version") or provider_version),
                    signature_ref=signature_ref,
                )
            )
            coverage_values.append(float(metric_cov))
            if signature_ref is not None:
                signature_refs.setdefault(metric_name, signature_ref)
        if coverage is None:
            coverage = float(sum(coverage_values) / max(1, len(coverage_values))) if coverage_values else 0.0
        summary_node = dict(raw.get("summary") or {}) if isinstance(raw.get("summary"), dict) else {}
        if len(summary_node) == 0:
            summary_node = {
                "candidate_count": raw.get("candidate_count"),
                "guidance": list(raw.get("guidance") or [])[:4],
            }
    else:
        core_metric_names = [
            "parser_boundary_alignment",
            "parser_visible_body_alignment",
            "parser_consensus_score",
            "enhanced_selection_score",
            "garment_coverage_ratio",
            "upper_cloth_coverage",
            "lower_cloth_coverage",
            "neckline_depth_ratio",
            "shoulder_cloth_balance",
            "visible_body_ratio",
            "visible_face_ratio",
            "visible_arm_ratio",
            "visible_leg_ratio",
        ]
        available_metric_count = sum(1 for key in core_metric_names if _float_or_none(raw.get(key)) is not None)
        if coverage is None:
            coverage = float(available_metric_count / max(1, len(core_metric_names)))
        metric_rows = [
            _metric_record(
                "parser_confidence",
                confidence,
                confidence=confidence,
                coverage=1.0 if confidence is not None else 0.0,
                lane_scope=lane_scope,
                failure_reason=failure_reason,
                provider_name=provider_name,
                provider_family=provider_family,
                provider_version=provider_version,
            ),
            _metric_record(
                "parser_boundary_alignment",
                raw.get("parser_boundary_alignment"),
                confidence=confidence,
                coverage=1.0 if _float_or_none(raw.get("parser_boundary_alignment")) is not None else 0.0,
                lane_scope=lane_scope,
                failure_reason=failure_reason,
                provider_name=provider_name,
                provider_family=provider_family,
                provider_version=provider_version,
                signature_ref=boundary_signature_ref,
            ),
            _metric_record(
                "parser_visible_body_alignment",
                raw.get("parser_visible_body_alignment"),
                confidence=confidence,
                coverage=1.0 if _float_or_none(raw.get("parser_visible_body_alignment")) is not None else 0.0,
                lane_scope=lane_scope,
                failure_reason=failure_reason,
                provider_name=provider_name,
                provider_family=provider_family,
                provider_version=provider_version,
                signature_ref=visible_signature_ref,
            ),
            _metric_record(
                "parser_consensus_score",
                raw.get("parser_consensus_score"),
                confidence=confidence,
                coverage=1.0 if _float_or_none(raw.get("parser_consensus_score")) is not None else 0.0,
                lane_scope=lane_scope,
                failure_reason=failure_reason,
                provider_name=provider_name,
                provider_family=provider_family,
                provider_version=provider_version,
            ),
            _metric_record(
                "enhanced_selection_score",
                raw.get("enhanced_selection_score"),
                confidence=confidence,
                coverage=1.0 if _float_or_none(raw.get("enhanced_selection_score")) is not None else 0.0,
                lane_scope=lane_scope,
                failure_reason=failure_reason,
                provider_name=provider_name,
                provider_family=provider_family,
                provider_version=provider_version,
            ),
        ]
        for metric_name in [
            "garment_coverage_ratio",
            "upper_cloth_coverage",
            "lower_cloth_coverage",
            "neckline_depth_ratio",
            "shoulder_cloth_balance",
            "visible_body_ratio",
            "visible_face_ratio",
            "visible_arm_ratio",
            "visible_leg_ratio",
            "hem_depth_ratio",
        ]:
            metric_rows.append(
                _metric_record(
                    metric_name,
                    raw.get(metric_name),
                    confidence=confidence,
                    coverage=1.0 if _float_or_none(raw.get(metric_name)) is not None else 0.0,
                    lane_scope=lane_scope,
                    failure_reason=failure_reason,
                    provider_name=provider_name,
                    provider_family=provider_family,
                    provider_version=provider_version,
                )
            )
        signature_refs = {
            "boundary_signature": boundary_signature_ref,
            "visible_body_signature": visible_signature_ref,
        }
        summary_node = {
            "parser_consensus_score": _round_or_none(_float_or_none(raw.get("parser_consensus_score"))),
            "enhanced_selection_score": _round_or_none(_float_or_none(raw.get("enhanced_selection_score"))),
            "garment_coverage_ratio": _round_or_none(_float_or_none(raw.get("garment_coverage_ratio"))),
            "upper_cloth_coverage": _round_or_none(_float_or_none(raw.get("upper_cloth_coverage"))),
            "lower_cloth_coverage": _round_or_none(_float_or_none(raw.get("lower_cloth_coverage"))),
            "neckline_depth_ratio": _round_or_none(_float_or_none(raw.get("neckline_depth_ratio"))),
            "shoulder_cloth_balance": _round_or_none(_float_or_none(raw.get("shoulder_cloth_balance"))),
            "visible_body_ratio": _round_or_none(_float_or_none(raw.get("visible_body_ratio"))),
            "visible_face_ratio": _round_or_none(_float_or_none(raw.get("visible_face_ratio"))),
            "visible_arm_ratio": _round_or_none(_float_or_none(raw.get("visible_arm_ratio"))),
            "visible_leg_ratio": _round_or_none(_float_or_none(raw.get("visible_leg_ratio"))),
            "rank_in_heavy_review": raw.get("rank_in_heavy_review"),
            "consensus_top_image": raw.get("consensus_top_image"),
            "candidate_count": raw.get("candidate_count"),
            "cache_hit_count": raw.get("cache_hit_count"),
            "cache_miss_count": raw.get("cache_miss_count"),
            "cache_write_count": raw.get("cache_write_count"),
            "guidance": list(raw.get("guidance") or [])[:4],
        }

    if coverage is None:
        coverage = 0.0

    available = bool(raw.get("ok")) or confidence is not None or len(reasons) == 0
    return {
        "schema_version": HEAVY_EVIDENCE_SCHEMA,
        "provider_name": provider_name,
        "provider_family": provider_family,
        "provider_version": provider_version,
        "model_id": model_id,
        "device": raw.get("device"),
        "mode": str(mode),
        "advisory_only": bool(advisory_only),
        "lane_scope": str(lane_scope),
        "record_key": str(record_key or raw.get("record_key") or "").strip(),
        "image": str(image or raw.get("image") or "").strip(),
        "source_path": str(raw.get("source_path") or "").strip() or None,
        "cache_key": str(raw.get("cache_key") or "").strip() or None,
        "cache_file": str(raw.get("cache_file") or "").strip() or None,
        "cache_state": str(raw.get("cache_state") or "").strip() or None,
        "available": bool(available),
        "failure_reason": str(failure_reason or "").strip() or None,
        "confidence": _round_or_none(confidence),
        "coverage": _round_or_none(coverage),
        "metrics": metric_rows,
        "signature_refs": signature_refs,
        "summary": summary_node,
        "reasons": reasons[:8],
    }


def normalize_heavy_evidence_bundle(node: Any) -> Dict[str, Any]:
    if isinstance(node, dict) and str(node.get("schema_version", "")).strip() == HEAVY_EVIDENCE_SCHEMA:
        return {
            **node,
            "provider_name": str(node.get("provider_name") or _HEAVY_PROVIDER_NAME),
            "provider_family": str(node.get("provider_family") or _HEAVY_PROVIDER_FAMILY),
            "provider_version": str(node.get("provider_version") or _HEAVY_PROVIDER_VERSION),
            "mode": str(node.get("mode") or "shortlist_only_advisory"),
            "advisory_only": bool(node.get("advisory_only", True)),
            "source_path": str(node.get("source_path") or "").strip() or None,
            "cache_key": str(node.get("cache_key") or "").strip() or None,
            "cache_file": str(node.get("cache_file") or "").strip() or None,
            "cache_state": str(node.get("cache_state") or "").strip() or None,
            "metrics": list(node.get("metrics") or []),
            "summary": dict(node.get("summary") or {}),
            "signature_refs": dict(node.get("signature_refs") or {}),
            "reasons": [str(reason) for reason in node.get("reasons", []) if str(reason).strip()],
        }
    if not isinstance(node, dict):
        return build_heavy_evidence_bundle(
            {"ok": False, "reasons": ["HEAVY_EVIDENCE_MISSING"]},
            lane_scope="report_item",
        )
    return build_heavy_evidence_bundle(node, lane_scope=str(node.get("lane_scope") or "report_item"))


class SegformerHeavyEvidenceProvider(HeavyEvidenceProvider):
    provider_name = _HEAVY_PROVIDER_NAME
    provider_family = _HEAVY_PROVIDER_FAMILY
    provider_version = _HEAVY_PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        bundle = _load_heavy_bundle()
        return {
            "enabled": bool(bundle.get("available")),
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": bundle.get("model_id") or _HEAVY_MODEL_ID,
            "device": bundle.get("device"),
            "reason": bundle.get("reason"),
            "cache_schema_version": HEAVY_CACHE_SCHEMA,
            "evidence_schema_version": HEAVY_EVIDENCE_SCHEMA,
        }

    def get_heavy_evidence_metrics(
        self,
        runtime: Any,
        image_path: Path,
    ) -> Dict[str, Any]:
        bundle = _load_heavy_bundle()
        resolved_path = Path(image_path).resolve()
        if not bool(bundle.get("available")):
            return {
                "ok": False,
                "provider_name": self.provider_name,
                "provider_family": self.provider_family,
                "provider_version": self.provider_version,
                "model_id": bundle.get("model_id") or _HEAVY_MODEL_ID,
                "device": bundle.get("device"),
                "source_path": str(resolved_path),
                "reasons": [f"HEAVY_REVIEW_UNAVAILABLE:{bundle.get('reason') or 'unavailable'}"],
            }
        metrics = _extract_parser_metrics_cached(runtime, resolved_path)
        metrics.setdefault("provider_name", self.provider_name)
        metrics.setdefault("provider_family", self.provider_family)
        metrics.setdefault("provider_version", self.provider_version)
        metrics.setdefault("model_id", bundle.get("model_id") or _HEAVY_MODEL_ID)
        metrics.setdefault("device", bundle.get("device"))
        metrics.setdefault("source_path", str(resolved_path))
        return metrics


def _load_heavy_bundle() -> Dict[str, Any]:
    global _HEAVY_BUNDLE
    if _HEAVY_BUNDLE is not None:
        return _HEAVY_BUNDLE

    bundle: Dict[str, Any] = {
        "available": False,
        "model_id": _HEAVY_MODEL_ID,
        "reason": None,
        "device": "cpu",
        "processor": None,
        "model": None,
        "id2label": {},
    }
    if torch is None or torch_f is None or Image is None or SegformerImageProcessor is None or AutoModelForSemanticSegmentation is None:
        bundle["reason"] = "missing_dependencies"
        _HEAVY_BUNDLE = bundle
        return bundle

    try:
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        processor = SegformerImageProcessor.from_pretrained(_HEAVY_MODEL_ID, local_files_only=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(_HEAVY_MODEL_ID, local_files_only=True)
        model.to(device)
        model.eval()
        bundle.update(
            {
                "available": True,
                "device": device,
                "processor": processor,
                "model": model,
                "id2label": dict(getattr(model.config, "id2label", {}) or {}),
            }
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        bundle["reason"] = f"model_load_failed:{exc}"
    _HEAVY_BUNDLE = bundle
    return bundle


def _heavy_cache_dir(runtime: Any) -> Path:
    config = getattr(runtime, "config", None)
    paths = getattr(config, "paths", None)
    cache_dir = getattr(paths, "dir_heavy_cache", None)
    if cache_dir is None:
        output_dir = getattr(paths, "dir_output", Path.cwd() / "outputs")
        cache_dir = Path(output_dir) / "heavy_evidence_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _heavy_standardization_config(runtime: Any) -> Dict[str, Any]:
    standardization = getattr(getattr(runtime, "config", None), "standardization", None)
    return {
        "enabled": bool(getattr(standardization, "enabled", True)),
        "long_side": int(getattr(standardization, "long_side", 0) or 0),
        "upscale_small_input": bool(getattr(standardization, "upscale_small_input", False)),
    }


def _build_heavy_cache_key(runtime: Any, image_path: Path) -> tuple[str, Dict[str, Any]]:
    resolved = image_path.resolve()
    stat = resolved.stat()
    standardization = _heavy_standardization_config(runtime)
    payload = {
        "provider_name": _HEAVY_PROVIDER_NAME,
        "provider_version": _HEAVY_PROVIDER_VERSION,
        "source_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "standardization": standardization,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest, payload


def _serialize_parser_metrics(
    metrics: Dict[str, Any],
    *,
    image_path: Path,
    cache_key: str,
    runtime: Any,
) -> Dict[str, Any]:
    source_meta = _build_heavy_cache_key(runtime, image_path)[1]
    return {
        "schema_version": HEAVY_CACHE_SCHEMA,
        "provider_name": _HEAVY_PROVIDER_NAME,
        "provider_family": _HEAVY_PROVIDER_FAMILY,
        "provider_version": _HEAVY_PROVIDER_VERSION,
        "cache_key": cache_key,
        "source": source_meta,
        "metrics": _json_ready_heavy_value(metrics),
    }


def _restore_parser_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    metrics["boundary_signature"] = _normalize_embedding(metrics.get("boundary_signature"))
    metrics["visible_body_signature"] = _normalize_embedding(metrics.get("visible_body_signature"))
    metrics["cache_key"] = str(payload.get("cache_key") or "").strip() or None
    metrics["source_path"] = str(((payload.get("source") or {}).get("source_path")) or "").strip() or None
    return metrics


def _load_cached_parser_metrics(runtime: Any, image_path: Path) -> tuple[Optional[Dict[str, Any]], str, Path]:
    cache_key, _ = _build_heavy_cache_key(runtime, image_path)
    cache_file = _heavy_cache_dir(runtime) / f"{cache_key}.json"
    if not cache_file.exists():
        return None, cache_key, cache_file
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None, cache_key, cache_file
    if str(payload.get("schema_version") or "") != HEAVY_CACHE_SCHEMA:
        return None, cache_key, cache_file
    if str(payload.get("provider_version") or "") != _HEAVY_PROVIDER_VERSION:
        return None, cache_key, cache_file
    metrics = _restore_parser_metrics(payload)
    metrics["cache_key"] = cache_key
    metrics["cache_file"] = str(cache_file)
    metrics["cache_state"] = "hit"
    return metrics, cache_key, cache_file


def _write_cached_parser_metrics(
    runtime: Any,
    image_path: Path,
    cache_key: str,
    cache_file: Path,
    metrics: Dict[str, Any],
) -> bool:
    try:
        payload = _serialize_parser_metrics(metrics, image_path=image_path, cache_key=cache_key, runtime=runtime)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False


def _extract_parser_metrics_cached(runtime: Any, image_path: Path) -> Dict[str, Any]:
    cached, cache_key, cache_file = _load_cached_parser_metrics(runtime, image_path)
    if cached is not None:
        return cached
    img = image_read_bgr(image_path, runtime.config.standardization)
    if img is None:
        return {
            "ok": False,
            "model_id": _HEAVY_MODEL_ID,
            "device": None,
            "reasons": ["HEAVY_REVIEW_IMAGE_READ_ERROR"],
            "cache_key": cache_key,
            "cache_file": str(cache_file),
            "cache_state": "miss",
            "source_path": str(image_path.resolve()),
        }
    metrics = _extract_parser_metrics(img)
    metrics["cache_key"] = cache_key
    metrics["cache_file"] = str(cache_file)
    metrics["cache_state"] = "miss"
    metrics["source_path"] = str(image_path.resolve())
    if bool(metrics.get("ok")) and _write_cached_parser_metrics(runtime, image_path, cache_key, cache_file, metrics):
        metrics["cache_state"] = "write"
    return metrics


def _mask_ratio(mask: np.ndarray, roi: Tuple[int, int, int, int], subject_mask: np.ndarray) -> Optional[float]:
    x1, y1, x2, y2 = roi
    subject_crop = subject_mask[y1:y2, x1:x2]
    if subject_crop.size == 0:
        return None
    subject_pixels = int(np.count_nonzero(subject_crop))
    if subject_pixels <= 0:
        return None
    mask_crop = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(mask_crop) / max(1, subject_pixels))


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w <= 1 or h <= 1:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def _clip_roi(roi: Tuple[float, float, float, float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    x1 = int(max(0, min(width - 1, round(float(x1)))))
    y1 = int(max(0, min(height - 1, round(float(y1)))))
    x2 = int(max(1, min(width, round(float(x2)))))
    y2 = int(max(1, min(height, round(float(y2)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _label_mask(seg_map: np.ndarray, id2label: Dict[int, str], label_names: set[str]) -> np.ndarray:
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for label_id, label_name in id2label.items():
        if str(label_name) in label_names:
            mask[seg_map == int(label_id)] = 1
    return mask


def _first_and_last_rows(mask: np.ndarray, x1: int, x2: int) -> tuple[Optional[int], Optional[int]]:
    if x2 <= x1:
        return None, None
    crop = mask[:, x1:x2]
    if crop.size == 0:
        return None, None
    rows = np.where(np.count_nonzero(crop, axis=1) > 0)[0]
    if rows.size == 0:
        return None, None
    return int(rows[0]), int(rows[-1])


def _extract_parser_metrics(img_bgr: np.ndarray) -> Dict[str, Any]:
    bundle = _load_heavy_bundle()
    out: Dict[str, Any] = {
        "ok": False,
        "model_id": bundle.get("model_id"),
        "device": bundle.get("device"),
        "confidence": 0.0,
        "parser_boundary_alignment": None,
        "parser_visible_body_alignment": None,
        "garment_coverage_ratio": None,
        "upper_cloth_coverage": None,
        "lower_cloth_coverage": None,
        "neckline_depth_ratio": None,
        "shoulder_cloth_balance": None,
        "visible_body_ratio": None,
        "visible_face_ratio": None,
        "visible_arm_ratio": None,
        "visible_leg_ratio": None,
        "hem_depth_ratio": None,
        "boundary_signature": None,
        "visible_body_signature": None,
        "reasons": [],
    }
    if not bundle.get("available"):
        out["reasons"] = [f"HEAVY_REVIEW_UNAVAILABLE:{bundle.get('reason') or 'disabled'}"]
        return out

    processor = bundle["processor"]
    model = bundle["model"]
    id2label = bundle["id2label"]
    height, width = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    with torch.inference_mode():
        inputs = processor(images=pil_image, return_tensors="pt")
        device = str(bundle.get("device") or "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits
        logits = torch_f.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.softmax(logits, dim=1)
        conf_map, label_map = torch.max(probs, dim=1)

    seg_map = label_map[0].detach().cpu().numpy().astype(np.int32)
    conf_map_np = conf_map[0].detach().cpu().numpy().astype(np.float32)
    subject_mask = np.where(seg_map != 0, 1, 0).astype(np.uint8)
    subject_bbox = _bbox_from_mask(subject_mask)
    if subject_bbox is None:
        out["reasons"] = ["HEAVY_REVIEW_SUBJECT_MISSING"]
        return out

    garment_mask = _label_mask(seg_map, id2label, _GARMENT_LABELS)
    upper_mask = _label_mask(seg_map, id2label, _UPPER_LABELS)
    lower_mask = _label_mask(seg_map, id2label, _LOWER_LABELS)
    visible_body_mask = _label_mask(seg_map, id2label, _VISIBLE_BODY_LABELS)
    face_mask = _label_mask(seg_map, id2label, _FACE_LABELS)
    arm_mask = _label_mask(seg_map, id2label, _ARM_LABELS)
    leg_mask = _label_mask(seg_map, id2label, _LEG_LABELS)

    x1, y1, x2, y2 = subject_bbox
    box_w = float(x2 - x1)
    box_h = float(y2 - y1)
    subject_pixels = int(np.count_nonzero(subject_mask))
    if subject_pixels <= 0:
        out["reasons"] = ["HEAVY_REVIEW_SUBJECT_EMPTY"]
        return out

    torso_roi = _clip_roi((x1 + box_w * 0.18, y1 + box_h * 0.18, x2 - box_w * 0.18, y1 + box_h * 0.60), width, height)
    lower_roi = _clip_roi((x1 + box_w * 0.22, y1 + box_h * 0.58, x2 - box_w * 0.22, y2 - box_h * 0.04), width, height)
    shoulder_left_roi = _clip_roi((x1 + box_w * 0.06, y1 + box_h * 0.10, x1 + box_w * 0.38, y1 + box_h * 0.28), width, height)
    shoulder_right_roi = _clip_roi((x2 - box_w * 0.38, y1 + box_h * 0.10, x2 - box_w * 0.06, y1 + box_h * 0.28), width, height)

    garment_coverage_ratio = float(np.count_nonzero(garment_mask) / max(1, subject_pixels))
    upper_cloth_coverage = _mask_ratio(upper_mask, torso_roi, subject_mask) if torso_roi is not None else None
    lower_cloth_coverage = _mask_ratio(lower_mask, lower_roi, subject_mask) if lower_roi is not None else None
    visible_body_ratio = float(np.count_nonzero(visible_body_mask) / max(1, subject_pixels))
    visible_face_ratio = float(np.count_nonzero(face_mask) / max(1, subject_pixels))
    visible_arm_ratio = float(np.count_nonzero(arm_mask) / max(1, subject_pixels))
    visible_leg_ratio = float(np.count_nonzero(leg_mask) / max(1, subject_pixels))

    left_upper = _mask_ratio(upper_mask, shoulder_left_roi, subject_mask) if shoulder_left_roi is not None else None
    right_upper = _mask_ratio(upper_mask, shoulder_right_roi, subject_mask) if shoulder_right_roi is not None else None
    shoulder_cloth_balance = None
    if left_upper is not None and right_upper is not None:
        hi = max(float(left_upper), float(right_upper))
        if hi > 1e-6:
            shoulder_cloth_balance = float(min(float(left_upper), float(right_upper)) / hi)

    band_x1 = int(x1 + box_w * 0.38)
    band_x2 = int(x2 - box_w * 0.38)
    face_y1, face_y2 = None, None
    face_bbox = _bbox_from_mask(face_mask)
    if face_bbox is not None:
        _, face_y1, _, face_y2 = face_bbox
    first_garment_row, last_garment_row = _first_and_last_rows(upper_mask | lower_mask, band_x1, band_x2)
    neckline_depth_ratio = None
    if face_y2 is not None and first_garment_row is not None and box_h > 1e-6:
        neckline_gap = max(0.0, float(first_garment_row - face_y2))
        neckline_depth_ratio = _clamp(neckline_gap / max(1.0, box_h * 0.28), 0.0, 1.0)
    hem_depth_ratio = None
    if last_garment_row is not None and box_h > 1e-6:
        hem_depth_ratio = _clamp((float(last_garment_row) - float(y1)) / max(1.0, box_h), 0.0, 1.0)

    boundary_vector = np.asarray(
        [
            (garment_coverage_ratio - 0.82) / 0.18,
            ((upper_cloth_coverage or 0.0) - 0.80) / 0.18,
            ((lower_cloth_coverage or 0.0) - 0.94) / 0.12,
            ((neckline_depth_ratio or 0.0) - 0.06) / 0.12,
            ((shoulder_cloth_balance or 0.0) - 0.94) / 0.10,
            ((hem_depth_ratio or 0.0) - 0.88) / 0.14,
        ],
        dtype=np.float32,
    )
    visible_vector = np.asarray(
        [
            (visible_body_ratio - 0.16) / 0.16,
            (visible_face_ratio - 0.06) / 0.05,
            (visible_arm_ratio - 0.06) / 0.06,
            (visible_leg_ratio - 0.06) / 0.06,
            ((shoulder_cloth_balance or 0.0) - 0.94) / 0.10,
        ],
        dtype=np.float32,
    )
    boundary_signature = _normalize_embedding(boundary_vector)
    visible_body_signature = _normalize_embedding(visible_vector)

    subject_conf = float(np.mean(conf_map_np[subject_mask > 0])) if np.count_nonzero(subject_mask) > 0 else 0.0
    garment_conf = float(np.mean(conf_map_np[garment_mask > 0])) if np.count_nonzero(garment_mask) > 0 else subject_conf
    out.update(
        {
            "ok": True,
            "confidence": _clamp(_weighted_mean([(subject_conf, 0.45), (garment_conf, 0.55)]) or 0.0, 0.0, 1.0),
            "garment_coverage_ratio": garment_coverage_ratio,
            "upper_cloth_coverage": upper_cloth_coverage,
            "lower_cloth_coverage": lower_cloth_coverage,
            "neckline_depth_ratio": neckline_depth_ratio,
            "shoulder_cloth_balance": shoulder_cloth_balance,
            "visible_body_ratio": visible_body_ratio,
            "visible_face_ratio": visible_face_ratio,
            "visible_arm_ratio": visible_arm_ratio,
            "visible_leg_ratio": visible_leg_ratio,
            "hem_depth_ratio": hem_depth_ratio,
            "boundary_signature": boundary_signature,
            "visible_body_signature": visible_body_signature,
        }
    )
    if out["confidence"] < 0.58:
        out["reasons"].append("HEAVY_REVIEW_CONFIDENCE_LOW")
    return out


def _resolve_candidate_path(runtime: Any, row: Dict[str, Any]) -> Optional[Path]:
    input_dir = getattr(getattr(runtime, "config", None), "paths", None)
    input_dir = getattr(input_dir, "dir_input", None)
    record_key = str(row.get("record_key") or "").strip()
    image_name = str(row.get("image") or "").strip()
    candidates = []
    if input_dir is not None and record_key:
        candidates.append(Path(input_dir) / record_key)
    if input_dir is not None and image_name:
        candidates.append(Path(input_dir) / image_name)
    for path in candidates:
        if path.exists():
            return path
    return None


def apply_shortlist_heavy_review(
    runtime: Any,
    report_items: Sequence[Dict[str, Any]],
    shot_selection: Dict[str, Any],
    target_profile: Optional[str] = None,
    max_candidates: int = 5,
) -> Dict[str, Any]:
    del target_profile
    groups = shot_selection.get("groups") or []
    provider_status = (
        runtime.providers.describe_heavy_evidence()
        if hasattr(getattr(runtime, "providers", None), "describe_heavy_evidence")
        else SegformerHeavyEvidenceProvider().get_provider_status()
    )
    summary: Dict[str, Any] = {
        "enabled": bool(provider_status.get("enabled")),
        "advisory_only": True,
        "mode": "shortlist_only",
        "evidence_schema_version": str(provider_status.get("evidence_schema_version") or HEAVY_EVIDENCE_SCHEMA),
        "provider_name": str(provider_status.get("provider_name") or _HEAVY_PROVIDER_NAME),
        "provider_family": str(provider_status.get("provider_family") or _HEAVY_PROVIDER_FAMILY),
        "provider_version": str(provider_status.get("provider_version") or _HEAVY_PROVIDER_VERSION),
        "model_id": provider_status.get("model_id"),
        "device": provider_status.get("device"),
        "reason": provider_status.get("reason"),
        "group_count": len(groups),
        "processed_group_count": 0,
        "processed_candidate_count": 0,
        "cache_dir": str(_heavy_cache_dir(runtime)),
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "cache_write_count": 0,
    }
    if not bool(provider_status.get("enabled")):
        shot_selection["heavy_review_summary"] = summary
        shot_selection["heavy_evidence_summary"] = build_heavy_evidence_bundle(
            {
                "ok": False,
                "provider_name": summary.get("provider_name"),
                "provider_family": summary.get("provider_family"),
                "provider_version": summary.get("provider_version"),
                "model_id": summary.get("model_id"),
                "device": summary.get("device"),
                "reasons": [f"HEAVY_REVIEW_UNAVAILABLE:{summary.get('reason') or 'unavailable'}"],
            },
            lane_scope="shot_selection",
            advisory_only=True,
            mode="shortlist_only_advisory",
        )
        for group in groups:
            group["heavy_review"] = {
                "enabled": False,
                "reason": summary.get("reason") or "unavailable",
            }
            group["heavy_evidence"] = build_heavy_evidence_bundle(
                {
                    "ok": False,
                    "provider_name": summary.get("provider_name"),
                    "provider_family": summary.get("provider_family"),
                    "provider_version": summary.get("provider_version"),
                    "model_id": summary.get("model_id"),
                    "device": summary.get("device"),
                    "reasons": [f"HEAVY_REVIEW_UNAVAILABLE:{summary.get('reason') or 'unavailable'}"],
                },
                lane_scope="shortlist_group",
                advisory_only=True,
                mode="shortlist_only_advisory",
            )
        return shot_selection

    item_by_key: Dict[str, Dict[str, Any]] = {}
    for item in report_items:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        if record_key:
            item_by_key[record_key] = item

    boundary_cohesions: List[float] = []
    visible_cohesions: List[float] = []
    consensus_matches = 0

    for group in groups:
        shortlist = list(group.get("shortlist") or [])
        if len(shortlist) == 0:
            group["heavy_review"] = {"enabled": False, "reason": "empty_shortlist"}
            group["heavy_evidence"] = build_heavy_evidence_bundle(
                {"ok": False, "reasons": ["HEAVY_REVIEW_EMPTY_SHORTLIST"]},
                lane_scope="shortlist_group",
                advisory_only=True,
                mode="shortlist_only_advisory",
            )
            continue
        process_count = min(len(shortlist), max(1, max_candidates))
        candidate_rows: List[Dict[str, Any]] = []
        for row in shortlist[:process_count]:
            image_path = _resolve_candidate_path(runtime, row)
            if image_path is None:
                candidate_rows.append(
                    {
                        "record_key": row.get("record_key"),
                        "image": row.get("image"),
                        "ok": False,
                        "reasons": ["HEAVY_REVIEW_IMAGE_MISSING"],
                    }
                )
                continue
            if hasattr(getattr(runtime, "providers", None), "get_heavy_evidence"):
                metrics = runtime.providers.get_heavy_evidence(runtime, image_path)
            else:
                metrics = SegformerHeavyEvidenceProvider().get_heavy_evidence_metrics(runtime, image_path)
            metrics["record_key"] = row.get("record_key")
            metrics["image"] = row.get("image")
            metrics["base_selection_score"] = row.get("selection_score")
            cache_state = str(metrics.get("cache_state") or "").strip().lower()
            if cache_state == "hit":
                summary["cache_hit_count"] = int(summary.get("cache_hit_count", 0)) + 1
            elif cache_state == "write":
                summary["cache_miss_count"] = int(summary.get("cache_miss_count", 0)) + 1
                summary["cache_write_count"] = int(summary.get("cache_write_count", 0)) + 1
            elif cache_state == "miss":
                summary["cache_miss_count"] = int(summary.get("cache_miss_count", 0)) + 1
            candidate_rows.append(metrics)

        boundary_centroid = _normalize_embedding(
            _weighted_sum(
                [
                    (row["boundary_signature"], max(0.1, float(row.get("confidence", 0.0) or 0.0)))
                    for row in candidate_rows
                    if row.get("ok") and row.get("boundary_signature") is not None
                ]
            )
        )
        visible_centroid = _normalize_embedding(
            _weighted_sum(
                [
                    (row["visible_body_signature"], max(0.1, float(row.get("confidence", 0.0) or 0.0)))
                    for row in candidate_rows
                    if row.get("ok") and row.get("visible_body_signature") is not None
                ]
            )
        )

        boundary_sims: List[float] = []
        visible_sims: List[float] = []
        for row in candidate_rows:
            row["parser_boundary_alignment"] = _cosine(row.get("boundary_signature"), boundary_centroid)
            row["parser_visible_body_alignment"] = _cosine(row.get("visible_body_signature"), visible_centroid)
            if isinstance(row.get("parser_boundary_alignment"), (int, float)):
                boundary_sims.append(float(row["parser_boundary_alignment"]))
            if isinstance(row.get("parser_visible_body_alignment"), (int, float)):
                visible_sims.append(float(row["parser_visible_body_alignment"]))
            row["parser_consensus_score"] = _weighted_geometric_mean(
                [
                    (row.get("parser_boundary_alignment"), 0.55),
                    (row.get("parser_visible_body_alignment"), 0.25),
                    (row.get("confidence"), 0.20),
                ]
            )
            row["enhanced_selection_score"] = _weighted_mean(
                [
                    (row.get("base_selection_score"), 0.76),
                    (row.get("parser_consensus_score"), 0.16),
                    (row.get("confidence"), 0.08),
                ]
            )

        boundary_cohesion = _mean(boundary_sims)
        visible_cohesion = _mean(visible_sims)
        if isinstance(boundary_cohesion, (int, float)):
            boundary_cohesions.append(float(boundary_cohesion))
        if isinstance(visible_cohesion, (int, float)):
            visible_cohesions.append(float(visible_cohesion))

        advisory_rows = sorted(
            [row for row in candidate_rows if row.get("ok")],
            key=lambda row: (
                1 if row.get("enhanced_selection_score") is None else 0,
                0.0 if row.get("enhanced_selection_score") is None else -float(row.get("enhanced_selection_score")),
                str(row.get("image") or ""),
            ),
        )
        consensus_top = advisory_rows[0].get("image") if len(advisory_rows) > 0 else None
        if consensus_top and consensus_top == group.get("top_ranked_image"):
            consensus_matches += 1

        shortlist_index = {str(row.get("record_key") or ""): row for row in shortlist}
        for rank, advisory in enumerate(advisory_rows, start=1):
            shortlist_row = shortlist_index.get(str(advisory.get("record_key") or ""))
            evidence_bundle = build_heavy_evidence_bundle(
                {
                    **advisory,
                    "rank_in_heavy_review": rank,
                },
                lane_scope="shortlist_candidate",
                advisory_only=True,
                mode="shortlist_only_advisory",
                record_key=str(advisory.get("record_key") or ""),
                image=str(advisory.get("image") or ""),
            )
            heavy_node = {
                "parser_confidence": _round_or_none(advisory.get("confidence")),
                "parser_boundary_alignment": _round_or_none(advisory.get("parser_boundary_alignment")),
                "parser_visible_body_alignment": _round_or_none(advisory.get("parser_visible_body_alignment")),
                "parser_consensus_score": _round_or_none(advisory.get("parser_consensus_score")),
                "enhanced_selection_score": _round_or_none(advisory.get("enhanced_selection_score")),
                "garment_coverage_ratio": _round_or_none(advisory.get("garment_coverage_ratio")),
                "upper_cloth_coverage": _round_or_none(advisory.get("upper_cloth_coverage")),
                "lower_cloth_coverage": _round_or_none(advisory.get("lower_cloth_coverage")),
                "neckline_depth_ratio": _round_or_none(advisory.get("neckline_depth_ratio")),
                "shoulder_cloth_balance": _round_or_none(advisory.get("shoulder_cloth_balance")),
                "visible_body_ratio": _round_or_none(advisory.get("visible_body_ratio")),
                "visible_face_ratio": _round_or_none(advisory.get("visible_face_ratio")),
                "visible_arm_ratio": _round_or_none(advisory.get("visible_arm_ratio")),
                "visible_leg_ratio": _round_or_none(advisory.get("visible_leg_ratio")),
                "rank_in_heavy_review": rank,
                "coverage": evidence_bundle.get("coverage"),
                "provider_name": evidence_bundle.get("provider_name"),
                "provider_version": evidence_bundle.get("provider_version"),
                "cache_state": evidence_bundle.get("cache_state"),
                "reasons": list(advisory.get("reasons") or []),
            }
            if shortlist_row is not None:
                shortlist_row["heavy_review"] = heavy_node
                shortlist_row["heavy_evidence"] = evidence_bundle
            item = item_by_key.get(str(advisory.get("record_key") or ""))
            if item is not None:
                item_debug = item.setdefault("debug", {})
                item_debug["heavy_review"] = heavy_node
                item_debug["heavy_evidence"] = evidence_bundle

        heavy_guidance: List[str] = []
        if consensus_top and consensus_top != group.get("top_ranked_image"):
            heavy_guidance.append("重解析复核的首选与基础排序不一致，建议至少人工对比前两名的领口、肩线和可见肢体比例。")
            group["manual_review_window"] = max(int(group.get("manual_review_window", 1) or 1), min(3, len(shortlist)))
        elif consensus_top:
            heavy_guidance.append("基础排序与重解析复核的首选一致，可优先从该候选开始人工复核。")
        if isinstance(boundary_cohesion, (int, float)) and float(boundary_cohesion) < 0.88:
            heavy_guidance.append("shortlist 的服装边界一致性仍偏松，人工复核时要重点看领口和肩线是否在漂。")
        if isinstance(visible_cohesion, (int, float)) and float(visible_cohesion) < 0.86:
            heavy_guidance.append("shortlist 的可见身体比例仍有波动，人工复核时要注意脸面积和露臂露腿比例是否突然变化。")

        group["review_guidance"] = list(dict.fromkeys(list(group.get("review_guidance") or []) + heavy_guidance))[:6]
        group_metrics = {
            "ok": True,
            "confidence": _mean([row.get("confidence") for row in advisory_rows]),
            "parser_boundary_alignment": boundary_cohesion,
            "parser_visible_body_alignment": visible_cohesion,
            "parser_consensus_score": _mean([row.get("parser_consensus_score") for row in advisory_rows]),
            "candidate_count": len(advisory_rows),
            "consensus_top_image": consensus_top,
            "cache_hit_count": sum(1 for row in advisory_rows if str(row.get("cache_state") or "").strip().lower() == "hit"),
            "cache_miss_count": sum(1 for row in advisory_rows if str(row.get("cache_state") or "").strip().lower() in {"miss", "write"}),
            "cache_write_count": sum(1 for row in advisory_rows if str(row.get("cache_state") or "").strip().lower() == "write"),
            "guidance": heavy_guidance[:4],
            "reasons": [],
        }
        group["heavy_review"] = {
            "enabled": True,
            "advisory_only": True,
            "candidate_count": len(advisory_rows),
            "consensus_top_image": consensus_top,
            "parser_boundary_cohesion": _round_or_none(boundary_cohesion),
            "parser_visible_body_cohesion": _round_or_none(visible_cohesion),
            "parser_confidence_mean": _round_or_none(_mean([row.get("confidence") for row in advisory_rows])),
            "cache_hit_count": group_metrics.get("cache_hit_count"),
            "cache_miss_count": group_metrics.get("cache_miss_count"),
            "cache_write_count": group_metrics.get("cache_write_count"),
            "guidance": heavy_guidance[:4],
        }
        group["heavy_evidence"] = build_heavy_evidence_bundle(
            group_metrics,
            lane_scope="shortlist_group",
            advisory_only=True,
            mode="shortlist_only_advisory",
        )
        summary["processed_group_count"] = int(summary.get("processed_group_count", 0)) + 1
        summary["processed_candidate_count"] = int(summary.get("processed_candidate_count", 0)) + len(advisory_rows)

    summary["parser_boundary_cohesion_mean"] = _round_or_none(_mean(boundary_cohesions))
    summary["parser_visible_body_cohesion_mean"] = _round_or_none(_mean(visible_cohesions))
    summary["consensus_top_match_ratio"] = _round_or_none(
        float(consensus_matches / max(1, int(summary.get("processed_group_count", 0) or 0)))
    )
    shot_selection["heavy_review_summary"] = summary
    shot_selection["heavy_evidence_summary"] = build_heavy_evidence_bundle(
        {
            "ok": bool(summary.get("enabled")),
            "confidence": summary.get("parser_boundary_cohesion_mean"),
            "parser_boundary_alignment": summary.get("parser_boundary_cohesion_mean"),
            "parser_visible_body_alignment": summary.get("parser_visible_body_cohesion_mean"),
            "parser_consensus_score": summary.get("consensus_top_match_ratio"),
            "candidate_count": summary.get("processed_candidate_count"),
            "cache_hit_count": summary.get("cache_hit_count"),
            "cache_miss_count": summary.get("cache_miss_count"),
            "cache_write_count": summary.get("cache_write_count"),
            "guidance": [],
            "reasons": [] if bool(summary.get("enabled")) else [str(summary.get("reason") or "HEAVY_REVIEW_UNAVAILABLE")],
        },
        lane_scope="shot_selection",
        advisory_only=True,
        mode="shortlist_only_advisory",
    )
    return shot_selection
