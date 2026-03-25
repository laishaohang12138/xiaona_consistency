from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .providers import HeavyEvidenceProvider
from .qa_heavy_body_measure import BodyMeasureHeavyEvidenceProvider
from .qa_heavy_review import (
    SegformerHeavyEvidenceProvider,
    build_heavy_evidence_bundle,
    normalize_heavy_evidence_bundle,
)
from .qa_utils import dedupe_keep_order

_PROVIDER_NAME = "segformer_body_fusion"
_PROVIDER_FAMILY = "multi_evidence_fusion"
_PROVIDER_VERSION = "segformer_body_fusion_v1"
_MODEL_ID = "mattmdjaga/segformer_b2_clothes+mediapipe_pose+body_measure_lite_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _weighted_mean(items: List[tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None or weight <= 0.0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 1e-8:
        return None
    return float(numerator / denominator)


def _bundle_metric_specs(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for metric in bundle.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        spec: Dict[str, Any] = {
            "metric_name": str(metric.get("metric_name") or "").strip(),
            "metric_value": metric.get("metric_value"),
            "confidence": metric.get("confidence"),
            "coverage": metric.get("coverage"),
            "lane_scope": metric.get("lane_scope"),
            "failure_reason": metric.get("failure_reason"),
            "provider_name": metric.get("provider_name"),
            "provider_family": metric.get("provider_family"),
            "provider_version": metric.get("provider_version"),
        }
        signature_ref = metric.get("feature_vector_or_signature")
        if isinstance(signature_ref, dict):
            spec["signature_ref"] = signature_ref
        if spec["metric_name"]:
            specs.append(spec)
    return specs


def _merge_signature_refs(*bundles: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for bundle in bundles:
        for key, value in dict(bundle.get("signature_refs") or {}).items():
            if key not in merged:
                merged[str(key)] = value
    return merged


def _merge_guidance(*bundles: Dict[str, Any]) -> List[str]:
    guidance: List[str] = []
    for bundle in bundles:
        summary = bundle.get("summary") or {}
        guidance.extend(str(item) for item in summary.get("guidance", []) if str(item).strip())
    return dedupe_keep_order(guidance)[:6]


def _aggregate_cache_counts(*bundles: Dict[str, Any]) -> Dict[str, int]:
    hit = 0
    miss = 0
    write = 0
    for bundle in bundles:
        summary = bundle.get("summary") or {}
        hit += int(summary.get("cache_hit_count", 0) or 0)
        miss += int(summary.get("cache_miss_count", 0) or 0)
        write += int(summary.get("cache_write_count", 0) or 0)
    return {
        "cache_hit_count": hit,
        "cache_miss_count": miss,
        "cache_write_count": write,
    }


def _resolve_cache_state(*bundles: Dict[str, Any]) -> Optional[str]:
    states = [str(bundle.get("cache_state") or "").strip().lower() for bundle in bundles if str(bundle.get("cache_state") or "").strip()]
    if len(states) == 0:
        return None
    if all(state == "hit" for state in states):
        return "hit"
    if any(state == "write" for state in states):
        return "write"
    if any(state == "miss" for state in states):
        return "miss"
    return states[0]


class SegformerBodyFusionHeavyEvidenceProvider(HeavyEvidenceProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def __init__(self) -> None:
        self.segformer_provider = SegformerHeavyEvidenceProvider()
        self.body_provider = BodyMeasureHeavyEvidenceProvider()

    def get_provider_status(self) -> Dict[str, Any]:
        seg_status = self.segformer_provider.get_provider_status()
        body_status = self.body_provider.get_provider_status()
        enabled = bool(seg_status.get("enabled")) or bool(body_status.get("enabled"))
        reason_parts = []
        if not bool(seg_status.get("enabled")):
            reason_parts.append(f"segformer:{seg_status.get('reason') or 'disabled'}")
        if not bool(body_status.get("enabled")):
            reason_parts.append(f"body_measure:{body_status.get('reason') or 'disabled'}")
        return {
            "enabled": enabled,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": seg_status.get("device") if seg_status.get("device") == body_status.get("device") else "mixed",
            "reason": None if len(reason_parts) == 0 else ";".join(reason_parts),
            "evidence_schema_version": "heavy_evidence_v1",
            "component_providers": [
                {
                    "provider_name": seg_status.get("provider_name"),
                    "provider_version": seg_status.get("provider_version"),
                    "enabled": bool(seg_status.get("enabled")),
                },
                {
                    "provider_name": body_status.get("provider_name"),
                    "provider_version": body_status.get("provider_version"),
                    "enabled": bool(body_status.get("enabled")),
                },
            ],
        }

    def get_heavy_evidence_metrics(
        self,
        runtime: Any,
        image_path: Path,
    ) -> Dict[str, Any]:
        resolved_path = Path(image_path).resolve()
        seg_raw = self.segformer_provider.get_heavy_evidence_metrics(runtime, resolved_path)
        body_raw = self.body_provider.get_heavy_evidence_metrics(runtime, resolved_path)
        seg_bundle = normalize_heavy_evidence_bundle(
            build_heavy_evidence_bundle(
                seg_raw,
                lane_scope="report_item",
                advisory_only=True,
                mode="provider_component",
                image=str(resolved_path.name),
            )
        )
        body_bundle = normalize_heavy_evidence_bundle(
            build_heavy_evidence_bundle(
                body_raw,
                lane_scope="report_item",
                advisory_only=True,
                mode="provider_component",
                image=str(resolved_path.name),
            )
        )

        seg_available = bool(seg_bundle.get("available"))
        body_available = bool(body_bundle.get("available"))
        confidence = _weighted_mean(
            [
                (_safe_float(seg_bundle.get("confidence")), 0.46 if seg_available else 0.0),
                (_safe_float(body_bundle.get("confidence")), 0.54 if body_available else 0.0),
            ]
        )
        coverage = _weighted_mean(
            [
                (_safe_float(seg_bundle.get("coverage")), 0.48 if seg_available else 0.0),
                (_safe_float(body_bundle.get("coverage")), 0.52 if body_available else 0.0),
            ]
        )
        cache_counts = _aggregate_cache_counts(seg_bundle, body_bundle)
        guidance = _merge_guidance(seg_bundle, body_bundle)
        if seg_available and body_available:
            guidance.insert(0, "服装边界证据与体态几何证据都已到位，可作为工业默认复核模式。")
        elif body_available:
            guidance.insert(0, "当前仅保留体态几何证据，领口/肩线边界判断会偏弱。")
        elif seg_available:
            guidance.insert(0, "当前仅保留服装边界证据，体态/3D 几何判断会偏弱。")
        else:
            guidance.insert(0, "融合证据不可用，当前没有可依赖的重型支撑读数。")
        guidance = dedupe_keep_order(guidance)[:6]

        reasons = dedupe_keep_order(
            list(seg_bundle.get("reasons") or [])
            + list(body_bundle.get("reasons") or [])
        )[:16]

        summary = {
            "component_providers": [
                {
                    "provider_name": seg_bundle.get("provider_name"),
                    "provider_version": seg_bundle.get("provider_version"),
                    "available": seg_available,
                    "confidence": seg_bundle.get("confidence"),
                    "coverage": seg_bundle.get("coverage"),
                    "cache_state": seg_bundle.get("cache_state"),
                },
                {
                    "provider_name": body_bundle.get("provider_name"),
                    "provider_version": body_bundle.get("provider_version"),
                    "available": body_available,
                    "confidence": body_bundle.get("confidence"),
                    "coverage": body_bundle.get("coverage"),
                    "cache_state": body_bundle.get("cache_state"),
                },
            ],
            "segformer_summary": dict(seg_bundle.get("summary") or {}),
            "body_measure_summary": dict(body_bundle.get("summary") or {}),
            **cache_counts,
            "candidate_count": max(
                int((seg_bundle.get("summary") or {}).get("candidate_count", 0) or 0),
                int((body_bundle.get("summary") or {}).get("candidate_count", 0) or 0),
                1,
            ),
            "guidance": guidance,
        }

        return {
            "ok": seg_available or body_available,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": seg_bundle.get("device") if seg_bundle.get("device") == body_bundle.get("device") else "mixed",
            "source_path": str(resolved_path),
            "cache_state": _resolve_cache_state(seg_bundle, body_bundle),
            "confidence": confidence,
            "coverage": coverage,
            "metric_specs": _bundle_metric_specs(seg_bundle) + _bundle_metric_specs(body_bundle),
            "summary": summary,
            "signature_refs": _merge_signature_refs(seg_bundle, body_bundle),
            "reasons": reasons,
        }
