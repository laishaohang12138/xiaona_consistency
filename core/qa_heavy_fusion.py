from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .providers import HeavyEvidenceProvider
from .qa_heavy_body_canonical import BodyCanonicalHeavyEvidenceProvider
from .qa_heavy_body_measure import BodyMeasureHeavyEvidenceProvider
from .qa_heavy_review import (
    SegformerHeavyEvidenceProvider,
    build_heavy_evidence_bundle,
    normalize_heavy_evidence_bundle,
)
from .qa_heavy_surface_occlusion import ClothingSurfaceOcclusionBridgeProvider
from .qa_utils import dedupe_keep_order

_FUSION_SCHEMA = "heavy_evidence_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _weighted_mean(items: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
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
    return dedupe_keep_order(guidance)[:8]


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
    states = [
        str(bundle.get("cache_state") or "").strip().lower()
        for bundle in bundles
        if str(bundle.get("cache_state") or "").strip()
    ]
    if len(states) == 0:
        return None
    if all(state == "hit" for state in states):
        return "hit"
    if any(state == "write" for state in states):
        return "write"
    if any(state == "miss" for state in states):
        return "miss"
    return states[0]


def _component_bundle(provider: HeavyEvidenceProvider, runtime: Any, image_path: Path) -> Dict[str, Any]:
    raw = provider.get_heavy_evidence_metrics(runtime, image_path)
    return normalize_heavy_evidence_bundle(
        build_heavy_evidence_bundle(
            raw,
            lane_scope="report_item",
            advisory_only=True,
            mode="provider_component",
            image=str(image_path.name),
        )
    )


class _MultiComponentFusionHeavyEvidenceProvider(HeavyEvidenceProvider):
    provider_name = "multi_component_fusion"
    provider_family = "multi_evidence_fusion"
    provider_version = "multi_component_fusion_v1"
    model_id = "fusion"

    def __init__(
        self,
        *,
        components: Sequence[Tuple[HeavyEvidenceProvider, float, float, str]],
        positive_guidance: Sequence[str],
        degraded_guidance: Sequence[Tuple[str, str]],
        unavailable_guidance: str,
    ) -> None:
        self.components = list(components)
        self.positive_guidance = list(positive_guidance)
        self.degraded_guidance = list(degraded_guidance)
        self.unavailable_guidance = str(unavailable_guidance)

    def get_provider_status(self) -> Dict[str, Any]:
        component_statuses = []
        reason_parts = []
        enabled = False
        device_values: List[str] = []
        for provider, _, _, component_key in self.components:
            status = provider.get_provider_status()
            component_enabled = bool(status.get("enabled"))
            enabled = enabled or component_enabled
            if not component_enabled:
                reason_parts.append(f"{component_key}:{status.get('reason') or 'disabled'}")
            device = str(status.get("device") or "").strip()
            if device:
                device_values.append(device)
            component_statuses.append(
                {
                    "provider_name": status.get("provider_name"),
                    "provider_version": status.get("provider_version"),
                    "enabled": component_enabled,
                    "component_key": component_key,
                }
            )
        device = device_values[0] if len(set(device_values)) == 1 else "mixed"
        return {
            "enabled": enabled,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": self.model_id,
            "device": device,
            "reason": None if len(reason_parts) == 0 else ";".join(reason_parts),
            "evidence_schema_version": _FUSION_SCHEMA,
            "component_providers": component_statuses,
        }

    def _inject_guidance(self, component_rows: Sequence[Dict[str, Any]], guidance: List[str]) -> List[str]:
        available_keys = {
            str(row.get("component_key") or "").strip()
            for row in component_rows
            if bool(row.get("available"))
        }
        if len(available_keys) == len(self.components):
            guidance = list(self.positive_guidance) + guidance
        elif len(available_keys) > 0:
            for component_key, message in self.degraded_guidance:
                if component_key not in available_keys:
                    guidance.insert(0, message)
        else:
            guidance.insert(0, self.unavailable_guidance)
        return dedupe_keep_order(guidance)[:8]

    def get_heavy_evidence_metrics(
        self,
        runtime: Any,
        image_path: Path,
    ) -> Dict[str, Any]:
        resolved_path = Path(image_path).resolve()
        bundle_rows: List[Dict[str, Any]] = []
        component_rows: List[Dict[str, Any]] = []
        confidence_items: List[Tuple[Optional[float], float]] = []
        coverage_items: List[Tuple[Optional[float], float]] = []
        reasons: List[str] = []
        candidate_count = 1

        for provider, confidence_weight, coverage_weight, component_key in self.components:
            bundle = _component_bundle(provider, runtime, resolved_path)
            bundle_rows.append(bundle)
            available = bool(bundle.get("available"))
            if available:
                confidence_items.append((_safe_float(bundle.get("confidence")), confidence_weight))
                coverage_items.append((_safe_float(bundle.get("coverage")), coverage_weight))
            reasons.extend(list(bundle.get("reasons") or []))
            summary = bundle.get("summary") or {}
            candidate_count = max(candidate_count, int(summary.get("candidate_count", 0) or 0), 1)
            component_rows.append(
                {
                    "component_key": component_key,
                    "provider_name": bundle.get("provider_name"),
                    "provider_version": bundle.get("provider_version"),
                    "available": available,
                    "confidence": bundle.get("confidence"),
                    "coverage": bundle.get("coverage"),
                    "cache_state": bundle.get("cache_state"),
                }
            )

        guidance = _merge_guidance(*bundle_rows)
        guidance = self._inject_guidance(component_rows, guidance)
        cache_counts = _aggregate_cache_counts(*bundle_rows)

        component_summaries = {
            f"{str(row.get('component_key') or '').strip()}_summary": dict((bundle_rows[idx].get("summary") or {}))
            for idx, row in enumerate(component_rows)
            if str(row.get("component_key") or "").strip()
        }
        ok = any(bool(bundle.get("available")) for bundle in bundle_rows)
        devices = [str(bundle.get("device") or "").strip() for bundle in bundle_rows if str(bundle.get("device") or "").strip()]
        device = devices[0] if len(set(devices)) == 1 else "mixed"

        return {
            "ok": ok,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": self.model_id,
            "device": device,
            "source_path": str(resolved_path),
            "cache_state": _resolve_cache_state(*bundle_rows),
            "confidence": _weighted_mean(confidence_items),
            "coverage": _weighted_mean(coverage_items),
            "metric_specs": [spec for bundle in bundle_rows for spec in _bundle_metric_specs(bundle)],
            "summary": {
                "component_providers": component_rows,
                **component_summaries,
                **cache_counts,
                "candidate_count": candidate_count,
                "guidance": guidance,
            },
            "signature_refs": _merge_signature_refs(*bundle_rows),
            "reasons": dedupe_keep_order(reasons)[:20],
        }


class SegformerBodyFusionHeavyEvidenceProvider(_MultiComponentFusionHeavyEvidenceProvider):
    provider_name = "segformer_body_fusion"
    provider_family = "multi_evidence_fusion"
    provider_version = "segformer_body_fusion_v1"
    model_id = "mattmdjaga/segformer_b2_clothes+mediapipe_pose+body_measure_lite_v1"

    def __init__(self) -> None:
        super().__init__(
            components=[
                (SegformerHeavyEvidenceProvider(), 0.46, 0.48, "segformer"),
                (BodyMeasureHeavyEvidenceProvider(), 0.54, 0.52, "body_measure"),
            ],
            positive_guidance=[
                "Garment boundary evidence and body geometry evidence are both available for review.",
            ],
            degraded_guidance=[
                ("segformer", "Segformer boundary evidence is missing; neckline and cloth-edge review will be weaker."),
                ("body_measure", "Body geometry evidence is missing; 3D and constitution review will be weaker."),
            ],
            unavailable_guidance="All heavy evidence components are unavailable for this image.",
        )


class SegformerBodyTruthFusionHeavyEvidenceProvider(_MultiComponentFusionHeavyEvidenceProvider):
    provider_name = "segformer_body_truth_fusion"
    provider_family = "multi_evidence_fusion"
    provider_version = "segformer_body_truth_fusion_v1"
    model_id = "mattmdjaga/segformer_b2_clothes+body_measure_lite+body_canonical_hmr2_v1"

    def __init__(self) -> None:
        super().__init__(
            components=[
                (SegformerHeavyEvidenceProvider(), 0.20, 0.25, "segformer"),
                (BodyMeasureHeavyEvidenceProvider(), 0.25, 0.25, "body_measure"),
                (BodyCanonicalHeavyEvidenceProvider(), 0.45, 0.40, "body_canonical"),
                (ClothingSurfaceOcclusionBridgeProvider(), 0.10, 0.10, "surface_occlusion"),
            ],
            positive_guidance=[
                "Boundary, geometry, canonical 116-1 body truth, and surface occlusion evidence are all available for review.",
            ],
            degraded_guidance=[
                ("body_canonical", "Canonical 116-1 body truth evidence is missing; this fusion falls back to boundary and geometry only."),
                ("segformer", "Segformer boundary evidence is missing; garment-edge review will be weaker."),
                ("body_measure", "Body geometry evidence is missing; constitution review will be weaker."),
                ("surface_occlusion", "DensePose/SAM2 surface occlusion sidecar is missing; clothing-invariant review falls back to parser and body topology."),
            ],
            unavailable_guidance="All heavy evidence components are unavailable for this image.",
        )
