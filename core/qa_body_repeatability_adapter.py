from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .qa_body_evidence_contract import (
    BODY_CORE_MEASUREMENT_ORDER,
    canonical_vertex_delta_vector,
    log_ratio_residual_vector,
)
from .qa_provider_contract import build_body_provider_contract


BODY_REPEATABILITY_ADAPTER_SCHEMA = "body_repeatability_measurement_adapter_v0_2"


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"image decode failed: {path}")
    return image


def _load_artifact(path_value: Any) -> Optional[Dict[str, Any]]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    artifact = payload.get("artifact")
    return dict(artifact) if isinstance(artifact, dict) else payload


def _candidate_artifact_path(result: Dict[str, Any]) -> Optional[str]:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    nested = (
        summary.get("body_canonical_summary")
        if isinstance(summary.get("body_canonical_summary"), dict)
        else summary
    )
    for value in [
        nested.get("candidate_artifact_path"),
        result.get("candidate_artifact_path"),
        result.get("cache_file"),
    ]:
        path_text = str(value or "").strip()
        if path_text:
            return path_text
    return None


def _normalized_bbox(artifact: Optional[Dict[str, Any]], *, width: int, height: int) -> Any:
    if not isinstance(artifact, dict) or width <= 0 or height <= 0:
        return None
    conversion_meta = (
        artifact.get("conversion_meta")
        if isinstance(artifact.get("conversion_meta"), dict)
        else {}
    )
    try:
        bbox = np.asarray(conversion_meta.get("bbox_xyxy"), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if bbox.size != 4 or not bool(np.all(np.isfinite(bbox))):
        return None
    return [
        float(bbox[0]) / float(width),
        float(bbox[1]) / float(height),
        float(bbox[2]) / float(width),
        float(bbox[3]) / float(height),
    ]


class BodyCanonicalRepeatabilityAdapter:
    """Re-execute the active HMR2 body provider for one controlled image."""

    def __init__(self, runtime: Any, *, execution_context: Optional[Dict[str, Any]] = None) -> None:
        self.runtime = runtime
        self._execution_context = dict(execution_context or {})

    def describe(self) -> Dict[str, Any]:
        provider_status = self.runtime.providers.describe_heavy_evidence()
        return {
            "schema_version": BODY_REPEATABILITY_ADAPTER_SCHEMA,
            "adapter_version": "0.2",
            "adapter_implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "body_canonical_provider": _json_ready(provider_status),
            "execution_context": _json_ready(self._execution_context),
            "axes": ["body_core_shape", "body_topology"],
            "measurement_order": list(BODY_CORE_MEASUREMENT_ORDER),
            "topology_contract": {
                "source_field": "canonical_smpl_vertices",
                "required_vertex_count": 6890,
                "required_coordinate_count": 20670,
                "coordinate_axis_order": ["x", "y", "z"],
                "translation_centered": True,
                "rotation_fit_applied": False,
                "scale_fit_applied": False,
                "procrustes_fit_applied": False,
            },
            "serialized_execution": True,
            "component_aggregation_allowed": False,
            "decision_influence": "NONE",
        }

    def measure(self, image_path: Path) -> Dict[str, Any]:
        image_path = Path(image_path).resolve()
        image = _load_image(image_path)
        result = self.runtime.providers.get_heavy_evidence(self.runtime, image_path)
        try:
            cooldown_seconds = float(
                self._execution_context.get("inter_execution_cooldown_seconds") or 0.0
            )
        except (TypeError, ValueError):
            cooldown_seconds = 0.0
        if cooldown_seconds > 0.0:
            time.sleep(cooldown_seconds)
        artifact_path = _candidate_artifact_path(result)
        artifact = _load_artifact(artifact_path)
        measurements = (
            dict(artifact.get("canonical_measurements") or {})
            if isinstance(artifact, dict)
            else {}
        )
        availability = log_ratio_residual_vector(measurements, measurements)
        provider_contract = build_body_provider_contract(artifact, axis="body_core_shape")
        vertices = artifact.get("canonical_smpl_vertices") if artifact else None
        topology_availability = canonical_vertex_delta_vector(vertices, vertices)
        topology_provider_contract = build_body_provider_contract(
            artifact,
            axis="body_topology",
        )
        topology_errors = list(topology_availability.get("errors") or [])
        if topology_provider_contract.get("completeness_state") != "COMPLETE":
            topology_errors.append("BODY_TOPOLOGY_PROVIDER_CONTRACT_NOT_COMPLETE")
        topology_available = bool(topology_availability.get("available")) and not topology_errors
        height, width = image.shape[:2]
        chain_observation = {
            "image_shape_hw": [int(height), int(width)],
            "body_provider": str(result.get("provider_name") or "unknown"),
            "body_provider_version": str(result.get("provider_version") or "unknown"),
            "artifact_available": artifact is not None,
            "body_bbox_normalized_xyxy": _normalized_bbox(
                artifact,
                width=width,
                height=height,
            ),
            "body_canonical_coverage": artifact.get("coverage") if artifact else None,
            "body_fit_confidence": artifact.get("fit_confidence") if artifact else None,
            "pose_vector": _json_ready(artifact.get("pose_vector")) if artifact else None,
            "pose_vector_dimension": (
                len(artifact.get("pose_vector"))
                if artifact and isinstance(artifact.get("pose_vector"), list)
                else None
            ),
            "available_measurement_components": list(availability.get("used_components") or []),
            "native_topology_available": topology_available,
            "canonical_smpl_vertex_count": (
                topology_availability.get("vertex_count")
                if topology_availability.get("available")
                else None
            ),
        }
        reasons = [str(reason) for reason in list(result.get("reasons") or [])]
        reasons.extend(str(error) for error in list(availability.get("errors") or []))
        reasons.extend(str(error) for error in topology_errors)
        return {
            "schema_version": BODY_REPEATABILITY_ADAPTER_SCHEMA,
            "source_path": str(image_path),
            "image_shape_hw": [int(height), int(width)],
            "chain_observation": chain_observation,
            "chain_signature": _canonical_sha256(chain_observation),
            "body_core_shape": {
                "available": bool(availability.get("available")),
                "value": _json_ready(measurements),
                "measurement_order": list(BODY_CORE_MEASUREMENT_ORDER),
                "used_components": list(availability.get("used_components") or []),
                "provider_contract": provider_contract,
                "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
                "errors": list(availability.get("errors") or []),
            },
            "body_topology": {
                "available": topology_available,
                "value": _json_ready(vertices) if topology_available else None,
                "vertex_count": topology_availability.get("vertex_count"),
                "coordinate_count": topology_availability.get("coordinate_count"),
                "coordinate_axis_order": ["x", "y", "z"],
                "provider_contract": topology_provider_contract,
                "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
                "errors": list(dict.fromkeys(topology_errors)),
            },
            "canonical_result_provenance": {
                "provider_name": result.get("provider_name"),
                "provider_version": result.get("provider_version"),
                "candidate_artifact_path": artifact_path,
                "cache_file": result.get("cache_file"),
            },
            "errors": list(dict.fromkeys(reasons)),
            "calibration_state": "SHADOW_UNCALIBRATED",
            "decision_influence": "NONE",
        }
