from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .qa_face_pose_canonical_3ddfa import describe_face_measurement_runtime_contract
from .qa_features import extract_face_feat
from .qa_provider_contract import build_face_provider_contract


FACE_REPEATABILITY_ADAPTER_SCHEMA = "face_repeatability_measurement_adapter_v0_1"


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


def _artifact_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ["cache_file", "candidate_artifact_path"]:
        artifact = _load_artifact(result.get(key))
        if artifact is not None:
            return artifact
    return None


def _artifact_landmarks(artifact: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(artifact, dict):
        return None
    for key in ["canonical_landmarks", "landmarks_2d", "landmarks"]:
        if artifact.get(key) is not None:
            return artifact.get(key)
    return None


def _artifact_visibility(artifact: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(artifact, dict):
        return None
    for key in ["landmark_visibility_weights", "landmark_weights", "landmark_confidence"]:
        if artifact.get(key) is not None:
            return artifact.get(key)
    return None


def _normalized_kps5(value: Any, *, width: int, height: int) -> Any:
    if value is None or width <= 0 or height <= 0:
        return None
    try:
        points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None
    if points.shape[0] != 5 or not bool(np.all(np.isfinite(points))):
        return None
    normalized = points.copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)
    return normalized.tolist()


class FaceCanonicalRepeatabilityAdapter:
    """Re-execute the existing face detector and canonical provider for one image."""

    def __init__(self, runtime: Any, *, execution_context: Optional[Dict[str, Any]] = None) -> None:
        self.runtime = runtime
        self._runtime_contract = describe_face_measurement_runtime_contract(runtime)
        self._execution_context = dict(execution_context or {})

    def describe(self) -> Dict[str, Any]:
        provider_status = self.runtime.providers.describe_face_canonical()
        engines = self.runtime.engines
        return {
            "schema_version": FACE_REPEATABILITY_ADAPTER_SCHEMA,
            "adapter_version": "0.1",
            "adapter_implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "face_engine_mode": str(getattr(engines, "face_mode", "unknown")),
            "face_engine_reason": getattr(engines, "face_reason", None),
            "face_canonical_provider": _json_ready(provider_status),
            "measurement_runtime_contract": _json_ready(self._runtime_contract),
            "execution_context": _json_ready(self._execution_context),
            "axes": ["face_identity", "face_shape"],
            "serialized_execution": True,
            "decision_influence": "NONE",
        }

    def measure(self, image_path: Path) -> Dict[str, Any]:
        image_path = Path(image_path).resolve()
        image = _load_image(image_path)
        face_feat = extract_face_feat(self.runtime, image, source_path=image_path)
        canonical_result = self.runtime.providers.analyze_face_canonical(
            self.runtime,
            image_path,
            img_bgr=image,
            face_feat=face_feat,
        )
        artifact = _artifact_from_result(canonical_result)
        identity_artifact = dict(artifact or {})
        if identity_artifact.get("runtime_face_embedding_raw") is None and face_feat.embedding is not None:
            identity_artifact["runtime_face_embedding_raw"] = _json_ready(face_feat.embedding)
            identity_artifact["runtime_face_embedding_contract"] = dict(
                self._runtime_contract.get("identity") or {}
            )
        landmarks = _artifact_landmarks(artifact)
        identity_contract = build_face_provider_contract(identity_artifact, axis="face_identity")
        shape_contract = build_face_provider_contract(artifact, axis="face_shape")
        height, width = image.shape[:2]
        bbox = list(face_feat.bbox_xyxy) if face_feat.bbox_xyxy is not None else None
        normalized_bbox = (
            [
                float(bbox[0]) / width,
                float(bbox[1]) / height,
                float(bbox[2]) / width,
                float(bbox[3]) / height,
            ]
            if bbox is not None and width > 0 and height > 0
            else None
        )
        chain_observation = {
            "image_shape_hw": [int(height), int(width)],
            "face_engine_mode": str(getattr(self.runtime.engines, "face_mode", "unknown")),
            "face_detected": bool(face_feat.ok),
            "face_bbox_xyxy": bbox,
            "face_bbox_normalized_xyxy": normalized_bbox,
            "face_bbox_area_ratio": float(face_feat.bbox_area_ratio),
            "kps5_available": face_feat.kps5 is not None,
            "kps5_normalized_xy": _normalized_kps5(face_feat.kps5, width=width, height=height),
            "canonical_provider": str(canonical_result.get("provider_name") or "unknown"),
            "canonical_available": bool(canonical_result.get("available")),
            "artifact_available": artifact is not None,
            "landmark_schema_id": artifact.get("landmark_schema_id") if artifact else None,
            "canonical_pose_euler_deg": _json_ready(artifact.get("pose_euler_deg")) if artifact else None,
        }
        errors = [str(reason) for reason in list(face_feat.reasons or []) if "MISSING" in str(reason)]
        errors.extend(str(reason) for reason in list(canonical_result.get("reasons") or []))
        return {
            "schema_version": FACE_REPEATABILITY_ADAPTER_SCHEMA,
            "source_path": str(image_path),
            "image_shape_hw": [int(height), int(width)],
            "chain_observation": chain_observation,
            "chain_signature": _canonical_sha256(chain_observation),
            "face_identity": {
                "available": face_feat.embedding is not None,
                "value": _json_ready(face_feat.embedding),
                "provider_contract": identity_contract,
                "errors": [] if face_feat.embedding is not None else ["FACE_EMBEDDING_UNAVAILABLE"],
            },
            "face_shape": {
                "available": landmarks is not None,
                "value": _json_ready(landmarks),
                "visibility_weights": _json_ready(_artifact_visibility(artifact)),
                "landmark_schema_id": artifact.get("landmark_schema_id") if artifact else None,
                "provider_contract": shape_contract,
                "errors": [] if landmarks is not None else ["CANONICAL_LANDMARKS_UNAVAILABLE"],
            },
            "canonical_result_provenance": {
                "provider_name": canonical_result.get("provider_name"),
                "provider_version": canonical_result.get("provider_version"),
                "candidate_artifact_path": canonical_result.get("candidate_artifact_path"),
                "cache_file": canonical_result.get("cache_file"),
                "guidance": list(canonical_result.get("guidance") or []),
            },
            "errors": list(dict.fromkeys(errors)),
            "calibration_state": "SHADOW_UNCALIBRATED",
            "decision_influence": "NONE",
        }
