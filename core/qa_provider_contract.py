from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Optional


PROVIDER_CONTRACT_SCHEMA = "measurement_provider_contract_v0_1"
PROVIDER_COMPARISON_SCHEMA = "measurement_provider_comparison_v0_1"

_AXIS_REQUIRED_FIELDS = {
    "face_identity": [
        "provider_name",
        "provider_version",
        "model_id",
        "model_sha256",
        "execution_backend",
        "detector_contract_id",
        "alignment_contract_id",
        "preprocessing_contract_id",
        "dimension",
        "source_field",
        "measurement_normalization",
    ],
    "face_shape": [
        "provider_name",
        "provider_version",
        "model_id",
        "model_sha256",
        "implementation_sha256",
        "execution_backend",
        "landmark_schema_id",
        "coordinate_convention",
        "preprocessing_contract_id",
        "landmark_count",
        "source_field",
    ],
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNRESOLVED_MARKERS = (
    "unknown",
    "unrecorded",
    "unresolved",
    "unhashed",
    "not_recorded",
    "legacy_alias",
)


def _text(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _embedding_dimension(value: Any) -> Optional[int]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return len(value)


def _landmark_count(value: Any) -> Optional[int]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if isinstance(value[0], (list, tuple)):
        return len(value) if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value) else None
    return len(value) // 2 if len(value) >= 6 and len(value) % 2 == 0 else None


def _canonical_sha256(value: Dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _is_resolved(field: str, value: Any) -> bool:
    if value is None:
        return False
    if field.endswith("sha256"):
        return bool(_SHA256_RE.fullmatch(str(value).strip().lower()))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized or any(marker in normalized for marker in _UNRESOLVED_MARKERS):
            return False
    return True


def _identity_fields(artifact: Dict[str, Any]) -> Dict[str, Any]:
    source_contract = _mapping(artifact.get("runtime_face_embedding_contract"))
    embedding = artifact.get("runtime_face_embedding_raw")
    if embedding is None:
        embedding = artifact.get("canonical_identity_vector")
    return {
        "provider_name": _text(source_contract.get("provider_name")),
        "provider_version": _text(source_contract.get("provider_version")),
        "model_id": _text(source_contract.get("model_id") or source_contract.get("model_pack_id")),
        "model_sha256": _text(source_contract.get("model_sha256")),
        "execution_backend": _text(source_contract.get("execution_backend")),
        "detector_contract_id": _text(source_contract.get("detector_contract_id")),
        "alignment_contract_id": _text(source_contract.get("alignment_contract_id")),
        "preprocessing_contract_id": _text(source_contract.get("preprocessing_contract_id")),
        "dimension": _embedding_dimension(embedding),
        "source_field": _text(source_contract.get("source_field"))
        or "runtime_face_embedding_raw_or_legacy_alias",
        "measurement_normalization": "l2_at_measurement_adapter",
    }


def _shape_fields(artifact: Dict[str, Any]) -> Dict[str, Any]:
    shape_contract = _mapping(artifact.get("canonical_landmark_contract"))
    landmarks = artifact.get("canonical_landmarks")
    if landmarks is None:
        landmarks = artifact.get("landmarks_2d") or artifact.get("landmarks")
    return {
        "provider_name": _text(shape_contract.get("provider_name") or artifact.get("provider_name")),
        "provider_version": _text(shape_contract.get("provider_version") or artifact.get("provider_version")),
        "model_id": _text(shape_contract.get("model_id") or artifact.get("model_id")),
        "model_sha256": _text(shape_contract.get("model_sha256") or artifact.get("model_sha256")),
        "implementation_sha256": _text(
            shape_contract.get("implementation_sha256") or artifact.get("provider_implementation_sha256")
        ),
        "execution_backend": _text(
            shape_contract.get("execution_backend") or artifact.get("provider_execution_backend")
        ),
        "landmark_schema_id": _text(shape_contract.get("landmark_schema_id") or artifact.get("landmark_schema_id")),
        "coordinate_convention": _text(
            shape_contract.get("coordinate_convention") or artifact.get("landmark_coordinate_convention")
        ),
        "preprocessing_contract_id": _text(
            shape_contract.get("preprocessing_contract_id")
            or artifact.get("canonical_preprocessing_contract_id")
        ),
        "landmark_count": _landmark_count(landmarks),
        "source_field": _text(shape_contract.get("source_field") or artifact.get("landmark_source_field"))
        or "canonical_landmarks_or_legacy_alias",
    }


def build_face_provider_contract(artifact: Any, *, axis: str) -> Dict[str, Any]:
    normalized_axis = str(axis).strip().lower()
    if normalized_axis not in _AXIS_REQUIRED_FIELDS:
        raise ValueError(f"unsupported provider contract axis: {axis!r}")
    artifact_node = artifact if isinstance(artifact, dict) else {}
    fields = _identity_fields(artifact_node) if normalized_axis == "face_identity" else _shape_fields(artifact_node)
    required_fields = list(_AXIS_REQUIRED_FIELDS[normalized_axis])
    missing_fields = [field for field in required_fields if not _is_resolved(field, fields.get(field))]
    source_available = bool(
        fields.get("dimension") if normalized_axis == "face_identity" else fields.get("landmark_count")
    )
    if not source_available:
        completeness = "UNAVAILABLE"
    elif missing_fields:
        completeness = "PARTIAL"
    else:
        completeness = "COMPLETE"
    observed_fingerprint = _canonical_sha256({"axis": normalized_axis, "fields": fields})
    return {
        "schema_version": PROVIDER_CONTRACT_SCHEMA,
        "axis": normalized_axis,
        "completeness_state": completeness,
        "required_fields": required_fields,
        "missing_fields": missing_fields,
        "fields": fields,
        "observed_contract_sha256": observed_fingerprint,
        "comparable_contract_sha256": observed_fingerprint if completeness == "COMPLETE" else None,
        "decision_influence": "NONE",
    }


def compare_provider_contracts(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    reference_axis = str(reference.get("axis") or "")
    candidate_axis = str(candidate.get("axis") or "")
    if reference_axis != candidate_axis or reference_axis not in _AXIS_REQUIRED_FIELDS:
        raise ValueError("provider contracts must use the same supported axis")
    reference_fields = _mapping(reference.get("fields"))
    candidate_fields = _mapping(candidate.get("fields"))
    conflicts = []
    compared_fields = sorted(set(reference_fields) | set(candidate_fields))
    for field in compared_fields:
        reference_value = reference_fields.get(field)
        candidate_value = candidate_fields.get(field)
        if (
            _is_resolved(field, reference_value)
            and _is_resolved(field, candidate_value)
            and reference_value != candidate_value
        ):
            conflicts.append(
                {
                    "field": field,
                    "reference": reference_value,
                    "candidate": candidate_value,
                }
            )

    reference_state = str(reference.get("completeness_state") or "UNAVAILABLE")
    candidate_state = str(candidate.get("completeness_state") or "UNAVAILABLE")
    if conflicts:
        comparison_state = "MISMATCH"
    elif "UNAVAILABLE" in {reference_state, candidate_state}:
        comparison_state = "UNAVAILABLE"
    elif reference_state == "COMPLETE" and candidate_state == "COMPLETE":
        comparison_state = (
            "MATCH"
            if reference.get("comparable_contract_sha256") == candidate.get("comparable_contract_sha256")
            else "MISMATCH"
        )
    else:
        comparison_state = "PARTIAL_MATCH"

    return {
        "schema_version": PROVIDER_COMPARISON_SCHEMA,
        "axis": reference_axis,
        "comparison_state": comparison_state,
        "reference_completeness_state": reference_state,
        "candidate_completeness_state": candidate_state,
        "conflicts": conflicts,
        "comparable": comparison_state == "MATCH",
        "decision_influence": "NONE",
    }


def provider_contract_gap_codes(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    comparison: Dict[str, Any],
) -> list[str]:
    axis_prefix = str(comparison.get("axis") or "provider").upper()
    gaps: list[str] = []
    for role, contract in [("REFERENCE", reference), ("CANDIDATE", candidate)]:
        state = str(contract.get("completeness_state") or "UNAVAILABLE")
        if state != "COMPLETE":
            gaps.append(f"{axis_prefix}_{role}_PROVIDER_CONTRACT_{state}")
    for conflict in list(comparison.get("conflicts") or []):
        if isinstance(conflict, dict):
            field = str(conflict.get("field") or "UNKNOWN").upper()
            gaps.append(f"{axis_prefix}_PROVIDER_CONTRACT_CONFLICT_{field}")
    return list(dict.fromkeys(gaps))
