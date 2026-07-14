from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .qa_procrustes_shape import weighted_irls_procrustes
from .qa_repeatability_shadow import empty_repeatability_contract


IDENTITY_EVIDENCE_CONTRACT_VERSION = "identity_evidence_contract_vnext_shadow_v0_1"
MEASUREMENT_RECORD_SCHEMA = "identity_evidence_measurement_v0_1"
OBSERVATION_RECORD_SCHEMA = "identity_evidence_observation_v0_1"

ELIGIBILITY_STATES = {
    "MEASURABLE",
    "CONDITIONAL",
    "PRIOR_DEPENDENT",
    "UNOBSERVABLE",
    "UNAVAILABLE",
    "UNASSESSED",
}
CHAIN_STATES = {"CHAIN_VALID", "CHAIN_INVALID", "CHAIN_UNASSESSED"}


def shadow_governance() -> Dict[str, Any]:
    return {
        "mode": "SHADOW",
        "decision_influence": "NONE",
        "may_affect_ranking": False,
        "may_affect_review_route": False,
        "may_affect_winner_bank": False,
        "may_modify_truth": False,
        "parameter_fitting_allowed": False,
    }


def l2_unit_embedding(
    value: Any,
    *,
    epsilon: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    if value is None:
        return None, None, "EMBEDDING_MISSING"
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None, None, "EMBEDDING_INVALID"
    if vector.size == 0:
        return None, None, "EMBEDDING_EMPTY"
    if not bool(np.all(np.isfinite(vector))):
        return None, None, "EMBEDDING_NONFINITE"
    norm = float(np.linalg.norm(vector, ord=2))
    if norm <= float(epsilon):
        return None, norm, "EMBEDDING_ZERO_NORM"
    return vector / norm, norm, None


def angular_distance_radians(reference: Any, candidate: Any) -> Dict[str, Any]:
    reference_unit, reference_norm, reference_error = l2_unit_embedding(reference)
    candidate_unit, candidate_norm, candidate_error = l2_unit_embedding(candidate)
    errors = [error for error in [reference_error, candidate_error] if error]
    if reference_unit is not None and candidate_unit is not None and reference_unit.shape != candidate_unit.shape:
        errors.append("EMBEDDING_DIMENSION_MISMATCH")
    if errors:
        return {
            "available": False,
            "residual": None,
            "unit": "radian",
            "reference_l2_norm": reference_norm,
            "candidate_l2_norm": candidate_norm,
            "dimension": None,
            "cosine": None,
            "errors": errors,
        }

    assert reference_unit is not None and candidate_unit is not None
    cosine = float(np.clip(np.dot(reference_unit, candidate_unit), -1.0, 1.0))
    residual = float(math.acos(cosine))
    return {
        "available": True,
        "residual": residual,
        "unit": "radian",
        "reference_l2_norm": reference_norm,
        "candidate_l2_norm": candidate_norm,
        "dimension": int(reference_unit.size),
        "cosine": cosine,
        "errors": [],
    }


def build_face_identity_measurement(
    reference: Any,
    candidate: Any,
    *,
    provider_name: str,
    provider_version: str,
    model_id: str,
) -> Dict[str, Any]:
    result = angular_distance_radians(reference, candidate)
    return {
        "schema_version": MEASUREMENT_RECORD_SCHEMA,
        "contract_version": IDENTITY_EVIDENCE_CONTRACT_VERSION,
        "axis": "face_identity",
        "evidence_family": "face_identity",
        "native_space": "unit_hypersphere",
        "measurement": "runtime_aligned_face_angular_distance",
        "residual": result["residual"],
        "unit": "radian",
        "direction": "lower_is_more_consistent",
        "available": bool(result["available"]),
        "calibration_state": "SHADOW_UNCALIBRATED",
        "embedding_contract": {
            "source": "insightface_runtime_embedding",
            "normalization": "l2",
            "dimension": result["dimension"],
            "reference_l2_norm": result["reference_l2_norm"],
            "candidate_l2_norm": result["candidate_l2_norm"],
            "cosine": result["cosine"],
            "provider_name": str(provider_name),
            "provider_version": str(provider_version),
            "model_id": str(model_id),
        },
        "errors": list(result["errors"]),
        "repeatability": empty_repeatability_contract(),
        "independence_contract": {
            "native_evidence_unit": "runtime_face_embedding",
            "diagnostics_are_independent_evidence": False,
        },
        "decision_influence": "NONE",
    }


def build_face_projection_shape_measurement(
    reference: Any,
    candidate: Any,
    *,
    visibility_weights: Any,
    provider_name: str,
    provider_version: str,
    model_id: str,
    visibility_weight_source: str = "uniform_unweighted_fallback",
    landmark_schema_id: Optional[str] = None,
    correspondence_contract_state: str = "UNRECORDED",
) -> Dict[str, Any]:
    result = weighted_irls_procrustes(
        reference,
        candidate,
        visibility_weights=visibility_weights,
    )
    available = bool(result.get("available"))
    error = str(result.get("error") or "").strip()
    return {
        "schema_version": MEASUREMENT_RECORD_SCHEMA,
        "contract_version": IDENTITY_EVIDENCE_CONTRACT_VERSION,
        "axis": "face_shape",
        "evidence_family": "face_geometry",
        "native_space": "similarity_shape_space_2d",
        "measurement": "weighted_irls_procrustes_projection_residual",
        "residual": result.get("residual") if available else None,
        "unit": "normalized_shape_distance",
        "direction": "lower_is_more_consistent",
        "available": available,
        "calibration_state": "SHADOW_UNCALIBRATED",
        "shape_contract": {
            "solver_schema_version": result.get("schema_version"),
            "geometry_semantics": "canonical_projection_geometry_not_absolute_3d_truth",
            "translation_removed": True,
            "uniform_scale_removed": True,
            "rotation_group": "SO(2)",
            "reflection_allowed": False,
            "landmark_count": result.get("landmark_count"),
            "landmark_schema_id": str(landmark_schema_id or "").strip() or None,
            "correspondence_contract_state": str(correspondence_contract_state).strip().upper(),
            "visibility_coverage": result.get("visibility_coverage"),
            "visibility_weight_source": str(visibility_weight_source),
            "effective_weight_share": result.get("effective_weight_share"),
            "huber_delta": result.get("huber_delta"),
            "max_iterations": result.get("max_iterations"),
            "tolerance": result.get("tolerance"),
            "iterations": result.get("iterations"),
            "converged": result.get("converged"),
            "partition_semantics": "coordinate_quantile_diagnostics_not_anatomical_regions",
            "provider_name": str(provider_name),
            "provider_version": str(provider_version),
            "model_id": str(model_id),
        },
        "raw_rms_residual": result.get("raw_rms_residual"),
        "alignment_transform": {
            "rotation_matrix": result.get("rotation_matrix"),
        },
        "partition_diagnostics": dict(result.get("partition_diagnostics") or {}),
        "errors": [error] if error else [],
        "repeatability": empty_repeatability_contract(),
        "independence_contract": {
            "native_evidence_unit": "face_projection_shape",
            "partition_diagnostics_are_independent_evidence": False,
        },
        "decision_influence": "NONE",
    }


def build_axis_observation(
    *,
    axis: str,
    eligibility: str,
    scope_state: str,
    chain_state: str,
    raw_observations: Dict[str, Any],
    reasons: Optional[list[str]] = None,
) -> Dict[str, Any]:
    normalized_eligibility = str(eligibility).strip().upper()
    normalized_chain_state = str(chain_state).strip().upper()
    if normalized_eligibility not in ELIGIBILITY_STATES:
        raise ValueError(f"unsupported eligibility state: {eligibility!r}")
    if normalized_chain_state not in CHAIN_STATES:
        raise ValueError(f"unsupported chain state: {chain_state!r}")
    return {
        "schema_version": OBSERVATION_RECORD_SCHEMA,
        "contract_version": IDENTITY_EVIDENCE_CONTRACT_VERSION,
        "axis": str(axis),
        "eligibility": normalized_eligibility,
        "scope_state": str(scope_state).strip().upper(),
        "observation_chain_state": normalized_chain_state,
        "raw_observations": dict(raw_observations),
        "reasons": list(dict.fromkeys(reasons or [])),
        "decision_influence": "NONE",
    }


def validate_shadow_axis_record(record: Dict[str, Any]) -> list[str]:
    issues: list[str] = []
    observation = record.get("observation") if isinstance(record.get("observation"), dict) else {}
    measurement = record.get("measurement") if isinstance(record.get("measurement"), dict) else {}
    observation_axis = str(observation.get("axis") or "")
    measurement_axis = str(measurement.get("axis") or "")
    if observation.get("decision_influence") != "NONE":
        issues.append("OBSERVATION_DECISION_INFLUENCE_MUST_BE_NONE")
    if measurement and measurement.get("decision_influence") != "NONE":
        issues.append("MEASUREMENT_DECISION_INFLUENCE_MUST_BE_NONE")
    if observation.get("eligibility") not in ELIGIBILITY_STATES:
        issues.append("OBSERVATION_ELIGIBILITY_INVALID")
    if observation.get("observation_chain_state") not in CHAIN_STATES:
        issues.append("OBSERVATION_CHAIN_STATE_INVALID")
    if measurement and measurement.get("calibration_state") != "SHADOW_UNCALIBRATED":
        issues.append("MEASUREMENT_CALIBRATION_STATE_INVALID")
    if measurement and observation_axis != measurement_axis:
        issues.append("OBSERVATION_MEASUREMENT_AXIS_MISMATCH")
    residual = measurement.get("residual") if measurement else None
    if measurement.get("available"):
        try:
            residual_is_finite = residual is not None and math.isfinite(float(residual))
        except (TypeError, ValueError):
            residual_is_finite = False
        if not residual_is_finite:
            issues.append("AVAILABLE_MEASUREMENT_REQUIRES_FINITE_RESIDUAL")
    elif residual is not None:
        issues.append("UNAVAILABLE_MEASUREMENT_MUST_WITHHOLD_RESIDUAL")
    if observation.get("eligibility") == "UNOBSERVABLE" and measurement.get("available"):
        issues.append("UNOBSERVABLE_AXIS_MUST_WITHHOLD_MEASUREMENT")
    repeatability = measurement.get("repeatability") if isinstance(measurement.get("repeatability"), dict) else {}
    domains = repeatability.get("domains") if isinstance(repeatability.get("domains"), dict) else {}
    required_repeatability_domains = {
        "numerical_repeatability",
        "preprocessing_repeatability",
        "admissible_perturbation_stability",
    }
    if set(domains) != required_repeatability_domains:
        issues.append("REPEATABILITY_DOMAINS_INCOMPLETE")
    if repeatability.get("combined_repeatability_score") is not None:
        issues.append("COMBINED_REPEATABILITY_SCORE_NOT_ALLOWED")
    if repeatability.get("decision_influence") != "NONE":
        issues.append("REPEATABILITY_DECISION_INFLUENCE_MUST_BE_NONE")
    provider_contracts = (
        measurement.get("provider_contracts")
        if isinstance(measurement.get("provider_contracts"), dict)
        else {}
    )
    if provider_contracts:
        for role in ["reference", "candidate", "comparison"]:
            node = provider_contracts.get(role) if isinstance(provider_contracts.get(role), dict) else {}
            if node.get("decision_influence") != "NONE":
                issues.append(f"{role.upper()}_PROVIDER_CONTRACT_DECISION_INFLUENCE_MUST_BE_NONE")
    return issues
