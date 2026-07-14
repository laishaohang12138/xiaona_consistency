from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from .qa_identity_evidence_contract import CHAIN_STATES, ELIGIBILITY_STATES, shadow_governance
from .qa_repeatability_shadow import empty_body_repeatability_contract


BODY_EVIDENCE_CONTRACT_VERSION = "body_evidence_contract_vnext_shadow_v0_2"
BODY_OBSERVATION_RECORD_SCHEMA = "body_evidence_observation_v0_1"
BODY_MEASUREMENT_RECORD_SCHEMA = "body_evidence_measurement_v0_1"

BODY_CORE_MEASUREMENT_ORDER = [
    "shoulder_width_to_torso",
    "hip_width_to_torso",
    "shoulder_to_hip_ratio",
    "upper_to_lower_leg_ratio",
    "foot_length_to_leg",
]
MIN_BODY_CORE_COMPONENTS = 3


def body_shadow_governance() -> Dict[str, Any]:
    return {
        **shadow_governance(),
        "body_truth_semantics": "pose_gait_conditioned_absolute_116_1",
        "pose_gait_is_condition_not_truth": True,
        "surface_occlusion_is_reliability_not_identity": True,
        "scalar_aggregation_allowed": False,
    }


def _empty_body_topology_repeatability_contract() -> Dict[str, Any]:
    contract = empty_body_repeatability_contract()
    contract.update(
        {
            "measurement_axis": "body_topology",
            "protocol_scope": "SHARED_HMR2_EXECUTION_DISTINCT_TOPOLOGY_SUMMARY",
        }
    )
    return contract


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def log_ratio_residual_vector(
    reference_measurements: Any,
    candidate_measurements: Any,
    *,
    measurement_order: Sequence[str] = BODY_CORE_MEASUREMENT_ORDER,
    min_components: int = MIN_BODY_CORE_COMPONENTS,
) -> Dict[str, Any]:
    reference = reference_measurements if isinstance(reference_measurements, dict) else {}
    candidate = candidate_measurements if isinstance(candidate_measurements, dict) else {}
    order = [str(name).strip() for name in measurement_order if str(name).strip()]
    component_residuals: Dict[str, float] = {}
    missing_components = []
    invalid_components = []
    for name in order:
        if name not in reference or name not in candidate:
            missing_components.append(name)
            continue
        reference_value = _finite_positive(reference.get(name))
        candidate_value = _finite_positive(candidate.get(name))
        if reference_value is None or candidate_value is None:
            invalid_components.append(name)
            continue
        component_residuals[name] = float(math.log(candidate_value / reference_value))

    used_components = [name for name in order if name in component_residuals]
    available = len(used_components) >= max(1, int(min_components))
    return {
        "available": available,
        "residual_vector": [component_residuals[name] for name in used_components]
        if available
        else None,
        "component_residuals": component_residuals if available else {},
        "used_components": used_components if available else [],
        "missing_components": missing_components,
        "invalid_components": invalid_components,
        "component_count": len(used_components),
        "required_component_count": max(1, int(min_components)),
        "coverage": float(len(used_components) / max(1, len(order))),
        "unit": "natural_log_ratio",
        "errors": [] if available else ["INSUFFICIENT_VALID_BODY_CORE_COMPONENTS"],
    }


def build_body_core_shape_measurement(
    reference_measurements: Any,
    candidate_measurements: Any,
    *,
    provider_name: str,
    provider_version: str,
    model_id: str,
) -> Dict[str, Any]:
    result = log_ratio_residual_vector(reference_measurements, candidate_measurements)
    return {
        "schema_version": BODY_MEASUREMENT_RECORD_SCHEMA,
        "contract_version": BODY_EVIDENCE_CONTRACT_VERSION,
        "axis": "body_core_shape",
        "evidence_family": "body_shape_geometry",
        "native_space": "positive_anthropometric_ratio_log_space",
        "measurement": "componentwise_body_core_log_ratio_residual",
        "residual": None,
        "residual_vector": result["residual_vector"],
        "component_residuals": result["component_residuals"],
        "used_components": result["used_components"],
        "missing_components": result["missing_components"],
        "invalid_components": result["invalid_components"],
        "component_count": result["component_count"],
        "required_component_count": result["required_component_count"],
        "coverage": result["coverage"],
        "unit": result["unit"],
        "direction": "two_sided_zero_is_same_shape_ratio",
        "available": bool(result["available"]),
        "calibration_state": "SHADOW_UNCALIBRATED",
        "scalar_residual_authorized": False,
        "provider_descriptor": {
            "provider_name": str(provider_name),
            "provider_version": str(provider_version),
            "model_id": str(model_id),
            "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
        },
        "errors": list(result["errors"]),
        "repeatability": empty_body_repeatability_contract(),
        "independence_contract": {
            "native_evidence_unit": "body_core_shape_measurement_vector",
            "components_are_independent_votes": False,
            "partition_diagnostics_are_independent_evidence": False,
        },
        "decision_influence": "NONE",
    }


def _canonical_vertex_matrix(value: Any) -> Optional[list[list[float]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    vertices: list[list[float]] = []
    for raw_vertex in value:
        if not isinstance(raw_vertex, (list, tuple)) or len(raw_vertex) != 3:
            return None
        try:
            vertex = [float(component) for component in raw_vertex]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(component) for component in vertex):
            return None
        vertices.append(vertex)
    return vertices


def canonical_vertex_delta_vector(
    reference_vertices: Any,
    candidate_vertices: Any,
) -> Dict[str, Any]:
    reference = _canonical_vertex_matrix(reference_vertices)
    candidate = _canonical_vertex_matrix(candidate_vertices)
    errors: list[str] = []
    if reference is None:
        errors.append("REFERENCE_CANONICAL_SMPL_VERTICES_INVALID_OR_UNAVAILABLE")
    if candidate is None:
        errors.append("CANDIDATE_CANONICAL_SMPL_VERTICES_INVALID_OR_UNAVAILABLE")
    if reference is not None and candidate is not None and len(reference) != len(candidate):
        errors.append("CANONICAL_SMPL_VERTEX_COUNT_MISMATCH")
    if errors:
        return {
            "available": False,
            "residual_vector": None,
            "vertex_count": None,
            "coordinate_count": None,
            "reference_centroid": None,
            "candidate_centroid": None,
            "errors": errors,
        }

    assert reference is not None and candidate is not None
    vertex_count = len(reference)
    reference_centroid = [
        sum(vertex[axis] for vertex in reference) / vertex_count for axis in range(3)
    ]
    candidate_centroid = [
        sum(vertex[axis] for vertex in candidate) / vertex_count for axis in range(3)
    ]
    residual_vector = [
        (candidate[index][axis] - candidate_centroid[axis])
        - (reference[index][axis] - reference_centroid[axis])
        for index in range(vertex_count)
        for axis in range(3)
    ]
    return {
        "available": True,
        "residual_vector": residual_vector,
        "vertex_count": vertex_count,
        "coordinate_count": vertex_count * 3,
        "reference_centroid": reference_centroid,
        "candidate_centroid": candidate_centroid,
        "errors": [],
    }


def build_body_topology_measurement(
    reference_vertices: Any,
    candidate_vertices: Any,
    *,
    provider_name: str,
    provider_version: str,
    model_id: str,
) -> Dict[str, Any]:
    result = canonical_vertex_delta_vector(reference_vertices, candidate_vertices)
    return {
        "schema_version": BODY_MEASUREMENT_RECORD_SCHEMA,
        "contract_version": BODY_EVIDENCE_CONTRACT_VERSION,
        "axis": "body_topology",
        "evidence_family": "body_shape_geometry",
        "native_space": "smpl_neutral_zero_pose_model_space_translation_centered",
        "measurement": "componentwise_corresponding_vertex_coordinate_delta",
        "residual": None,
        "residual_vector": result["residual_vector"],
        "component_residuals": {},
        "used_components": [],
        "vertex_count": result["vertex_count"],
        "coordinate_count": result["coordinate_count"],
        "reference_centroid": result["reference_centroid"],
        "candidate_centroid": result["candidate_centroid"],
        "coverage": 1.0 if result["available"] else 0.0,
        "unit": "smpl_model_length_unit",
        "direction": "signed_candidate_minus_reference_xyz_zero_is_same_topology",
        "available": bool(result["available"]),
        "calibration_state": "SHADOW_UNCALIBRATED",
        "scalar_residual_authorized": False,
        "alignment_contract": {
            "translation": "centroid_removed_independently",
            "rotation_fit_applied": False,
            "scale_fit_applied": False,
            "procrustes_fit_applied": False,
            "pose_fit_applied": False,
        },
        "provider_descriptor": {
            "provider_name": str(provider_name),
            "provider_version": str(provider_version),
            "model_id": str(model_id),
            "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
        },
        "errors": list(result["errors"]),
        "repeatability": _empty_body_topology_repeatability_contract(),
        "independence_contract": {
            "native_evidence_unit": "zero_pose_canonical_smpl_vertex_delta_vector",
            "components_are_independent_votes": False,
            "independent_from_body_core_shape": False,
            "shared_provider_family": "HMR2_SMPL_RECONSTRUCTION",
        },
        "decision_influence": "NONE",
    }


def build_body_axis_observation(
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
        "schema_version": BODY_OBSERVATION_RECORD_SCHEMA,
        "contract_version": BODY_EVIDENCE_CONTRACT_VERSION,
        "axis": str(axis),
        "eligibility": normalized_eligibility,
        "scope_state": str(scope_state).strip().upper(),
        "observation_chain_state": normalized_chain_state,
        "raw_observations": dict(raw_observations),
        "reasons": list(dict.fromkeys(reasons or [])),
        "decision_influence": "NONE",
    }


def body_core_observation_scope(
    lane_family: str,
    *,
    measurement_available: bool,
) -> tuple[str, str, list[str]]:
    lane = str(lane_family or "unknown").strip().lower()
    if not measurement_available:
        return (
            "UNAVAILABLE",
            "BODY_CORE_MEASUREMENT_UNAVAILABLE",
            ["BODY_CANONICAL_ARTIFACT_OR_COMPONENTS_UNAVAILABLE"],
        )
    if lane == "front":
        return (
            "PRIOR_DEPENDENT",
            "FRONT_HMR2_PRIOR_DEPENDENT",
            ["HMR2_SMPL_RECONSTRUCTION_PRIOR_DEPENDENT"],
        )
    if lane == "three_quarter":
        return (
            "PRIOR_DEPENDENT",
            "THREE_QUARTER_HMR2_PRIOR_DEPENDENT",
            ["VIEW_AND_POSE_CONDITIONED_BODY_RECONSTRUCTION"],
        )
    if lane == "side":
        return (
            "PRIOR_DEPENDENT",
            "SIDE_HMR2_PRIOR_DEPENDENT",
            ["SIDE_BODY_RECONSTRUCTION_PRIOR_DEPENDENT"],
        )
    if lane == "back":
        return (
            "PRIOR_DEPENDENT",
            "BACK_HMR2_PRIOR_DEPENDENT",
            ["BACK_BODY_RECONSTRUCTION_PRIOR_DEPENDENT"],
        )
    return "UNASSESSED", "BODY_LANE_UNASSESSED", ["BODY_LANE_UNASSESSED"]


def body_topology_observation_scope(
    lane_family: str,
    *,
    measurement_available: bool,
) -> tuple[str, str, list[str]]:
    eligibility, _core_scope, reasons = body_core_observation_scope(
        lane_family,
        measurement_available=measurement_available,
    )
    if not measurement_available:
        return (
            eligibility,
            "BODY_TOPOLOGY_MEASUREMENT_UNAVAILABLE",
            ["NATIVE_CANONICAL_SMPL_TOPOLOGY_UNAVAILABLE"],
        )
    lane = str(lane_family or "unknown").strip().upper()
    topology_scope = (
        f"{lane}_NATIVE_TOPOLOGY_HMR2_PRIOR_DEPENDENT"
        if lane in {"FRONT", "THREE_QUARTER", "SIDE", "BACK"}
        else "BODY_TOPOLOGY_LANE_UNASSESSED"
    )
    return eligibility, topology_scope, list(
        dict.fromkeys(reasons + ["SHARED_HMR2_SMPL_PRIOR_WITH_BODY_CORE_SHAPE"])
    )


def build_pose_gait_condition(reference_pose: Any, candidate_pose: Any) -> Dict[str, Any]:
    reference = list(reference_pose) if isinstance(reference_pose, (list, tuple)) else []
    candidate = list(candidate_pose) if isinstance(candidate_pose, (list, tuple)) else []
    errors = []
    delta_vector = None
    if not reference or not candidate:
        errors.append("POSE_VECTOR_UNAVAILABLE")
    elif len(reference) != len(candidate):
        errors.append("POSE_VECTOR_DIMENSION_MISMATCH")
    else:
        try:
            values = [float(c) - float(r) for r, c in zip(reference, candidate)]
            if all(math.isfinite(value) for value in values):
                delta_vector = values
            else:
                errors.append("POSE_VECTOR_NONFINITE")
        except (TypeError, ValueError):
            errors.append("POSE_VECTOR_INVALID")
    return {
        "schema_version": "pose_gait_condition_v0_1",
        "condition_role": "NUISANCE_DESCRIPTOR_NOT_CONSISTENCY_VOTE",
        "available": delta_vector is not None,
        "native_space": "provider_model_pose_parameter_space",
        "delta_vector": delta_vector,
        "dimension": len(delta_vector) if delta_vector is not None else None,
        "unit": "provider_native_pose_parameter",
        "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
        "errors": errors,
        "decision_influence": "NONE",
    }


def build_surface_occlusion_condition(
    *,
    body_coverage: Any = None,
    visible_body_ratio: Any = None,
    garment_coverage_ratio: Any = None,
) -> Dict[str, Any]:
    observations: Dict[str, float] = {}
    for name, value in {
        "body_canonical_coverage": body_coverage,
        "visible_body_ratio": visible_body_ratio,
        "garment_coverage_ratio": garment_coverage_ratio,
    }.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            observations[name] = number
    return {
        "schema_version": "surface_occlusion_condition_v0_1",
        "condition_role": "OBSERVATION_RELIABILITY_NOT_IDENTITY_VOTE",
        "available": bool(observations),
        "observations": observations,
        "threshold_classification": None,
        "calibration_state": "SHADOW_UNCALIBRATED",
        "decision_influence": "NONE",
    }


def validate_body_shadow_axis_record(record: Dict[str, Any]) -> list[str]:
    issues: list[str] = []
    observation = record.get("observation") if isinstance(record.get("observation"), dict) else {}
    measurement = record.get("measurement") if isinstance(record.get("measurement"), dict) else {}
    if observation.get("decision_influence") != "NONE":
        issues.append("OBSERVATION_DECISION_INFLUENCE_MUST_BE_NONE")
    if measurement.get("decision_influence") != "NONE":
        issues.append("MEASUREMENT_DECISION_INFLUENCE_MUST_BE_NONE")
    if observation.get("axis") != measurement.get("axis"):
        issues.append("OBSERVATION_MEASUREMENT_AXIS_MISMATCH")
    if observation.get("eligibility") not in ELIGIBILITY_STATES:
        issues.append("OBSERVATION_ELIGIBILITY_INVALID")
    if observation.get("observation_chain_state") not in CHAIN_STATES:
        issues.append("OBSERVATION_CHAIN_STATE_INVALID")
    if measurement.get("calibration_state") != "SHADOW_UNCALIBRATED":
        issues.append("MEASUREMENT_CALIBRATION_STATE_INVALID")
    axis_code = "BODY_TOPOLOGY" if measurement.get("axis") == "body_topology" else "BODY_CORE"
    if measurement.get("residual") is not None:
        issues.append(f"SCALAR_{axis_code}_RESIDUAL_NOT_ALLOWED")
    vector = measurement.get("residual_vector")
    if measurement.get("available"):
        if not isinstance(vector, list) or not vector:
            issues.append("AVAILABLE_MEASUREMENT_REQUIRES_RESIDUAL_VECTOR")
        else:
            try:
                if not all(math.isfinite(float(value)) for value in vector):
                    issues.append(f"{axis_code}_RESIDUAL_VECTOR_NONFINITE")
            except (TypeError, ValueError):
                issues.append(f"{axis_code}_RESIDUAL_VECTOR_INVALID")
    elif vector is not None:
        issues.append("UNAVAILABLE_MEASUREMENT_MUST_WITHHOLD_RESIDUAL_VECTOR")
    if measurement.get("scalar_residual_authorized") is not False:
        issues.append(f"SCALAR_{axis_code}_RESIDUAL_AUTHORIZATION_INVALID")
    repeatability = measurement.get("repeatability") if isinstance(measurement.get("repeatability"), dict) else {}
    domains = repeatability.get("domains") if isinstance(repeatability.get("domains"), dict) else {}
    if set(domains) != {
        "numerical_repeatability",
        "preprocessing_repeatability",
        "admissible_perturbation_stability",
    }:
        issues.append("REPEATABILITY_DOMAINS_INCOMPLETE")
    if repeatability.get("combined_repeatability_score") is not None:
        issues.append("COMBINED_REPEATABILITY_SCORE_NOT_ALLOWED")
    if repeatability.get("decision_influence") != "NONE":
        issues.append("REPEATABILITY_DECISION_INFLUENCE_MUST_BE_NONE")
    provider_contracts = measurement.get("provider_contracts")
    if isinstance(provider_contracts, dict):
        for role in ["reference", "candidate", "comparison"]:
            node = provider_contracts.get(role) if isinstance(provider_contracts.get(role), dict) else {}
            if node.get("decision_influence") != "NONE":
                issues.append(f"{role.upper()}_PROVIDER_CONTRACT_DECISION_INFLUENCE_MUST_BE_NONE")
    return issues
