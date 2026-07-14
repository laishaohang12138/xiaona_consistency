from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_evidence_lineage import EvidenceLineageGraph
from .qa_identity_evidence_contract import (
    IDENTITY_EVIDENCE_CONTRACT_VERSION,
    build_axis_observation,
    build_face_identity_measurement,
    build_face_projection_shape_measurement,
    shadow_governance,
    validate_shadow_axis_record,
)
from .qa_io import atomic_write_json
from .qa_provider_contract import (
    build_face_provider_contract,
    compare_provider_contracts,
    provider_contract_gap_codes,
)
from .qa_repeatability_shadow import repeatability_protocol_snapshot
from .qa_runtime import anchor_registry_snapshot


IDENTITY_EVIDENCE_SHADOW_SCHEMA = "identity_evidence_shadow_v0_1"
_OUTPUT_NAME = "identity_evidence_shadow.json"


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _load_artifact(path_value: Any) -> Optional[Dict[str, Any]]:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    nested = payload.get("artifact")
    return nested if isinstance(nested, dict) else payload


def _artifact_embedding(artifact: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(artifact, dict):
        return None
    for key in [
        "runtime_face_embedding_raw",
        "canonical_identity_vector",
        "identity_vector",
        "face_embedding",
    ]:
        value = artifact.get(key)
        if value is not None:
            return value
    return None


def _artifact_landmarks(artifact: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(artifact, dict):
        return None
    for key in ["canonical_landmarks", "landmarks_2d", "landmarks"]:
        value = artifact.get(key)
        if value is not None:
            return value
    return None


def _landmark_count(value: Any) -> Optional[int]:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if isinstance(value[0], (list, tuple)):
        return len(value) if all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value) else None
    return len(value) // 2 if len(value) >= 6 and len(value) % 2 == 0 else None


def _artifact_landmark_weights(artifact: Optional[Dict[str, Any]], count: Optional[int]) -> Optional[List[float]]:
    if not isinstance(artifact, dict) or count is None:
        return None
    value = None
    for key in ["landmark_visibility_weights", "landmark_weights", "landmark_confidence"]:
        if artifact.get(key) is not None:
            value = artifact.get(key)
            break
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    try:
        weights = [max(0.0, float(weight)) for weight in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(weight) for weight in weights) or sum(weight > 0.0 for weight in weights) < 3:
        return None
    return weights


def _pairwise_landmark_weights(
    master_artifact: Optional[Dict[str, Any]],
    candidate_artifact: Optional[Dict[str, Any]],
    reference_landmarks: Any,
    candidate_landmarks: Any,
) -> tuple[Optional[List[float]], str, List[str]]:
    count = _landmark_count(reference_landmarks)
    candidate_count = _landmark_count(candidate_landmarks)
    if count is None or candidate_count is None or count != candidate_count:
        return None, "unavailable", []
    master_weights = _artifact_landmark_weights(master_artifact, count)
    candidate_weights = _artifact_landmark_weights(candidate_artifact, count)
    if master_weights is not None and candidate_weights is not None:
        return (
            [min(reference, candidate) for reference, candidate in zip(master_weights, candidate_weights)],
            "pairwise_min_visibility",
            [],
        )
    if master_weights is not None:
        return master_weights, "master_visibility_only", ["CANDIDATE_LANDMARK_VISIBILITY_WEIGHTS_UNAVAILABLE"]
    if candidate_weights is not None:
        return candidate_weights, "candidate_visibility_only", ["MASTER_LANDMARK_VISIBILITY_WEIGHTS_UNAVAILABLE"]
    return None, "uniform_unweighted_fallback", ["PER_LANDMARK_VISIBILITY_WEIGHTS_UNAVAILABLE"]


def _pairwise_landmark_schema(
    master_artifact: Optional[Dict[str, Any]],
    candidate_artifact: Optional[Dict[str, Any]],
) -> tuple[Optional[str], str, List[str]]:
    master_schema = (
        str(master_artifact.get("landmark_schema_id") or "").strip()
        if isinstance(master_artifact, dict)
        else ""
    )
    candidate_schema = (
        str(candidate_artifact.get("landmark_schema_id") or "").strip()
        if isinstance(candidate_artifact, dict)
        else ""
    )
    if master_schema and candidate_schema and master_schema != candidate_schema:
        return None, "MISMATCH", ["LANDMARK_SCHEMA_MISMATCH"]
    if master_schema and candidate_schema:
        return master_schema, "MATCHED", []
    if master_schema or candidate_schema:
        return (
            master_schema or candidate_schema,
            "PARTIAL",
            ["LANDMARK_SCHEMA_ID_PARTIALLY_RECORDED"],
        )
    return None, "UNRECORDED", ["LANDMARK_CORRESPONDENCE_CONTRACT_UNRECORDED"]


def _withhold_shape_measurement(measurement: Dict[str, Any], error: str) -> None:
    measurement["available"] = False
    measurement["residual"] = None
    measurement["raw_rms_residual"] = None
    measurement["partition_diagnostics"] = {}
    alignment = measurement.get("alignment_transform")
    if isinstance(alignment, dict):
        alignment["rotation_matrix"] = None
    measurement["errors"] = [str(error)]


def _withhold_identity_measurement(measurement: Dict[str, Any], error: str) -> None:
    measurement["available"] = False
    measurement["residual"] = None
    embedding_contract = measurement.get("embedding_contract")
    if isinstance(embedding_contract, dict):
        embedding_contract["cosine"] = None
    measurement["errors"] = [str(error)]


def _attach_provider_contracts(
    measurement: Dict[str, Any],
    reference_contract: Dict[str, Any],
    candidate_contract: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    measurement["provider_contracts"] = {
        "reference": reference_contract,
        "candidate": candidate_contract,
        "comparison": comparison,
    }


def _lane_family(item: Dict[str, Any]) -> str:
    breakdown = item.get("review_only_breakdown_v2")
    if isinstance(breakdown, dict):
        lane = str(breakdown.get("observed_lane_family") or "").strip().lower()
        if lane:
            return lane
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    detail = str(debug.get("view_lane_detail") or debug.get("view_lane") or "").strip().lower()
    if detail == "front":
        return "front"
    if "three" in detail or "quarter" in detail or detail == "3q":
        return "three_quarter"
    if "side" in detail or "profile" in detail:
        return "side"
    if "back" in detail:
        return "back"
    return "unknown"


def _face_identity_scope(lane_family: str, measurement_available: bool) -> tuple[str, str, List[str]]:
    if lane_family == "back":
        return "UNOBSERVABLE", "BACK_FACE_UNOBSERVABLE", ["FACE_NOT_OBSERVABLE_IN_BACK_VIEW"]
    if not measurement_available:
        return "UNAVAILABLE", "FACE_MEASUREMENT_UNAVAILABLE", ["FACE_IDENTITY_ARTIFACT_UNAVAILABLE"]
    if lane_family == "front":
        return "MEASURABLE", "FRONT_SUPPORTED", []
    if lane_family == "three_quarter":
        return "CONDITIONAL", "THREE_QUARTER_CONDITIONAL", ["POSE_CONDITIONED_FACE_OBSERVATION"]
    if lane_family == "side":
        return "PRIOR_DEPENDENT", "SIDE_PRIOR_DEPENDENT", ["SIDE_IDENTITY_ENCODER_PRIOR_DEPENDENT"]
    return "UNASSESSED", "LANE_UNASSESSED", ["FACE_LANE_UNASSESSED"]


def _face_shape_scope(lane_family: str, measurement_available: bool) -> tuple[str, str, List[str]]:
    if lane_family == "back":
        return "UNOBSERVABLE", "BACK_FACE_SHAPE_UNOBSERVABLE", ["FACE_SHAPE_NOT_OBSERVABLE_IN_BACK_VIEW"]
    if not measurement_available:
        return "UNAVAILABLE", "FACE_SHAPE_MEASUREMENT_UNAVAILABLE", ["FACE_LANDMARK_ARTIFACT_UNAVAILABLE"]
    if lane_family == "front":
        return "MEASURABLE", "FRONT_PROJECTION_SUPPORTED", []
    if lane_family == "three_quarter":
        return "CONDITIONAL", "THREE_QUARTER_PROJECTION_CONDITIONAL", ["POSE_CONDITIONED_FACE_SHAPE_OBSERVATION"]
    if lane_family == "side":
        return "PRIOR_DEPENDENT", "SIDE_CANONICALIZATION_PRIOR_DEPENDENT", ["SIDE_FACE_SHAPE_MODEL_PRIOR_DEPENDENT"]
    return "UNASSESSED", "FACE_SHAPE_LANE_UNASSESSED", ["FACE_SHAPE_LANE_UNASSESSED"]


def _raw_metric(
    value: Any,
    *,
    unit: str,
    algorithm_id: str,
    input_region: str,
    image_scale: str,
    provider_version: str,
) -> Optional[Dict[str, Any]]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return {
        "value": numeric,
        "unit": unit,
        "algorithm_id": algorithm_id,
        "input_region": input_region,
        "image_scale": image_scale,
        "provider_version": provider_version,
    }


def _build_raw_observations(debug: Dict[str, Any], face_shadow: Dict[str, Any]) -> Dict[str, Any]:
    input_shape = debug.get("input_shape") if isinstance(debug.get("input_shape"), list) else []
    image_scale = "x".join(str(value) for value in input_shape[:2]) if input_shape else "unknown"
    face_provider_version = str(face_shadow.get("provider_version") or "unrecorded")
    candidates = {
        "face_area_ratio": _raw_metric(
            debug.get("candidate_face_bbox_area_ratio"),
            unit="image_area_ratio",
            algorithm_id="runtime_face_detector_bbox_v1",
            input_region="full_image",
            image_scale=image_scale,
            provider_version="runtime_face_engine_unrecorded",
        ),
        "face_laplacian_variance": _raw_metric(
            debug.get("candidate_face_lap_var"),
            unit="laplacian_variance",
            algorithm_id="opencv_laplacian_v1",
            input_region="detected_face_crop",
            image_scale="128x128",
            provider_version="qa_features_v1",
        ),
        "face_high_frequency_energy": _raw_metric(
            debug.get("candidate_face_hf_energy"),
            unit="high_frequency_energy",
            algorithm_id="qa_high_frequency_energy_v1",
            input_region="detected_face_crop",
            image_scale="128x128",
            provider_version="qa_features_v1",
        ),
        "visible_face_coverage_estimate": _raw_metric(
            face_shadow.get("visible_face_coverage"),
            unit="provider_estimated_ratio",
            algorithm_id="face_canonical_visible_coverage_v1",
            input_region="face_canonical_artifact",
            image_scale="provider_native",
            provider_version=face_provider_version,
        ),
    }
    return {key: value for key, value in candidates.items() if value is not None}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_identity_evidence_shadow(runtime: Any, report_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph = EvidenceLineageGraph()
    anchor_snapshot = anchor_registry_snapshot(runtime.config)
    truth_anchors = anchor_snapshot.get("truth_anchors") if isinstance(anchor_snapshot.get("truth_anchors"), dict) else {}
    face_truth = truth_anchors.get("face_identity") if isinstance(truth_anchors.get("face_identity"), dict) else {}
    face_truth_path = str(face_truth.get("resolved_path") or "")
    face_truth_node_id = _stable_id("truth_face", str(face_truth.get("actual_sha256") or face_truth_path))
    graph.add_node(
        face_truth_node_id,
        "TRUTH_ASSET",
        evidence_family=None,
        attributes={
            "path": face_truth_path,
            "sha256": face_truth.get("actual_sha256"),
            "authority": face_truth.get("authority"),
        },
    )

    item_records: List[Dict[str, Any]] = []
    identity_measurement_count = 0
    shape_measurement_count = 0
    gap_count = 0
    validation_issues: List[str] = []
    provider_comparison_counts: Dict[str, Dict[str, int]] = {
        "face_identity": {},
        "face_shape": {},
    }

    for item in report_items:
        debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
        face_shadow = debug.get("face_canonical_shadow") if isinstance(debug.get("face_canonical_shadow"), dict) else {}
        source_path = str(debug.get("source_path") or item.get("image") or "")
        image_name = str(item.get("image") or Path(source_path).name)
        master_artifact = _load_artifact(face_shadow.get("master_artifact_path"))
        candidate_artifact_path = str(face_shadow.get("cache_file") or "")
        candidate_artifact = _load_artifact(candidate_artifact_path)
        if candidate_artifact is None:
            candidate_artifact_path = str(face_shadow.get("candidate_artifact_path") or "")
            candidate_artifact = _load_artifact(candidate_artifact_path)

        lane_family = _lane_family(item)
        reference_embedding = _artifact_embedding(master_artifact)
        candidate_embedding = _artifact_embedding(candidate_artifact)
        reference_landmarks = _artifact_landmarks(master_artifact)
        candidate_landmarks = _artifact_landmarks(candidate_artifact)
        visibility_weights, visibility_weight_source, visibility_gaps = _pairwise_landmark_weights(
            master_artifact,
            candidate_artifact,
            reference_landmarks,
            candidate_landmarks,
        )
        landmark_schema_id, correspondence_state, correspondence_gaps = _pairwise_landmark_schema(
            master_artifact,
            candidate_artifact,
        )
        reference_identity_contract = build_face_provider_contract(master_artifact, axis="face_identity")
        candidate_identity_contract = build_face_provider_contract(candidate_artifact, axis="face_identity")
        identity_provider_comparison = compare_provider_contracts(
            reference_identity_contract,
            candidate_identity_contract,
        )
        reference_shape_contract = build_face_provider_contract(master_artifact, axis="face_shape")
        candidate_shape_contract = build_face_provider_contract(candidate_artifact, axis="face_shape")
        shape_provider_comparison = compare_provider_contracts(
            reference_shape_contract,
            candidate_shape_contract,
        )
        for axis, comparison in [
            ("face_identity", identity_provider_comparison),
            ("face_shape", shape_provider_comparison),
        ]:
            state = str(comparison.get("comparison_state") or "UNAVAILABLE")
            provider_comparison_counts[axis][state] = provider_comparison_counts[axis].get(state, 0) + 1

        measurement = build_face_identity_measurement(
            reference_embedding,
            candidate_embedding,
            provider_name="insightface_runtime_embedding",
            provider_version="unrecorded_in_face_artifact_v1",
            model_id="unrecorded_in_face_artifact_v1",
        )
        _attach_provider_contracts(
            measurement,
            reference_identity_contract,
            candidate_identity_contract,
            identity_provider_comparison,
        )
        if identity_provider_comparison["comparison_state"] == "MISMATCH":
            _withhold_identity_measurement(measurement, "IDENTITY_PROVIDER_CONTRACT_MISMATCH")
        identity_chain_valid = bool(measurement["available"])
        if lane_family == "back":
            _withhold_identity_measurement(measurement, "MEASUREMENT_WITHHELD_UNOBSERVABLE_SCOPE")
        contract_gaps: List[str] = provider_contract_gap_codes(
            reference_identity_contract,
            candidate_identity_contract,
            identity_provider_comparison,
        )
        if measurement["available"]:
            identity_measurement_count += 1
        measurement["contract_gaps"] = contract_gaps

        shape_measurement = build_face_projection_shape_measurement(
            reference_landmarks,
            candidate_landmarks,
            visibility_weights=visibility_weights,
            provider_name=str(face_shadow.get("provider_name") or "face_canonical_artifact"),
            provider_version=str(face_shadow.get("provider_version") or "unrecorded_in_face_artifact_v1"),
            model_id=str(face_shadow.get("model_id") or "unrecorded_in_face_artifact_v1"),
            visibility_weight_source=visibility_weight_source,
            landmark_schema_id=landmark_schema_id,
            correspondence_contract_state=correspondence_state,
        )
        _attach_provider_contracts(
            shape_measurement,
            reference_shape_contract,
            candidate_shape_contract,
            shape_provider_comparison,
        )
        if correspondence_state == "MISMATCH":
            _withhold_shape_measurement(shape_measurement, "LANDMARK_SCHEMA_MISMATCH")
        elif shape_provider_comparison["comparison_state"] == "MISMATCH":
            _withhold_shape_measurement(shape_measurement, "SHAPE_PROVIDER_CONTRACT_MISMATCH")
        shape_chain_valid = bool(shape_measurement["available"])
        if lane_family == "back":
            _withhold_shape_measurement(shape_measurement, "MEASUREMENT_WITHHELD_UNOBSERVABLE_SCOPE")
        shape_contract_gaps = (
            list(visibility_gaps)
            + list(correspondence_gaps)
            + provider_contract_gap_codes(
                reference_shape_contract,
                candidate_shape_contract,
                shape_provider_comparison,
            )
        )
        if shape_measurement["available"]:
            shape_contract_gaps.extend(
                [
                    "CANONICAL_PROJECTION_MODEL_PRIOR_DEPENDENT",
                    "HUBER_DELTA_UNCALIBRATED",
                ]
            )
            shape_measurement_count += 1
        shape_measurement["contract_gaps"] = list(dict.fromkeys(shape_contract_gaps))

        identity_eligibility, identity_scope_state, identity_scope_reasons = _face_identity_scope(
            lane_family,
            bool(measurement["available"]),
        )
        raw_observations = _build_raw_observations(debug, face_shadow)
        observation = build_axis_observation(
            axis="face_identity",
            eligibility=identity_eligibility,
            scope_state=identity_scope_state,
            chain_state="CHAIN_VALID" if identity_chain_valid else "CHAIN_INVALID",
            raw_observations=raw_observations,
            reasons=identity_scope_reasons,
        )
        shape_eligibility, shape_scope_state, shape_scope_reasons = _face_shape_scope(
            lane_family,
            bool(shape_measurement["available"]),
        )
        shape_observation = build_axis_observation(
            axis="face_shape",
            eligibility=shape_eligibility,
            scope_state=shape_scope_state,
            chain_state="CHAIN_VALID" if shape_chain_valid else "CHAIN_INVALID",
            raw_observations=raw_observations,
            reasons=shape_scope_reasons,
        )

        record_key = source_path or image_name
        image_node_id = graph.add_node(
            _stable_id("image", record_key),
            "OBSERVATION_ASSET",
            evidence_family=None,
            attributes={"image": image_name, "source_path": source_path},
        )
        master_artifact_path = str(face_shadow.get("master_artifact_path") or "")
        master_artifact_id = graph.add_node(
            _stable_id("face_master_artifact", master_artifact_path or face_truth_path),
            "PROVIDER_ARTIFACT",
            evidence_family=None,
            attributes={"path": master_artifact_path, "role": "master_truth"},
        )
        candidate_artifact_id = graph.add_node(
            _stable_id("face_candidate_artifact", candidate_artifact_path or record_key),
            "PROVIDER_ARTIFACT",
            evidence_family=None,
            attributes={"path": candidate_artifact_path, "role": "candidate"},
        )
        master_embedding_id = graph.add_node(
            _stable_id("face_master_embedding", master_artifact_path or face_truth_path),
            "SOURCE_MEASUREMENT",
            evidence_family="face_identity",
            attributes={
                "source_field": "runtime_face_embedding_raw_or_legacy_alias",
                "role": "master_truth",
                "available": reference_embedding is not None,
                "provider_contract_sha256": reference_identity_contract.get("observed_contract_sha256"),
                "provider_contract_state": reference_identity_contract.get("completeness_state"),
            },
        )
        candidate_embedding_id = graph.add_node(
            _stable_id("face_candidate_embedding", candidate_artifact_path or record_key),
            "SOURCE_MEASUREMENT",
            evidence_family="face_identity",
            attributes={
                "source_field": "runtime_face_embedding_raw_or_legacy_alias",
                "role": "candidate",
                "available": candidate_embedding is not None,
                "provider_contract_sha256": candidate_identity_contract.get("observed_contract_sha256"),
                "provider_contract_state": candidate_identity_contract.get("completeness_state"),
            },
        )
        master_landmarks_id = graph.add_node(
            _stable_id("face_master_landmarks", master_artifact_path or face_truth_path),
            "SOURCE_MEASUREMENT",
            evidence_family="face_geometry",
            attributes={
                "source_field": "canonical_landmarks_or_legacy_alias",
                "role": "master_truth",
                "available": reference_landmarks is not None,
                "provider_contract_sha256": reference_shape_contract.get("observed_contract_sha256"),
                "provider_contract_state": reference_shape_contract.get("completeness_state"),
            },
        )
        candidate_landmarks_id = graph.add_node(
            _stable_id("face_candidate_landmarks", candidate_artifact_path or record_key),
            "SOURCE_MEASUREMENT",
            evidence_family="face_geometry",
            attributes={
                "source_field": "canonical_landmarks_or_legacy_alias",
                "role": "candidate",
                "available": candidate_landmarks is not None,
                "provider_contract_sha256": candidate_shape_contract.get("observed_contract_sha256"),
                "provider_contract_state": candidate_shape_contract.get("completeness_state"),
            },
        )
        measurement_id = graph.add_node(
            _stable_id("face_identity_measurement", record_key),
            "NATIVE_MEASUREMENT",
            evidence_family="face_identity",
            attributes={"axis": "face_identity", "decision_influence": "NONE"},
        )
        shape_measurement_id = graph.add_node(
            _stable_id("face_shape_measurement", record_key),
            "NATIVE_MEASUREMENT",
            evidence_family="face_geometry",
            attributes={
                "axis": "face_shape",
                "decision_influence": "NONE",
                "partition_diagnostics_independent_evidence": False,
            },
        )
        graph.add_edge(face_truth_node_id, master_artifact_id, "OBSERVED_FROM")
        graph.add_edge(image_node_id, candidate_artifact_id, "OBSERVED_FROM")
        graph.add_edge(master_artifact_id, master_embedding_id, "TRANSFORMED_FROM")
        graph.add_edge(candidate_artifact_id, candidate_embedding_id, "TRANSFORMED_FROM")
        graph.add_edge(master_embedding_id, measurement_id, "DERIVED_FROM")
        graph.add_edge(candidate_embedding_id, measurement_id, "DERIVED_FROM")
        graph.add_edge(master_artifact_id, master_landmarks_id, "TRANSFORMED_FROM")
        graph.add_edge(candidate_artifact_id, candidate_landmarks_id, "TRANSFORMED_FROM")
        graph.add_edge(master_landmarks_id, shape_measurement_id, "DERIVED_FROM")
        graph.add_edge(candidate_landmarks_id, shape_measurement_id, "DERIVED_FROM")

        axis_record = {
            "observation": observation,
            "measurement": measurement,
            "lineage_measurement_id": measurement_id,
        }
        axis_issues = validate_shadow_axis_record(axis_record)
        if axis_issues:
            validation_issues.extend(f"{image_name}:{issue}" for issue in axis_issues)
        shape_axis_record = {
            "observation": shape_observation,
            "measurement": shape_measurement,
            "lineage_measurement_id": shape_measurement_id,
        }
        shape_axis_issues = validate_shadow_axis_record(shape_axis_record)
        if shape_axis_issues:
            validation_issues.extend(f"{image_name}:face_shape:{issue}" for issue in shape_axis_issues)
        evidence_gaps = list(
            dict.fromkeys(
                identity_scope_reasons
                + contract_gaps
                + list(measurement.get("errors") or [])
                + shape_scope_reasons
                + shape_contract_gaps
                + list(shape_measurement.get("errors") or [])
            )
        )
        gap_count += len(evidence_gaps)
        item_records.append(
            {
                "image": image_name,
                "source_path": source_path,
                "observed_lane_family": lane_family,
                "axes": {
                    "face_identity": axis_record,
                    "face_shape": shape_axis_record,
                },
                "evidence_gaps": evidence_gaps,
                "decision_influence": "NONE",
            }
        )

    lineage = graph.to_dict()
    validation_issues.extend(lineage.get("issues") or [])
    repeatability_protocol = repeatability_protocol_snapshot()
    if repeatability_protocol.get("validation_status") != "VALID":
        validation_issues.extend(repeatability_protocol.get("validation_issues") or [])
    payload = {
        "schema_version": IDENTITY_EVIDENCE_SHADOW_SCHEMA,
        "contract_version": IDENTITY_EVIDENCE_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "governance": shadow_governance(),
        "repeatability_protocol": {
            "protocol_id": repeatability_protocol.get("protocol_id"),
            "protocol_sha256": repeatability_protocol.get("protocol_sha256"),
            "protocol_execution_state": "NOT_EXECUTED",
            "validation_status": repeatability_protocol.get("validation_status"),
            "validation_issues": list(repeatability_protocol.get("validation_issues") or []),
            "decision_influence": "NONE",
        },
        "truth_registry": {
            "face_truth": truth_anchors.get("face_identity"),
            "body_truth": truth_anchors.get("body_master"),
            "winner_bank": {
                "authority": "REVIEW_MEMORY_ONLY",
                "mutable": True,
                "may_modify_truth": False,
            },
        },
        "summary": {
            "item_count": len(item_records),
            "available_face_identity_measurements": identity_measurement_count,
            "available_face_shape_measurements": shape_measurement_count,
            "provider_contract_comparison_counts": provider_comparison_counts,
            "evidence_gap_count": gap_count,
            "validation_status": "VALID" if not validation_issues else "INVALID",
            "validation_issues": list(dict.fromkeys(validation_issues)),
        },
        "items": item_records,
        "lineage": lineage,
    }
    return payload


def write_identity_evidence_shadow(runtime: Any, report_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_path = Path(runtime.config.paths.dir_output) / _OUTPUT_NAME
    payload = build_identity_evidence_shadow(runtime, report_items)
    atomic_write_json(output_path, payload)
    return {
        "schema_version": IDENTITY_EVIDENCE_SHADOW_SCHEMA,
        "contract_version": IDENTITY_EVIDENCE_CONTRACT_VERSION,
        "path": str(output_path.resolve()),
        "sha256": _sha256_file(output_path),
        "mode": "SHADOW",
        "decision_influence": "NONE",
        "summary": payload["summary"],
    }
