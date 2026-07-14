from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_body_evidence_contract import (
    BODY_EVIDENCE_CONTRACT_VERSION,
    body_core_observation_scope,
    body_shadow_governance,
    body_topology_observation_scope,
    build_body_axis_observation,
    build_body_core_shape_measurement,
    build_body_topology_measurement,
    build_pose_gait_condition,
    build_surface_occlusion_condition,
    validate_body_shadow_axis_record,
)
from .qa_evidence_lineage import EvidenceLineageGraph
from .qa_io import atomic_write_json
from .qa_provider_contract import (
    build_body_provider_contract,
    compare_provider_contracts,
    provider_contract_gap_codes,
)
from .qa_repeatability_shadow import body_repeatability_protocol_snapshot
from .qa_runtime import anchor_registry_snapshot


BODY_EVIDENCE_SHADOW_SCHEMA = "body_evidence_shadow_v0_2"
_OUTPUT_NAME = "body_evidence_shadow.json"


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
    artifact = payload.get("artifact")
    return dict(artifact) if isinstance(artifact, dict) else payload


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


def _body_summary(heavy_evidence: Dict[str, Any]) -> Dict[str, Any]:
    summary = (
        dict(heavy_evidence.get("summary"))
        if isinstance(heavy_evidence.get("summary"), dict)
        else {}
    )
    nested = summary.get("body_canonical_summary")
    if isinstance(nested, dict):
        return dict(nested)
    if str(heavy_evidence.get("provider_name") or "") == "body_canonical_hmr2":
        return summary
    return {}


def _metric_value(heavy_evidence: Dict[str, Any], metric_name: str) -> Any:
    for row in list(heavy_evidence.get("metrics") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("metric_name") or "") == metric_name:
            return row.get("metric_value")
    return None


def _raw_metric(
    value: Any,
    *,
    unit: str,
    algorithm_id: str,
    input_region: str,
    provider_version: str,
) -> Optional[Dict[str, Any]]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return {
        "value": number,
        "unit": unit,
        "algorithm_id": algorithm_id,
        "input_region": input_region,
        "image_scale": "provider_native",
        "provider_version": provider_version,
    }


def _body_raw_observations(
    artifact: Optional[Dict[str, Any]],
    heavy_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    node = artifact if isinstance(artifact, dict) else {}
    provider_version = str(node.get("provider_version") or "unrecorded")
    observations = {
        "body_canonical_coverage": _raw_metric(
            node.get("coverage"),
            unit="provider_estimated_ratio",
            algorithm_id="hmr2_body25_keypoint_frame_coverage_v2",
            input_region="full_body_reconstruction",
            provider_version=provider_version,
        ),
        "body_fit_confidence": _raw_metric(
            node.get("fit_confidence"),
            unit="provider_native_quality",
            algorithm_id="hmr2_finite_output_quality_v1",
            input_region="full_body_reconstruction",
            provider_version=provider_version,
        ),
        "visible_body_ratio": _raw_metric(
            _metric_value(heavy_evidence, "visible_body_ratio"),
            unit="provider_estimated_ratio",
            algorithm_id="segformer_visible_body_ratio_v1",
            input_region="full_image",
            provider_version=str(heavy_evidence.get("provider_version") or "unrecorded"),
        ),
        "garment_coverage_ratio": _raw_metric(
            _metric_value(heavy_evidence, "garment_coverage_ratio"),
            unit="provider_estimated_ratio",
            algorithm_id="segformer_garment_coverage_ratio_v1",
            input_region="full_image",
            provider_version=str(heavy_evidence.get("provider_version") or "unrecorded"),
        ),
    }
    return {key: value for key, value in observations.items() if value is not None}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _canonical_vertex_shape(artifact: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not isinstance(artifact, dict):
        return None
    vertices = artifact.get("canonical_smpl_vertices")
    if not isinstance(vertices, list) or not vertices:
        return None
    if not all(
        isinstance(vertex, list)
        and len(vertex) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vertex)
        for vertex in vertices
    ):
        return None
    return [len(vertices), 3]


def _native_topology_readiness(
    reference_artifact: Optional[Dict[str, Any]],
    candidate_artifact: Optional[Dict[str, Any]],
    provider_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    reference_shape = _canonical_vertex_shape(reference_artifact)
    candidate_shape = _canonical_vertex_shape(candidate_artifact)
    blockers: List[str] = []
    if reference_shape is None:
        blockers.append("REFERENCE_CANONICAL_SMPL_VERTICES_UNAVAILABLE")
    if candidate_shape is None:
        blockers.append("CANDIDATE_CANONICAL_SMPL_VERTICES_UNAVAILABLE")
    if reference_shape is not None and candidate_shape is not None and reference_shape != candidate_shape:
        blockers.append("CANONICAL_SMPL_VERTEX_SHAPE_MISMATCH")
    if str(provider_comparison.get("comparison_state") or "") != "MATCH":
        blockers.append("BODY_TOPOLOGY_PROVIDER_CONTRACT_NOT_MATCHED")
    return {
        "schema_version": "native_body_topology_readiness_v0_1",
        "readiness_state": "READY" if not blockers else "BLOCKED",
        "required_source_field": "canonical_smpl_vertices",
        "reference_vertex_shape": reference_shape,
        "candidate_vertex_shape": candidate_shape,
        "requires_exact_smpl_vertex_correspondence": True,
        "requires_zero_pose_canonicalization": True,
        "requires_matching_body_model_sha256": True,
        "blockers": blockers,
        "decision_influence": "NONE",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _withhold_measurement(measurement: Dict[str, Any], error: str) -> None:
    measurement["available"] = False
    measurement["residual_vector"] = None
    measurement["component_residuals"] = {}
    measurement["used_components"] = []
    measurement["errors"] = list(
        dict.fromkeys(list(measurement.get("errors") or []) + [str(error)])
    )


def build_body_evidence_shadow(runtime: Any, report_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph = EvidenceLineageGraph()
    anchor_snapshot = anchor_registry_snapshot(runtime.config)
    truth_anchors = (
        anchor_snapshot.get("truth_anchors")
        if isinstance(anchor_snapshot.get("truth_anchors"), dict)
        else {}
    )
    body_truth = (
        truth_anchors.get("body_master")
        if isinstance(truth_anchors.get("body_master"), dict)
        else {}
    )
    body_truth_path = str(body_truth.get("resolved_path") or "")
    body_truth_node_id = graph.add_node(
        _stable_id("truth_body", str(body_truth.get("actual_sha256") or body_truth_path)),
        "TRUTH_ASSET",
        evidence_family=None,
        attributes={
            "path": body_truth_path,
            "sha256": body_truth.get("actual_sha256"),
            "authority": body_truth.get("authority"),
            "pose_gait_semantics": "conditioned_observation_not_new_anchor",
        },
    )

    item_records: List[Dict[str, Any]] = []
    measurement_count = 0
    topology_measurement_count = 0
    topology_ready_count = 0
    gap_count = 0
    validation_issues: List[str] = []
    provider_comparison_counts: Dict[str, Dict[str, int]] = {
        "body_core_shape": {},
        "body_topology": {},
    }

    for item in report_items:
        debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
        heavy_evidence = (
            debug.get("heavy_evidence") if isinstance(debug.get("heavy_evidence"), dict) else {}
        )
        body_summary = _body_summary(heavy_evidence)
        source_path = str(debug.get("source_path") or item.get("image") or "")
        image_name = str(item.get("image") or Path(source_path).name)
        lane_family = _lane_family(item)

        master_artifact_path = str(body_summary.get("master_artifact_path") or "")
        candidate_artifact_path = str(body_summary.get("candidate_artifact_path") or "")
        if not candidate_artifact_path and str(heavy_evidence.get("provider_name") or "") == "body_canonical_hmr2":
            candidate_artifact_path = str(heavy_evidence.get("cache_file") or "")
        master_artifact = _load_artifact(master_artifact_path)
        candidate_artifact = _load_artifact(candidate_artifact_path)

        reference_contract = build_body_provider_contract(master_artifact, axis="body_core_shape")
        candidate_contract = build_body_provider_contract(candidate_artifact, axis="body_core_shape")
        comparison = compare_provider_contracts(reference_contract, candidate_contract)
        comparison_state = str(comparison.get("comparison_state") or "UNAVAILABLE")
        core_counts = provider_comparison_counts["body_core_shape"]
        core_counts[comparison_state] = core_counts.get(comparison_state, 0) + 1

        reference_topology_contract = build_body_provider_contract(
            master_artifact,
            axis="body_topology",
        )
        candidate_topology_contract = build_body_provider_contract(
            candidate_artifact,
            axis="body_topology",
        )
        topology_comparison = compare_provider_contracts(
            reference_topology_contract,
            candidate_topology_contract,
        )
        topology_comparison_state = str(
            topology_comparison.get("comparison_state") or "UNAVAILABLE"
        )
        topology_counts = provider_comparison_counts["body_topology"]
        topology_counts[topology_comparison_state] = (
            topology_counts.get(topology_comparison_state, 0) + 1
        )
        topology_contract_gaps = provider_contract_gap_codes(
            reference_topology_contract,
            candidate_topology_contract,
            topology_comparison,
        )
        topology_readiness = _native_topology_readiness(
            master_artifact,
            candidate_artifact,
            topology_comparison,
        )
        if topology_readiness["readiness_state"] == "READY":
            topology_ready_count += 1

        topology_measurement = build_body_topology_measurement(
            (master_artifact or {}).get("canonical_smpl_vertices"),
            (candidate_artifact or {}).get("canonical_smpl_vertices"),
            provider_name=str(
                (candidate_artifact or {}).get("provider_name") or "body_canonical_hmr2"
            ),
            provider_version=str(
                (candidate_artifact or {}).get("provider_version") or "unrecorded"
            ),
            model_id=str((candidate_artifact or {}).get("model_id") or "unrecorded"),
        )
        topology_measurement["provider_contracts"] = {
            "reference": reference_topology_contract,
            "candidate": candidate_topology_contract,
            "comparison": topology_comparison,
        }
        topology_measurement["contract_gaps"] = topology_contract_gaps
        if topology_readiness["readiness_state"] != "READY":
            _withhold_measurement(
                topology_measurement,
                "BODY_TOPOLOGY_NATIVE_READINESS_BLOCKED",
            )
        if topology_measurement.get("available"):
            topology_measurement_count += 1

        measurement = build_body_core_shape_measurement(
            (master_artifact or {}).get("canonical_measurements"),
            (candidate_artifact or {}).get("canonical_measurements"),
            provider_name=str((candidate_artifact or {}).get("provider_name") or "body_canonical_hmr2"),
            provider_version=str((candidate_artifact or {}).get("provider_version") or "unrecorded"),
            model_id=str((candidate_artifact or {}).get("model_id") or "unrecorded"),
        )
        measurement["provider_contracts"] = {
            "reference": reference_contract,
            "candidate": candidate_contract,
            "comparison": comparison,
        }
        contract_gaps = provider_contract_gap_codes(reference_contract, candidate_contract, comparison)
        if comparison_state == "MISMATCH":
            _withhold_measurement(measurement, "BODY_CORE_PROVIDER_CONTRACT_MISMATCH")
        measurement["contract_gaps"] = contract_gaps
        if measurement.get("available"):
            measurement_count += 1

        eligibility, scope_state, scope_reasons = body_core_observation_scope(
            lane_family,
            measurement_available=bool(measurement.get("available")),
        )
        raw_observations = _body_raw_observations(candidate_artifact, heavy_evidence)
        observation = build_body_axis_observation(
            axis="body_core_shape",
            eligibility=eligibility,
            scope_state=scope_state,
            chain_state="CHAIN_VALID" if measurement.get("available") else "CHAIN_INVALID",
            raw_observations=raw_observations,
            reasons=scope_reasons,
        )
        topology_eligibility, topology_scope_state, topology_scope_reasons = (
            body_topology_observation_scope(
                lane_family,
                measurement_available=bool(topology_measurement.get("available")),
            )
        )
        topology_observation = build_body_axis_observation(
            axis="body_topology",
            eligibility=topology_eligibility,
            scope_state=topology_scope_state,
            chain_state=(
                "CHAIN_VALID" if topology_measurement.get("available") else "CHAIN_INVALID"
            ),
            raw_observations=raw_observations,
            reasons=topology_scope_reasons,
        )
        pose_gait_condition = build_pose_gait_condition(
            (master_artifact or {}).get("pose_vector"),
            (candidate_artifact or {}).get("pose_vector"),
        )
        surface_condition = build_surface_occlusion_condition(
            body_coverage=(candidate_artifact or {}).get("coverage"),
            visible_body_ratio=_metric_value(heavy_evidence, "visible_body_ratio"),
            garment_coverage_ratio=_metric_value(heavy_evidence, "garment_coverage_ratio"),
        )

        record_key = source_path or image_name
        image_node_id = graph.add_node(
            _stable_id("image", record_key),
            "OBSERVATION_ASSET",
            evidence_family=None,
            attributes={"image": image_name, "source_path": source_path},
        )
        master_artifact_id = graph.add_node(
            _stable_id("body_master_artifact", master_artifact_path or body_truth_path),
            "PROVIDER_ARTIFACT",
            evidence_family=None,
            attributes={"path": master_artifact_path, "role": "master_truth"},
        )
        candidate_artifact_id = graph.add_node(
            _stable_id("body_candidate_artifact", candidate_artifact_path or record_key),
            "PROVIDER_ARTIFACT",
            evidence_family=None,
            attributes={"path": candidate_artifact_path, "role": "candidate"},
        )
        master_measurements_id = graph.add_node(
            _stable_id("body_master_core_measurements", master_artifact_path or body_truth_path),
            "SOURCE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "source_field": "canonical_measurements",
                "role": "master_truth",
                "provider_contract_state": reference_contract.get("completeness_state"),
            },
        )
        candidate_measurements_id = graph.add_node(
            _stable_id("body_candidate_core_measurements", candidate_artifact_path or record_key),
            "SOURCE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "source_field": "canonical_measurements",
                "role": "candidate",
                "provider_contract_state": candidate_contract.get("completeness_state"),
            },
        )
        measurement_id = graph.add_node(
            _stable_id("body_core_shape_measurement", record_key),
            "NATIVE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "axis": "body_core_shape",
                "decision_influence": "NONE",
                "components_are_independent_votes": False,
            },
        )
        master_topology_id = graph.add_node(
            _stable_id(
                "body_master_canonical_vertices",
                master_artifact_path or body_truth_path,
            ),
            "SOURCE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "source_field": "canonical_smpl_vertices",
                "role": "master_truth",
                "provider_contract_state": reference_topology_contract.get(
                    "completeness_state"
                ),
            },
        )
        candidate_topology_id = graph.add_node(
            _stable_id(
                "body_candidate_canonical_vertices",
                candidate_artifact_path or record_key,
            ),
            "SOURCE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "source_field": "canonical_smpl_vertices",
                "role": "candidate",
                "provider_contract_state": candidate_topology_contract.get(
                    "completeness_state"
                ),
            },
        )
        topology_measurement_id = graph.add_node(
            _stable_id("body_topology_measurement", record_key),
            "NATIVE_MEASUREMENT",
            evidence_family="body_shape_geometry",
            attributes={
                "axis": "body_topology",
                "decision_influence": "NONE",
                "components_are_independent_votes": False,
                "independent_from_body_core_shape": False,
            },
        )
        graph.add_edge(body_truth_node_id, master_artifact_id, "OBSERVED_FROM")
        graph.add_edge(image_node_id, candidate_artifact_id, "OBSERVED_FROM")
        graph.add_edge(master_artifact_id, master_measurements_id, "TRANSFORMED_FROM")
        graph.add_edge(candidate_artifact_id, candidate_measurements_id, "TRANSFORMED_FROM")
        graph.add_edge(master_measurements_id, measurement_id, "DERIVED_FROM")
        graph.add_edge(candidate_measurements_id, measurement_id, "DERIVED_FROM")
        graph.add_edge(master_artifact_id, master_topology_id, "TRANSFORMED_FROM")
        graph.add_edge(candidate_artifact_id, candidate_topology_id, "TRANSFORMED_FROM")
        graph.add_edge(master_topology_id, topology_measurement_id, "DERIVED_FROM")
        graph.add_edge(candidate_topology_id, topology_measurement_id, "DERIVED_FROM")

        axis_record = {
            "observation": observation,
            "measurement": measurement,
            "lineage_measurement_id": measurement_id,
        }
        axis_issues = validate_body_shadow_axis_record(axis_record)
        validation_issues.extend(f"{image_name}:{issue}" for issue in axis_issues)
        topology_axis_record = {
            "implementation_state": "NATIVE_ZERO_POSE_VERTEX_RESIDUAL_SHADOW",
            "observation": topology_observation,
            "measurement": topology_measurement,
            "lineage_measurement_id": topology_measurement_id,
            "decision_influence": "NONE",
            "legacy_source_available": bool(
                (candidate_artifact or {}).get("body_topology_signature")
            ),
            "native_measurement_readiness": topology_readiness,
            "provider_contracts": {
                "reference": reference_topology_contract,
                "candidate": candidate_topology_contract,
                "comparison": topology_comparison,
            },
            "contract_gaps": topology_contract_gaps,
        }
        topology_axis_issues = validate_body_shadow_axis_record(topology_axis_record)
        validation_issues.extend(
            f"{image_name}:{issue}" for issue in topology_axis_issues
        )
        evidence_gaps = list(
            dict.fromkeys(
                scope_reasons
                + topology_scope_reasons
                + contract_gaps
                + topology_contract_gaps
                + list(measurement.get("errors") or [])
                + list(topology_measurement.get("errors") or [])
                + list(topology_readiness.get("blockers") or [])
                + ([] if surface_condition.get("available") else ["SURFACE_OCCLUSION_CONDITION_UNASSESSED"])
                + ([] if pose_gait_condition.get("available") else ["POSE_GAIT_CONDITION_UNAVAILABLE"])
            )
        )
        gap_count += len(evidence_gaps)
        item_records.append(
            {
                "image": image_name,
                "source_path": source_path,
                "observed_lane_family": lane_family,
                "axes": {
                    "body_core_shape": axis_record,
                    "body_topology": topology_axis_record,
                },
                "conditions": {
                    "pose_gait_condition": pose_gait_condition,
                    "surface_occlusion": surface_condition,
                },
                "evidence_gaps": evidence_gaps,
                "decision_influence": "NONE",
            }
        )

    lineage = graph.to_dict()
    validation_issues.extend(lineage.get("issues") or [])
    repeatability_protocol = body_repeatability_protocol_snapshot()
    if repeatability_protocol.get("validation_status") != "VALID":
        validation_issues.extend(repeatability_protocol.get("validation_issues") or [])
    payload = {
        "schema_version": BODY_EVIDENCE_SHADOW_SCHEMA,
        "contract_version": BODY_EVIDENCE_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "governance": body_shadow_governance(),
        "repeatability_protocol": {
            "protocol_id": repeatability_protocol.get("protocol_id"),
            "protocol_sha256": repeatability_protocol.get("protocol_sha256"),
            "protocol_execution_state": "NOT_EXECUTED",
            "validation_status": repeatability_protocol.get("validation_status"),
            "validation_issues": list(repeatability_protocol.get("validation_issues") or []),
            "decision_influence": "NONE",
        },
        "truth_registry": {
            "body_truth": body_truth,
            "body_truth_observation_semantics": "pose_gait_conditioned_absolute_116_1",
            "pose_gait_creates_new_anchor": False,
            "winner_bank": {
                "authority": "REVIEW_MEMORY_ONLY",
                "mutable": True,
                "may_modify_truth": False,
            },
        },
        "summary": {
            "item_count": len(item_records),
            "available_body_core_shape_measurements": measurement_count,
            "body_topology_native_measurements": topology_measurement_count,
            "body_topology_native_measurement_ready_items": topology_ready_count,
            "provider_contract_comparison_counts": provider_comparison_counts,
            "evidence_gap_count": gap_count,
            "validation_status": "VALID" if not validation_issues else "INVALID",
            "validation_issues": list(dict.fromkeys(validation_issues)),
        },
        "items": item_records,
        "lineage": lineage,
    }
    return payload


def write_body_evidence_shadow(runtime: Any, report_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    output_path = Path(runtime.config.paths.dir_output) / _OUTPUT_NAME
    payload = build_body_evidence_shadow(runtime, report_items)
    atomic_write_json(output_path, payload)
    return {
        "schema_version": BODY_EVIDENCE_SHADOW_SCHEMA,
        "contract_version": BODY_EVIDENCE_CONTRACT_VERSION,
        "path": str(output_path.resolve()),
        "sha256": _sha256_file(output_path),
        "mode": "SHADOW",
        "decision_influence": "NONE",
        "summary": payload["summary"],
    }
