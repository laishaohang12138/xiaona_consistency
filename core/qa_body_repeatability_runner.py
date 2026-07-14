from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .qa_body_evidence_contract import (
    BODY_CORE_MEASUREMENT_ORDER,
    canonical_vertex_delta_vector,
    log_ratio_residual_vector,
)
from .qa_identity_repeatability_runner import run_repeatability_shadow_engine
from .qa_io import atomic_write_json
from .qa_repeatability_shadow import (
    REPEATABILITY_DOMAINS,
    load_body_repeatability_protocol,
    summarize_repeatability_cohort,
    summarize_repeatability_trials,
)


BODY_REPEATABILITY_RUN_SCHEMA = "body_repeatability_shadow_run_v0_2"
BODY_REPEATABILITY_TRIAL_SCHEMA = "body_repeatability_shadow_trial_v0_2"
BODY_REPEATABILITY_AXES = ("body_core_shape", "body_topology")
TOPOLOGY_COORDINATE_AXES = ("x", "y", "z")
TOPOLOGY_SIGNED_QUANTILES = (
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
)
TOPOLOGY_ABSOLUTE_QUANTILES = (
    ("q50", 0.50),
    ("q90", 0.90),
    ("q95", 0.95),
    ("q99", 0.99),
    ("q100", 1.00),
)


def _axis_node(observation: Dict[str, Any], axis: str) -> Dict[str, Any]:
    node = observation.get(axis)
    return dict(node) if isinstance(node, dict) else {"available": False, "errors": ["AXIS_MISSING"]}


def _provider_contract_sha(
    node: Dict[str, Any],
    *,
    comparable_only: bool = False,
) -> Optional[str]:
    contract = node.get("provider_contract")
    if not isinstance(contract, dict) or not contract:
        return None
    keys = ["comparable_contract_sha256"]
    if not comparable_only:
        keys.append("observed_contract_sha256")
    for key in keys:
        value = str(contract.get(key) or "").strip()
        if value:
            return value
    return None


def build_body_native_repeatability_residual(
    baseline: Dict[str, Any],
    trial: Dict[str, Any],
    axis: str,
) -> Dict[str, Any]:
    normalized_axis = str(axis)
    if normalized_axis not in BODY_REPEATABILITY_AXES:
        raise ValueError(f"unsupported body repeatability axis: {axis!r}")
    reference = _axis_node(baseline, normalized_axis)
    candidate = _axis_node(trial, normalized_axis)
    errors: List[str] = []
    if not bool(reference.get("available")):
        errors.append("BASELINE_MEASUREMENT_UNAVAILABLE")
    if not bool(candidate.get("available")):
        errors.append("TRIAL_MEASUREMENT_UNAVAILABLE")
    comparable_only = normalized_axis == "body_topology"
    reference_contract_sha = _provider_contract_sha(
        reference,
        comparable_only=comparable_only,
    )
    candidate_contract_sha = _provider_contract_sha(
        candidate,
        comparable_only=comparable_only,
    )
    if not reference_contract_sha or not candidate_contract_sha:
        errors.append("MEASUREMENT_PROVIDER_CONTRACT_UNAVAILABLE")
    elif reference_contract_sha != candidate_contract_sha:
        errors.append("MEASUREMENT_PROVIDER_CONTRACT_MISMATCH")
    if errors:
        return {
            "available": False,
            "residual": None,
            "residual_vector": None,
            "component_residuals": {},
            "unit": (
                "smpl_model_length_unit"
                if normalized_axis == "body_topology"
                else "natural_log_ratio"
            ),
            "errors": errors,
            "baseline_provider_contract_sha256": reference_contract_sha,
            "trial_provider_contract_sha256": candidate_contract_sha,
            "decision_influence": "NONE",
        }
    if normalized_axis == "body_topology":
        result = canonical_vertex_delta_vector(
            reference.get("value"),
            candidate.get("value"),
        )
        if bool(result.get("available")) and (
            result.get("vertex_count") != 6890
            or result.get("coordinate_count") != 20670
        ):
            return {
                "available": False,
                "residual": None,
                "residual_vector": None,
                "component_residuals": {},
                "unit": "smpl_model_length_unit",
                "errors": ["BODY_TOPOLOGY_VERTEX_CONTRACT_MISMATCH"],
                "baseline_provider_contract_sha256": reference_contract_sha,
                "trial_provider_contract_sha256": candidate_contract_sha,
                "decision_influence": "NONE",
            }
        return {
            **result,
            "residual": None,
            "component_residuals": {},
            "unit": "smpl_model_length_unit",
            "coordinate_axis_order": list(TOPOLOGY_COORDINATE_AXES),
            "scalar_residual_authorized": False,
            "alignment_contract": {
                "translation": "centroid_removed_independently",
                "rotation_fit_applied": False,
                "scale_fit_applied": False,
                "procrustes_fit_applied": False,
                "pose_fit_applied": False,
            },
            "baseline_provider_contract_sha256": reference_contract_sha,
            "trial_provider_contract_sha256": candidate_contract_sha,
            "decision_influence": "NONE",
        }
    result = log_ratio_residual_vector(reference.get("value"), candidate.get("value"))
    return {
        **result,
        "residual": None,
        "scalar_residual_authorized": False,
        "baseline_provider_contract_sha256": reference_contract_sha,
        "trial_provider_contract_sha256": candidate_contract_sha,
        "decision_influence": "NONE",
    }


def _finite_scalar(node: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = float(node.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_body_chain_diagnostics(
    baseline_observation: Dict[str, Any],
    trial_observation: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = (
        baseline_observation.get("chain_observation")
        if isinstance(baseline_observation.get("chain_observation"), dict)
        else {}
    )
    trial = (
        trial_observation.get("chain_observation")
        if isinstance(trial_observation.get("chain_observation"), dict)
        else {}
    )
    diagnostics: Dict[str, Any] = {
        "schema_version": "body_reconstruction_chain_diagnostics_v0_1",
        "bbox_iou": None,
        "bbox_center_displacement_image_fraction": None,
        "bbox_width_relative_delta": None,
        "bbox_height_relative_delta": None,
        "body_canonical_coverage_delta": None,
        "body_fit_confidence_delta": None,
        "pose_parameter_rms_delta": None,
        "decision_influence": "NONE",
    }

    def vector(node: Dict[str, Any], key: str, size: Optional[int] = None) -> Optional[np.ndarray]:
        try:
            value = np.asarray(node.get(key), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if not value.size or (size is not None and value.size != size):
            return None
        return value if bool(np.all(np.isfinite(value))) else None

    reference_bbox = vector(baseline, "body_bbox_normalized_xyxy", 4)
    trial_bbox = vector(trial, "body_bbox_normalized_xyxy", 4)
    if reference_bbox is not None and trial_bbox is not None:
        intersection_width = max(
            0.0,
            min(reference_bbox[2], trial_bbox[2]) - max(reference_bbox[0], trial_bbox[0]),
        )
        intersection_height = max(
            0.0,
            min(reference_bbox[3], trial_bbox[3]) - max(reference_bbox[1], trial_bbox[1]),
        )
        intersection = intersection_width * intersection_height
        reference_width = max(0.0, reference_bbox[2] - reference_bbox[0])
        reference_height = max(0.0, reference_bbox[3] - reference_bbox[1])
        trial_width = max(0.0, trial_bbox[2] - trial_bbox[0])
        trial_height = max(0.0, trial_bbox[3] - trial_bbox[1])
        union = reference_width * reference_height + trial_width * trial_height - intersection
        if union > 0.0:
            diagnostics["bbox_iou"] = float(intersection / union)
        reference_center = np.asarray(
            [
                (reference_bbox[0] + reference_bbox[2]) / 2.0,
                (reference_bbox[1] + reference_bbox[3]) / 2.0,
            ]
        )
        trial_center = np.asarray(
            [
                (trial_bbox[0] + trial_bbox[2]) / 2.0,
                (trial_bbox[1] + trial_bbox[3]) / 2.0,
            ]
        )
        diagnostics["bbox_center_displacement_image_fraction"] = float(
            np.linalg.norm(trial_center - reference_center)
        )
        if reference_width > 0.0:
            diagnostics["bbox_width_relative_delta"] = float(trial_width / reference_width - 1.0)
        if reference_height > 0.0:
            diagnostics["bbox_height_relative_delta"] = float(trial_height / reference_height - 1.0)

    for source_key, output_key in [
        ("body_canonical_coverage", "body_canonical_coverage_delta"),
        ("body_fit_confidence", "body_fit_confidence_delta"),
    ]:
        reference_value = _finite_scalar(baseline, source_key)
        trial_value = _finite_scalar(trial, source_key)
        if reference_value is not None and trial_value is not None:
            diagnostics[output_key] = float(trial_value - reference_value)

    reference_pose = vector(baseline, "pose_vector")
    trial_pose = vector(trial, "pose_vector")
    if (
        reference_pose is not None
        and trial_pose is not None
        and reference_pose.size == trial_pose.size
    ):
        diagnostics["pose_parameter_rms_delta"] = float(
            np.sqrt(np.mean((trial_pose - reference_pose) ** 2))
        )
    return diagnostics


def _descriptor(values: Iterable[float]) -> Dict[str, Optional[float]]:
    rows: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            rows.append(numeric)
    if not rows:
        return {"min": None, "median": None, "max": None, "spread": None}
    minimum = min(rows)
    maximum = max(rows)
    return {
        "min": minimum,
        "median": float(statistics.median(rows)),
        "max": maximum,
        "spread": maximum - minimum,
    }


def _component_rows(
    results: Iterable[Dict[str, Any]],
    *,
    domain: str,
    component: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        if str(result.get("domain") or "") != domain:
            continue
        residuals = result.get("residuals") if isinstance(result.get("residuals"), dict) else {}
        axis_result = (
            residuals.get("body_core_shape")
            if isinstance(residuals.get("body_core_shape"), dict)
            else {}
        )
        component_residuals = (
            axis_result.get("component_residuals")
            if isinstance(axis_result.get("component_residuals"), dict)
            else {}
        )
        value = component_residuals.get(component) if axis_result.get("available") else None
        rows.append(
            {
                "trial_id": result.get("trial_id"),
                "perturbation_family": result.get("perturbation_family"),
                "signed_strength": result.get("signed_strength"),
                "native_residual": value,
                "baseline_chain_signature": result.get("baseline_chain_signature"),
                "trial_chain_signature": result.get("chain_signature"),
                "chain_diagnostics": result.get("chain_diagnostics"),
            }
        )
    return rows


def _component_domain_summary(
    results: Iterable[Dict[str, Any]],
    *,
    domain: str,
    component: str,
) -> Dict[str, Any]:
    rows = _component_rows(results, domain=domain, component=component)
    summary = summarize_repeatability_trials(
        rows,
        domain=domain,
        residual_unit="signed_natural_log_ratio",
    )
    signed_values = [
        float(row["native_residual"])
        for row in rows
        if isinstance(row.get("native_residual"), (int, float))
        and math.isfinite(float(row["native_residual"]))
    ]
    summary["native_absolute_residual_descriptor"] = _descriptor(
        abs(value) for value in signed_values
    )
    summary["direction_preserved"] = True
    return summary


def _topology_trial_descriptor(axis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not bool(axis_result.get("available")):
        return None
    try:
        residual = np.asarray(axis_result.get("residual_vector"), dtype=np.float64)
    except Exception:
        return None
    if residual.size != 20670:
        return None
    residual = residual.reshape(6890, 3)
    if not bool(np.all(np.isfinite(residual))):
        return None
    coordinate_axes: Dict[str, Any] = {}
    for coordinate_index, coordinate_name in enumerate(TOPOLOGY_COORDINATE_AXES):
        values = residual[:, coordinate_index]
        coordinate_axes[coordinate_name] = {
            "signed_quantiles": {
                label: float(np.quantile(values, quantile))
                for label, quantile in TOPOLOGY_SIGNED_QUANTILES
            },
            "absolute_quantiles": {
                label: float(np.quantile(np.abs(values), quantile))
                for label, quantile in TOPOLOGY_ABSOLUTE_QUANTILES
            },
        }
    return {
        "vertex_count": 6890,
        "coordinate_count": 20670,
        "coordinate_axis_order": list(TOPOLOGY_COORDINATE_AXES),
        "coordinate_axes": coordinate_axes,
        "coordinate_axis_aggregation_allowed": False,
        "vertex_norm_aggregation_allowed": False,
        "combined_topology_score": None,
    }


def _topology_trial_rows(
    results: Iterable[Dict[str, Any]],
    *,
    domain: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        if str(result.get("domain") or "") != domain:
            continue
        residuals = result.get("residuals") if isinstance(result.get("residuals"), dict) else {}
        axis_result = (
            residuals.get("body_topology")
            if isinstance(residuals.get("body_topology"), dict)
            else {}
        )
        rows.append(
            {
                "trial_id": result.get("trial_id"),
                "perturbation_family": result.get("perturbation_family"),
                "signed_strength": result.get("signed_strength"),
                "coordinate_descriptors": _topology_trial_descriptor(axis_result),
                "raw_residual_vector_retained_in_trial_artifact": bool(
                    axis_result.get("available")
                    and isinstance(axis_result.get("residual_vector"), list)
                ),
                "baseline_chain_signature": result.get("baseline_chain_signature"),
                "trial_chain_signature": result.get("chain_signature"),
                "chain_diagnostics": result.get("chain_diagnostics"),
            }
        )
    return rows


def _topology_domain_summary(
    results: Iterable[Dict[str, Any]],
    *,
    domain: str,
) -> Dict[str, Any]:
    rows = _topology_trial_rows(results, domain=domain)
    available_rows = [
        row for row in rows if isinstance(row.get("coordinate_descriptors"), dict)
    ]
    assessed_transition_count = 0
    chain_transition_count = 0
    chain_diagnostic_values: Dict[str, List[float]] = {}
    for row in rows:
        baseline_signature = str(row.get("baseline_chain_signature") or "").strip()
        trial_signature = str(row.get("trial_chain_signature") or "").strip()
        if baseline_signature and trial_signature:
            assessed_transition_count += 1
            chain_transition_count += int(baseline_signature != trial_signature)
        diagnostics = (
            row.get("chain_diagnostics")
            if isinstance(row.get("chain_diagnostics"), dict)
            else {}
        )
        for name in [
            "bbox_iou",
            "bbox_center_displacement_image_fraction",
            "bbox_width_relative_delta",
            "bbox_height_relative_delta",
            "body_canonical_coverage_delta",
            "body_fit_confidence_delta",
            "pose_parameter_rms_delta",
        ]:
            value = _finite_scalar(diagnostics, name)
            if value is not None:
                chain_diagnostic_values.setdefault(name, []).append(value)
    coordinate_axes: Dict[str, Any] = {}
    for coordinate_name in TOPOLOGY_COORDINATE_AXES:
        signed_quantiles: Dict[str, Any] = {}
        absolute_quantiles: Dict[str, Any] = {}
        for label, quantile in TOPOLOGY_SIGNED_QUANTILES:
            values = [
                row["coordinate_descriptors"]["coordinate_axes"][coordinate_name][
                    "signed_quantiles"
                ][label]
                for row in available_rows
            ]
            signed_quantiles[label] = {
                "quantile": quantile,
                "across_trial_descriptor": _descriptor(values),
            }
        for label, quantile in TOPOLOGY_ABSOLUTE_QUANTILES:
            values = [
                row["coordinate_descriptors"]["coordinate_axes"][coordinate_name][
                    "absolute_quantiles"
                ][label]
                for row in available_rows
            ]
            absolute_quantiles[label] = {
                "quantile": quantile,
                "across_trial_descriptor": _descriptor(values),
            }
        coordinate_axes[coordinate_name] = {
            "signed_quantiles": signed_quantiles,
            "absolute_quantiles": absolute_quantiles,
        }
    if not rows:
        measurement_state = "NOT_MEASURED"
    elif available_rows:
        measurement_state = "OBSERVED_UNCALIBRATED"
    else:
        measurement_state = "MEASUREMENT_UNAVAILABLE"
    return {
        "schema_version": "body_topology_repeatability_domain_v0_1",
        "domain": domain,
        "measurement_state": measurement_state,
        "trial_count": len(rows),
        "available_residual_count": len(available_rows),
        "native_residual_unit": "smpl_model_length_unit",
        "raw_residual_vector_retained_per_trial": True,
        "detector_chain_transition_count": (
            chain_transition_count if assessed_transition_count else None
        ),
        "detector_chain_assessed_count": assessed_transition_count,
        "chain_diagnostic_descriptors": {
            name: _descriptor(values)
            for name, values in sorted(chain_diagnostic_values.items())
        },
        "coordinate_axes": coordinate_axes,
        "coordinate_axis_aggregation_allowed": False,
        "vertex_norm_aggregation_allowed": False,
        "combined_topology_score": None,
        "stable_unstable_classification": None,
        "trials": rows,
        "calibration_state": "SHADOW_UNCALIBRATED",
        "decision_influence": "NONE",
    }


def _axis_trial_availability(
    results: Iterable[Dict[str, Any]],
    *,
    axis: str,
) -> Dict[str, Dict[str, int]]:
    availability: Dict[str, Dict[str, int]] = {}
    for domain in REPEATABILITY_DOMAINS:
        domain_results = [
            result
            for result in results
            if str(result.get("domain") or "") == domain
        ]
        available_count = 0
        for result in domain_results:
            residuals = (
                result.get("residuals")
                if isinstance(result.get("residuals"), dict)
                else {}
            )
            axis_result = residuals.get(axis) if isinstance(residuals.get(axis), dict) else {}
            available_count += int(bool(axis_result.get("available")))
        availability[domain] = {
            "trial_count": len(domain_results),
            "available_residual_count": available_count,
        }
    return availability


def build_body_item_repeatability_summary(
    source: Dict[str, Any],
    baseline: Dict[str, Any],
    results: List[Dict[str, Any]],
    axes: Sequence[str],
) -> Dict[str, Any]:
    if tuple(axes) != BODY_REPEATABILITY_AXES:
        raise ValueError(f"body repeatability axes must be {BODY_REPEATABILITY_AXES!r}")
    state_counts: Dict[str, int] = {}
    for result in results:
        state = str(result.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1
    components = {
        component: {
            domain: _component_domain_summary(
                results,
                domain=domain,
                component=component,
            )
            for domain in REPEATABILITY_DOMAINS
        }
        for component in BODY_CORE_MEASUREMENT_ORDER
    }
    baseline_observation = (
        baseline.get("observation") if isinstance(baseline.get("observation"), dict) else {}
    )
    baseline_topology = _axis_node(baseline_observation, "body_topology")
    baseline_body_core = _axis_node(baseline_observation, "body_core_shape")
    topology_domains = {
        domain: _topology_domain_summary(results, domain=domain)
        for domain in REPEATABILITY_DOMAINS
    }
    return {
        "source": source,
        "baseline_state": baseline.get("state"),
        "trial_state_counts": state_counts,
        "axes": {
            "body_core_shape": {
                "baseline_available": bool(baseline_body_core.get("available")),
                "components": components,
                "measurement_order": list(BODY_CORE_MEASUREMENT_ORDER),
                "trial_availability": _axis_trial_availability(
                    results,
                    axis="body_core_shape",
                ),
                "component_aggregation_allowed": False,
                "combined_repeatability_score": None,
                "stable_unstable_classification": None,
                "decision_influence": "NONE",
            },
            "body_topology": {
                "baseline_available": bool(baseline_topology.get("available")),
                "domains": topology_domains,
                "coordinate_axis_order": list(TOPOLOGY_COORDINATE_AXES),
                "raw_residual_vector_retained_per_trial": True,
                "coordinate_axis_aggregation_allowed": False,
                "vertex_norm_aggregation_allowed": False,
                "combined_topology_score": None,
                "stable_unstable_classification": None,
                "decision_influence": "NONE",
            },
        },
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "decision_influence": "NONE",
    }


def _topology_cohort_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    coordinate_axes: Dict[str, Any] = {}
    quantile_groups = {
        "signed_quantiles": TOPOLOGY_SIGNED_QUANTILES,
        "absolute_quantiles": TOPOLOGY_ABSOLUTE_QUANTILES,
    }
    for coordinate_name in TOPOLOGY_COORDINATE_AXES:
        group_summaries: Dict[str, Any] = {}
        for group_name, quantiles in quantile_groups.items():
            quantile_summaries: Dict[str, Any] = {}
            for label, quantile in quantiles:
                domain_summaries: Dict[str, Any] = {}
                for domain in REPEATABILITY_DOMAINS:
                    source_medians: List[float] = []
                    source_maxima: List[float] = []
                    source_spreads: List[float] = []
                    for item in items:
                        axes = item.get("axes") if isinstance(item.get("axes"), dict) else {}
                        topology = (
                            axes.get("body_topology")
                            if isinstance(axes.get("body_topology"), dict)
                            else {}
                        )
                        domains = (
                            topology.get("domains")
                            if isinstance(topology.get("domains"), dict)
                            else {}
                        )
                        domain_node = (
                            domains.get(domain)
                            if isinstance(domains.get(domain), dict)
                            else {}
                        )
                        coordinate_node = (
                            domain_node.get("coordinate_axes", {}).get(coordinate_name, {})
                            if isinstance(domain_node.get("coordinate_axes"), dict)
                            else {}
                        )
                        quantile_node = (
                            coordinate_node.get(group_name, {}).get(label, {})
                            if isinstance(coordinate_node.get(group_name), dict)
                            else {}
                        )
                        descriptor = (
                            quantile_node.get("across_trial_descriptor")
                            if isinstance(
                                quantile_node.get("across_trial_descriptor"),
                                dict,
                            )
                            else {}
                        )
                        for key, target in [
                            ("median", source_medians),
                            ("max", source_maxima),
                            ("spread", source_spreads),
                        ]:
                            value = _finite_scalar(descriptor, key)
                            if value is not None:
                                target.append(value)
                    domain_summaries[domain] = {
                        "source_count": len(items),
                        "observed_source_count": len(source_medians),
                        "source_trial_median_descriptor": _descriptor(source_medians),
                        "source_trial_maximum_descriptor": _descriptor(source_maxima),
                        "source_trial_spread_descriptor": _descriptor(source_spreads),
                        "combined_score": None,
                    }
                quantile_summaries[label] = {
                    "quantile": quantile,
                    "domains": domain_summaries,
                }
            group_summaries[group_name] = quantile_summaries
        coordinate_axes[coordinate_name] = group_summaries
    return {
        "schema_version": "body_topology_repeatability_cohort_v0_1",
        "source_count": len(items),
        "coordinate_axis_order": list(TOPOLOGY_COORDINATE_AXES),
        "coordinate_axes": coordinate_axes,
        "coordinate_axis_aggregation_allowed": False,
        "vertex_norm_aggregation_allowed": False,
        "combined_topology_score": None,
        "stable_unstable_classification": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }


def build_body_repeatability_cohort_summary(
    items: List[Dict[str, Any]],
    axes: Sequence[str],
) -> Dict[str, Any]:
    if tuple(axes) != BODY_REPEATABILITY_AXES:
        raise ValueError(f"body repeatability axes must be {BODY_REPEATABILITY_AXES!r}")
    transformed_items: List[Dict[str, Any]] = []
    for item in items:
        axis_node = (
            (item.get("axes") or {}).get("body_core_shape")
            if isinstance(item.get("axes"), dict)
            else {}
        )
        components = (
            axis_node.get("components")
            if isinstance(axis_node, dict) and isinstance(axis_node.get("components"), dict)
            else {}
        )
        transformed_items.append(
            {
                **{key: value for key, value in item.items() if key != "axes"},
                "axes": {
                    component: dict(components.get(component) or {})
                    for component in BODY_CORE_MEASUREMENT_ORDER
                },
            }
        )
    component_cohort = summarize_repeatability_cohort(
        transformed_items,
        axes=BODY_CORE_MEASUREMENT_ORDER,
    )
    body_core_summary = {
        "axis": "body_core_shape",
        "components": component_cohort.get("axes") or {},
        "component_aggregation_allowed": False,
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }
    body_topology_summary = _topology_cohort_summary(items)
    return {
        "schema_version": "body_repeatability_cohort_descriptor_v0_2",
        "source_count": len(items),
        "axis": "body_core_shape",
        "components": body_core_summary["components"],
        "component_aggregation_allowed": False,
        "axes": {
            "body_core_shape": body_core_summary,
            "body_topology": body_topology_summary,
        },
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }


def run_body_repeatability_shadow(
    *,
    image_paths: Sequence[Path],
    output_root: Path,
    adapter: Any,
    run_id: Optional[str] = None,
    retry_failed: bool = False,
) -> Dict[str, Any]:
    summary = run_repeatability_shadow_engine(
        image_paths=image_paths,
        output_root=output_root,
        adapter=adapter,
        axes=BODY_REPEATABILITY_AXES,
        protocol=load_body_repeatability_protocol(),
        run_schema=BODY_REPEATABILITY_RUN_SCHEMA,
        trial_schema=BODY_REPEATABILITY_TRIAL_SCHEMA,
        runner_implementation_path=Path(__file__).resolve(),
        native_residual_builder=build_body_native_repeatability_residual,
        chain_diagnostics_builder=build_body_chain_diagnostics,
        item_summary_builder=build_body_item_repeatability_summary,
        cohort_summary_builder=build_body_repeatability_cohort_summary,
        run_id=run_id,
        retry_failed=retry_failed,
    )
    axis_availability: Dict[str, Any] = {}
    source_count = int(summary.get("source_count") or 0)
    planned_trial_count = int(summary.get("planned_trial_count") or 0)
    items = [item for item in list(summary.get("items") or []) if isinstance(item, dict)]
    for axis in BODY_REPEATABILITY_AXES:
        baseline_available_source_count = 0
        trial_count = 0
        available_trial_count = 0
        for item in items:
            axes = item.get("axes") if isinstance(item.get("axes"), dict) else {}
            axis_node = axes.get(axis) if isinstance(axes.get(axis), dict) else {}
            baseline_available_source_count += int(bool(axis_node.get("baseline_available")))
            domain_availability = (
                axis_node.get("trial_availability")
                if axis == "body_core_shape"
                else axis_node.get("domains")
            )
            domain_availability = (
                domain_availability if isinstance(domain_availability, dict) else {}
            )
            for domain in REPEATABILITY_DOMAINS:
                domain_node = (
                    domain_availability.get(domain)
                    if isinstance(domain_availability.get(domain), dict)
                    else {}
                )
                trial_count += int(domain_node.get("trial_count") or 0)
                available_trial_count += int(
                    domain_node.get("available_residual_count") or 0
                )
        fully_observed = (
            source_count > 0
            and baseline_available_source_count == source_count
            and trial_count == planned_trial_count
            and available_trial_count == planned_trial_count
        )
        axis_availability[axis] = {
            "source_count": source_count,
            "baseline_available_source_count": baseline_available_source_count,
            "planned_trial_count": planned_trial_count,
            "observed_trial_count": trial_count,
            "available_trial_count": available_trial_count,
            "fully_observed": fully_observed,
            "decision_influence": "NONE",
        }
    summary["axis_availability"] = axis_availability
    if summary.get("status") == "COMPLETE" and not all(
        axis_availability[axis]["fully_observed"] for axis in BODY_REPEATABILITY_AXES
    ):
        summary["status"] = "COMPLETE_WITH_UNAVAILABLE_MEASUREMENTS"
    atomic_write_json(Path(str(summary["run_dir"])) / "run_summary.json", summary)
    return summary
