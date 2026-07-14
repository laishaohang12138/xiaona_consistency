from __future__ import annotations

import math
import hashlib
import json
import statistics
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPEATABILITY_CONTRACT_SCHEMA = "measurement_repeatability_contract_v0_1"
REPEATABILITY_PROTOCOL_ID = "identity_repeatability_probe_protocol_v0_1"
REPEATABILITY_DOMAINS = (
    "numerical_repeatability",
    "preprocessing_repeatability",
    "admissible_perturbation_stability",
)


def _protocol_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "identity_repeatability_protocol.yaml"


def _read_protocol_payload() -> Any:
    import yaml

    return yaml.safe_load(_protocol_path().read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _descriptor(values: Iterable[float]) -> Dict[str, Optional[float]]:
    rows = [float(value) for value in values]
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


def validate_repeatability_protocol(payload: Any) -> list[str]:
    node = payload if isinstance(payload, dict) else {}
    governance = node.get("governance") if isinstance(node.get("governance"), dict) else {}
    execution = node.get("execution") if isinstance(node.get("execution"), dict) else {}
    domains = node.get("domains") if isinstance(node.get("domains"), dict) else {}
    reporting = node.get("reporting") if isinstance(node.get("reporting"), dict) else {}
    transform_contract = (
        node.get("transform_contract") if isinstance(node.get("transform_contract"), dict) else {}
    )
    issues: list[str] = []
    if node.get("schema_version") != REPEATABILITY_PROTOCOL_ID:
        issues.append("REPEATABILITY_PROTOCOL_SCHEMA_INVALID")
    if node.get("protocol_id") != REPEATABILITY_PROTOCOL_ID:
        issues.append("REPEATABILITY_PROTOCOL_ID_INVALID")
    if node.get("protocol_state") != "PREREGISTERED_NOT_EXECUTED":
        issues.append("REPEATABILITY_PROTOCOL_STATE_INVALID")
    if governance.get("mode") != "SHADOW" or governance.get("decision_influence") != "NONE":
        issues.append("REPEATABILITY_PROTOCOL_SHADOW_GOVERNANCE_INVALID")
    if governance.get("parameter_fitting_allowed") is not False:
        issues.append("REPEATABILITY_PROTOCOL_PARAMETER_FITTING_MUST_BE_FALSE")
    if governance.get("threshold_fitting_allowed") is not False:
        issues.append("REPEATABILITY_PROTOCOL_THRESHOLD_FITTING_MUST_BE_FALSE")
    for field in ["may_affect_ranking", "may_affect_review_route", "may_affect_winner_bank"]:
        if governance.get(field) is not False:
            issues.append(f"REPEATABILITY_PROTOCOL_{field.upper()}_MUST_BE_FALSE")
    if execution.get("enabled_by_default") is not False:
        issues.append("REPEATABILITY_PROTOCOL_DEFAULT_EXECUTION_MUST_BE_FALSE")
    if execution.get("requires_explicit_workflow") is not True:
        issues.append("REPEATABILITY_PROTOCOL_EXPLICIT_WORKFLOW_REQUIRED")
    if execution.get("max_concurrent_gpu_jobs") != 1:
        issues.append("REPEATABILITY_PROTOCOL_GPU_CONCURRENCY_MUST_BE_ONE")
    if execution.get("retain_completed_materialized_images") is not False:
        issues.append("REPEATABILITY_PROTOCOL_COMPLETED_IMAGES_MUST_BE_RECONSTRUCTABLE")
    if execution.get("retain_failed_materialized_images") is not True:
        issues.append("REPEATABILITY_PROTOCOL_FAILED_IMAGES_MUST_BE_RETAINED")
    if set(domains) != set(REPEATABILITY_DOMAINS):
        issues.append("REPEATABILITY_PROTOCOL_DOMAINS_INVALID")
    trial_ids: list[str] = []
    for domain in REPEATABILITY_DOMAINS:
        domain_node = domains.get(domain) if isinstance(domains.get(domain), dict) else {}
        transforms = domain_node.get("transforms") if isinstance(domain_node.get("transforms"), list) else []
        if not transforms:
            issues.append(f"REPEATABILITY_PROTOCOL_{domain.upper()}_TRANSFORMS_MISSING")
        for transform in transforms:
            transform_node = transform if isinstance(transform, dict) else {}
            trial_id = str(transform_node.get("id") or "").strip()
            if not trial_id:
                issues.append(f"REPEATABILITY_PROTOCOL_{domain.upper()}_TRIAL_ID_MISSING")
            else:
                trial_ids.append(trial_id)
            if _safe_float(transform_node.get("signed_strength")) is None:
                issues.append(f"REPEATABILITY_PROTOCOL_{domain.upper()}_STRENGTH_INVALID")
    if len(trial_ids) != len(set(trial_ids)):
        issues.append("REPEATABILITY_PROTOCOL_TRIAL_IDS_MUST_BE_UNIQUE")
    if reporting.get("combined_repeatability_score") is not None:
        issues.append("REPEATABILITY_PROTOCOL_COMBINED_SCORE_NOT_ALLOWED")
    if reporting.get("stable_unstable_threshold") is not None:
        issues.append("REPEATABILITY_PROTOCOL_STABILITY_THRESHOLD_NOT_ALLOWED")
    if transform_contract.get("schema_version") != "identity_repeatability_transform_contract_v0_1":
        issues.append("REPEATABILITY_PROTOCOL_TRANSFORM_CONTRACT_INVALID")
    if transform_contract.get("preserve_output_dimensions") is not True:
        issues.append("REPEATABILITY_PROTOCOL_OUTPUT_DIMENSIONS_MUST_BE_PRESERVED")
    return list(dict.fromkeys(issues))


@lru_cache(maxsize=1)
def repeatability_protocol_snapshot() -> Dict[str, Any]:
    protocol_path = _protocol_path()
    payload: Any = None
    load_error = None
    try:
        payload = _read_protocol_payload()
    except Exception as exc:
        load_error = f"{type(exc).__name__}:{exc}"
    issues = validate_repeatability_protocol(payload)
    if load_error:
        issues.insert(0, "REPEATABILITY_PROTOCOL_LOAD_FAILED")
    protocol_sha256 = None
    if isinstance(payload, dict):
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        protocol_sha256 = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return {
        "protocol_id": REPEATABILITY_PROTOCOL_ID,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "validation_status": "VALID" if not issues else "INVALID",
        "validation_issues": issues,
        "load_error": load_error,
    }


def load_repeatability_protocol() -> Dict[str, Any]:
    """Return a validated protocol copy suitable for deterministic execution."""
    payload = _read_protocol_payload()
    issues = validate_repeatability_protocol(payload)
    if issues:
        raise ValueError("invalid repeatability protocol: " + ", ".join(issues))
    if not isinstance(payload, dict):
        raise ValueError("invalid repeatability protocol payload")
    return copy.deepcopy(payload)


def empty_repeatability_contract() -> Dict[str, Any]:
    protocol = repeatability_protocol_snapshot()
    return {
        "schema_version": REPEATABILITY_CONTRACT_SCHEMA,
        "protocol_id": REPEATABILITY_PROTOCOL_ID,
        "protocol_execution_state": "NOT_EXECUTED",
        "protocol_sha256": protocol.get("protocol_sha256"),
        "protocol_validation_status": protocol.get("validation_status"),
        "protocol_validation_issues": list(protocol.get("validation_issues") or []),
        "domains": {
            domain: {
                "measurement_state": "NOT_MEASURED",
                "trial_count": 0,
                "available_residual_count": 0,
                "native_residual_descriptor": _descriptor([]),
                "detector_chain_transition_count": None,
                "perturbation_families": {},
                "calibration_state": "SHADOW_UNCALIBRATED",
                "decision_influence": "NONE",
            }
            for domain in REPEATABILITY_DOMAINS
        },
        "combined_repeatability_score": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }


def summarize_repeatability_trials(
    trials: Iterable[Dict[str, Any]],
    *,
    domain: str,
    residual_unit: str,
) -> Dict[str, Any]:
    normalized_domain = str(domain).strip().lower()
    if normalized_domain not in REPEATABILITY_DOMAINS:
        raise ValueError(f"unsupported repeatability domain: {domain!r}")
    rows: List[Dict[str, Any]] = [dict(row) for row in trials if isinstance(row, dict)]
    residuals: List[float] = []
    grouped: Dict[str, List[float]] = {}
    chain_diagnostic_values: Dict[str, List[float]] = {}
    transition_count = 0
    assessed_transition_count = 0
    normalized_trials: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        residual = _safe_float(row.get("native_residual"))
        family = str(row.get("perturbation_family") or "unclassified").strip() or "unclassified"
        baseline_signature = str(row.get("baseline_chain_signature") or "").strip()
        trial_signature = str(row.get("trial_chain_signature") or "").strip()
        chain_transition = None
        chain_diagnostics = row.get("chain_diagnostics") if isinstance(row.get("chain_diagnostics"), dict) else {}
        for diagnostic_name in [
            "bbox_iou",
            "bbox_center_displacement_image_fraction",
            "bbox_width_relative_delta",
            "bbox_height_relative_delta",
            "kps5_raw_rms_image_fraction",
            "kps5_similarity_shape_residual",
            "canonical_pose_l2_delta_deg",
        ]:
            diagnostic_value = _safe_float(chain_diagnostics.get(diagnostic_name))
            if diagnostic_value is not None:
                chain_diagnostic_values.setdefault(diagnostic_name, []).append(diagnostic_value)
        if baseline_signature and trial_signature:
            assessed_transition_count += 1
            chain_transition = baseline_signature != trial_signature
            transition_count += int(chain_transition)
        if residual is not None:
            residuals.append(residual)
            grouped.setdefault(family, []).append(residual)
        normalized_trials.append(
            {
                "trial_id": str(row.get("trial_id") or f"trial_{index + 1:04d}"),
                "perturbation_family": family,
                "signed_strength": _safe_float(row.get("signed_strength")),
                "native_residual": residual,
                "unit": str(residual_unit),
                "chain_transition": chain_transition,
                "chain_diagnostics": dict(chain_diagnostics),
                "measurement_available": residual is not None,
            }
        )

    return {
        "schema_version": REPEATABILITY_CONTRACT_SCHEMA,
        "domain": normalized_domain,
        "measurement_state": "OBSERVED_UNCALIBRATED" if rows else "NOT_MEASURED",
        "trial_count": len(rows),
        "available_residual_count": len(residuals),
        "native_residual_unit": str(residual_unit),
        "native_residual_descriptor": _descriptor(residuals),
        "detector_chain_transition_count": transition_count if assessed_transition_count else None,
        "detector_chain_assessed_count": assessed_transition_count,
        "chain_diagnostic_descriptors": {
            name: _descriptor(values)
            for name, values in sorted(chain_diagnostic_values.items())
        },
        "perturbation_families": {
            family: {
                "trial_count": len(values),
                "native_residual_descriptor": _descriptor(values),
            }
            for family, values in sorted(grouped.items())
        },
        "trials": normalized_trials,
        "calibration_state": "SHADOW_UNCALIBRATED",
        "stable_unstable_classification": None,
        "decision_influence": "NONE",
    }


def summarize_repeatability_cohort(
    items: Iterable[Dict[str, Any]],
    *,
    axes: Sequence[str],
) -> Dict[str, Any]:
    """Describe cross-source variation without pooling it into a score."""
    item_rows = [dict(item) for item in items if isinstance(item, dict)]
    axes_summary: Dict[str, Any] = {}
    for axis in axes:
        domain_summaries: Dict[str, Any] = {}
        for domain in REPEATABILITY_DOMAINS:
            observed_source_count = 0
            fully_observed_source_count = 0
            units: set[str] = set()
            source_medians: List[float] = []
            source_maxima: List[float] = []
            source_spreads: List[float] = []
            trial_values: Dict[str, Dict[str, Any]] = {}
            chain_source_medians: Dict[str, List[float]] = {}

            for item in item_rows:
                item_axes = item.get("axes") if isinstance(item.get("axes"), dict) else {}
                axis_node = item_axes.get(axis) if isinstance(item_axes.get(axis), dict) else {}
                domain_node = axis_node.get(domain) if isinstance(axis_node.get(domain), dict) else {}
                unit = str(domain_node.get("native_residual_unit") or "").strip()
                if unit:
                    units.add(unit)
                descriptor = (
                    domain_node.get("native_residual_descriptor")
                    if isinstance(domain_node.get("native_residual_descriptor"), dict)
                    else {}
                )
                source_median = _safe_float(descriptor.get("median"))
                source_maximum = _safe_float(descriptor.get("max"))
                source_spread = _safe_float(descriptor.get("spread"))
                if source_median is not None:
                    observed_source_count += 1
                    source_medians.append(source_median)
                if source_maximum is not None:
                    source_maxima.append(source_maximum)
                if source_spread is not None:
                    source_spreads.append(source_spread)
                trial_count = int(domain_node.get("trial_count") or 0)
                available_count = int(domain_node.get("available_residual_count") or 0)
                if trial_count > 0 and available_count == trial_count:
                    fully_observed_source_count += 1

                chain_descriptors = (
                    domain_node.get("chain_diagnostic_descriptors")
                    if isinstance(domain_node.get("chain_diagnostic_descriptors"), dict)
                    else {}
                )
                for name, chain_descriptor in chain_descriptors.items():
                    if not isinstance(chain_descriptor, dict):
                        continue
                    value = _safe_float(chain_descriptor.get("median"))
                    if value is not None:
                        chain_source_medians.setdefault(str(name), []).append(value)

                trials = domain_node.get("trials") if isinstance(domain_node.get("trials"), list) else []
                for trial in trials:
                    if not isinstance(trial, dict):
                        continue
                    trial_id = str(trial.get("trial_id") or "").strip()
                    if not trial_id:
                        continue
                    trial_node = trial_values.setdefault(
                        trial_id,
                        {
                            "perturbation_family": str(trial.get("perturbation_family") or "unclassified"),
                            "signed_strength": _safe_float(trial.get("signed_strength")),
                            "values": [],
                        },
                    )
                    residual = _safe_float(trial.get("native_residual"))
                    if residual is not None:
                        trial_node["values"].append(residual)

            unit_state = "CONSISTENT" if len(units) <= 1 else "MISMATCH"
            residual_unit = next(iter(units)) if len(units) == 1 else None
            domain_summaries[domain] = {
                "schema_version": "repeatability_cross_source_descriptor_v0_1",
                "domain": domain,
                "source_count": len(item_rows),
                "observed_source_count": observed_source_count,
                "fully_observed_source_count": fully_observed_source_count,
                "native_residual_unit": residual_unit,
                "native_residual_unit_state": unit_state,
                "source_native_residual_descriptors": {
                    "source_medians": _descriptor(source_medians),
                    "source_maxima": _descriptor(source_maxima),
                    "source_spreads": _descriptor(source_spreads),
                },
                "trial_descriptors": {
                    trial_id: {
                        "perturbation_family": node["perturbation_family"],
                        "signed_strength": node["signed_strength"],
                        "observed_source_count": len(node["values"]),
                        "native_residual_descriptor": _descriptor(node["values"]),
                    }
                    for trial_id, node in sorted(trial_values.items())
                },
                "source_median_chain_diagnostic_descriptors": {
                    name: _descriptor(values)
                    for name, values in sorted(chain_source_medians.items())
                },
                "aggregation_policy": "DESCRIPTIVE_ONLY_SOURCE_LEVEL_NO_POOLING_NO_FITTING",
                "evidence_independence": "DETECTOR_CHAIN_DIAGNOSTICS_ARE_NOT_INDEPENDENT_VOTES",
                "calibration_state": "SHADOW_UNCALIBRATED",
                "stable_unstable_classification": None,
                "decision_influence": "NONE",
            }
        axes_summary[str(axis)] = domain_summaries

    return {
        "schema_version": "repeatability_cohort_descriptor_v0_1",
        "source_count": len(item_rows),
        "axes": axes_summary,
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }
