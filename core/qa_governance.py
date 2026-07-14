from __future__ import annotations

import copy
from typing import Any, Dict, Optional


RELEASE_GATE_SCHEMA_VERSION = "qa_release_gates_v2"
LEGACY_RELEASE_GATE_SCHEMA_VERSION = "qa_release_gates_v1"
LEGACY_ADMISSION_FIELDS_STATE = "DEPRECATED_FORCED_FALSE"

EXTERNAL_REVIEW_ROUTES = {
    "PRIORITY_REVIEW",
    "STANDARD_REVIEW",
    "HOLD_FOR_MORE_EVIDENCE",
    "SHADOW_EVIDENCE_ONLY",
    "NOT_APPLICABLE",
}


def _schema_state(source_schema_version: str) -> str:
    if source_schema_version == RELEASE_GATE_SCHEMA_VERSION:
        return "VALID_V2"
    if source_schema_version == LEGACY_RELEASE_GATE_SCHEMA_VERSION:
        return "LEGACY_MIGRATED_FAIL_CLOSED"
    if not source_schema_version:
        return "MISSING_SCHEMA_FAIL_CLOSED"
    return "UNKNOWN_SCHEMA_FAIL_CLOSED"


def _default_external_review_route(release_state: str) -> str:
    if release_state == "primary":
        return "PRIORITY_REVIEW"
    if release_state in {"shadow", "filter_only"}:
        return "SHADOW_EVIDENCE_ONLY"
    if release_state == "review":
        return "STANDARD_REVIEW"
    return "HOLD_FOR_MORE_EVIDENCE"


def fail_closed_release_gate(
    raw_gate: Optional[Dict[str, Any]],
    *,
    target_bucket: str = "",
    source_schema_version: Optional[str] = None,
    fallback_gate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one gate while making local admission and fitting impossible."""
    raw = raw_gate if isinstance(raw_gate, dict) else {}
    fallback = fallback_gate if isinstance(fallback_gate, dict) else {}
    source_schema = str(
        source_schema_version
        if source_schema_version is not None
        else raw.get("schema_version") or ""
    ).strip()
    schema_state = _schema_state(source_schema)

    release_state = str(
        raw.get("release_state", fallback.get("release_state", "review"))
    ).strip().lower() or "review"
    ceiling = str(
        raw.get("machine_status_ceiling", fallback.get("machine_status_ceiling", "WARN"))
    ).strip().upper()
    if ceiling not in {"FAIL", "WARN", "PASS"}:
        ceiling = "WARN"

    requested_route = str(
        raw.get("external_review_route", fallback.get("external_review_route", ""))
    ).strip().upper()
    if schema_state == "VALID_V2" and requested_route in EXTERNAL_REVIEW_ROUTES:
        external_review_route = requested_route
    elif schema_state == "LEGACY_MIGRATED_FAIL_CLOSED":
        external_review_route = _default_external_review_route(release_state)
    elif schema_state == "VALID_V2":
        external_review_route = _default_external_review_route(release_state)
    else:
        external_review_route = "HOLD_FOR_MORE_EVIDENCE"

    required_lane_families = raw.get(
        "required_lane_families", fallback.get("required_lane_families", [])
    )
    if isinstance(required_lane_families, list):
        normalized_lanes = [
            str(item).strip() for item in required_lane_families if str(item).strip()
        ]
    elif required_lane_families is None:
        normalized_lanes = []
    else:
        lane = str(required_lane_families).strip()
        normalized_lanes = [lane] if lane else []

    overrides = []
    if bool(raw.get("training_admission_allowed")):
        overrides.append("training_admission_allowed_forced_false")
    if bool(raw.get("manual_training_admission_required")):
        overrides.append("manual_training_admission_required_retired")
    if bool(raw.get("may_emit_final_admission")):
        overrides.append("may_emit_final_admission_forced_false")
    if bool(raw.get("may_emit_final_image_set_membership")):
        overrides.append("may_emit_final_image_set_membership_forced_false")
    if bool(raw.get("optuna_fit_allowed")) or bool(raw.get("parameter_fitting_allowed")):
        overrides.append("parameter_fitting_forced_false")

    return {
        "schema_version": RELEASE_GATE_SCHEMA_VERSION,
        "source_schema_version": source_schema,
        "schema_state": schema_state,
        "target_bucket": str(target_bucket or raw.get("target_bucket") or "").strip(),
        "release_state": release_state,
        "machine_status_ceiling": ceiling,
        "local_decision_authority": "NONE",
        "external_review_route": external_review_route,
        "external_human_review_required": external_review_route != "NOT_APPLICABLE",
        "may_emit_final_admission": False,
        "may_emit_final_image_set_membership": False,
        "training_admission_allowed": False,
        "manual_training_admission_required": False,
        "legacy_admission_fields_state": LEGACY_ADMISSION_FIELDS_STATE,
        "parameter_fitting_allowed": False,
        "optuna_fit_allowed": False,
        "requires_frozen_benchmark": bool(
            raw.get(
                "requires_frozen_benchmark",
                fallback.get("requires_frozen_benchmark", False),
            )
        ),
        "requires_curated_winner_bank": bool(
            raw.get(
                "requires_curated_winner_bank",
                fallback.get("requires_curated_winner_bank", False),
            )
        ),
        "required_lane_families": normalized_lanes,
        "notes": str(raw.get("notes", fallback.get("notes", ""))).strip(),
        "governance_overrides_applied": overrides,
    }


def normalize_release_gate_config(
    data: Any,
    default_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize a release-gate document without importing the vision runtime."""
    base = copy.deepcopy(default_config)
    if not isinstance(data, dict):
        return base
    gates_node = data.get("release_gates", data)
    if not isinstance(gates_node, dict):
        return base

    raw_schema_version = str(data.get("schema_version", "")).strip()
    normalized_gates: Dict[str, Dict[str, Any]] = copy.deepcopy(
        base.get("release_gates", {})
    )
    for bucket_name, raw_node in gates_node.items():
        if not isinstance(raw_node, dict):
            continue
        bucket_key = str(bucket_name).strip()
        if not bucket_key:
            continue
        normalized_gates[bucket_key] = fail_closed_release_gate(
            raw_node,
            target_bucket=bucket_key,
            source_schema_version=raw_schema_version,
            fallback_gate=normalized_gates.get(bucket_key, {}),
        )
    return {
        "schema_version": RELEASE_GATE_SCHEMA_VERSION,
        "source_schema_version": raw_schema_version,
        "release_gates": normalized_gates,
    }
