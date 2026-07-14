from __future__ import annotations

import re
from typing import Any, Dict, List

from .qa_runtime import anchor_registry_snapshot


TRUTH_INTEGRITY_SCHEMA = "truth_integrity_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_TRUTHS = {
    "face_identity": ("face_truth_anchor", "FACE_MASTER"),
    "body_master": ("body_truth_anchor", "FULL_BODY_MASTER"),
}


def validate_truth_integrity(config: Any) -> Dict[str, Any]:
    snapshot = anchor_registry_snapshot(config)
    rules = snapshot.get("rules") if isinstance(snapshot.get("rules"), dict) else {}
    entries = snapshot.get("entries") if isinstance(snapshot.get("entries"), dict) else {}
    checks: List[Dict[str, Any]] = []
    issues: List[str] = []
    truth_ids: List[str] = []

    for truth_label, (rule_key, expected_role) in _REQUIRED_TRUTHS.items():
        anchor_id = str(rules.get(rule_key, "")).strip()
        raw_node = entries.get(anchor_id) if anchor_id else None
        node = raw_node if isinstance(raw_node, dict) else {}
        truth_ids.append(anchor_id)
        check_issues: List[str] = []

        if not anchor_id:
            check_issues.append(f"{truth_label.upper()}_RULE_MISSING")
        if not node:
            check_issues.append(f"{truth_label.upper()}_ANCHOR_MISSING")
        if str(node.get("role", "")).upper() != expected_role:
            check_issues.append(f"{truth_label.upper()}_ROLE_INVALID")
        if str(node.get("anchor_tier", "")).lower() != "absolute":
            check_issues.append(f"{truth_label.upper()}_TIER_INVALID")
        if str(node.get("authority", "")).upper() != "ABSOLUTE_FROZEN":
            check_issues.append(f"{truth_label.upper()}_AUTHORITY_INVALID")
        if bool(node.get("mutable", True)):
            check_issues.append(f"{truth_label.upper()}_MUST_BE_IMMUTABLE")
        if bool(node.get("may_modify_truth", True)):
            check_issues.append(f"{truth_label.upper()}_MAY_MODIFY_TRUTH_INVALID")
        if not bool(node.get("exists", False)):
            check_issues.append(f"{truth_label.upper()}_FILE_MISSING")

        expected_sha256 = str(node.get("expected_sha256") or "").strip().lower()
        actual_sha256 = str(node.get("actual_sha256") or "").strip().lower()
        if not _SHA256_RE.fullmatch(expected_sha256):
            check_issues.append(f"{truth_label.upper()}_SHA256_NOT_PINNED")
        elif actual_sha256 != expected_sha256:
            check_issues.append(f"{truth_label.upper()}_SHA256_MISMATCH")

        checks.append(
            {
                "truth_label": truth_label,
                "rule_key": rule_key,
                "anchor_id": anchor_id,
                "resolved_path": str(node.get("resolved_path") or ""),
                "role": str(node.get("role") or ""),
                "expected_role": expected_role,
                "anchor_tier": str(node.get("anchor_tier") or ""),
                "authority": str(node.get("authority") or ""),
                "mutable": bool(node.get("mutable", False)),
                "may_modify_truth": bool(node.get("may_modify_truth", False)),
                "expected_sha256": expected_sha256 or None,
                "actual_sha256": actual_sha256 or None,
                "status": "PASS" if not check_issues else "FAIL",
                "issues": check_issues,
            }
        )
        issues.extend(check_issues)

    populated_truth_ids = [anchor_id for anchor_id in truth_ids if anchor_id]
    if len(populated_truth_ids) != len(set(populated_truth_ids)):
        issues.append("FACE_AND_BODY_TRUTH_MUST_BE_DISTINCT")

    return {
        "schema_version": TRUTH_INTEGRITY_SCHEMA,
        "status": "PASS" if not issues else "FAIL",
        "gpu_initialization_allowed": not issues,
        "checks": checks,
        "issues": list(dict.fromkeys(issues)),
    }


def assert_truth_integrity(config: Any) -> Dict[str, Any]:
    result = validate_truth_integrity(config)
    if result["status"] != "PASS":
        reasons = ", ".join(result.get("issues") or ["UNKNOWN_TRUTH_INTEGRITY_FAILURE"])
        raise RuntimeError(f"TRUTH_INTEGRITY_CHECK_FAILED: {reasons}")
    return result
