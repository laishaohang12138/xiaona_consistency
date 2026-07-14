from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .qa_admission import resolve_target_bucket
from .qa_governance import fail_closed_release_gate
from .qa_io import atomic_write_json


def training_admission_manifest_path(output_dir: Path) -> Path:
    return output_dir / "training_admission_manifest.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest_payload(manifest_file: Path) -> Dict[str, Any]:
    if not manifest_file.exists():
        return {"available": False, "entries": [], "reason": "missing_manifest_file"}
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "entries": [], "reason": f"invalid_json:{exc}"}
    entries = list(payload.get("entries") or [])
    return {
        "available": len(entries) > 0,
        "entries": entries,
        "payload": payload if isinstance(payload, dict) else {},
        "reason": None if len(entries) > 0 else "empty_manifest",
    }


def load_training_admission_manifest(manifest_file: Path) -> Dict[str, Any]:
    payload = _load_manifest_payload(manifest_file)
    payload["file"] = str(manifest_file)
    return payload


def _recent_entries(entries: Sequence[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    ordered = sorted(
        [dict(entry) for entry in entries if isinstance(entry, dict)],
        key=lambda row: str(
            (row.get("external_audit") or {}).get("recorded_at_utc")
            or (row.get("human_seal") or {}).get("sealed_at_utc")
            or ""
        ),
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for entry in ordered[: max(0, int(limit))]:
        external_audit = entry.get("external_audit") or {}
        human_seal = entry.get("human_seal") or {}
        out.append(
            {
                "image": entry.get("image"),
                "record_key": entry.get("record_key"),
                "target_bucket": entry.get("target_bucket"),
                "recorded_at_utc": external_audit.get("recorded_at_utc")
                or human_seal.get("sealed_at_utc"),
                "sealed_at_utc": None,
                "owner": external_audit.get("owner") or human_seal.get("owner"),
                "note": external_audit.get("note") or human_seal.get("note"),
            }
        )
    return out


def load_training_admission_manifest_summary(manifest_file: Path) -> Dict[str, Any]:
    payload = _load_manifest_payload(manifest_file)
    entries = list(payload.get("entries") or [])
    bucket_counts: Dict[str, int] = {}
    for entry in entries:
        bucket = str(entry.get("target_bucket") or "").strip()
        if not bucket:
            continue
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    last_entry = _recent_entries(entries, limit=1)
    return {
        "manifest_file": str(manifest_file),
        "scope": "external_training_admission_audit_only",
        "local_decision_participation": False,
        "available": bool(payload.get("available")),
        "reason": payload.get("reason"),
        "entry_count": len(entries),
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda row: (-row[1], row[0]))),
        "last_recorded_at_utc": (last_entry[0].get("recorded_at_utc") if last_entry else None),
        "last_sealed_at_utc": None,
        "legacy_seal_fields_state": "DEPRECATED_FORCED_EMPTY",
        "recent_entries": _recent_entries(entries, limit=3),
    }


def _gate_summary(release_gate: Optional[Dict[str, Any]], target_bucket: str) -> Dict[str, Any]:
    gate = release_gate or {}
    return fail_closed_release_gate(
        gate,
        target_bucket=target_bucket,
        source_schema_version=str(gate.get("schema_version") or "").strip(),
    )


def seal_training_admission_entry(
    candidate_entry: Dict[str, Any],
    manifest_file: Path,
    *,
    release_gate: Optional[Dict[str, Any]] = None,
    admission_advice: Optional[Dict[str, Any]] = None,
    batch_preflight: Optional[Dict[str, Any]] = None,
    evidence_completeness: Optional[Dict[str, Any]] = None,
    threshold_hash: Optional[str] = None,
    anchor_snapshot: Optional[Dict[str, Any]] = None,
    source_batch: Optional[Dict[str, Any]] = None,
    source_files: Optional[Dict[str, Any]] = None,
    manual_owner: Optional[str] = None,
    manual_note: Optional[str] = None,
    external_decision_confirmed: bool = False,
) -> Dict[str, Any]:
    owner = str(manual_owner or "").strip()
    if not owner:
        raise ValueError("manual_owner is required to record external training admission audit")
    if not external_decision_confirmed:
        return {
            "status": "blocked",
            "reason": "external_decision_confirmation_required",
            "manifest_file": str(manifest_file),
            "local_decision_authority": "NONE",
        }

    target_profile = str(candidate_entry.get("target_profile") or "").strip()
    target_bucket = resolve_target_bucket(target_profile)
    gate_summary = _gate_summary(release_gate, target_bucket)
    batch_advice = admission_advice or {}
    preflight_summary = batch_preflight or {}
    evidence_summary = evidence_completeness or {}
    evidence_warnings: List[str] = []
    evidence_warnings.extend(str(item) for item in (batch_advice.get("blockers") or [])[:8])
    preflight_status = str(preflight_summary.get("status") or "").strip().upper()
    if preflight_status == "FAIL":
        evidence_warnings.append("BATCH_PREFLIGHT_FAILED_AT_AUDIT_TIME")
    evidence_status = str(evidence_summary.get("status") or "").strip().upper()
    if evidence_status == "FAIL":
        evidence_warnings.append("EVIDENCE_COMPLETENESS_FAILED_AT_AUDIT_TIME")
    if evidence_summary and not bool(evidence_summary.get("replay_ready")):
        evidence_warnings.append("EVIDENCE_NOT_REPLAY_READY_AT_AUDIT_TIME")
    evidence_warnings = list(dict.fromkeys(item for item in evidence_warnings if item))[:12]

    payload = _load_manifest_payload(manifest_file)
    entries = list(payload.get("entries") or [])
    audit_timestamp = _utcnow_iso()
    audit_entry = {
        "schema_version": "external_admission_audit_entry_v2",
        "record_role": "external_training_admission_audit_only",
        "local_decision_participation": False,
        "local_decision_authority": "NONE",
        "sealed_for_training": False,
        "external_decision_recorded": True,
        "final_decision_owner": "external_training_decision_flow",
        "final_image_set_decision_owner": "external_dataset_curation_flow",
        "target_profile": target_profile,
        "target_bucket": target_bucket,
        "layer_tag": candidate_entry.get("layer_tag"),
        "look_key": candidate_entry.get("look_key"),
        "group_key": candidate_entry.get("group_key"),
        "group_source": candidate_entry.get("group_source"),
        "image": candidate_entry.get("image"),
        "record_key": candidate_entry.get("record_key"),
        "rank": candidate_entry.get("rank"),
        "review_bucket": candidate_entry.get("review_bucket"),
        "status": candidate_entry.get("status"),
        "selection_score": candidate_entry.get("selection_score"),
        "winner_reasons": list(candidate_entry.get("winner_reasons") or [])[:6],
        "caution_reasons": list(candidate_entry.get("caution_reasons") or [])[:6],
        "release_gate": gate_summary,
        "admission_advice": batch_advice,
        "batch_preflight": preflight_summary,
        "evidence_completeness": evidence_summary,
        "threshold_hash": str(threshold_hash or "").strip(),
        "anchor_snapshot": anchor_snapshot or {},
        "source_batch": source_batch or {},
        "source_files": source_files or {},
        "external_decision_context": {
            "confirmation_source": "explicit_caller_assertion",
            "local_evidence_is_advisory": True,
            "local_evidence_warnings": evidence_warnings,
        },
        "external_audit": {
            "owner": owner,
            "note": str(manual_note or "").strip(),
            "recorded_at_utc": audit_timestamp,
        },
    }

    candidate_id = str(audit_entry.get("record_key") or audit_entry.get("image") or "").strip()
    existing_index: Optional[int] = None
    for index, entry in enumerate(entries):
        entry_id = str(entry.get("record_key") or entry.get("image") or "").strip()
        if candidate_id and entry_id == candidate_id:
            existing_index = index
            break

    if existing_index is None:
        entries.append(audit_entry)
        action = "added"
    else:
        entries[existing_index] = audit_entry
        action = "updated"

    manifest_payload = {
        "schema_version": "external_admission_audit_manifest_v2",
        "updated_at_utc": _utcnow_iso(),
        "entry_count": len(entries),
        "entries": entries,
        "policy": {
            "manifest_role": "external_training_admission_audit_ledger",
            "local_decision_participation": False,
            "local_decision_authority": "NONE",
            "external_decision_confirmation_required": True,
            "winner_bank_equals_training_admission": False,
            "final_decision_owner": "external_training_decision_flow",
            "final_image_set_decision_owner": "external_dataset_curation_flow",
            "local_release_gate_is_advisory": True,
            "local_evidence_is_recorded_not_enforced": True,
            "does_not_decide": [
                "final training-set admission",
                "final image-set membership",
            ],
        },
    }
    atomic_write_json(manifest_file, manifest_payload)
    return {
        "status": "ok",
        "action": action,
        "manifest_file": str(manifest_file),
        "entry_count": len(entries),
        "record_role": "external_training_admission_audit_only",
        "local_decision_authority": "NONE",
        "recorded_image": audit_entry.get("image"),
        "recorded_record_key": audit_entry.get("record_key"),
        "sealed_for_training": False,
        "legacy_seal_fields_state": "DEPRECATED_NOT_EMITTED",
        "target_bucket": target_bucket,
        "local_evidence_warnings": evidence_warnings,
    }
