from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_io import atomic_write_json
from .qa_winner_bank_policy import WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER, winner_bank_bootstrap_policy


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_brief(entry: Dict[str, Any]) -> Dict[str, Any]:
    master = entry.get("master_consistency_card") or {}
    return {
        "rank": entry.get("rank"),
        "image": entry.get("image"),
        "record_key": entry.get("record_key"),
        "status": entry.get("status"),
        "selection_score": _round_or_none(_safe_float(entry.get("selection_score"))),
        "enhanced_selection_score": _round_or_none(_safe_float(entry.get("enhanced_selection_score"))),
        "lane_family": master.get("lane_family"),
        "hybrid_master_alignment": _round_or_none(_safe_float(master.get("hybrid_master_alignment"))),
        "face_master_alignment": _round_or_none(_safe_float(master.get("face_master_alignment"))),
        "body_truth_alignment": _round_or_none(_safe_float(master.get("body_truth_alignment"))),
        "topology_signature_alignment": _round_or_none(_safe_float(master.get("topology_signature_alignment"))),
        "body_topology_alignment": _round_or_none(_safe_float(master.get("body_topology_alignment"))),
        "winner_reasons": list(entry.get("winner_reasons") or [])[:4],
        "caution_reasons": list(entry.get("caution_reasons") or [])[:4],
        "manual_focus": list(master.get("manual_focus") or [])[:4],
    }


def _promotion_blockers(
    winner_bank_report: Dict[str, Any],
    review_packet: Dict[str, Any],
    preflight_payload: Dict[str, Any],
) -> List[str]:
    blockers: List[str] = []
    batch_summary = review_packet.get("batch_summary") or review_packet.get("batch") or {}
    preflight = (
        preflight_payload.get("batch_preflight")
        or batch_summary.get("batch_preflight")
        or {}
    )
    if str(preflight.get("status") or "").upper() == "FAIL":
        blockers.append("BATCH_PREFLIGHT_FAIL")
    admission = batch_summary.get("admission_advice") or {}
    if list(admission.get("blockers") or []):
        blockers.extend(
            [str(item).strip() for item in (admission.get("blockers") or []) if str(item).strip()]
        )
    return blockers


def build_winner_bank_review_packet(
    *,
    candidate_file: Path,
    winner_bank_report_file: Path,
    review_packet_file: Path,
    preflight_file: Path,
    output_file: Path,
) -> Dict[str, Any]:
    candidate_payload = _load_json(candidate_file)
    winner_bank_report = _load_json(winner_bank_report_file)
    review_packet = _load_json(review_packet_file)
    preflight_payload = _load_json(preflight_file)
    batch_summary = review_packet.get("batch_summary") or review_packet.get("batch") or {}
    target_profile = str(batch_summary.get("target_profile") or "").strip() or "body_gold_fullbody"
    artifact_root = output_file.parent.resolve()
    artifact_arg = ""
    if artifact_root.name != "outputs":
        artifact_arg = f' --artifacts-dir "{artifact_root}"'

    entries = list(candidate_payload.get("entries") or [])
    top_candidates = [_candidate_brief(entry) for entry in entries[:3]]
    recommended = top_candidates[0] if top_candidates else {}
    blockers = _promotion_blockers(winner_bank_report, review_packet, preflight_payload)
    bootstrap_policy = winner_bank_bootstrap_policy()
    if str(bootstrap_policy.get("state") or "") == "deferred" and WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER not in blockers:
        blockers.append(WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER)
    promotion_ready = len(blockers) == 0 and bool(recommended)

    payload = {
        "schema_version": "winner_bank_review_packet_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(entries),
        "curated_bank_available": bool(winner_bank_report.get("curated_bank_available")),
        "curated_entry_count": int(winner_bank_report.get("curated_entry_count") or 0),
        "bank_bootstrap_required": not bool(winner_bank_report.get("curated_bank_available")),
        "batch_preflight_status": str(
            (
                preflight_payload.get("batch_preflight")
                or (batch_summary.get("batch_preflight"))
                or {}
            ).get("status")
            or ""
        ).strip(),
        "promotion_ready": promotion_ready,
        "promotion_blockers": blockers,
        "decision_boundary": {
            "packet_role": "winner_bank_review_memory_packet",
            "winner_bank_role": "mutable_review_memory_only",
            "does_not_decide": [
                "final image-set membership",
                "final training-set admission",
                "identity truth",
                "body truth",
            ],
        },
        "winner_bank_bootstrap_policy": bootstrap_policy,
        "recommended_candidate": recommended,
        "top_candidates": top_candidates,
        "suggested_commands": {
            "inspect_review_packet": f"python check_consistency.py --workflow inspect_review_packet{artifact_arg}",
            "winner_bank_status": f"python check_consistency.py --workflow winner_bank_status{artifact_arg}",
            "promote_rank_1": "deferred_by_policy_do_not_run",
            "rerun_shot_review": f"python check_consistency.py --workflow shot_review --profile {target_profile}",
        },
        "manual_next_step": (
            "Do not promote from the current batch until batch_preflight is clean."
            if "BATCH_PREFLIGHT_FAIL" in blockers
            else (
                "Winner bank bootstrap is deferred until review-only invariance and 3D topology consistency mature."
                if WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER in blockers
                else (
                "Review the recommended candidate, but clear the remaining blockers before recording it as mutable winner_bank memory."
                if blockers
                else "Review the recommended candidate and optionally record one candidate into winner_bank.json as mutable review memory."
                )
            )
        ),
    }
    atomic_write_json(output_file, payload)
    return payload
