from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


_STATUS_RANK = {
    "FAIL": 0,
    "WARN": 1,
    "PASS": 2,
}


def resolve_target_bucket(profile_name: Optional[str]) -> str:
    normalized = str(profile_name or "").strip()
    mapping = {
        "body_gold_fullbody": "BODY_GOLD.front_core",
        "body_gold_front_cal": "BODY_GOLD.front_core",
        "body_gold_threequarter_review": "BODY_GOLD.three_quarter_review",
        "body_gold_side90_shadow": "BODY_GOLD.side90_shadow",
        "body_gold_back180_shadow": "BODY_GOLD.back180_shadow",
        "bridge_simple_outfit": "BRIDGE.simple_outfit",
        "full_body_outfit": "BRIDGE.review",
        "upper_body_product": "UPPER_BODY_PRODUCT",
        "identity_lock": "IDENTITY_LOCK",
        "lora_dataset": "IDENTITY_FILTER_ONLY",
    }
    return mapping.get(normalized, normalized or "UNSPECIFIED")


def _dominant_lane_family(lane_detail_counts: Dict[str, int]) -> Optional[str]:
    if not lane_detail_counts:
        return None
    row = sorted(lane_detail_counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]
    if row.startswith("strict_side_90") or row.startswith("side_like"):
        return "side"
    if row in {"strict_back_180", "back_like"}:
        return "back"
    return row


def _release_gate_summary(batch_summary: Dict[str, Any], target_bucket: str) -> Dict[str, Any]:
    raw_gate = batch_summary.get("release_gate") or {}
    if not isinstance(raw_gate, dict):
        raw_gate = {}
    return {
        "target_bucket": target_bucket,
        "schema_version": str(raw_gate.get("schema_version") or "").strip(),
        "release_state": str(raw_gate.get("release_state") or "review").strip() or "review",
        "machine_status_ceiling": str(raw_gate.get("machine_status_ceiling") or "WARN").strip().upper() or "WARN",
        "training_admission_allowed": bool(raw_gate.get("training_admission_allowed")),
        "manual_training_admission_required": bool(raw_gate.get("manual_training_admission_required", True)),
        "optuna_fit_allowed": bool(raw_gate.get("optuna_fit_allowed")),
        "requires_frozen_benchmark": bool(raw_gate.get("requires_frozen_benchmark")),
        "requires_curated_winner_bank": bool(raw_gate.get("requires_curated_winner_bank")),
        "required_lane_families": list(raw_gate.get("required_lane_families") or []),
        "notes": str(raw_gate.get("notes") or "").strip(),
    }


def _status_within_ceiling(status: Any, ceiling: Any) -> bool:
    ceiling_rank = _STATUS_RANK.get(str(ceiling or "").strip().upper())
    if ceiling_rank is None:
        return True
    status_rank = _STATUS_RANK.get(str(status or "").strip().upper())
    if status_rank is None:
        return False
    return status_rank >= ceiling_rank


def build_batch_admission_advice(
    batch_summary: Dict[str, Any],
    winner_bank_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_profile = batch_summary.get("target_profile")
    target_bucket = resolve_target_bucket(target_profile)
    lane_family = _dominant_lane_family(batch_summary.get("lane_detail_counts") or {})
    batch_gate = batch_summary.get("batch_gate") or {}
    identity_summary = batch_summary.get("identity_summary") or {}
    geometry_summary = batch_summary.get("geometry_summary") or {}
    engine_status = batch_summary.get("engine_status") or {}
    release_gate = _release_gate_summary(batch_summary, target_bucket)
    training_governance = batch_summary.get("training_admission_governance") or {}
    batch_preflight = batch_summary.get("batch_preflight") or {}
    evidence_completeness = batch_summary.get("evidence_completeness") or {}

    blockers: List[str] = []
    supports: List[str] = []
    if bool(engine_status.get("fatal")):
        blockers.append("ENGINE_FATAL_UNAVAILABLE")
    if batch_gate.get("enabled") and batch_gate.get("status") not in {None, "pass", "disabled"}:
        blockers.extend(list(batch_gate.get("reasons") or []))
    preflight_status = str(batch_preflight.get("status") or "").strip().upper()
    if preflight_status == "FAIL":
        blockers.append("BATCH_PREFLIGHT_FAILED")
    elif preflight_status == "WARN":
        blockers.append("BATCH_PREFLIGHT_WARN")
    if lane_family in {"side", "back"} and target_bucket == "BODY_GOLD.front_core":
        blockers.append("NON_FRONT_BATCH_FOR_FRONT_CORE")
    if release_gate.get("release_state") in {"shadow", "review", "filter_only"}:
        blockers.append(f"RELEASE_GATE_{str(release_gate.get('release_state') or '').upper()}_ONLY")
    if not bool(release_gate.get("training_admission_allowed")):
        blockers.append("RELEASE_GATE_TRAINING_ADMISSION_DENIED")
    required_lane_families = list(release_gate.get("required_lane_families") or [])
    if lane_family and required_lane_families and lane_family not in required_lane_families:
        blockers.append("LANE_FAMILY_OUTSIDE_RELEASE_GATE")
    if isinstance(identity_summary.get("batch_identity_cohesion"), (int, float)) and float(identity_summary["batch_identity_cohesion"]) < 0.86:
        blockers.append("BATCH_FACE_IDENTITY_STILL_WEAK")
    if isinstance(identity_summary.get("batch_hybrid_identity_cohesion"), (int, float)) and float(identity_summary["batch_hybrid_identity_cohesion"]) >= 0.86:
        supports.append("HYBRID_IDENTITY_STABLE")
    if isinstance(identity_summary.get("batch_clothfree_identity_cohesion"), (int, float)) and float(identity_summary["batch_clothfree_identity_cohesion"]) >= 0.90:
        supports.append("CLOTHFREE_IDENTITY_STABLE")
    if isinstance(geometry_summary.get("batch_world3d_cohesion"), (int, float)) and float(geometry_summary["batch_world3d_cohesion"]) >= 0.95:
        supports.append("WORLD3D_STABLE")
    if isinstance(geometry_summary.get("routing_consistency"), (int, float)) and float(geometry_summary["routing_consistency"]) >= 0.95:
        supports.append("ROUTING_CONSISTENT")
    evidence_status = str(evidence_completeness.get("status") or "").strip().upper()
    if evidence_status == "FAIL":
        blockers.append("EVIDENCE_COMPLETENESS_FAILED")
    if bool(release_gate.get("requires_frozen_benchmark")) and not bool(evidence_completeness.get("replay_ready")):
        blockers.append("EVIDENCE_NOT_REPLAY_READY_FOR_FROZEN_BENCHMARK")
    if bool(evidence_completeness.get("replay_ready")):
        supports.append("EVIDENCE_REPLAY_READY")
    if bool(evidence_completeness.get("gpt_review_ready")):
        supports.append("EVIDENCE_GPT_REVIEW_READY")

    winner_report = (winner_bank_status or {}).get("report") or {}
    curated_bank_available = bool(winner_report.get("curated_bank_available"))
    if release_gate.get("requires_curated_winner_bank") and not curated_bank_available:
        blockers.append("CURATED_WINNER_BANK_REQUIRED")
    if curated_bank_available and int(winner_report.get("drift_row_count") or 0) > 0:
        drift_rows = list(winner_report.get("drift_rows") or [])
        if any(len(list(row.get("drift_flags") or [])) > 0 for row in drift_rows):
            blockers.append("WINNER_BANK_DRIFT_PENDING_REVIEW")

    if training_governance.get("manifest_summary"):
        manifest_summary = training_governance.get("manifest_summary") or {}
        if int(manifest_summary.get("entry_count") or 0) > 0:
            supports.append("TRAINING_ADMISSION_MANIFEST_AVAILABLE")

    eligible_for_training_seal = len(blockers) == 0 and bool(release_gate.get("training_admission_allowed"))
    if "ENGINE_FATAL_UNAVAILABLE" in blockers:
        suggested_action = "hold_batch_until_engine_recovered"
    elif "NON_FRONT_BATCH_FOR_FRONT_CORE" in blockers:
        suggested_action = "reroute_to_matching_lane_profile"
    elif "BATCH_PREFLIGHT_FAILED" in blockers:
        suggested_action = "split_batch_before_any_promotion"
    elif any(code.startswith("RELEASE_GATE_") for code in blockers):
        suggested_action = "do_not_seal_training_admission"
    elif "CURATED_WINNER_BANK_REQUIRED" in blockers:
        suggested_action = "prepare_curated_winner_bank_first"
    elif len(blockers) == 0:
        suggested_action = "manual_review_for_training_candidate"
    else:
        suggested_action = "manual_hold_until_review"

    return {
        "system_role": "advisory_only",
        "target_bucket": target_bucket,
        "dominant_lane_family": lane_family,
        "release_gate": release_gate,
        "batch_preflight": batch_preflight,
        "evidence_completeness": evidence_completeness,
        "machine_ceiling": release_gate.get("machine_status_ceiling"),
        "eligible_for_training_seal": eligible_for_training_seal,
        "suggested_action": suggested_action,
        "blockers": blockers[:8],
        "supports": supports[:8],
    }


def build_candidate_admission_advice(
    item_summary: Dict[str, Any],
    batch_admission: Dict[str, Any],
    drift_flags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    lane_detail = str(((item_summary.get("lane") or {}).get("view_lane_detail") or "")).strip()
    lane_family = str((item_summary.get("master_consistency_card") or {}).get("lane_family") or "")
    master_card = item_summary.get("master_consistency_card") or {}
    status = str(item_summary.get("status") or "").strip().upper()
    release_gate = dict(batch_admission.get("release_gate") or {})
    training_admission_allowed = bool(release_gate.get("training_admission_allowed"))
    batch_preflight = batch_admission.get("batch_preflight") or {}
    evidence_completeness = batch_admission.get("evidence_completeness") or {}
    blockers: List[str] = []
    supports: List[str] = []
    if status == "FAIL":
        blockers.append("ITEM_STATUS_FAIL")
    if lane_family in {"side", "back"} and batch_admission.get("target_bucket") == "BODY_GOLD.front_core":
        blockers.append("ITEM_LANE_OUTSIDE_FRONT_CORE")
    if release_gate.get("required_lane_families") and lane_family and lane_family not in set(release_gate.get("required_lane_families") or []):
        blockers.append("ITEM_LANE_OUTSIDE_RELEASE_GATE")
    if release_gate.get("machine_status_ceiling") and not _status_within_ceiling(status, release_gate.get("machine_status_ceiling")):
        blockers.append("ITEM_STATUS_EXCEEDS_RELEASE_GATE_CEILING")
    if str(batch_preflight.get("status") or "").strip().upper() == "FAIL":
        blockers.append("BATCH_PREFLIGHT_FAILED")
    if str(evidence_completeness.get("status") or "").strip().upper() == "FAIL":
        blockers.append("EVIDENCE_COMPLETENESS_FAILED")
    if master_card.get("advisory_status") == "shadow_review_only":
        blockers.append("MASTER_CARD_REVIEW_ONLY")
    if isinstance(master_card.get("hybrid_master_alignment"), (int, float)) and float(master_card["hybrid_master_alignment"]) >= 0.82:
        supports.append("MASTER_HYBRID_STABLE")
    if isinstance(master_card.get("body_master_alignment"), (int, float)) and float(master_card["body_master_alignment"]) >= 0.84:
        supports.append("BODY_MASTER_STABLE")
    face_drift = (master_card.get("face_drift") or {}).get("primary_bottleneck")
    if face_drift and face_drift not in {"stable", ""}:
        blockers.append(f"FACE_DRIFT_{str(face_drift).upper()}")
    if drift_flags:
        blockers.extend(list(drift_flags))
    if lane_detail in {"strict_side_90_left", "strict_side_90_right", "strict_back_180"}:
        supports.append("STRICT_LANE_SAMPLE")
    if bool(evidence_completeness.get("replay_ready")):
        supports.append("EVIDENCE_REPLAY_READY")

    eligible_for_training_seal = bool(batch_admission.get("eligible_for_training_seal")) and training_admission_allowed and len(blockers) == 0
    if "ITEM_STATUS_FAIL" in blockers:
        suggestion = "reject_tail"
    elif not training_admission_allowed:
        suggestion = "not_eligible_for_training_seal"
    elif "BATCH_PREFLIGHT_FAILED" in blockers:
        suggestion = "split_batch_before_candidate_promotion"
    elif "MASTER_CARD_REVIEW_ONLY" in blockers or "ITEM_LANE_OUTSIDE_FRONT_CORE" in blockers or "ITEM_LANE_OUTSIDE_RELEASE_GATE" in blockers:
        suggestion = "shadow_candidate_only"
    elif "ITEM_STATUS_EXCEEDS_RELEASE_GATE_CEILING" in blockers:
        suggestion = "manual_hold_candidate"
    elif len(blockers) == 0:
        suggestion = "manual_shortlist_candidate"
    else:
        suggestion = "manual_hold_candidate"

    return {
        "target_bucket": batch_admission.get("target_bucket"),
        "release_gate": release_gate,
        "batch_preflight": batch_preflight,
        "evidence_completeness": evidence_completeness,
        "suggestion": suggestion,
        "machine_ceiling": release_gate.get("machine_status_ceiling") or "human_confirmation_required",
        "eligible_for_training_seal": eligible_for_training_seal,
        "blockers": blockers[:8],
        "supports": supports[:8],
    }
