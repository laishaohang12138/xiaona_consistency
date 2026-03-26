from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _profile_target_bucket(profile_name: Optional[str]) -> str:
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


def build_batch_admission_advice(
    batch_summary: Dict[str, Any],
    winner_bank_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_profile = batch_summary.get("target_profile")
    target_bucket = _profile_target_bucket(target_profile)
    lane_family = _dominant_lane_family(batch_summary.get("lane_detail_counts") or {})
    batch_gate = batch_summary.get("batch_gate") or {}
    identity_summary = batch_summary.get("identity_summary") or {}
    geometry_summary = batch_summary.get("geometry_summary") or {}
    engine_status = batch_summary.get("engine_status") or {}

    blockers: List[str] = []
    supports: List[str] = []
    if bool(engine_status.get("fatal")):
        blockers.append("ENGINE_FATAL_UNAVAILABLE")
    if batch_gate.get("enabled") and batch_gate.get("status") not in {None, "pass", "disabled"}:
        blockers.extend(list(batch_gate.get("reasons") or []))
    if lane_family in {"side", "back"} and target_bucket == "BODY_GOLD.front_core":
        blockers.append("NON_FRONT_BATCH_FOR_FRONT_CORE")
    if target_bucket.endswith("shadow") or target_bucket.endswith("review"):
        blockers.append("SHADOW_OR_REVIEW_BUCKET_ONLY")
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

    winner_report = (winner_bank_status or {}).get("report") or {}
    if bool(winner_report.get("curated_bank_available")) and int(winner_report.get("drift_row_count") or 0) > 0:
        drift_rows = list(winner_report.get("drift_rows") or [])
        if any(len(list(row.get("drift_flags") or [])) > 0 for row in drift_rows):
            blockers.append("WINNER_BANK_DRIFT_PENDING_REVIEW")

    if "ENGINE_FATAL_UNAVAILABLE" in blockers:
        suggested_action = "hold_batch_until_engine_recovered"
    elif "SHADOW_OR_REVIEW_BUCKET_ONLY" in blockers:
        suggested_action = "shadow_only_manual_review"
    elif "NON_FRONT_BATCH_FOR_FRONT_CORE" in blockers:
        suggested_action = "reroute_to_matching_lane_profile"
    elif len(blockers) == 0:
        suggested_action = "manual_review_for_training_candidate"
    else:
        suggested_action = "manual_hold_until_review"

    return {
        "system_role": "advisory_only",
        "target_bucket": target_bucket,
        "dominant_lane_family": lane_family,
        "machine_ceiling": "never_auto_admit",
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
    status = str(item_summary.get("status") or "")
    blockers: List[str] = []
    supports: List[str] = []
    if status == "FAIL":
        blockers.append("ITEM_STATUS_FAIL")
    if lane_family in {"side", "back"} and batch_admission.get("target_bucket") == "BODY_GOLD.front_core":
        blockers.append("ITEM_LANE_OUTSIDE_FRONT_CORE")
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

    if "ITEM_STATUS_FAIL" in blockers:
        suggestion = "reject_tail"
    elif "MASTER_CARD_REVIEW_ONLY" in blockers or "ITEM_LANE_OUTSIDE_FRONT_CORE" in blockers:
        suggestion = "shadow_candidate_only"
    elif len(blockers) == 0:
        suggestion = "manual_shortlist_candidate"
    else:
        suggestion = "manual_hold_candidate"

    return {
        "target_bucket": batch_admission.get("target_bucket"),
        "suggestion": suggestion,
        "machine_ceiling": "human_confirmation_required",
        "blockers": blockers[:8],
        "supports": supports[:8],
    }
