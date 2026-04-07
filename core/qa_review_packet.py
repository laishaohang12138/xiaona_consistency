from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .qa_admission import build_batch_admission_advice, build_candidate_admission_advice


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _dedupe_keep_order(values: Sequence[str], limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _top_reason_counts(items: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    ignored = {
        "FULL_PASS",
        "FRAMING_OK",
        "FEET_IN_FRAME",
        "FACE_EMBEDDING_READY",
        "FACE_LANDMARKS_READY",
    }
    counts: Dict[str, int] = {}
    for item in items:
        for reason in item.get("reasons") or []:
            if not isinstance(reason, str) or reason in ignored or reason.endswith("_READY"):
                continue
            counts[reason] = counts.get(reason, 0) + 1
    rows = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    return [{"reason": reason, "count": count} for reason, count in rows[:limit]]


def _lane_family_from_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "back" in text:
        return "back"
    if "side" in text:
        return "side"
    if "3q" in text or "quarter" in text:
        return "three_quarter"
    if "front" in text or "frontal" in text:
        return "front"
    return "unknown"


def _dominant_lane_family(items: Sequence[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        debug = item.get("debug") or {}
        lane_text = debug.get("view_lane_detail") or debug.get("view_lane")
        family = _lane_family_from_text(lane_text)
        if family == "unknown":
            continue
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))[0][0]


def _build_lane_risk_focus(items: Sequence[Dict[str, Any]], limit: int = 6) -> Dict[str, Any]:
    dominant_lane = _dominant_lane_family(items)
    suppressed_by_lane = {
        "side": {
            "UPPER_FAIL",
            "FACE_LOW_CONFIDENCE",
            "FACE_LOW_CONF_NEEDS_REVIEW",
            "FACE_TOO_SMALL",
            "FACE_DARKER_THAN_TONE_ANCHOR",
            "FACE_FLIP_CANONICALIZED",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_NO_RELIABLE_SIGNAL",
        },
        "back": {
            "UPPER_FAIL",
            "FACE_LOW_CONFIDENCE",
            "FACE_LOW_CONF_NEEDS_REVIEW",
            "FACE_TOO_SMALL",
            "FACE_DARKER_THAN_TONE_ANCHOR",
            "FACE_FLIP_CANONICALIZED",
            "FACE_SOFTER_THAN_ANCHOR",
            "FACE_NO_RELIABLE_SIGNAL",
        },
        "three_quarter": {
            "UPPER_FAIL",
        },
    }
    review_focus_by_lane = {
        "front": ["脸部身份", "年龄感/表情", "肩颈与上身比例"],
        "three_quarter": ["脸型漂移", "年龄感", "肩颈与体态衔接"],
        "side": ["身材真相", "侧身轮廓", "depth3d/厚度", "腿线与站姿"],
        "back": ["后背轮廓", "肩线与骨盆", "腿轴与重心", "后侧体量"],
        "unknown": ["人工复核主图", "确认 lane 是否正确", "优先看明显漂移项"],
    }
    note_by_lane = {
        "front": "当前批次按 front 解释，脸部身份和上身结构是主信号。",
        "three_quarter": "当前批次按 3Q 解释，脸型漂移和年龄感比纯正脸更重要。",
        "side": "当前批次按 side 解释，脸部弱信号只做否决参考，主看身材真相、轮廓和空间结构。",
        "back": "当前批次按 back 解释，正脸相关 reason 会被降噪，主看后背轮廓、骨盆和腿轴。",
        "unknown": "当前批次 lane 不够稳定，先确认路由再解释风险。",
    }

    suppressed_reasons = suppressed_by_lane.get(dominant_lane, set())
    ignored = {
        "FULL_PASS",
        "FRAMING_OK",
        "FEET_IN_FRAME",
        "FACE_EMBEDDING_READY",
        "FACE_LANDMARKS_READY",
    }
    active_counts: Dict[str, int] = {}
    noise_counts: Dict[str, int] = {}
    for item in items:
        for reason in item.get("reasons") or []:
            if not isinstance(reason, str) or reason in ignored or reason.endswith("_READY"):
                continue
            if reason in suppressed_reasons:
                noise_counts[reason] = noise_counts.get(reason, 0) + 1
                continue
            active_counts[reason] = active_counts.get(reason, 0) + 1

    active_rows = sorted(active_counts.items(), key=lambda row: (-row[1], row[0]))
    if not active_rows:
        for row in _top_reason_counts(items, limit=limit):
            reason = str(row.get("reason") or "").strip()
            if reason:
                active_rows.append((reason, int(row.get("count") or 0)))
    noise_rows = sorted(noise_counts.items(), key=lambda row: (-row[1], row[0]))
    return {
        "dominant_lane_family": dominant_lane,
        "note": note_by_lane.get(dominant_lane, note_by_lane["unknown"]),
        "review_focus": review_focus_by_lane.get(dominant_lane, review_focus_by_lane["unknown"]),
        "primary_risks": [
            {"reason": reason, "count": count}
            for reason, count in active_rows[:limit]
        ],
        "suppressed_noise": [
            {"reason": reason, "count": count}
            for reason, count in noise_rows[: min(4, limit)]
        ],
        "suppressed_reason_codes": sorted(suppressed_reasons),
    }


def _status_counts(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _module_status_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    modules = ("face", "upper", "full")
    result: Dict[str, Dict[str, int]] = {}
    for module_name in modules:
        counts: Dict[str, int] = {}
        for item in items:
            state = str((item.get("module_state") or {}).get(module_name) or "").strip()
            if not state:
                continue
            counts[state] = counts.get(state, 0) + 1
        result[module_name] = dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))
    return result


def _lane_detail_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        debug = item.get("debug") or {}
        lane_detail = str(debug.get("view_lane_detail") or debug.get("view_lane") or "").strip()
        if not lane_detail:
            continue
        counts[lane_detail] = counts.get(lane_detail, 0) + 1
    return dict(sorted(counts.items(), key=lambda row: (-row[1], row[0])))


def _candidate_lookup(shot_selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for group in shot_selection.get("groups") or []:
        for row in group.get("candidates") or []:
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            if record_key:
                lookup[record_key] = row
    return lookup


def _drift_flag_lookup(winner_bank_report: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    report = winner_bank_report or {}
    for row in report.get("drift_rows") or []:
        record_key = str(row.get("record_key") or row.get("image") or "").strip()
        if record_key:
            lookup[record_key] = list(row.get("drift_flags") or [])
    return lookup


def _load_json_if_exists(path_str: Any) -> Optional[Dict[str, Any]]:
    try:
        path = Path(str(path_str))
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        cleaned[key] = value
    return cleaned


def _compact_selection_comparison(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    production_method = str(
        data.get("production_method") or data.get("preferred_method") or ""
    ).strip()
    legacy_retired = bool(data.get("legacy_retired_for_production")) or production_method == "review_only_score_v2"
    comparison_mode = str(data.get("comparison_mode") or "").strip()
    if legacy_retired and comparison_mode == "unlabeled_truth_proxy":
        comparison_mode = "legacy_monitor_only"
    compact = {
        "comparison_mode": comparison_mode,
        "production_method": production_method,
        "legacy_retired_for_production": legacy_retired,
        "legacy_monitor_preferred_method": str(data.get("legacy_monitor_preferred_method") or "").strip(),
        "truth_proxy_coverage": _round_or_none(data.get("truth_proxy_coverage")),
        "decision_reasons": list(data.get("decision_reasons") or [])[:4],
    }
    return _clean_dict(compact)


def _compact_release_gate(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return _clean_dict(
        {
            "target_bucket": str(data.get("target_bucket") or "").strip(),
            "release_state": str(data.get("release_state") or "").strip(),
            "machine_status_ceiling": str(data.get("machine_status_ceiling") or "").strip(),
            "training_admission_allowed": bool(data.get("training_admission_allowed")),
            "required_lane_families": list(data.get("required_lane_families") or []),
            "notes": str(data.get("notes") or "").strip(),
        }
    )


def _compact_admission_advice(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    suggested_action = str(
        data.get("suggested_action") or data.get("suggestion") or ""
    ).strip()
    return _clean_dict(
        {
            "target_bucket": str(data.get("target_bucket") or "").strip(),
            "suggested_action": suggested_action,
            "machine_ceiling": str(data.get("machine_ceiling") or "").strip(),
            "eligible_for_training_seal": data.get("eligible_for_training_seal"),
            "blockers": list(data.get("blockers") or [])[:6],
            "supports": list(data.get("supports") or [])[:4],
        }
    )


def _strip_legacy_review_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(row)
    for key in (
        "selection_score_legacy",
        "legacy_review_only_status",
        "legacy_review_only_confidence",
        "rank_legacy",
        "review_bucket_legacy",
    ):
        cleaned.pop(key, None)
    delta_vs_top = cleaned.get("delta_vs_top")
    if isinstance(delta_vs_top, dict):
        delta_cleaned = dict(delta_vs_top)
        delta_cleaned.pop("selection_score_legacy", None)
        cleaned["delta_vs_top"] = delta_cleaned
    return cleaned


def _compact_heavy_review_summary(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    compact = {
        "provider_name": str(data.get("provider_name") or "").strip(),
        "provider_version": str(data.get("provider_version") or "").strip(),
        "cache_state": str(data.get("cache_state") or "").strip(),
        "coverage": _round_or_none(data.get("coverage")),
        "parser_confidence": _round_or_none(data.get("parser_confidence")),
        "parser_consensus_score": _round_or_none(data.get("parser_consensus_score")),
        "enhanced_selection_score": _round_or_none(data.get("enhanced_selection_score")),
        "rank_in_heavy_review": data.get("rank_in_heavy_review"),
    }
    return _clean_dict(compact)


def _extract_canonical_truth_summary(heavy_evidence: Any) -> Dict[str, Any]:
    bundle = heavy_evidence if isinstance(heavy_evidence, dict) else {}
    metrics = list(bundle.get("metrics") or [])
    summary = dict(bundle.get("summary") or {}) if isinstance(bundle.get("summary"), dict) else {}
    component_rows = list(summary.get("component_providers") or [])

    def _metric_value(metric_name: str) -> Optional[float]:
        for row in metrics:
            if not isinstance(row, dict):
                continue
            if str(row.get("metric_name") or "").strip() != metric_name:
                continue
            value = row.get("metric_value")
            try:
                return float(value)
            except Exception:
                return None
        return None

    body_canonical_component = None
    for row in component_rows:
        if not isinstance(row, dict):
            continue
        provider_name = str(row.get("provider_name") or "").strip()
        component_key = str(row.get("component_key") or "").strip()
        if provider_name == "body_canonical_hmr2" or component_key == "body_canonical":
            body_canonical_component = dict(row)
            break

    available = None
    if isinstance(body_canonical_component, dict):
        available = bool(body_canonical_component.get("available"))
    elif str(bundle.get("provider_name") or "").strip() == "body_canonical_hmr2":
        available = bool(bundle.get("available")) if bundle.get("available") is not None else None
        body_canonical_component = {
            "provider_name": "body_canonical_hmr2",
            "available": available,
            "confidence": bundle.get("confidence"),
            "coverage": bundle.get("coverage"),
            "device": bundle.get("device"),
        }

    return {
        "canonical_truth_available": available,
        "body_canonical_provider_state": body_canonical_component or {},
        "body_shape_truth_alignment": _round_or_none(_metric_value("body_shape_truth_alignment")),
        "body_shape_beta_similarity": _round_or_none(_metric_value("body_shape_beta_similarity")),
        "canonical_measurement_similarity": _round_or_none(_metric_value("canonical_measurement_similarity")),
        "body_pose_delta_similarity": _round_or_none(_metric_value("body_pose_delta_similarity")),
        "body_mesh_fit_confidence": _round_or_none(_metric_value("body_mesh_fit_confidence")),
    }


def _extract_face_canonical_summary(debug: Any) -> Dict[str, Any]:
    payload = debug if isinstance(debug, dict) else {}
    shadow = payload.get("face_canonical_shadow") if isinstance(payload.get("face_canonical_shadow"), dict) else {}
    return {
        "available": bool(shadow.get("available")) if shadow else None,
        "provider_name": str(shadow.get("provider_name") or "") if shadow else "",
        "mode": str(shadow.get("mode") or "") if shadow else "",
        "face_pose_normalization_confidence": _round_or_none(shadow.get("face_pose_normalization_confidence")),
        "visible_face_coverage": _round_or_none(shadow.get("visible_face_coverage")),
        "frontalization_quality": _round_or_none(shadow.get("frontalization_quality")),
        "pose_fit_confidence": _round_or_none(shadow.get("pose_fit_confidence")),
        "canonical_face_landmark_similarity": _round_or_none(shadow.get("canonical_face_landmark_similarity")),
        "canonical_face_identity_similarity": _round_or_none(shadow.get("canonical_face_identity_similarity")),
        "pose_delta_similarity": _round_or_none(shadow.get("pose_delta_similarity")),
        "pose_delta_deg": _round_or_none(shadow.get("pose_delta_deg")),
        "guidance": list(shadow.get("guidance") or [])[:4] if shadow else [],
        "reasons": list(shadow.get("reasons") or [])[:6] if shadow else [],
    }


def _primary_group(shot_selection: Dict[str, Any]) -> Dict[str, Any]:
    groups = list(shot_selection.get("groups") or [])
    if len(groups) == 0:
        return {}
    return groups[0] if isinstance(groups[0], dict) else {}


def _build_batch_summary(report_payload: Dict[str, Any]) -> Dict[str, Any]:
    report_meta = report_payload.get("report_meta") or {}
    collection_aggregates = report_payload.get("collection_aggregates") or {}
    shot_selection = report_payload.get("shot_selection") or {}
    items = list(report_payload.get("items") or [])
    primary_group = _primary_group(shot_selection)
    collection_summary = collection_aggregates.get("summary") or {}
    batch_gate = collection_aggregates.get("batch_gate") or {}
    batch_reference = primary_group.get("batch_reference") or {}

    return {
        "target_profile": report_meta.get("active_profile"),
        "run_status": report_meta.get("run_status"),
        "input_count": report_meta.get("input_count"),
        "engine_status": report_meta.get("engine_status") or {},
        "anchor_truth": report_meta.get("anchor_governance") or {},
        "master_truth_reference": report_meta.get("master_truth_reference") or {},
        "master_truth_artifact_dir": report_meta.get("master_truth_artifact_dir"),
        "artifact_manifest_file": report_meta.get("artifact_manifest_file"),
        "artifact_manifest_summary": report_meta.get("artifact_manifest_summary") or {},
        "heavy_provider_status": report_meta.get("heavy_provider_status") or {},
        "view_classifier_status": report_meta.get("view_classifier_status") or {},
        "face_canonical_status": report_meta.get("face_canonical_status") or {},
        "release_gate": report_meta.get("release_gate") or {},
        "training_admission_governance": report_meta.get("training_admission_governance") or {},
        "heavy_evidence_summary": shot_selection.get("heavy_evidence_summary") or {},
        "canonical_truth_summary": _extract_canonical_truth_summary(shot_selection.get("heavy_evidence_summary") or {}),
        "group_count": shot_selection.get("group_count"),
        "status_counts": _status_counts(items, "status"),
        "module_status_counts": _module_status_counts(items),
        "lane_detail_counts": _lane_detail_counts(items),
        "primary_risks": _top_reason_counts(items),
        "lane_risk_focus": _build_lane_risk_focus(items),
        "batch_gate": {
            "enabled": batch_gate.get("enabled"),
            "applied": batch_gate.get("applied"),
            "status": batch_gate.get("status"),
            "reasons": list(batch_gate.get("reasons") or []),
            "recommendation": batch_gate.get("recommendation"),
        },
        "selection": {
            "selection_method": primary_group.get("selection_method") or shot_selection.get("selection_method"),
            "top_ranked_image": primary_group.get("top_ranked_image"),
            "selection_gap_top2": primary_group.get("selection_gap_top2"),
            "manual_review_window": primary_group.get("manual_review_window"),
            "shortlist_size": primary_group.get("shortlist_size"),
            "selection_comparison": _compact_selection_comparison(
                primary_group.get("selection_comparison") or shot_selection.get("review_only_score_v2_summary") or {}
            ),
        },
        "identity_summary": {
            "identity_continuity": batch_reference.get("identity_continuity"),
            "batch_identity_cohesion": batch_reference.get("batch_identity_cohesion"),
            "batch_clothfree_identity_cohesion": batch_reference.get("batch_clothfree_identity_cohesion"),
            "batch_hybrid_identity_cohesion": batch_reference.get("batch_hybrid_identity_cohesion"),
        },
        "geometry_summary": {
            "body_under_clothes_continuity": batch_reference.get("body_under_clothes_continuity"),
            "batch_3d_cohesion": batch_reference.get("batch_3d_cohesion"),
            "batch_world3d_cohesion": batch_reference.get("batch_world3d_cohesion"),
            "routing_consistency": batch_reference.get("routing_consistency"),
        },
        "garment_summary": {
            "garment_boundary_stability": batch_reference.get("garment_boundary_stability"),
            "groupable_items": collection_summary.get("groupable_items"),
            "look_groups": collection_summary.get("look_groups"),
            "layer_groups": collection_summary.get("layer_groups"),
        },
        "review_guidance": list(primary_group.get("review_guidance") or []),
    }


def _build_ranked_review_packet(
    shot_selection: Dict[str, Any],
    item_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups_out: List[Dict[str, Any]] = []
    for group in shot_selection.get("groups") or []:
        shortlist_rows: List[Dict[str, Any]] = []
        for row in group.get("shortlist") or []:
            enriched = _strip_legacy_review_fields(dict(row))
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            item_row = item_lookup.get(record_key) or {}
            enriched["heavy_review"] = _compact_heavy_review_summary(enriched.get("heavy_review") or {})
            enriched.pop("heavy_evidence", None)
            enriched["canonical_truth_summary"] = _extract_canonical_truth_summary(
                row.get("heavy_evidence") or (item_row.get("canonical_truth_summary") if isinstance(item_row, dict) else {})
            )
            if item_row:
                enriched["master_consistency_card"] = item_row.get("master_consistency_card")
                enriched["admission_advice"] = item_row.get("admission_advice")
                enriched["face_canonical_summary"] = item_row.get("face_canonical_summary")
            shortlist_rows.append(enriched)
        pairwise_rows: List[Dict[str, Any]] = []
        for card in group.get("pairwise_compare_cards") or []:
            enriched_card = dict(card)
            top_row = item_lookup.get(str(card.get("top_image") or "").strip())
            cand_row = item_lookup.get(str(card.get("candidate_image") or "").strip())
            top_master = (top_row or {}).get("master_consistency_card") or {}
            cand_master = (cand_row or {}).get("master_consistency_card") or {}
            if top_row:
                enriched_card["top_master_consistency"] = top_master
            if cand_row:
                enriched_card["candidate_master_consistency"] = cand_master
                enriched_card["candidate_admission_advice"] = cand_row.get("admission_advice")
            enriched_card["combined_manual_focus"] = _dedupe_keep_order(
                list(card.get("manual_focus") or [])
                + list(top_master.get("manual_focus") or [])
                + list(cand_master.get("manual_focus") or []),
                limit=8,
            )
            enriched_card["manual_review_prompts"] = _dedupe_keep_order(
                list(top_master.get("manual_review_prompts") or [])
                + list(cand_master.get("manual_review_prompts") or []),
                limit=6,
            )
            pairwise_rows.append(enriched_card)
        groups_out.append(
            {
                "group_key": group.get("group_key"),
                "group_source": group.get("group_source"),
                "layer_tag": group.get("layer_tag"),
                "look_key": group.get("look_key"),
                "image_count": group.get("image_count"),
                "top_ranked_image": group.get("top_ranked_image"),
                "selection_method": group.get("selection_method") or shot_selection.get("selection_method"),
                "selection_gap_top2": group.get("selection_gap_top2"),
                "manual_review_window": group.get("manual_review_window"),
                "shortlist_size": group.get("shortlist_size"),
                "review_guidance": list(group.get("review_guidance") or []),
                "selection_comparison": _compact_selection_comparison(group.get("selection_comparison") or {}),
                "batch_reference": dict(group.get("batch_reference") or {}),
                "shortlist": shortlist_rows,
                "pairwise_compare_cards": pairwise_rows,
            }
        )
    top_candidates: List[Dict[str, Any]] = []
    if groups_out:
        primary_group = groups_out[0] if isinstance(groups_out[0], dict) else {}
        for row in list(primary_group.get("shortlist") or [])[:3]:
            if isinstance(row, dict):
                top_candidates.append(dict(row))
    return {
        "mode": shot_selection.get("mode"),
        "final_decision_owner": shot_selection.get("final_decision_owner"),
        "target_profile": shot_selection.get("target_profile"),
        "group_count": shot_selection.get("group_count"),
        "top_candidates": top_candidates,
        "groups": groups_out,
    }


def _summarize_item(
    item: Dict[str, Any],
    candidate_row: Optional[Dict[str, Any]],
    batch_admission: Dict[str, Any],
    drift_flags: Optional[Sequence[str]],
) -> Dict[str, Any]:
    debug = item.get("debug") or {}
    diagnostics = debug.get("collection_diagnostics") or {}
    garment = debug.get("garment_metrics") or {}
    depth_metrics = debug.get("depth_3d_metrics") or {}
    master_consistency_card = debug.get("master_consistency_card") or {}
    heavy_evidence = debug.get("heavy_evidence") or {}
    record_key = (item.get("collection") or {}).get("input_relative_path") or item.get("image")
    row = {
        "image": item.get("image"),
        "record_key": record_key,
        "status": item.get("status"),
        "task_profile": item.get("task_profile"),
        "rank": (candidate_row or {}).get("rank"),
        "review_bucket": (candidate_row or {}).get("review_bucket"),
        "selection_method": (candidate_row or {}).get("selection_score_method"),
        "selection_score": (candidate_row or {}).get("selection_score"),
        "review_only_score_v2": (candidate_row or {}).get("review_only_score_v2"),
        "review_only_confidence_v2": (candidate_row or {}).get("review_only_confidence_v2"),
        "review_only_status_v2": (candidate_row or {}).get("review_only_status_v2"),
        "review_only_breakdown_v2": (candidate_row or {}).get("review_only_breakdown_v2"),
        "review_only_hard_vetoes_v2": list((candidate_row or {}).get("review_only_hard_vetoes_v2") or []),
        "review_only_soft_flags_v2": list((candidate_row or {}).get("review_only_soft_flags_v2") or []),
        "review_only_policy_note_v2": (candidate_row or {}).get("review_only_policy_note_v2"),
        "why_not_high_confidence_v2": list((candidate_row or {}).get("why_not_high_confidence_v2") or []),
        "delta_vs_top": (_strip_legacy_review_fields({"delta_vs_top": (candidate_row or {}).get("delta_vs_top")}).get("delta_vs_top")),
        "lane": {
            "view_lane": debug.get("view_lane"),
            "view_lane_detail": debug.get("view_lane_detail"),
            "view_lane_strictness_score": debug.get("view_lane_strictness_score"),
            "shadow_classifier": {
                "provider_name": ((debug.get("view_classifier_shadow") or {}).get("provider_name")),
                "lane": ((debug.get("view_classifier_shadow") or {}).get("lane")),
                "lane_detail": ((debug.get("view_classifier_shadow") or {}).get("lane_detail")),
                "confidence": ((debug.get("view_classifier_shadow") or {}).get("confidence")),
                "decision_margin": ((debug.get("view_classifier_shadow") or {}).get("decision_margin")),
                "disagrees_with_primary": debug.get("view_classifier_shadow_disagrees"),
            },
        },
        "face_canonical_summary": _extract_face_canonical_summary(debug),
        "scores": {
            "face": (item.get("scores") or {}).get("face"),
            "upper": (item.get("scores") or {}).get("upper"),
            "full": (item.get("scores") or {}).get("full"),
            "constitution": (item.get("scores") or {}).get("constitution"),
            "depth_3d": (item.get("scores") or {}).get("depth_3d"),
        },
        "identity_alignment": {
            "batch_face_alignment": diagnostics.get("identity_centroid_similarity"),
            "body_alignment": diagnostics.get("body_identity_centroid_similarity"),
            "depth_alignment": diagnostics.get("depth_identity_centroid_similarity"),
            "world3d_alignment": diagnostics.get("world3d_identity_centroid_similarity"),
            "clothfree_alignment": diagnostics.get("clothfree_identity_alignment"),
            "hybrid_alignment": diagnostics.get("hybrid_identity_alignment"),
        },
        "garment": {
            "clothing_coverage_ratio": garment.get("clothing_coverage_ratio"),
            "upper_cloth_coverage": garment.get("upper_cloth_coverage"),
            "lower_cloth_coverage": garment.get("lower_cloth_coverage"),
            "neckline_openness": garment.get("neckline_openness"),
            "shoulder_exposure_balance": garment.get("shoulder_exposure_balance"),
            "confidence": garment.get("confidence"),
        },
        "depth3d": {
            "depth_3d_score": depth_metrics.get("depth_3d_score"),
            "primary_bottleneck": depth_metrics.get("primary_bottleneck"),
            "reasons": list(depth_metrics.get("reasons") or []),
        },
        "canonical_truth_summary": _extract_canonical_truth_summary(heavy_evidence),
        "outliers": {
            "score": diagnostics.get("outlier_score"),
            "reasons": list(diagnostics.get("outlier_reasons") or []),
        },
        "master_consistency_card": master_consistency_card,
        "review_focus": {
            "manual_focus": list(master_consistency_card.get("manual_focus") or []),
            "manual_review_prompts": list(master_consistency_card.get("manual_review_prompts") or []),
        },
        "top_reasons": list(item.get("reasons") or [])[:10],
        "recommendations": list(item.get("recommendations") or [])[:6],
    }
    row["admission_advice"] = build_candidate_admission_advice(row, batch_admission, drift_flags=drift_flags)
    return row


def _build_item_analysis(
    report_payload: Dict[str, Any],
    batch_admission: Dict[str, Any],
    winner_bank_report: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    shot_selection = report_payload.get("shot_selection") or {}
    candidate_lookup = _candidate_lookup(shot_selection)
    drift_lookup = _drift_flag_lookup(winner_bank_report)
    rows: List[Dict[str, Any]] = []
    for item in report_payload.get("items") or []:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        rows.append(
            _summarize_item(
                item,
                candidate_lookup.get(record_key),
                batch_admission,
                drift_flags=drift_lookup.get(record_key),
            )
        )
    rows.sort(
        key=lambda row: (
            9999 if row.get("rank") is None else int(row.get("rank")),
            str(row.get("image") or ""),
        )
    )
    return rows


def _compact_face_canonical_for_gpt(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return _clean_dict(
        {
            "available": data.get("available"),
            "canonical_face_identity_similarity": _round_or_none(data.get("canonical_face_identity_similarity")),
            "canonical_face_landmark_similarity": _round_or_none(data.get("canonical_face_landmark_similarity")),
            "pose_delta_deg": _round_or_none(data.get("pose_delta_deg")),
            "face_pose_normalization_confidence": _round_or_none(data.get("face_pose_normalization_confidence")),
        }
    )


def _compact_candidate_for_gpt(row: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = row.get("review_only_breakdown_v2") or {}
    lane = row.get("lane") or {}
    truth_center = {
        "face_truth_support": _round_or_none((breakdown or {}).get("face_truth_support")),
        "body_truth_support": _round_or_none((breakdown or {}).get("body_truth_support")),
        "truth_center_score": _round_or_none((breakdown or {}).get("truth_center_score")),
        "support_only_score": _round_or_none((breakdown or {}).get("support_only_score")),
    }
    compact = {
        "image": row.get("image"),
        "rank": row.get("rank"),
        "review_bucket": row.get("review_bucket"),
        "review_only_status": row.get("review_only_status_v2"),
        "selection_score": _round_or_none(row.get("selection_score")),
        "review_only_confidence": _round_or_none(row.get("review_only_confidence_v2")),
        "lane": _clean_dict(
            {
                "view_lane": str((lane or {}).get("view_lane") or "").strip(),
                "view_lane_detail": str((lane or {}).get("view_lane_detail") or "").strip(),
            }
        ),
        "truth_center": _clean_dict(truth_center),
        "canonical_truth_summary": _clean_dict(dict(row.get("canonical_truth_summary") or {})),
        "face_canonical_summary": _compact_face_canonical_for_gpt(row.get("face_canonical_summary") or {}),
        "hard_vetoes": list(row.get("review_only_hard_vetoes_v2") or [])[:6],
        "soft_flags": list(row.get("review_only_soft_flags_v2") or [])[:6],
        "why_not_high_confidence": list(row.get("why_not_high_confidence_v2") or [])[:4],
        "review_focus": list(((row.get("review_focus") or {}).get("manual_focus")) or [])[:4],
        "top_reasons": list(row.get("top_reasons") or [])[:6],
        "admission_advice": _compact_admission_advice(row.get("admission_advice") or {}),
    }
    policy_note = str(row.get("review_only_policy_note_v2") or "").strip()
    if policy_note:
        compact["policy_note"] = policy_note
    return _clean_dict(compact)


def _first_rows_by_status(
    rows: Sequence[Dict[str, Any]],
    *,
    status: str,
    limit: int,
) -> List[Dict[str, Any]]:
    picked = [
        row
        for row in rows
        if str(row.get("review_only_status_v2") or "").strip().upper() == status
    ]
    picked.sort(
        key=lambda row: (
            9999 if row.get("rank") is None else int(row.get("rank")),
            str(row.get("image") or ""),
        )
    )
    return [_compact_candidate_for_gpt(row) for row in picked[:limit]]


def _build_gpt_review_packet(
    review_packet: Dict[str, Any],
    *,
    review_packet_file: Path,
    gpt_review_packet_file: Path,
) -> Dict[str, Any]:
    batch_summary = dict(review_packet.get("batch_summary") or {})
    ranked_review_packet = review_packet.get("ranked_review_packet") or {}
    items = list(review_packet.get("items") or [])
    selection = batch_summary.get("selection") or {}
    gpt_packet = {
        "schema_version": "gpt_review_packet_v1",
        "generated_at_utc": review_packet.get("generated_at_utc"),
        "system_role": review_packet.get("system_role"),
        "final_decision_owner": review_packet.get("final_decision_owner"),
        "analysis_scope": {
            "default_send_to_gpt": ["gpt_review_packet.json"],
            "optional_companion_files": {
                "winner_bank_report.json": "only when analyzing cross-batch drift",
                "training_admission_manifest.json": "only when analyzing sealed training admissions",
            },
            "do_not_send_by_default": [
                "qa_report.json",
                "ranked_candidates.json",
                "winner_bank_candidate.json",
                "outputs/heavy_evidence_cache/**",
            ],
        },
        "source_files": _clean_dict(
            {
                "review_packet": str(review_packet_file),
                "gpt_review_packet": str(gpt_review_packet_file),
                "winner_bank_report": (review_packet.get("source_files") or {}).get("winner_bank_report"),
                "training_admission_manifest": (review_packet.get("source_files") or {}).get("training_admission_manifest"),
            }
        ),
        "batch": {
            "target_profile": batch_summary.get("target_profile"),
            "run_status": batch_summary.get("run_status"),
            "input_count": batch_summary.get("input_count"),
            "selection_method": selection.get("selection_method"),
            "top_ranked_image": selection.get("top_ranked_image"),
            "selection_gap_top2": selection.get("selection_gap_top2"),
            "manual_review_window": selection.get("manual_review_window"),
            "shortlist_size": selection.get("shortlist_size"),
            "selection_monitor": _compact_selection_comparison(selection.get("selection_comparison") or {}),
            "review_only_status_counts": _status_counts(items, "review_only_status_v2"),
            "main_status_counts": batch_summary.get("status_counts") or {},
            "lane_detail_counts": batch_summary.get("lane_detail_counts") or {},
            "primary_risks": list(batch_summary.get("primary_risks") or [])[:6],
            "lane_risk_focus": _clean_dict(dict(batch_summary.get("lane_risk_focus") or {})),
            "release_gate": _compact_release_gate(batch_summary.get("release_gate") or {}),
            "admission_advice": _compact_admission_advice(batch_summary.get("admission_advice") or {}),
        },
        "priority_review_queue": {
            "pass_candidates": _first_rows_by_status(items, status="PASS", limit=12),
            "warn_watchlist": _first_rows_by_status(items, status="WARN", limit=8),
            "fail_watchlist": _first_rows_by_status(items, status="FAIL", limit=8),
        },
        "top_candidates": [
            _compact_candidate_for_gpt(row)
            for row in list(ranked_review_packet.get("top_candidates") or [])[:5]
            if isinstance(row, dict)
        ],
    }
    return gpt_packet


def _build_review_artifacts_index(
    *,
    output_dir: Path,
    review_packet_file: Path,
    gpt_review_packet_file: Path,
    report_file: Path,
    ranked_candidates_file: Path,
    winner_bank_report_file: Any,
    training_admission_manifest_file: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": "review_artifacts_v1",
        "default_send_to_gpt": [
            {
                "file": str(gpt_review_packet_file),
                "purpose": "default compact packet for GPT batch analysis",
            }
        ],
        "human_deep_review": [
            {
                "file": str(review_packet_file),
                "purpose": "rich structured packet for human or deep GPT review",
            }
        ],
        "optional_companion_files": [
            {
                "file": str(winner_bank_report_file or ""),
                "purpose": "attach only for cross-batch drift analysis",
            },
            {
                "file": str(training_admission_manifest_file or ""),
                "purpose": "attach only for training admission or seal analysis",
            },
        ],
        "internal_debug_only": [
            {
                "file": str(report_file),
                "purpose": "full internal replay/debug ledger",
            },
            {
                "file": str(ranked_candidates_file),
                "purpose": "full machine ranking and component details",
            },
            {
                "file": str((output_dir / "winner_bank_candidate.json")),
                "purpose": "machine candidate export, not for default GPT analysis",
            },
            {
                "file": str((output_dir / "heavy_evidence_cache")),
                "purpose": "heavy cache directory, never send by default",
            },
        ],
    }


def build_review_packet(
    report_payload: Dict[str, Any],
    output_dir: Path,
    report_file: Path,
    ranked_candidates_file: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_meta = report_payload.get("winner_bank_governance") or {}
    winner_bank_report = _load_json_if_exists(winner_meta.get("drift_report_file"))
    winner_bank_candidate = _load_json_if_exists(winner_meta.get("candidate_file"))
    batch_summary = _build_batch_summary(report_payload)
    batch_admission = build_batch_admission_advice(batch_summary, {"report": winner_bank_report})
    item_rows = _build_item_analysis(report_payload, batch_admission, winner_bank_report)
    item_lookup = {
        str(row.get("record_key") or row.get("image") or "").strip(): row
        for row in item_rows
        if str(row.get("record_key") or row.get("image") or "").strip()
    }
    review_packet_file = output_dir / "review_packet.json"
    gpt_review_packet_file = output_dir / "gpt_review_packet.json"
    review_packet = {
        "schema_version": "review_packet_v1_7",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_role": "evidence_only",
        "final_decision_owner": "custom_gpt_plus_human",
        "usage_protocol": {
            "machine_role": "provide explainable metrics, shortlist ranking, and drift evidence only",
            "human_role": "custom_gpt_plus_human decides the final winner and training admission",
            "manual_steps": [
                "review batch_summary first",
                "compare top candidates in ranked_review_packet and pairwise cards",
                "confirm the final winner manually",
                "promote the human-approved winner into winner_bank if needed",
                "seal the approved training item into training_admission_manifest if the release gate allows it",
            ],
        },
        "source_files": {
            "qa_report": str(report_file),
            "ranked_candidates": str(ranked_candidates_file),
            "gpt_review_packet": str(gpt_review_packet_file),
            "winner_bank_candidate": winner_meta.get("candidate_file"),
            "winner_bank_report": winner_meta.get("drift_report_file"),
            "training_admission_manifest": ((report_payload.get("report_meta") or {}).get("training_admission_governance") or {}).get("manifest_file"),
        },
        "batch_summary": {
            **batch_summary,
            "admission_advice": batch_admission,
        },
        "ranked_review_packet": _build_ranked_review_packet(report_payload.get("shot_selection") or {}, item_lookup),
        "winner_bank_status": {
            "enabled": winner_meta.get("enabled"),
            "mode": winner_meta.get("mode"),
            "curated_bank_file": winner_meta.get("curated_bank_file"),
            "curated_bank_available": winner_meta.get("curated_bank_available"),
            "curated_entry_count": winner_meta.get("curated_entry_count"),
            "candidate_entry_count": winner_meta.get("candidate_entry_count"),
            "drift_row_count": winner_meta.get("drift_row_count"),
            "top_drift_risks": list((winner_bank_report or {}).get("top_drift_risks") or []),
            "drift_highlights": [
                {
                    "image": row.get("image"),
                    "drift_flags": list(row.get("drift_flags") or []),
                    "drift_severity": row.get("drift_severity"),
                    "manual_focus": list(row.get("manual_focus") or []),
                }
                for row in list((winner_bank_report or {}).get("drift_rows") or [])[:3]
            ],
            "report": winner_bank_report,
            "candidate": {
                "entry_count": (winner_bank_candidate or {}).get("entry_count"),
                "target_profile": (winner_bank_candidate or {}).get("target_profile"),
            },
        },
        "training_admission_status": (batch_summary.get("training_admission_governance") or {}),
        "items": item_rows,
        "debug": {
            "report_meta": report_payload.get("report_meta"),
            "collection_summary": (report_payload.get("collection_aggregates") or {}).get("summary"),
        },
    }
    review_packet_file.write_text(json.dumps(review_packet, indent=2, ensure_ascii=False), encoding="utf-8")
    gpt_review_packet = _build_gpt_review_packet(
        review_packet,
        review_packet_file=review_packet_file,
        gpt_review_packet_file=gpt_review_packet_file,
    )
    gpt_review_packet_file.write_text(json.dumps(gpt_review_packet, indent=2, ensure_ascii=False), encoding="utf-8")
    review_artifacts = _build_review_artifacts_index(
        output_dir=output_dir,
        review_packet_file=review_packet_file,
        gpt_review_packet_file=gpt_review_packet_file,
        report_file=report_file,
        ranked_candidates_file=ranked_candidates_file,
        winner_bank_report_file=winner_meta.get("drift_report_file"),
        training_admission_manifest_file=((report_payload.get("report_meta") or {}).get("training_admission_governance") or {}).get("manifest_file"),
    )
    review_artifacts_file = output_dir / "review_artifacts.json"
    review_artifacts_file.write_text(json.dumps(review_artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_packet
