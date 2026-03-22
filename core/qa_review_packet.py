from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


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
        "input_count": report_meta.get("input_count"),
        "group_count": shot_selection.get("group_count"),
        "status_counts": _status_counts(items, "status"),
        "module_status_counts": _module_status_counts(items),
        "lane_detail_counts": _lane_detail_counts(items),
        "primary_risks": _top_reason_counts(items),
        "batch_gate": {
            "enabled": batch_gate.get("enabled"),
            "applied": batch_gate.get("applied"),
            "status": batch_gate.get("status"),
            "reasons": list(batch_gate.get("reasons") or []),
            "recommendation": batch_gate.get("recommendation"),
        },
        "selection": {
            "top_ranked_image": primary_group.get("top_ranked_image"),
            "selection_gap_top2": primary_group.get("selection_gap_top2"),
            "manual_review_window": primary_group.get("manual_review_window"),
            "shortlist_size": primary_group.get("shortlist_size"),
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


def _build_ranked_review_packet(shot_selection: Dict[str, Any]) -> Dict[str, Any]:
    groups_out: List[Dict[str, Any]] = []
    for group in shot_selection.get("groups") or []:
        groups_out.append(
            {
                "group_key": group.get("group_key"),
                "group_source": group.get("group_source"),
                "layer_tag": group.get("layer_tag"),
                "look_key": group.get("look_key"),
                "image_count": group.get("image_count"),
                "top_ranked_image": group.get("top_ranked_image"),
                "selection_gap_top2": group.get("selection_gap_top2"),
                "manual_review_window": group.get("manual_review_window"),
                "shortlist_size": group.get("shortlist_size"),
                "review_guidance": list(group.get("review_guidance") or []),
                "batch_reference": dict(group.get("batch_reference") or {}),
                "shortlist": list(group.get("shortlist") or []),
                "pairwise_compare_cards": list(group.get("pairwise_compare_cards") or []),
            }
        )
    return {
        "mode": shot_selection.get("mode"),
        "final_decision_owner": shot_selection.get("final_decision_owner"),
        "target_profile": shot_selection.get("target_profile"),
        "group_count": shot_selection.get("group_count"),
        "groups": groups_out,
    }


def _summarize_item(item: Dict[str, Any], candidate_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    debug = item.get("debug") or {}
    diagnostics = debug.get("collection_diagnostics") or {}
    garment = debug.get("garment_metrics") or {}
    depth_metrics = debug.get("depth_3d_metrics") or {}
    record_key = (item.get("collection") or {}).get("input_relative_path") or item.get("image")
    return {
        "image": item.get("image"),
        "record_key": record_key,
        "status": item.get("status"),
        "task_profile": item.get("task_profile"),
        "rank": (candidate_row or {}).get("rank"),
        "review_bucket": (candidate_row or {}).get("review_bucket"),
        "selection_score": (candidate_row or {}).get("selection_score"),
        "delta_vs_top": (candidate_row or {}).get("delta_vs_top"),
        "lane": {
            "view_lane": debug.get("view_lane"),
            "view_lane_detail": debug.get("view_lane_detail"),
            "view_lane_strictness_score": debug.get("view_lane_strictness_score"),
        },
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
        "outliers": {
            "score": diagnostics.get("outlier_score"),
            "reasons": list(diagnostics.get("outlier_reasons") or []),
        },
        "top_reasons": list(item.get("reasons") or [])[:10],
        "recommendations": list(item.get("recommendations") or [])[:6],
    }


def _build_item_analysis(report_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    shot_selection = report_payload.get("shot_selection") or {}
    candidate_lookup = _candidate_lookup(shot_selection)
    rows: List[Dict[str, Any]] = []
    for item in report_payload.get("items") or []:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        rows.append(_summarize_item(item, candidate_lookup.get(record_key)))
    rows.sort(
        key=lambda row: (
            9999 if row.get("rank") is None else int(row.get("rank")),
            str(row.get("image") or ""),
        )
    )
    return rows


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
    review_packet = {
        "schema_version": "review_packet_v1",
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
            ],
        },
        "source_files": {
            "qa_report": str(report_file),
            "ranked_candidates": str(ranked_candidates_file),
            "winner_bank_candidate": winner_meta.get("candidate_file"),
            "winner_bank_report": winner_meta.get("drift_report_file"),
        },
        "batch_summary": _build_batch_summary(report_payload),
        "ranked_review_packet": _build_ranked_review_packet(report_payload.get("shot_selection") or {}),
        "winner_bank_status": {
            "enabled": winner_meta.get("enabled"),
            "mode": winner_meta.get("mode"),
            "curated_bank_file": winner_meta.get("curated_bank_file"),
            "curated_bank_available": winner_meta.get("curated_bank_available"),
            "curated_entry_count": winner_meta.get("curated_entry_count"),
            "candidate_entry_count": winner_meta.get("candidate_entry_count"),
            "drift_row_count": winner_meta.get("drift_row_count"),
            "report": winner_bank_report,
            "candidate": {
                "entry_count": (winner_bank_candidate or {}).get("entry_count"),
                "target_profile": (winner_bank_candidate or {}).get("target_profile"),
            },
        },
        "items": _build_item_analysis(report_payload),
        "debug": {
            "report_meta": report_payload.get("report_meta"),
            "collection_summary": (report_payload.get("collection_aggregates") or {}).get("summary"),
        },
    }
    review_packet_file = output_dir / "review_packet.json"
    review_packet_file.write_text(json.dumps(review_packet, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_packet
