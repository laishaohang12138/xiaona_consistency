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
        "heavy_provider_status": report_meta.get("heavy_provider_status") or {},
        "view_classifier_status": report_meta.get("view_classifier_status") or {},
        "face_canonical_status": report_meta.get("face_canonical_status") or {},
        "heavy_evidence_summary": shot_selection.get("heavy_evidence_summary") or {},
        "canonical_truth_summary": _extract_canonical_truth_summary(shot_selection.get("heavy_evidence_summary") or {}),
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


def _build_ranked_review_packet(
    shot_selection: Dict[str, Any],
    item_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    groups_out: List[Dict[str, Any]] = []
    for group in shot_selection.get("groups") or []:
        shortlist_rows: List[Dict[str, Any]] = []
        for row in group.get("shortlist") or []:
            enriched = dict(row)
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            item_row = item_lookup.get(record_key) or {}
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
                "selection_gap_top2": group.get("selection_gap_top2"),
                "manual_review_window": group.get("manual_review_window"),
                "shortlist_size": group.get("shortlist_size"),
                "review_guidance": list(group.get("review_guidance") or []),
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
        "selection_score": (candidate_row or {}).get("selection_score"),
        "delta_vs_top": (candidate_row or {}).get("delta_vs_top"),
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
    review_packet = {
        "schema_version": "review_packet_v1_4",
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
        "items": item_rows,
        "debug": {
            "report_meta": report_payload.get("report_meta"),
            "collection_summary": (report_payload.get("collection_aggregates") or {}).get("summary"),
        },
    }
    review_packet_file = output_dir / "review_packet.json"
    review_packet_file.write_text(json.dumps(review_packet, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_packet
