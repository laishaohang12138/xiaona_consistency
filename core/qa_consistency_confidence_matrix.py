from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Iterable[Any]) -> Optional[float]:
    numbers: List[float] = []
    for value in values:
        number = _safe_float(value)
        if number is not None:
            numbers.append(number)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _confidence_band(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "UNAVAILABLE"
    if number >= 0.82:
        return "HIGH"
    if number >= 0.72:
        return "MEDIUM"
    if number >= 0.62:
        return "REVIEW"
    return "LOW"


def _dedupe(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _project_scope() -> Dict[str, Any]:
    return {
        "schema_version": "project_scope_v1",
        "role": "screening_and_evidence_only",
        "machine_role": "rank_candidates_explain_risks_route_review_priority_and_package_evidence",
        "training_admission_participation": False,
        "image_set_decision_participation": False,
        "final_training_decision_owner": "external_training_decision_flow",
        "final_image_set_decision_owner": "external_dataset_curation_flow",
        "outputs_are_not": [
            "final training-set admission",
            "final image-set decision",
            "dataset membership decision",
            "training sample seal",
            "frozen identity truth",
            "parameter-fitting data",
        ],
    }


def _rel(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _manifest_traceability(
    status_board: Dict[str, Any],
    lane_family: str,
) -> Tuple[Optional[float], List[str], Dict[str, Optional[float]]]:
    manifests = status_board.get("input_manifests") if isinstance(status_board.get("input_manifests"), dict) else {}
    normalized_lane = _safe_text(lane_family).lower()
    if normalized_lane == "front":
        manifest = manifests.get("input_split_front") if isinstance(manifests.get("input_split_front"), dict) else {}
    elif normalized_lane == "three_quarter":
        manifest = (
            manifests.get("input_split_three_quarter")
            if isinstance(manifests.get("input_split_three_quarter"), dict)
            else {}
        )
    else:
        manifest = {}
    if not manifest and manifests:
        coverage_values = []
        missing_fields: List[str] = []
        merged_coverage: Dict[str, Optional[float]] = {}
        for raw_manifest in manifests.values():
            if not isinstance(raw_manifest, dict):
                continue
            coverage = raw_manifest.get("required_field_coverage")
            if not isinstance(coverage, dict):
                continue
            for field in ["prompt_id", "seed", "anchor_source", "intended_view"]:
                value = _safe_float(coverage.get(field))
                if value is not None:
                    coverage_values.append(value)
                    current = merged_coverage.get(field)
                    merged_coverage[field] = value if current is None else min(float(current), value)
            missing_fields.extend(raw_manifest.get("missing_fields") or [])
        if not coverage_values:
            return None, [], {}
        return _mean(coverage_values), _dedupe(missing_fields), merged_coverage

    coverage = manifest.get("required_field_coverage") if isinstance(manifest.get("required_field_coverage"), dict) else {}
    field_values = {
        field: _round_or_none(coverage.get(field))
        for field in ["prompt_id", "seed", "anchor_source", "intended_view"]
    }
    return _mean(field_values.values()), _dedupe(manifest.get("missing_fields") or []), field_values


def _rank_stability(packet: Dict[str, Any], review_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = packet.get("batch") if isinstance(packet.get("batch"), dict) else {}
    top_gap = _safe_float(batch.get("selection_gap_top2"))
    if top_gap is None:
        groups = (
            packet.get("ranked_review_packet", {}).get("groups")
            if isinstance(packet.get("ranked_review_packet"), dict)
            else []
        )
        if isinstance(groups, list) and groups:
            group = groups[0] if isinstance(groups[0], dict) else {}
            top_gap = _safe_float(group.get("selection_gap_top2"))
    if top_gap is None and len(review_items) >= 2:
        first = _safe_float(review_items[0].get("selection_score"))
        second = _safe_float(review_items[1].get("selection_score"))
        if first is not None and second is not None:
            top_gap = max(0.0, first - second)

    top_score = _safe_float(review_items[0].get("selection_score")) if review_items else None
    cluster_delta = 0.015
    cluster_count = 0
    if top_score is not None:
        for row in review_items:
            score = _safe_float(row.get("selection_score"))
            if score is None:
                continue
            if top_score - score <= cluster_delta:
                cluster_count += 1
    stability_score = None if top_gap is None else _clamp01(top_gap / 0.03)
    if stability_score is None:
        stability_band = "UNAVAILABLE"
    elif stability_score >= 0.70:
        stability_band = "STABLE"
    elif stability_score >= 0.35:
        stability_band = "CLOSE_REVIEW"
    else:
        stability_band = "TIGHT_REVIEW"
    return {
        "selection_gap_top2": _round_or_none(top_gap),
        "stability_score": _round_or_none(stability_score),
        "stability_band": stability_band,
        "top_cluster_delta": cluster_delta,
        "top_cluster_count": cluster_count,
        "manual_review_window": _safe_int(batch.get("manual_review_window") or 3),
        "interpretation": "ranking is review-priority evidence only; close gaps require manual comparison",
    }


def _extract_review_items(review_packet: Dict[str, Any], gpt_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = review_packet.get("items") if isinstance(review_packet.get("items"), list) else []
    rows = [dict(item) for item in items if isinstance(item, dict)]
    if rows:
        rows.sort(key=lambda row: (_safe_int(row.get("rank")) or 999999, _safe_text(row.get("image"))))
        return rows

    seen: set[str] = set()
    fallback: List[Dict[str, Any]] = []
    queue = (
        gpt_packet.get("priority_review_queue")
        if isinstance(gpt_packet.get("priority_review_queue"), dict)
        else {}
    )
    for bucket_items in queue.values():
        if not isinstance(bucket_items, list):
            continue
        for item in bucket_items:
            if not isinstance(item, dict):
                continue
            image = _safe_text(item.get("image"))
            if image in seen:
                continue
            seen.add(image)
            fallback.append(dict(item))
    if not fallback:
        top = gpt_packet.get("top_candidates") if isinstance(gpt_packet.get("top_candidates"), list) else []
        fallback = [dict(item) for item in top if isinstance(item, dict)]
    fallback.sort(key=lambda row: (_safe_int(row.get("rank")) or 999999, _safe_text(row.get("image"))))
    return fallback


def _field(row: Dict[str, Any], *names: str) -> Any:
    for name in names:
        node: Any = row
        ok = True
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                ok = False
                break
            node = node.get(part)
        if ok and node is not None:
            return node
    return None


def _pose_gait_read_signal(read: str) -> Optional[float]:
    mapping = {
        "pose_gait_consistent": 0.85,
        "pose_explained_delta_possible": 0.62,
        "gait_tolerant_topology_margin_review": 0.55,
        "pose_sensitive_noise_possible": 0.50,
        "manual_review_required": 0.40,
        "unexplained_body_drift_risk": 0.20,
    }
    return mapping.get(_safe_text(read), None)


def _lighting_signal(reasons: Sequence[Any]) -> Tuple[float, List[str]]:
    reason_set = {_safe_text(reason) for reason in reasons if _safe_text(reason)}
    if "SKIN_LIGHTING_RISK_HIGH" in reason_set or "SKIN_SAMPLE_RISK_HIGH" in reason_set:
        return 0.20, ["lighting_high_risk_present"]
    if "SKIN_LIGHTING_RISK_WARN" in reason_set or "SKIN_SAMPLE_RISK_WARN" in reason_set:
        return 0.55, ["lighting_warn_risk_present"]
    return 0.85, []


def _row_to_matrix_entry(
    row: Dict[str, Any],
    *,
    rank_state: Dict[str, Any],
    status_board: Dict[str, Any],
) -> Dict[str, Any]:
    breakdown = row.get("review_only_breakdown_v2") if isinstance(row.get("review_only_breakdown_v2"), dict) else {}
    canonical = row.get("canonical_truth_summary") if isinstance(row.get("canonical_truth_summary"), dict) else {}
    face = row.get("face_canonical_summary") if isinstance(row.get("face_canonical_summary"), dict) else {}
    lane = row.get("lane") if isinstance(row.get("lane"), dict) else {}
    truth = row.get("truth_center") if isinstance(row.get("truth_center"), dict) else {}
    clothing = row.get("clothing_invariant") if isinstance(row.get("clothing_invariant"), dict) else {}
    if not truth:
        truth = breakdown
    if not clothing:
        clothing = breakdown

    lane_family = _safe_text(
        _field(row, "lane.observed_lane_family")
        or _field(row, "truth_center.observed_lane_family")
        or breakdown.get("observed_lane_family")
    )
    metadata_signal, missing_metadata, metadata_coverage = _manifest_traceability(status_board, lane_family)
    top_reasons = row.get("top_reasons") if isinstance(row.get("top_reasons"), list) else []
    soft_flags = (
        row.get("soft_flags")
        if isinstance(row.get("soft_flags"), list)
        else row.get("review_only_soft_flags_v2")
        if isinstance(row.get("review_only_soft_flags_v2"), list)
        else []
    )
    lighting_score, lighting_reasons = _lighting_signal(top_reasons)
    pose_read = _safe_text(canonical.get("body_truth_pose_gait_read")) or "unavailable"
    pose_read_signal = _pose_gait_read_signal(pose_read)

    face_identity = _mean(
        [
            truth.get("face_truth_support"),
            face.get("canonical_face_identity_similarity"),
        ]
    )
    head_topology = _mean(
        [
            truth.get("face_topology_support"),
            face.get("canonical_face_topology_similarity"),
            face.get("canonical_face_landmark_similarity"),
            face.get("head_topology_weakest_part_similarity"),
        ]
    )
    body_truth = _mean(
        [
            truth.get("body_truth_support"),
            truth.get("body_topology_support"),
            canonical.get("body_pose_independent_truth_alignment"),
            canonical.get("body_core_measurement_similarity"),
            canonical.get("body_gait_tolerant_topology_similarity"),
        ]
    )
    body_partition = _mean(
        [
            canonical.get("body_topology_partition_mean_similarity"),
            canonical.get("body_topology_weakest_part_similarity"),
        ]
    )
    pose_gait = _mean(
        [
            pose_read_signal,
            canonical.get("body_pose_delta_similarity"),
            canonical.get("body_mesh_fit_confidence"),
        ]
    )
    clothing_signal = _mean(
        [
            clothing.get("clothing_invariant_score"),
            clothing.get("clothing_invariant_confidence"),
            clothing.get("clothfree_identity_alignment"),
            clothing.get("occlusion_adjusted_truth_score"),
            None
            if _safe_float(clothing.get("garment_occlusion_index")) is None
            else 1.0 - float(clothing.get("garment_occlusion_index")),
        ]
    )
    lane_signal = _mean(
        [
            truth.get("lane_membership_confidence"),
            truth.get("angle_tolerance_score"),
            lane.get("view_lane_detail_confidence"),
            lane.get("view_lane_strictness_score"),
        ]
    )

    visual_axes = {
        "face_identity": face_identity,
        "head_topology": head_topology,
        "body_truth": body_truth,
        "body_topology_partition": body_partition,
        "pose_gait_explanation": pose_gait,
        "clothing_independence": clothing_signal,
        "lighting_robustness": lighting_score,
        "lane_pose_trace": lane_signal,
    }
    evidence_axes = dict(visual_axes)
    evidence_axes["metadata_traceability"] = metadata_signal
    visual_score = _mean(visual_axes.values())
    evidence_score = _mean(evidence_axes.values())

    gaps: List[str] = []
    if metadata_signal is not None and metadata_signal < 1.0:
        gaps.append("PROMPT_METADATA_TRACEABILITY_INCOMPLETE")
    if body_partition is None:
        gaps.append("BODY_TOPOLOGY_PARTITION_NOT_IN_CURRENT_PACKET")
    if _safe_float(face.get("head_topology_weakest_part_similarity")) is None:
        gaps.append("HEAD_TOPOLOGY_PARTITION_NOT_IN_CURRENT_PACKET")
    gaps.extend(lighting_reasons)
    if pose_read != "pose_gait_consistent":
        gaps.append(f"POSE_GAIT_READ_{pose_read.upper()}")
    if "GROUP_PASS_QUOTA_EXCEEDED" in {_safe_text(flag) for flag in soft_flags}:
        gaps.append("GROUP_PASS_QUOTA_EXCEEDED")

    score = _safe_float(row.get("selection_score") or row.get("review_only_score_v2"))
    top_score = _safe_float(rank_state.get("top_score"))
    delta_vs_top = None
    if score is not None and top_score is not None:
        delta_vs_top = score - top_score
    in_close_top_cluster = (
        delta_vs_top is not None
        and abs(delta_vs_top) <= float(rank_state.get("top_cluster_delta") or 0.015)
    )
    review_priority = "standard_review_priority"
    if in_close_top_cluster or any(gap.startswith("POSE_GAIT_READ_") for gap in gaps):
        review_priority = "high_manual_review_priority"
    if evidence_score is not None and evidence_score < 0.62:
        review_priority = "diagnostic_watchlist"

    axes = {
        key: {
            "score": _round_or_none(value),
            "band": _confidence_band(value),
        }
        for key, value in evidence_axes.items()
    }

    return {
        "image": _safe_text(row.get("image")),
        "rank": _safe_int(row.get("rank")),
        "review_only_status": _safe_text(row.get("review_only_status_v2") or row.get("review_only_status")),
        "selection_score": _round_or_none(score),
        "review_only_confidence": _round_or_none(
            row.get("review_only_confidence_v2") or row.get("review_only_confidence")
        ),
        "delta_vs_top": _round_or_none(delta_vs_top),
        "inside_close_top_cluster": bool(in_close_top_cluster),
        "observed_lane_family": lane_family,
        "body_yaw_deg": _round_or_none(lane.get("body_yaw_deg")),
        "face_pose_delta_deg": _round_or_none(face.get("pose_delta_deg")),
        "body_truth_pose_gait_read": pose_read,
        "body_topology_weakest_part": _safe_text(canonical.get("body_topology_weakest_part")),
        "head_topology_weakest_part": _safe_text(face.get("head_topology_weakest_part")),
        "consistency_signal_score": _round_or_none(visual_score),
        "evidence_confidence_score": _round_or_none(evidence_score),
        "evidence_confidence_band": _confidence_band(evidence_score),
        "review_priority": review_priority,
        "axes": axes,
        "traceability": {
            "metadata_coverage": metadata_coverage,
            "missing_metadata_fields": missing_metadata,
        },
        "unresolved_evidence_gaps": _dedupe(gaps),
        "top_reasons": _dedupe(top_reasons[:10]),
        "soft_flags": _dedupe(soft_flags),
    }


def _global_blockers(
    invariance_status: Dict[str, Any],
    status_board: Dict[str, Any],
    replay_collection: Dict[str, Any],
) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []
    gates = invariance_status.get("gates") if isinstance(invariance_status.get("gates"), dict) else {}
    for name, raw_gate in gates.items():
        gate = raw_gate if isinstance(raw_gate, dict) else {}
        status = _safe_text(gate.get("status")).upper()
        if status == "PASS":
            continue
        blockers.append(
            {
                "area": name,
                "status": status,
                "reasons": _dedupe(gate.get("reasons") or []),
            }
        )
    manifests = status_board.get("input_manifests") if isinstance(status_board.get("input_manifests"), dict) else {}
    for name, raw_manifest in manifests.items():
        manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
        if bool(manifest.get("required_field_ready")):
            continue
        blockers.append(
            {
                "area": name,
                "status": "WARN",
                "reasons": [f"MISSING_{field.upper()}" for field in manifest.get("missing_fields") or []],
            }
        )
    summary = replay_collection.get("summary") if isinstance(replay_collection.get("summary"), dict) else {}
    status_counts = (
        summary.get("status_counts_by_area")
        if isinstance(summary.get("status_counts_by_area"), dict)
        else {}
    )
    for area, raw_counts in status_counts.items():
        counts = raw_counts if isinstance(raw_counts, dict) else {}
        empty_count = _safe_int(counts.get("empty_needs_collection"))
        if empty_count > 0:
            blockers.append(
                {
                    "area": _safe_text(area),
                    "status": "EVIDENCE_GAP",
                    "reasons": [f"EMPTY_REPLAY_TASKS={empty_count}"],
                }
            )
    return blockers


def _axis_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    axis_names = [
        "face_identity",
        "head_topology",
        "body_truth",
        "body_topology_partition",
        "pose_gait_explanation",
        "clothing_independence",
        "lighting_robustness",
        "lane_pose_trace",
        "metadata_traceability",
    ]
    summary: Dict[str, Any] = {}
    for axis in axis_names:
        values = [
            ((row.get("axes") or {}).get(axis) or {}).get("score")
            for row in rows
            if isinstance(row.get("axes"), dict)
        ]
        numbers = [float(value) for value in values if _safe_float(value) is not None]
        summary[axis] = {
            "mean": _round_or_none(_mean(numbers)),
            "min": _round_or_none(min(numbers) if numbers else None),
            "available_count": len(numbers),
            "band": _confidence_band(_mean(numbers)),
        }
    return summary


def _front_diagnostic_summary(front_sheet: Dict[str, Any]) -> Dict[str, Any]:
    candidates = front_sheet.get("top_candidates") if isinstance(front_sheet.get("top_candidates"), list) else []
    rows: List[Dict[str, Any]] = []
    for raw in candidates[:5]:
        if not isinstance(raw, dict):
            continue
        score = _mean(
            [
                raw.get("face_master_alignment"),
                raw.get("face_topology_support"),
                raw.get("body_truth_support"),
                raw.get("body_topology_support"),
                raw.get("clothing_invariant_confidence"),
            ]
        )
        rows.append(
            {
                "rank": raw.get("rank"),
                "image": raw.get("image"),
                "evidence_confidence_score": _round_or_none(score),
                "evidence_confidence_band": _confidence_band(score),
                "face_master_alignment": _round_or_none(raw.get("face_master_alignment")),
                "body_truth_support": _round_or_none(raw.get("body_truth_support")),
                "body_topology_support": _round_or_none(raw.get("body_topology_support")),
                "clothing_invariant_confidence": _round_or_none(raw.get("clothing_invariant_confidence")),
                "review_risks": raw.get("review_risks") if isinstance(raw.get("review_risks"), list) else [],
            }
        )
    return {
        "available": bool(rows),
        "purpose": "front diagnostics only; mutable review memory candidates, not final image-set decisions",
        "top_candidates": rows,
    }


def build_consistency_confidence_matrix(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    review_packet_file = outputs_dir / "review_packet.json"
    gpt_packet_file = outputs_dir / "gpt_review_packet.json"
    status_board_file = outputs_dir / "review_status_board.json"
    invariance_file = outputs_dir / "review_invariance_status.json"
    replay_plan_file = outputs_dir / "replay_collection_plan.json"
    front_sheet_file = outputs_dir / "front_bootstrap_review_sheet.json"

    review_packet = _load_json(review_packet_file)
    gpt_packet = _load_json(gpt_packet_file)
    status_board = _load_json(status_board_file)
    invariance_status = _load_json(invariance_file)
    replay_collection = _load_json(replay_plan_file)
    front_sheet = _load_json(front_sheet_file)

    review_items = _extract_review_items(review_packet, gpt_packet)
    rank_state = _rank_stability(gpt_packet or review_packet, review_items)
    if review_items:
        rank_state["top_score"] = _safe_float(review_items[0].get("selection_score"))
    rows = [
        _row_to_matrix_entry(row, rank_state=rank_state, status_board=status_board)
        for row in review_items
    ]

    band_counts = Counter(_safe_text(row.get("evidence_confidence_band")) for row in rows)
    priority_counts = Counter(_safe_text(row.get("review_priority")) for row in rows)
    gap_counts: Counter[str] = Counter()
    pose_read_counts: Counter[str] = Counter()
    for row in rows:
        pose_read_counts[_safe_text(row.get("body_truth_pose_gait_read")) or "unavailable"] += 1
        for gap in row.get("unresolved_evidence_gaps") or []:
            gap_counts[_safe_text(gap)] += 1

    axis_summary = _axis_summary(rows)
    weakest_axes = sorted(
        [
            {"axis": axis, "mean": data.get("mean"), "band": data.get("band")}
            for axis, data in axis_summary.items()
            if data.get("mean") is not None
        ],
        key=lambda item: float(item.get("mean") or 0.0),
    )[:5]

    global_blockers = _global_blockers(invariance_status, status_board, replay_collection)
    next_actions = [
        "fill prompt_id and anchor_source before treating same-score differences as stable",
        "collect lighting replay for front and three_quarter before judging light-driven drift",
        "collect OUTER replay before changing clothing invariance gates",
        "refresh review artifacts after heavy topology partition evidence is available",
    ]
    if not any(gap.startswith("BODY_TOPOLOGY_PARTITION") for gap in gap_counts):
        next_actions = [action for action in next_actions if "topology partition" not in action]

    payload = {
        "schema_version": "consistency_confidence_matrix_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Candidate-level consistency confidence for screening evidence, risk routing, "
            "and manual review priority. This is not final dataset admission."
        ),
        "project_scope": _project_scope(),
        "truth_policy": {
            "face_truth_anchor": "A-Core_01_0deg_MASTER.png",
            "body_truth_anchor": "Task-63987060-116-1.png",
            "body_truth_policy": "pose_gait_aware_absolute_116_1",
            "winner_bank_role": "mutable_review_memory_only",
            "parameter_fitting_allowed": False,
        },
        "source_files": {
            "review_packet": _rel(review_packet_file, base_dir),
            "gpt_review_packet": _rel(gpt_packet_file, base_dir),
            "review_status_board": _rel(status_board_file, base_dir),
            "review_invariance_status": _rel(invariance_file, base_dir),
            "replay_collection_plan": _rel(replay_plan_file, base_dir),
            "front_bootstrap_review_sheet": _rel(front_sheet_file, base_dir),
        },
        "batch_confidence": {
            "item_count": len(rows),
            "overall_status": "NEEDS_EVIDENCE_HARDENING" if global_blockers else "EVIDENCE_READY_FOR_REVIEW",
            "evidence_confidence_band_counts": dict(sorted(band_counts.items())),
            "review_priority_counts": dict(sorted(priority_counts.items())),
            "pose_gait_read_counts": dict(sorted(pose_read_counts.items())),
            "top_unresolved_evidence_gaps": [
                {"gap": gap, "count": count}
                for gap, count in gap_counts.most_common(12)
            ],
            "weakest_axes": weakest_axes,
            "axis_summary": axis_summary,
            "ranking_stability": rank_state,
            "global_blockers": global_blockers,
        },
        "front_diagnostic_summary": _front_diagnostic_summary(front_sheet),
        "candidate_matrix": rows,
        "top_review_queue": rows[:12],
        "next_actions": next_actions,
        "explicit_holds": [
            "do not use this matrix as final image-set membership",
            "do not use this matrix as training-set admission",
            "do not freeze winner_bank from this matrix alone",
            "do not fit parameters from current review candidates",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
