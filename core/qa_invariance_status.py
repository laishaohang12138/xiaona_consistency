from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _risk_count(batch: Dict[str, Any], reason: str) -> int:
    risks = batch.get("primary_risks") if isinstance(batch.get("primary_risks"), list) else []
    for row in risks:
        if not isinstance(row, dict):
            continue
        if str(row.get("reason") or "").strip() == reason:
            try:
                return int(row.get("count") or 0)
            except Exception:
                return 0
    return 0


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _top_candidate_values(packet: Dict[str, Any], section: str, key: str) -> List[float]:
    candidates = packet.get("top_candidates") if isinstance(packet.get("top_candidates"), list) else []
    values: List[float] = []
    for candidate in candidates[:3]:
        if not isinstance(candidate, dict):
            continue
        node = candidate.get(section) if isinstance(candidate.get(section), dict) else {}
        value = _safe_float(node.get(key))
        if value is not None:
            values.append(value)
    return values


def _resolve_path(base_dir: Path, raw_path: Any) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_packet(root: Path) -> Dict[str, Any]:
    packet = _load_json(root / "gpt_review_packet.json")
    review_packet = _load_json(root / "review_packet.json")
    batch = packet.get("batch") if isinstance(packet.get("batch"), dict) else {}
    preflight = batch.get("batch_preflight") if isinstance(batch.get("batch_preflight"), dict) else {}
    evidence = batch.get("evidence_completeness") if isinstance(batch.get("evidence_completeness"), dict) else {}
    clothing = batch.get("clothing_invariant_summary") if isinstance(batch.get("clothing_invariant_summary"), dict) else {}
    review_items = review_packet.get("items") if isinstance(review_packet.get("items"), list) else []
    return {
        "root": str(root.resolve()),
        "packet": packet,
        "review_packet": review_packet,
        "review_items": review_items,
        "batch": batch,
        "preflight": preflight,
        "evidence": evidence,
        "clothing": clothing,
    }


def _body_topology_compare_state(base_dir: Path, three_quarter_root: Path) -> Dict[str, Any]:
    compare = _load_json(base_dir / "outputs" / "body_topology_truth_fusion_compare.json")
    if not compare:
        return {}
    truth_fusion_root = _resolve_path(base_dir, compare.get("truth_fusion_snapshot"))
    expected_root = three_quarter_root.resolve() if str(three_quarter_root) else Path("")
    compare_applied = bool(expected_root and truth_fusion_root and truth_fusion_root == expected_root)
    truth_fusion = compare.get("truth_fusion") if isinstance(compare.get("truth_fusion"), dict) else {}
    delta = compare.get("delta") if isinstance(compare.get("delta"), dict) else {}
    interpretation = compare.get("interpretation") if isinstance(compare.get("interpretation"), dict) else {}
    return {
        "compare_available": True,
        "compare_scope": str(compare.get("compare_scope") or "").strip(),
        "compare_applied": compare_applied,
        "truth_fusion_snapshot": str(truth_fusion_root) if truth_fusion_root else "",
        "body_topology_support_mean": _round_or_none(truth_fusion.get("body_topology_support_mean")),
        "body_truth_support_mean": _round_or_none(truth_fusion.get("body_truth_support_mean")),
        "body_core_measurement_similarity_mean": _round_or_none(truth_fusion.get("body_core_measurement_similarity_mean")),
        "body_pose_independent_truth_alignment_mean": _round_or_none(
            truth_fusion.get("body_pose_independent_truth_alignment_mean")
        ),
        "body_topology_support_mean_delta_vs_baseline": _round_or_none(delta.get("body_topology_support_mean")),
        "body_truth_support_mean_delta_vs_baseline": _round_or_none(delta.get("body_truth_support_mean")),
        "resolved_for_three_quarter_review": bool(interpretation.get("body_topology_resolved_for_three_quarter_review")),
        "remaining_primary_blockers": [
            str(item).strip()
            for item in (interpretation.get("remaining_primary_blockers") or [])
            if str(item).strip()
        ],
    }


def _outer_manifest_state(base_dir: Path) -> Dict[str, Any]:
    manifest_file = base_dir / "prompts" / "outer" / "manifest.yaml"
    if not manifest_file.exists():
        return {
            "manifest_available": False,
            "status": "",
            "runtime_active": False,
            "review_only_replay_allowed": False,
        }
    text = manifest_file.read_text(encoding="utf-8", errors="ignore")
    status = ""
    runtime_active = False
    review_only_replay_allowed = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("status:") and not status:
            status = line.split(":", 1)[1].strip()
        if line.startswith("runtime_active:"):
            runtime_active = line.split(":", 1)[1].strip().lower() == "true"
        if line.startswith("review_only_replay_allowed:"):
            review_only_replay_allowed = line.split(":", 1)[1].strip().lower() == "true"
    return {
        "manifest_available": True,
        "manifest_file": str(manifest_file.resolve()),
        "status": status,
        "runtime_active": runtime_active,
        "review_only_replay_allowed": review_only_replay_allowed,
    }


def _lighting_replay_pack_state(base_dir: Path) -> Dict[str, Any]:
    pack_file = base_dir / "outputs" / "lighting_replay_pack.json"
    pack = _load_json(pack_file)
    if not pack:
        return {
            "prepared": False,
            "pack_file": str(pack_file.resolve()),
            "replay_root": str((base_dir / "input_replay" / "lighting").resolve()),
        }
    return {
        "prepared": True,
        "pack_file": str(pack_file.resolve()),
        "replay_root": str(pack.get("replay_root") or ""),
        "lane_count": int(pack.get("lane_count") or 0),
        "variant_dir_count": int(pack.get("variant_dir_count") or 0),
        "total_current_images": int(pack.get("total_current_images") or 0),
    }


def _outer_replay_pack_state(base_dir: Path) -> Dict[str, Any]:
    pack_file = base_dir / "outputs" / "outer_replay_pack.json"
    pack = _load_json(pack_file)
    if not pack:
        return {
            "prepared": False,
            "pack_file": str(pack_file.resolve()),
            "replay_root": str((base_dir / "input_replay" / "outer").resolve()),
        }
    return {
        "prepared": True,
        "pack_file": str(pack_file.resolve()),
        "replay_root": str(pack.get("replay_root") or ""),
        "lane_count": int(pack.get("lane_count") or 0),
        "prompt_dir_count": int(pack.get("prompt_dir_count") or 0),
        "total_current_images": int(pack.get("total_current_images") or 0),
    }


def _gate(status: str, reasons: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": status,
        "metrics": metrics,
        "reasons": reasons,
    }


def _reason_count(items: List[Dict[str, Any]], reason: str) -> int:
    target = _safe_text(reason)
    if not target:
        return 0
    hits = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        reasons = item.get("top_reasons") if isinstance(item.get("top_reasons"), list) else []
        reason_set = {_safe_text(value) for value in reasons if _safe_text(value)}
        if target in reason_set:
            hits += 1
    return hits


def _reason_rate(items: List[Dict[str, Any]], reason: str) -> float:
    return round(_reason_count(items, reason) / max(1, len(items)), 4)


def _lane_angle_center_distance_target_deg(lane_name: str) -> float:
    normalized = _safe_text(lane_name).lower()
    if normalized == "front":
        return 35.0
    if normalized == "three_quarter":
        return 18.0
    if normalized == "side":
        return 14.0
    if normalized == "back":
        return 12.0
    return 18.0


def _clothing_lane_thresholds(lane_name: str) -> Dict[str, float]:
    normalized = _safe_text(lane_name).lower()
    if normalized == "front":
        return {
            "clothfree_identity_cohesion_min": 0.96,
            "body_under_clothes_continuity_min": 0.62,
            "garment_boundary_stability_min": 0.72,
            "garment_occlusion_index_max": 0.10,
            "surface_evidence_coverage_min": 0.95,
        }
    if normalized == "three_quarter":
        return {
            "clothfree_identity_cohesion_min": 0.96,
            "body_under_clothes_continuity_min": 0.58,
            "garment_boundary_stability_min": 0.70,
            "garment_occlusion_index_max": 0.42,
            "surface_evidence_coverage_min": 0.95,
        }
    if normalized == "side":
        return {
            "clothfree_identity_cohesion_min": 0.95,
            "body_under_clothes_continuity_min": 0.54,
            "garment_boundary_stability_min": 0.66,
            "garment_occlusion_index_max": 0.58,
            "surface_evidence_coverage_min": 0.92,
        }
    if normalized == "back":
        return {
            "clothfree_identity_cohesion_min": 0.95,
            "body_under_clothes_continuity_min": 0.52,
            "garment_boundary_stability_min": 0.64,
            "garment_occlusion_index_max": 0.68,
            "surface_evidence_coverage_min": 0.90,
        }
    return {
        "clothfree_identity_cohesion_min": 0.95,
        "body_under_clothes_continuity_min": 0.56,
        "garment_boundary_stability_min": 0.68,
        "garment_occlusion_index_max": 0.50,
        "surface_evidence_coverage_min": 0.92,
    }


def build_review_invariance_status(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    run_index = _load_json(base_dir / "outputs" / "review_run_index.json")
    recommended = run_index.get("recommended_runs") if isinstance(run_index.get("recommended_runs"), dict) else {}
    front_root = Path(str((recommended.get("front_bootstrap_snapshot") or {}).get("artifact_root") or ""))
    tq_root = Path(str((recommended.get("three_quarter_clean_snapshot") or {}).get("artifact_root") or ""))

    runs = {
        "front": _run_packet(front_root) if str(front_root) else {},
        "three_quarter": _run_packet(tq_root) if str(tq_root) else {},
    }
    valid_runs = [run for run in runs.values() if run and run.get("packet")]

    manifest_ready = all(
        bool((run.get("preflight") or {}).get("manifest_required_field_ready"))
        for run in valid_runs
    ) if valid_runs else False
    evidence_ready = all(
        str((run.get("evidence") or {}).get("status") or "").upper() == "PASS"
        and bool((run.get("evidence") or {}).get("replay_ready"))
        for run in valid_runs
    ) if valid_runs else False

    lane_purity_values = [
        float((run.get("preflight") or {}).get("lane_purity_score") or 0.0)
        for run in valid_runs
    ]
    intent_match_values = [
        float((run.get("preflight") or {}).get("intended_observed_lane_match_share") or 0.0)
        for run in valid_runs
    ]
    angle_distance_values = [
        float((run.get("preflight") or {}).get("observed_lane_center_distance_mean_deg") or 0.0)
        for run in valid_runs
    ]
    angle_distance_by_lane: Dict[str, float] = {}
    angle_target_by_lane: Dict[str, float] = {}
    angle_reasons: List[str] = []
    if len(valid_runs) < 2:
        angle_reasons.append("FRONT_AND_THREE_QUARTER_REPLAY_NOT_BOTH_AVAILABLE")
    if not evidence_ready:
        angle_reasons.append("REPLAY_EVIDENCE_NOT_READY")
    if not manifest_ready:
        angle_reasons.append("PROMPT_INTENT_FIELDS_INCOMPLETE")
    for lane_name, run in runs.items():
        if not run:
            continue
        observed_distance = float((run.get("preflight") or {}).get("observed_lane_center_distance_mean_deg") or 0.0)
        target_distance = _lane_angle_center_distance_target_deg(lane_name)
        angle_distance_by_lane[lane_name] = round(observed_distance, 4)
        angle_target_by_lane[lane_name] = round(target_distance, 4)
        if observed_distance > target_distance:
            angle_reasons.append(f"{lane_name.upper()}_ANGLE_CENTER_DISTANCE_ABOVE_LANE_TARGET")
    angle_status = "PASS" if not angle_reasons else "WARN"

    clothing_conf_values = [
        float((run.get("clothing") or {}).get("clothing_invariant_confidence_mean") or 0.0)
        for run in valid_runs
    ]
    surface_coverage_values = [
        float((run.get("clothing") or {}).get("surface_evidence_coverage") or 0.0)
        for run in valid_runs
    ]
    garment_occlusion_values = [
        float((run.get("clothing") or {}).get("garment_occlusion_index_mean") or 0.0)
        for run in valid_runs
    ]
    clothing_metrics_by_lane: Dict[str, Dict[str, Optional[float]]] = {}
    outer_state = _outer_manifest_state(base_dir)
    outer_replay_allowed = bool(outer_state.get("review_only_replay_allowed"))
    outer_active = bool(outer_state.get("runtime_active"))
    outer_pack_state = _outer_replay_pack_state(base_dir)
    outer_pack_prepared = bool(outer_pack_state.get("prepared"))
    outer_pack_image_count = int(outer_pack_state.get("total_current_images") or 0)
    clothing_reasons: List[str] = []
    for lane_name, run in runs.items():
        if not run:
            continue
        clothing = run.get("clothing") or {}
        thresholds = _clothing_lane_thresholds(lane_name)
        metrics = {
            "batch_clothfree_identity_cohesion": _round_or_none(clothing.get("batch_clothfree_identity_cohesion")),
            "body_under_clothes_continuity": _round_or_none(clothing.get("body_under_clothes_continuity")),
            "garment_boundary_stability": _round_or_none(clothing.get("garment_boundary_stability")),
            "garment_occlusion_index_mean": _round_or_none(clothing.get("garment_occlusion_index_mean")),
            "clothing_invariant_confidence_mean": _round_or_none(clothing.get("clothing_invariant_confidence_mean")),
            "surface_evidence_coverage": _round_or_none(clothing.get("surface_evidence_coverage")),
            "thresholds": {key: _round_or_none(value) for key, value in thresholds.items()},
        }
        clothing_metrics_by_lane[lane_name] = metrics
        clothfree_identity = _safe_float(clothing.get("batch_clothfree_identity_cohesion"))
        body_under_clothes = _safe_float(clothing.get("body_under_clothes_continuity"))
        garment_boundary_stability = _safe_float(clothing.get("garment_boundary_stability"))
        garment_occlusion = _safe_float(clothing.get("garment_occlusion_index_mean"))
        surface_coverage = _safe_float(clothing.get("surface_evidence_coverage"))
        if clothfree_identity is None or clothfree_identity < thresholds["clothfree_identity_cohesion_min"]:
            clothing_reasons.append(f"{lane_name.upper()}_CLOTHFREE_IDENTITY_COHESION_BELOW_TARGET")
        if body_under_clothes is None or body_under_clothes < thresholds["body_under_clothes_continuity_min"]:
            clothing_reasons.append(f"{lane_name.upper()}_BODY_UNDER_CLOTHES_CONTINUITY_BELOW_TARGET")
        if garment_boundary_stability is None or garment_boundary_stability < thresholds["garment_boundary_stability_min"]:
            clothing_reasons.append(f"{lane_name.upper()}_GARMENT_BOUNDARY_STABILITY_BELOW_TARGET")
        if garment_occlusion is None or garment_occlusion > thresholds["garment_occlusion_index_max"]:
            clothing_reasons.append(f"{lane_name.upper()}_GARMENT_OCCLUSION_ABOVE_TARGET")
        if surface_coverage is None or surface_coverage < thresholds["surface_evidence_coverage_min"]:
            clothing_reasons.append(f"{lane_name.upper()}_SURFACE_EVIDENCE_COVERAGE_BELOW_TARGET")
    if not outer_replay_allowed:
        clothing_reasons.append("OUTER_OCCLUSION_REPLAY_NOT_GOVERNED")
    elif not outer_pack_prepared:
        clothing_reasons.append("OUTER_OCCLUSION_REPLAY_PACK_NOT_PREPARED")
    elif outer_pack_image_count <= 0:
        clothing_reasons.append("OUTER_OCCLUSION_REPLAY_NOT_COLLECTED")
    clothing_status = "PASS" if not clothing_reasons else "WARN"

    lighting_warn_ratios: Dict[str, float] = {}
    lighting_high_ratios: Dict[str, float] = {}
    sample_warn_ratios: Dict[str, float] = {}
    sample_high_ratios: Dict[str, float] = {}
    lighting_pack_state = _lighting_replay_pack_state(base_dir)
    for name, run in runs.items():
        if not run:
            continue
        review_items = run.get("review_items") if isinstance(run.get("review_items"), list) else []
        lighting_warn_ratios[name] = _reason_rate(review_items, "SKIN_LIGHTING_RISK_WARN")
        lighting_high_ratios[name] = _reason_rate(review_items, "SKIN_LIGHTING_RISK_HIGH")
        sample_warn_ratios[name] = _reason_rate(review_items, "SKIN_SAMPLE_RISK_WARN")
        sample_high_ratios[name] = _reason_rate(review_items, "SKIN_SAMPLE_RISK_HIGH")
    lighting_reasons: List[str] = []
    if max(lighting_high_ratios.values() or [0.0]) > 0.05:
        lighting_reasons.append("SKIN_LIGHTING_RISK_HIGH_PRESENT")
    if max(lighting_warn_ratios.values() or [0.0]) > 0.35:
        lighting_reasons.append("SKIN_LIGHTING_RISK_WARN_RATE_HIGH")
    if max(sample_high_ratios.values() or [0.0]) > 0.08:
        lighting_reasons.append("SKIN_SAMPLE_RISK_HIGH_PRESENT")
    if "SKIN_LIGHTING_RISK_HIGH_PRESENT" in lighting_reasons or "SKIN_SAMPLE_RISK_HIGH_PRESENT" in lighting_reasons:
        lighting_status = "FAIL"
    elif lighting_reasons:
        lighting_status = "WARN"
    else:
        lighting_status = "PASS"

    face_topology_values: List[float] = []
    body_topology_values: List[float] = []
    lane_face_topology_means: Dict[str, Optional[float]] = {}
    lane_body_topology_means: Dict[str, Optional[float]] = {}
    lane_face_topology_counts: Dict[str, int] = {}
    lane_body_topology_counts: Dict[str, int] = {}
    for lane_name, run in runs.items():
        if not run:
            continue
        packet = run.get("packet") or {}
        face_values = _top_candidate_values(packet, "truth_center", "face_topology_support")
        body_values = _top_candidate_values(packet, "truth_center", "body_topology_support")
        face_topology_values.extend(face_values)
        body_topology_values.extend(body_values)
        lane_face_topology_means[lane_name] = _round_or_none(_mean(face_values))
        lane_body_topology_means[lane_name] = _round_or_none(_mean(body_values))
        lane_face_topology_counts[lane_name] = len(face_values)
        lane_body_topology_counts[lane_name] = len(body_values)
    topology_compare = _body_topology_compare_state(base_dir, tq_root)
    topology_reasons: List[str] = []
    front_face_topology_mean = _safe_float(lane_face_topology_means.get("front"))
    three_quarter_face_topology_mean = _safe_float(lane_face_topology_means.get("three_quarter"))
    front_body_topology_mean = _safe_float(lane_body_topology_means.get("front"))
    three_quarter_body_topology_mean = _safe_float(lane_body_topology_means.get("three_quarter"))
    if front_face_topology_mean is None or front_face_topology_mean < 0.995:
        topology_reasons.append("FRONT_FACE_TOPOLOGY_TOP3_MEAN_BELOW_TARGET")
    if three_quarter_face_topology_mean is None or three_quarter_face_topology_mean < 0.99:
        topology_reasons.append("THREE_QUARTER_FACE_TOPOLOGY_TOP3_MEAN_BELOW_TARGET")
    if front_body_topology_mean is None or front_body_topology_mean < 0.68:
        topology_reasons.append("FRONT_BODY_TOPOLOGY_TOP3_MEAN_BELOW_TARGET")
    if three_quarter_body_topology_mean is None or three_quarter_body_topology_mean < 0.72:
        topology_reasons.append("THREE_QUARTER_BODY_TOPOLOGY_TOP3_MEAN_BELOW_TARGET")
    topology_status = "PASS" if not topology_reasons else "WARN"

    gates = {
        "angle_invariance": _gate(
            angle_status,
            angle_reasons,
            {
                "clean_run_count": len(valid_runs),
                "manifest_ready": manifest_ready,
                "evidence_ready": evidence_ready,
                "lane_purity_mean": _round_or_none(_mean(lane_purity_values)),
                "intended_observed_match_mean": _round_or_none(_mean(intent_match_values)),
                "observed_lane_center_distance_max_deg": _round_or_none(max(angle_distance_values or [0.0])),
                "observed_lane_center_distance_deg_by_lane": angle_distance_by_lane,
                "observed_lane_center_distance_target_deg_by_lane": angle_target_by_lane,
            },
        ),
        "clothing_invariance": _gate(
            clothing_status,
            clothing_reasons,
            {
                "clothing_invariant_confidence_min": _round_or_none(min(clothing_conf_values or [0.0])),
                "surface_evidence_coverage_min": _round_or_none(min(surface_coverage_values or [0.0])),
                "garment_occlusion_index_max": _round_or_none(max(garment_occlusion_values or [0.0])),
                "clothing_metrics_by_lane": clothing_metrics_by_lane,
                "outer_runtime_pack_active": outer_active,
                "outer_review_only_replay_allowed": outer_replay_allowed,
                "outer_manifest_state": outer_state,
                "outer_replay_pack": outer_pack_state,
            },
        ),
        "lighting_invariance": _gate(
            lighting_status,
            lighting_reasons,
            {
                "skin_lighting_risk_warn_rate": lighting_warn_ratios,
                "skin_lighting_risk_high_rate": lighting_high_ratios,
                "skin_sample_risk_warn_rate": sample_warn_ratios,
                "skin_sample_risk_high_rate": sample_high_ratios,
                "lighting_replay_pack": lighting_pack_state,
            },
        ),
        "topology_consistency": _gate(
            topology_status,
            topology_reasons,
            {
                "face_topology_top3_mean": _round_or_none(_mean(face_topology_values)),
                "body_topology_top3_mean": _round_or_none(_mean(body_topology_values)),
                "face_topology_sample_count": len(face_topology_values),
                "body_topology_sample_count": len(body_topology_values),
                "face_topology_top3_mean_by_lane": lane_face_topology_means,
                "body_topology_top3_mean_by_lane": lane_body_topology_means,
                "face_topology_sample_count_by_lane": lane_face_topology_counts,
                "body_topology_sample_count_by_lane": lane_body_topology_counts,
                "three_quarter_truth_fusion_compare": topology_compare,
            },
        ),
    }

    overall_ready = all(str(gate.get("status") or "") == "PASS" for gate in gates.values())
    next_actions: List[str] = []
    if not manifest_ready:
        next_actions.append("complete prompt_id / seed-or-unavailable / anchor_source in split input manifests")
        next_actions.append("rerun clean front and three_quarter replay after manifest fields are completed")
    if not outer_replay_allowed:
        next_actions.append("turn OUTER raw shortlist into a governed inactive runtime pack")
    elif not outer_pack_prepared:
        next_actions.append("prepare an OUTER review-only occlusion replay pack before changing clothing gates")
    elif outer_pack_image_count <= 0:
        next_actions.append("collect OUTER occlusion replay images into input_replay/outer before changing clothing gates")
    else:
        next_actions.append("run OUTER review-only occlusion replay before activating any OUTER runtime pack")
    if str(gates["lighting_invariance"]["status"]) != "PASS":
        next_actions.append("run or collect lighting-variant replay batches before judging identity drift under exposure changes")
    if str(gates["topology_consistency"]["status"]) != "PASS":
        next_actions.append("tighten body topology support or add replay cases before winner_bank freezing")
    elif bool((topology_compare or {}).get("resolved_for_three_quarter_review")):
        next_actions.append("keep the body truth-fusion chain on three_quarter and validate the same topology gain on side/back before promotion")
    if str(gates["lighting_invariance"]["status"]) != "PASS":
        if bool(lighting_pack_state.get("prepared")):
            next_actions.append("collect controlled front / three_quarter lighting variants into input_replay/lighting before adjusting lighting gates")
        else:
            next_actions.append("prepare a controlled lighting replay pack before adjusting lighting gates")

    payload = {
        "schema_version": "review_invariance_status_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "READY" if overall_ready else "NOT_READY",
        "winner_bank_bootstrap_allowed": bool(overall_ready),
        "winner_bank_freeze_allowed": bool(overall_ready),
        "winner_bank_mutable_memory_allowed": True,
        "parameter_fitting_allowed": False,
        "gates": gates,
        "next_actions": next_actions,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
