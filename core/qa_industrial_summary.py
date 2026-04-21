from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return float(sum(rows) / max(1, len(rows)))


def _lane_family_from_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "back" in text:
        return "back"
    if "side" in text or "profile" in text:
        return "side"
    if "three_quarter" in text:
        return "three_quarter"
    if "front" in text:
        return "front"
    return "unknown"


def _extract_lane_family(item: Dict[str, Any]) -> str:
    lane = item.get("lane") if isinstance(item.get("lane"), dict) else {}
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    for candidate in (
        lane.get("view_lane_detail"),
        lane.get("view_lane"),
        debug.get("view_lane_detail"),
        debug.get("view_lane"),
    ):
        family = _lane_family_from_value(candidate)
        if family != "unknown":
            return family
    return "unknown"


def _extract_collection(item: Dict[str, Any]) -> Dict[str, Any]:
    direct = item.get("collection")
    if isinstance(direct, dict):
        return direct
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    payload = debug.get("collection_metadata")
    return payload if isinstance(payload, dict) else {}


def _extract_intended_lane_family(item: Dict[str, Any]) -> str:
    collection = _extract_collection(item)
    for candidate in (
        collection.get("view_expected_family"),
        collection.get("view_expected"),
    ):
        family = _lane_family_from_value(candidate)
        if family != "unknown":
            return family
    return "unknown"


def _extract_prompt_intent_source(item: Dict[str, Any]) -> str:
    collection = _extract_collection(item)
    value = str(collection.get("prompt_intent_metadata_source") or "").strip().lower()
    return value or "none"


def _extract_manifest_entry_present(item: Dict[str, Any]) -> bool:
    collection = _extract_collection(item)
    return bool(collection.get("manifest_entry_present"))


def _has_required_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _extract_prompt_intent_field(item: Dict[str, Any], field_name: str) -> Any:
    collection = _extract_collection(item)
    if field_name == "intended_view":
        return collection.get("view_expected")
    return collection.get(field_name)


def _extract_observed_lane_center_distance(item: Dict[str, Any]) -> Optional[float]:
    breakdown = _extract_breakdown(item)
    return _safe_float(
        breakdown.get("observed_lane_center_distance_deg", breakdown.get("body_angle_delta_deg"))
    )


def _extract_breakdown(item: Dict[str, Any]) -> Dict[str, Any]:
    direct = item.get("review_only_breakdown_v2")
    if isinstance(direct, dict):
        return direct
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    payload = debug.get("review_only_score_v2") if isinstance(debug.get("review_only_score_v2"), dict) else {}
    breakdown = payload.get("breakdown")
    return breakdown if isinstance(breakdown, dict) else {}


def _extract_face_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    direct = item.get("face_canonical_summary")
    if isinstance(direct, dict):
        return direct
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    payload = debug.get("face_pose_canonical")
    if not isinstance(payload, dict):
        payload = debug.get("face_canonical_shadow")
    return payload if isinstance(payload, dict) else {}


def _extract_heavy_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    direct = item.get("canonical_truth_summary")
    if isinstance(direct, dict):
        return direct
    debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
    payload = debug.get("heavy_evidence")
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def _component_available(summary: Dict[str, Any], component_key: str) -> bool:
    providers = summary.get("component_providers")
    if isinstance(providers, list):
        for row in providers:
            if not isinstance(row, dict):
                continue
            if str(row.get("component_key") or "").strip() == component_key and bool(row.get("available")):
                return True
    return False


def _provider_declares_component(heavy_provider: Dict[str, Any], component_key: str) -> bool:
    providers = heavy_provider.get("component_providers")
    if isinstance(providers, list):
        for row in providers:
            if not isinstance(row, dict):
                continue
            if str(row.get("component_key") or "").strip() == component_key:
                return True
    return False


def _provider_expectations(active_heavy_name: str, heavy_provider: Dict[str, Any]) -> Dict[str, bool]:
    normalized = str(active_heavy_name or "").strip().lower()
    expects_body_canonical = (
        "canonical" in normalized
        or "truth_fusion" in normalized
        or _provider_declares_component(heavy_provider, "body_canonical")
    )
    expects_surface_evidence = (
        "surface" in normalized
        or "occlusion" in normalized
        or "truth_fusion" in normalized
        or _provider_declares_component(heavy_provider, "surface_occlusion")
    )
    return {
        "expects_body_canonical": bool(expects_body_canonical),
        "expects_surface_evidence": bool(expects_surface_evidence),
    }


def _coverage_ratio(items: Sequence[Dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    hits = 0
    for item in items:
        try:
            if predicate(item):
                hits += 1
        except Exception:
            continue
    return float(hits / max(1, len(items)))


def build_batch_preflight_summary(
    items: Sequence[Dict[str, Any]],
    report_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = report_meta if isinstance(report_meta, dict) else {}
    release_gate = meta.get("release_gate") if isinstance(meta.get("release_gate"), dict) else {}
    lane_counts: Dict[str, int] = {}
    intended_lane_counts: Dict[str, int] = {}
    prompt_intent_source_counts: Dict[str, int] = {}
    manifest_entry_count = 0
    intended_pairs = 0
    intended_matches = 0
    observed_center_distances: List[float] = []
    for item in items:
        family = _extract_lane_family(item)
        lane_counts[family] = lane_counts.get(family, 0) + 1
        prompt_source = _extract_prompt_intent_source(item)
        prompt_intent_source_counts[prompt_source] = prompt_intent_source_counts.get(prompt_source, 0) + 1
        if _extract_manifest_entry_present(item):
            manifest_entry_count += 1
        intended_family = _extract_intended_lane_family(item)
        if intended_family != "unknown":
            intended_lane_counts[intended_family] = intended_lane_counts.get(intended_family, 0) + 1
            if family != "unknown":
                intended_pairs += 1
                if family == intended_family:
                    intended_matches += 1
        observed_center_distance = _extract_observed_lane_center_distance(item)
        if observed_center_distance is not None:
            observed_center_distances.append(float(observed_center_distance))
    total = int(sum(lane_counts.values()))
    dominant_lane_family = "unknown"
    dominant_lane_count = 0
    if lane_counts:
        dominant_lane_family, dominant_lane_count = sorted(
            lane_counts.items(),
            key=lambda row: (-int(row[1]), str(row[0])),
        )[0]
    dominant_lane_share = float(dominant_lane_count / max(1, total))
    dominant_intended_lane_family = "unknown"
    dominant_intended_lane_count = 0
    if intended_lane_counts:
        dominant_intended_lane_family, dominant_intended_lane_count = sorted(
            intended_lane_counts.items(),
            key=lambda row: (-int(row[1]), str(row[0])),
        )[0]
    intended_lane_share = float(dominant_intended_lane_count / max(1, total))
    intended_lane_coverage = float(sum(intended_lane_counts.values()) / max(1, total))
    manifest_entry_coverage = float(manifest_entry_count / max(1, total))
    prompt_intent_source = "none"
    if prompt_intent_source_counts:
        prompt_intent_source, _ = sorted(
            prompt_intent_source_counts.items(),
            key=lambda row: (-int(row[1]), str(row[0])),
        )[0]
    manifest_required_field_coverage = {
        "prompt_id": _coverage_ratio(items, lambda item: _has_required_text(_extract_prompt_intent_field(item, "prompt_id"))),
        "seed": _coverage_ratio(items, lambda item: _extract_prompt_intent_field(item, "seed") is not None),
        "anchor_source": _coverage_ratio(items, lambda item: _has_required_text(_extract_prompt_intent_field(item, "anchor_source"))),
        "intended_view": _coverage_ratio(items, lambda item: _has_required_text(_extract_prompt_intent_field(item, "intended_view"))),
    }
    manifest_required_field_ready = min(manifest_required_field_coverage.values(), default=0.0) >= 0.80
    intended_observed_match_share = (
        float(intended_matches / max(1, intended_pairs))
        if intended_pairs > 0
        else None
    )
    required_lane_families = [
        str(value).strip().lower()
        for value in (release_gate.get("required_lane_families") or [])
        if str(value).strip()
    ]
    in_gate_count = 0
    unknown_count = 0
    for family, count in lane_counts.items():
        if family == "unknown":
            unknown_count += int(count)
        if not required_lane_families or family in required_lane_families:
            in_gate_count += int(count)
    in_gate_share = float(in_gate_count / max(1, total))
    outside_gate_share = _clamp(1.0 - in_gate_share)
    entropy = 0.0
    for count in lane_counts.values():
        ratio = float(count / max(1, total))
        if ratio > 0.0:
            entropy -= ratio * math.log(ratio, 2)
    max_entropy = math.log(max(2, len([key for key, count in lane_counts.items() if count > 0])), 2)
    entropy_score = 1.0 - _clamp(entropy / max(1e-6, max_entropy)) if max_entropy > 0.0 else 1.0
    lane_purity_score = _clamp(0.64 * dominant_lane_share + 0.24 * in_gate_share + 0.12 * entropy_score)

    release_state = str(release_gate.get("release_state") or "review").strip().lower()
    purity_floor = {
        "primary": 0.88,
        "review": 0.78,
        "shadow": 0.72,
        "filter_only": 0.82,
    }.get(release_state, 0.78)
    reasons: List[str] = []
    if dominant_lane_share < purity_floor:
        reasons.append("DOMINANT_LANE_SHARE_BELOW_PROFILE_FLOOR")
    if outside_gate_share > 0.0:
        reasons.append("LANE_FAMILY_OUTSIDE_RELEASE_GATE_PRESENT")
    if unknown_count > 0:
        reasons.append("UNKNOWN_LANE_PRESENT")
    if entropy_score < 0.55:
        reasons.append("LANE_DISTRIBUTION_TOO_MIXED")
    if intended_lane_coverage < 0.50:
        reasons.append("PROMPT_INTENT_METADATA_MISSING")
    elif not manifest_required_field_ready:
        reasons.append("PROMPT_INTENT_FIELDS_INCOMPLETE")
    if (
        intended_observed_match_share is not None
        and intended_lane_coverage >= 0.50
        and intended_observed_match_share < 0.55
    ):
        reasons.append("PROMPT_INTENT_AND_OBSERVED_LANE_DIVERGE")

    if outside_gate_share >= 0.22 or dominant_lane_share < max(0.50, purity_floor - 0.18):
        status = "FAIL"
        recommended_action = "split_batch_before_next_training_or_benchmark_run"
    elif outside_gate_share >= 0.08 or dominant_lane_share < purity_floor:
        status = "WARN"
        recommended_action = "keep_review_only_and_split_lane_before_promoting"
    else:
        status = "PASS"
        recommended_action = "lane_purity_is_stable_enough_for_current_profile"
    if (
        status == "PASS"
        and intended_observed_match_share is not None
        and intended_lane_coverage >= 0.50
        and intended_observed_match_share < 0.55
    ):
        status = "WARN"
        recommended_action = "retag_or_split_by_observed_lane_before_benchmark_or_training"
    elif (
        recommended_action == "lane_purity_is_stable_enough_for_current_profile"
        and intended_lane_coverage < 0.50
    ):
        recommended_action = "record_prompt_intent_metadata_and_keep_observed_lane_as_governance_source"

    return {
        "schema_version": "industrial_batch_preflight_v1",
        "input_count": total,
        "governance_lane_source": "observed_lane_family",
        "prompt_intent_source": prompt_intent_source,
        "prompt_intent_source_counts": prompt_intent_source_counts,
        "prompt_intent_is_weak_prior": True,
        "manifest_entry_coverage": _round_or_none(manifest_entry_coverage),
        "manifest_required_fields": ["prompt_id", "seed", "anchor_source", "intended_view"],
        "manifest_required_field_coverage": {
            key: _round_or_none(value)
            for key, value in manifest_required_field_coverage.items()
        },
        "manifest_required_field_ready": bool(manifest_required_field_ready),
        "lane_counts": lane_counts,
        "dominant_lane_family": dominant_lane_family,
        "dominant_lane_share": _round_or_none(dominant_lane_share),
        "intended_lane_counts": intended_lane_counts,
        "dominant_intended_lane_family": dominant_intended_lane_family,
        "dominant_intended_lane_share": _round_or_none(intended_lane_share),
        "intended_lane_coverage": _round_or_none(intended_lane_coverage),
        "intent_diagnostics_ready": intended_lane_coverage >= 0.50,
        "intended_observed_lane_match_share": _round_or_none(intended_observed_match_share),
        "observed_lane_center_distance_mean_deg": _round_or_none(_mean_or_none(observed_center_distances)),
        "required_lane_families": required_lane_families,
        "inside_release_gate_share": _round_or_none(in_gate_share),
        "outside_release_gate_share": _round_or_none(outside_gate_share),
        "lane_entropy_score": _round_or_none(entropy_score),
        "lane_purity_score": _round_or_none(lane_purity_score),
        "unknown_lane_count": unknown_count,
        "status": status,
        "recommended_action": recommended_action,
        "split_batch_recommended": status != "PASS",
        "reasons": reasons,
    }


def build_evidence_completeness_summary(
    items: Sequence[Dict[str, Any]],
    report_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = report_meta if isinstance(report_meta, dict) else {}
    heavy_provider = meta.get("heavy_provider_status") if isinstance(meta.get("heavy_provider_status"), dict) else {}
    active_heavy_name = str(
        heavy_provider.get("active_provider_name")
        or heavy_provider.get("provider_name")
        or (((meta.get("provider_policy") or {}) if isinstance(meta.get("provider_policy"), dict) else {}).get("heavy_evidence"))
        or ""
    ).strip()
    expectations = _provider_expectations(active_heavy_name, heavy_provider)
    expects_surface_evidence = bool(expectations.get("expects_surface_evidence"))
    expects_body_canonical = bool(expectations.get("expects_body_canonical"))

    coverage = {
        "review_only_score_coverage": _coverage_ratio(items, lambda item: bool(_extract_breakdown(item))),
        "face_truth_coverage": _coverage_ratio(items, lambda item: _safe_float(_extract_breakdown(item).get("face_truth_support")) is not None),
        "body_truth_coverage": _coverage_ratio(items, lambda item: _safe_float(_extract_breakdown(item).get("body_truth_support")) is not None),
        "face_topology_coverage": _coverage_ratio(items, lambda item: _safe_float(_extract_breakdown(item).get("face_topology_support")) is not None),
        "body_topology_coverage": _coverage_ratio(items, lambda item: _safe_float(_extract_breakdown(item).get("body_topology_support")) is not None),
        "surface_evidence_coverage": _coverage_ratio(items, lambda item: _safe_float(_extract_breakdown(item).get("surface_evidence_support")) is not None),
        "face_canonical_coverage": _coverage_ratio(
            items,
            lambda item: bool((_extract_face_summary(item).get("available")))
            or _safe_float(_extract_face_summary(item).get("canonical_face_topology_similarity")) is not None,
        ),
        "body_canonical_coverage": _coverage_ratio(
            items,
            lambda item: _component_available(_extract_heavy_summary(item), "body_canonical")
            or bool((_extract_heavy_summary(item).get("body_canonical_summary") or {}).get("integration_state"))
            or _safe_float((_extract_heavy_summary(item).get("body_canonical_summary") or {}).get("body_shape_truth_alignment")) is not None,
        ),
        "master_consistency_coverage": _coverage_ratio(
            items,
            lambda item: bool(
                item.get("master_consistency_card")
                if isinstance(item.get("master_consistency_card"), dict)
                else ((item.get("debug") or {}) if isinstance(item.get("debug"), dict) else {}).get("master_consistency_card")
            ),
        ),
    }
    weights = {
        "review_only_score_coverage": 0.16,
        "face_truth_coverage": 0.12,
        "body_truth_coverage": 0.12,
        "face_topology_coverage": 0.10,
        "body_topology_coverage": 0.10,
        "surface_evidence_coverage": 0.10 if expects_surface_evidence else 0.04,
        "face_canonical_coverage": 0.12,
        "body_canonical_coverage": 0.10 if expects_body_canonical else 0.04,
        "master_consistency_coverage": 0.08,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        weighted_sum += float(coverage.get(key, 0.0)) * float(weight)
        total_weight += float(weight)
    completeness_score = float(weighted_sum / max(1e-6, total_weight))

    reasons: List[str] = []
    if coverage["review_only_score_coverage"] < 0.99:
        reasons.append("REVIEW_ONLY_SCORE_INCOMPLETE")
    if coverage["master_consistency_coverage"] < 0.95:
        reasons.append("MASTER_CONSISTENCY_COVERAGE_WEAK")
    if coverage["face_canonical_coverage"] < 0.95:
        reasons.append("FACE_CANONICAL_COVERAGE_WEAK")
    if expects_body_canonical and coverage["body_canonical_coverage"] < 0.95:
        reasons.append("BODY_CANONICAL_COVERAGE_WEAK")
    if expects_surface_evidence and coverage["surface_evidence_coverage"] < 0.90:
        reasons.append("SURFACE_EVIDENCE_COVERAGE_WEAK")

    if completeness_score >= 0.92 and not reasons:
        status = "PASS"
    elif completeness_score >= 0.78:
        status = "WARN"
    else:
        status = "FAIL"

    replay_ready = (
        coverage["review_only_score_coverage"] >= 0.99
        and coverage["face_canonical_coverage"] >= 0.95
        and (not expects_body_canonical or coverage["body_canonical_coverage"] >= 0.95)
        and coverage["master_consistency_coverage"] >= 0.95
        and (not expects_surface_evidence or coverage["surface_evidence_coverage"] >= 0.90)
    )

    return {
        "schema_version": "industrial_evidence_completeness_v1",
        "item_count": len(items),
        "active_heavy_provider": active_heavy_name,
        "expects_body_canonical": expects_body_canonical,
        "expects_surface_evidence": expects_surface_evidence,
        "coverage": {key: _round_or_none(value) for key, value in coverage.items()},
        "completeness_score": _round_or_none(completeness_score),
        "status": status,
        "replay_ready": replay_ready,
        "gpt_review_ready": bool(completeness_score >= 0.80),
        "reasons": reasons,
    }
