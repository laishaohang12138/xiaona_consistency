from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .qa_heavy_review import normalize_heavy_evidence_bundle


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    numeric = _safe_float(value)
    return None if numeric is None else round(numeric, digits)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    numeric = [_safe_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if len(numeric) == 0:
        return None
    return float(sum(numeric) / len(numeric))


def _weighted_mean(items: Sequence[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        numeric = _safe_float(value)
        if numeric is None:
            continue
        weight_value = float(weight)
        if weight_value <= 0.0:
            continue
        numerator += numeric * weight_value
        denominator += weight_value
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _rank_status(status: Any) -> int:
    normalized = str(status or "").strip().upper()
    if normalized == "PASS":
        return 2
    if normalized == "WARN":
        return 1
    return 0


def _status_for_score(score: Optional[float], *, pass_th: float, warn_th: float) -> str:
    numeric = _safe_float(score)
    if numeric is None:
        return "FAIL"
    if numeric >= pass_th:
        return "PASS"
    if numeric >= warn_th:
        return "WARN"
    return "FAIL"


def _percentile(values: Sequence[Optional[float]], quantile: float) -> Optional[float]:
    numeric = sorted(value for value in (_safe_float(row) for row in values) if value is not None)
    if len(numeric) == 0:
        return None
    if len(numeric) == 1:
        return numeric[0]
    position = _clamp(float(quantile), 0.0, 1.0) * float(len(numeric) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numeric[lower]
    ratio = position - float(lower)
    return float(numeric[lower] * (1.0 - ratio) + numeric[upper] * ratio)


def _correlation(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x_value = _safe_float(raw_x)
        y_value = _safe_float(raw_y)
        if x_value is None or y_value is None:
            continue
        pairs.append((x_value, y_value))
    if len(pairs) < 2:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    if x_mean is None or y_mean is None:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    denominator_y = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if denominator_x <= 0.0 or denominator_y <= 0.0:
        return None
    return float(numerator / (denominator_x * denominator_y))


def _metric_map(bundle: Any) -> Dict[str, Optional[float]]:
    normalized = normalize_heavy_evidence_bundle(bundle)
    metrics: Dict[str, Optional[float]] = {}
    for row in normalized.get("metrics") or []:
        if not isinstance(row, dict):
            continue
        metric_name = str(row.get("metric_name") or "").strip()
        if not metric_name:
            continue
        metrics[metric_name] = _safe_float(row.get("metric_value"))
    summary = normalized.get("summary") or {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            metrics.setdefault(str(key), _safe_float(value))
    return metrics


def _lane_family(item: Dict[str, Any], candidate_row: Dict[str, Any]) -> str:
    debug = item.get("debug") or {}
    master = debug.get("master_consistency_card") or {}
    lane_family = str(master.get("lane_family") or "").strip().lower()
    if lane_family in {"front", "three_quarter", "side", "back"}:
        return lane_family
    view_lane = str(
        (debug.get("view_lane") or debug.get("view_lane_detail") or (candidate_row.get("lane") or {}).get("view_lane") or "")
    ).strip().lower()
    if "three" in view_lane:
        return "three_quarter"
    if "side" in view_lane:
        return "side"
    if "back" in view_lane:
        return "back"
    return "front"


def _lane_thresholds(lane_family: str) -> Dict[str, float]:
    mapping = {
        "front": {"pass_score": 0.80, "warn_score": 0.66, "pass_conf": 0.62, "warn_conf": 0.44},
        "three_quarter": {"pass_score": 0.76, "warn_score": 0.62, "pass_conf": 0.58, "warn_conf": 0.42},
        "side": {"pass_score": 0.72, "warn_score": 0.58, "pass_conf": 0.54, "warn_conf": 0.40},
        "back": {"pass_score": 0.70, "warn_score": 0.56, "pass_conf": 0.52, "warn_conf": 0.38},
    }
    return dict(mapping.get(lane_family, mapping["three_quarter"]))


def _legacy_thresholds(lane_family: str) -> Dict[str, float]:
    mapping = {
        "front": {"pass_score": 0.78, "warn_score": 0.64},
        "three_quarter": {"pass_score": 0.72, "warn_score": 0.60},
        "side": {"pass_score": 0.68, "warn_score": 0.56},
        "back": {"pass_score": 0.66, "warn_score": 0.54},
    }
    return dict(mapping.get(lane_family, mapping["three_quarter"]))


def _legacy_confidence_proxy(item: Dict[str, Any], lane_family: str) -> Optional[float]:
    confidence = item.get("confidence") or {}
    if lane_family == "side":
        return _weighted_mean(
            [
                (confidence.get("full"), 0.28),
                (confidence.get("constitution"), 0.26),
                (confidence.get("depth_3d"), 0.24),
                (confidence.get("upper"), 0.14),
                (confidence.get("face"), 0.08),
            ]
        )
    if lane_family == "three_quarter":
        return _weighted_mean(
            [
                (confidence.get("face"), 0.24),
                (confidence.get("full"), 0.22),
                (confidence.get("constitution"), 0.20),
                (confidence.get("depth_3d"), 0.18),
                (confidence.get("upper"), 0.16),
            ]
        )
    return _weighted_mean(
        [
            (confidence.get("face"), 0.28),
            (confidence.get("full"), 0.24),
            (confidence.get("upper"), 0.18),
            (confidence.get("constitution"), 0.16),
            (confidence.get("depth_3d"), 0.14),
        ]
    )


def _component_deltas(
    top_components: Dict[str, Optional[float]],
    candidate_components: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    rows: Dict[str, Optional[float]] = {}
    for key in sorted(set(top_components.keys()) | set(candidate_components.keys())):
        top_value = _safe_float(top_components.get(key))
        candidate_value = _safe_float(candidate_components.get(key))
        if top_value is None or candidate_value is None:
            rows[key] = None
            continue
        rows[key] = _round_or_none(candidate_value - top_value)
    return rows


def _compute_review_only_v2(
    item: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Any]:
    debug = item.get("debug") or {}
    diagnostics = debug.get("collection_diagnostics") or {}
    master = debug.get("master_consistency_card") or {}
    face_shadow = debug.get("face_canonical_shadow") or {}
    heavy_bundle = candidate_row.get("heavy_evidence") or debug.get("heavy_evidence") or {}
    heavy_metrics = _metric_map(heavy_bundle)

    lane_family = _lane_family(item, candidate_row)
    thresholds = _lane_thresholds(lane_family)

    face_support_score = _weighted_mean(
        [
            (face_shadow.get("canonical_face_identity_similarity"), 0.62),
            (face_shadow.get("canonical_face_landmark_similarity"), 0.23),
            (face_shadow.get("frontalization_quality"), 0.15),
        ]
    )
    face_truth_weights = {
        "front": {"absolute": 0.50, "canonical": 0.32, "landmark": 0.12, "frontal": 0.06},
        "three_quarter": {"absolute": 0.34, "canonical": 0.42, "landmark": 0.16, "frontal": 0.08},
        "side": {"absolute": 0.18, "canonical": 0.48, "landmark": 0.24, "frontal": 0.10},
        "back": {"absolute": 0.00, "canonical": 0.00, "landmark": 0.00, "frontal": 0.00},
    }
    face_weights = face_truth_weights.get(lane_family, face_truth_weights["three_quarter"])
    face_truth_support = _weighted_mean(
        [
            (master.get("face_master_alignment"), face_weights["absolute"]),
            (face_shadow.get("canonical_face_identity_similarity"), face_weights["canonical"]),
            (face_shadow.get("canonical_face_landmark_similarity"), face_weights["landmark"]),
            (face_shadow.get("frontalization_quality"), face_weights["frontal"]),
        ]
    )

    body_truth_support = _weighted_mean(
        [
            (master.get("body_master_alignment"), 0.26),
            (heavy_metrics.get("body_shape_truth_alignment"), 0.28),
            (heavy_metrics.get("canonical_measurement_similarity"), 0.22),
            (master.get("world3d_master_alignment"), 0.14),
            (heavy_metrics.get("body_mesh_fit_confidence"), 0.10),
        ]
    )
    absolute_truth_support = _weighted_mean(
        [
            (face_truth_support, 0.36 if lane_family in {"front", "three_quarter"} else 0.14 if lane_family == "side" else 0.0),
            (body_truth_support, 0.50 if lane_family in {"front", "three_quarter"} else 0.76 if lane_family == "side" else 0.84),
            (master.get("world3d_master_alignment"), 0.14 if lane_family in {"front", "three_quarter"} else 0.10 if lane_family == "side" else 0.16),
        ]
    )
    canonical_invariant_score = _weighted_mean(
        [
            (heavy_metrics.get("body_shape_truth_alignment"), 0.30),
            (heavy_metrics.get("canonical_measurement_similarity"), 0.24),
            (heavy_metrics.get("body_pose_delta_similarity"), 0.08),
            (heavy_metrics.get("body_mesh_fit_confidence"), 0.10),
            (face_truth_support, 0.28 if lane_family in {"front", "three_quarter"} else 0.14 if lane_family == "side" else 0.0),
        ]
    )
    outfit_invariant_score = _weighted_mean(
        [
            (diagnostics.get("clothfree_identity_alignment"), 0.44),
            (heavy_metrics.get("parser_boundary_alignment"), 0.24),
            (heavy_metrics.get("parser_visible_body_alignment"), 0.24),
            (heavy_metrics.get("parser_consensus_score"), 0.08),
        ]
    )
    batch_relative_score = _weighted_mean(
        [
            (diagnostics.get("hybrid_identity_alignment"), 0.44),
            (diagnostics.get("world3d_identity_centroid_similarity"), 0.26),
            (diagnostics.get("body_identity_centroid_similarity"), 0.18),
            (diagnostics.get("clothfree_identity_alignment"), 0.12),
        ]
    )
    lane_validity = _safe_float(master.get("lane_validity"))
    truth_center_weights = {
        "front": {"face": 0.54, "body": 0.46},
        "three_quarter": {"face": 0.42, "body": 0.58},
        "side": {"face": 0.18, "body": 0.82},
        "back": {"face": 0.00, "body": 1.00},
    }
    truth_weights = truth_center_weights.get(lane_family, truth_center_weights["three_quarter"])
    truth_center_score = _weighted_mean(
        [
            (face_truth_support, truth_weights["face"]),
            (body_truth_support, truth_weights["body"]),
        ]
    )

    support_only_score = _weighted_mean(
        [
            (canonical_invariant_score, 0.42),
            (outfit_invariant_score, 0.20),
            (lane_validity, 0.20),
            (batch_relative_score, 0.10),
            (face_support_score, 0.08 if lane_family != "back" else 0.0),
        ]
    )
    review_only_score = _weighted_mean(
        [
            (truth_center_score, 0.78 if lane_family in {"front", "three_quarter"} else 0.84 if lane_family == "side" else 0.88),
            (support_only_score, 0.22 if lane_family in {"front", "three_quarter"} else 0.16 if lane_family == "side" else 0.12),
        ]
    )

    heavy_confidence = _safe_float((heavy_bundle or {}).get("confidence"))
    heavy_coverage = _safe_float((heavy_bundle or {}).get("coverage"))
    evidence_agreement_score = _weighted_mean(
        [
            (truth_center_score, 0.46),
            (canonical_invariant_score, 0.24),
            (lane_validity, 0.12),
            (outfit_invariant_score, 0.10),
            (batch_relative_score, 0.08),
        ]
    )
    review_only_confidence = _weighted_mean(
        [
            (evidence_agreement_score, 0.58),
            (truth_center_score, 0.14),
            (heavy_confidence, 0.10),
            (heavy_coverage, 0.10),
            (heavy_metrics.get("body_mesh_fit_confidence"), 0.04),
            (face_shadow.get("face_pose_normalization_confidence"), 0.04),
        ]
    )
    truth_proxy = _weighted_mean(
        [
            (face_truth_support, 0.28 if lane_family in {"front", "three_quarter"} else 0.10 if lane_family == "side" else 0.0),
            (body_truth_support, 0.50 if lane_family in {"front", "three_quarter"} else 0.78 if lane_family == "side" else 0.88),
            (master.get("world3d_master_alignment"), 0.12),
            (heavy_metrics.get("body_pose_delta_similarity"), 0.06),
            (lane_validity, 0.04),
        ]
    )
    truth_proxy_confidence = _weighted_mean(
        [
            (heavy_metrics.get("body_mesh_fit_confidence"), 0.32),
            (heavy_confidence, 0.28),
            (heavy_coverage, 0.18),
            (face_shadow.get("face_pose_normalization_confidence"), 0.12),
            (lane_validity, 0.10),
        ]
    )

    hard_vetoes: List[str] = []
    soft_flags: List[str] = []
    body_truth_alignment = _safe_float(heavy_metrics.get("body_shape_truth_alignment"))
    body_measurement_similarity = _safe_float(heavy_metrics.get("canonical_measurement_similarity"))
    world3d_master_alignment = _safe_float(master.get("world3d_master_alignment"))
    mesh_confidence = _safe_float(heavy_metrics.get("body_mesh_fit_confidence"))

    if lane_validity is not None and lane_validity < 0.42:
        hard_vetoes.append("LANE_INVALID_FOR_REVIEW")
    if lane_family in {"front", "three_quarter", "side"}:
        face_master_alignment = _safe_float(master.get("face_master_alignment"))
        canonical_face_alignment = _safe_float(face_shadow.get("canonical_face_identity_similarity"))
        if face_master_alignment is not None and canonical_face_alignment is not None:
            if face_master_alignment < 0.48 and canonical_face_alignment < 0.54:
                hard_vetoes.append("FACE_TRUTH_STRONG_CONFLICT")
    if mesh_confidence is not None and mesh_confidence >= 0.60:
        if body_truth_alignment is not None and body_truth_alignment < 0.50:
            hard_vetoes.append("BODY_SHAPE_TRUTH_STRONG_CONFLICT")
        if body_measurement_similarity is not None and body_measurement_similarity < 0.50:
            hard_vetoes.append("BODY_MEASUREMENT_TRUTH_STRONG_CONFLICT")
        if world3d_master_alignment is not None and world3d_master_alignment < 0.50:
            hard_vetoes.append("WORLD3D_MASTER_STRONG_CONFLICT")
        if body_truth_support is not None and body_truth_support < 0.56:
            hard_vetoes.append("BODY_TRUTH_COMPOSITE_STRONG_CONFLICT")
    if review_only_score is None:
        hard_vetoes.append("REVIEW_ONLY_SCORE_UNAVAILABLE")

    if mesh_confidence is not None and mesh_confidence < 0.56:
        soft_flags.append("BODY_MESH_CONFIDENCE_LOW")
    if heavy_coverage is not None and heavy_coverage < 0.60:
        soft_flags.append("HEAVY_EVIDENCE_COVERAGE_LOW")
    if face_support_score is not None and face_support_score < 0.56 and lane_family in {"front", "three_quarter", "side"}:
        soft_flags.append("FACE_CANONICAL_SUPPORT_WEAK")
    truth_soft_floors = {
        "front": {"face": 0.70, "body": 0.70},
        "three_quarter": {"face": 0.64, "body": 0.68},
        "side": {"face": 0.54, "body": 0.66},
        "back": {"face": 0.00, "body": 0.66},
    }
    truth_floor = truth_soft_floors.get(lane_family, truth_soft_floors["three_quarter"])
    if face_truth_support is not None and face_truth_support < truth_floor["face"] and lane_family != "back":
        soft_flags.append("FACE_TRUTH_SUPPORT_WEAK")
    if body_truth_support is not None and body_truth_support < truth_floor["body"]:
        soft_flags.append("BODY_TRUTH_SUPPORT_WEAK")
    if outfit_invariant_score is not None and outfit_invariant_score < 0.62:
        soft_flags.append("OUTFIT_INVARIANT_SUPPORT_WEAK")
    if batch_relative_score is not None and batch_relative_score < 0.66:
        soft_flags.append("BATCH_RELATIVE_ALIGNMENT_WEAK")

    status = "FAIL"
    if len(hard_vetoes) == 0:
        if review_only_score is not None and review_only_confidence is not None:
            if review_only_score >= thresholds["pass_score"] and review_only_confidence >= thresholds["pass_conf"]:
                status = "PASS"
            elif review_only_score >= thresholds["warn_score"] and review_only_confidence >= thresholds["warn_conf"]:
                status = "WARN"
            elif review_only_score >= thresholds["warn_score"]:
                status = "WARN"
            else:
                status = "FAIL"
        elif review_only_score is not None and review_only_score >= thresholds["warn_score"]:
            status = "WARN"

    policy_note = None
    if lane_family in {"side", "back"} and status == "PASS":
        policy_note = "shadow lane PASS means priority review only; release and admission remain governed outside review_only"

    why_not_high: List[str] = []
    if len(hard_vetoes) > 0:
        why_not_high.extend(hard_vetoes[:2])
    if status != "PASS":
        why_not_high.extend(soft_flags[:2])
    if review_only_confidence is not None and review_only_confidence < thresholds["pass_conf"]:
        why_not_high.append("REVIEW_CONFIDENCE_BELOW_PASS_RANGE")
    if review_only_score is not None and review_only_score < thresholds["pass_score"]:
        why_not_high.append("REVIEW_SCORE_BELOW_PASS_RANGE")

    return {
        "lane_family": lane_family,
        "review_only_score_v2": _round_or_none(review_only_score),
        "review_only_confidence_v2": _round_or_none(review_only_confidence),
        "review_only_status_v2": status,
        "review_only_breakdown_v2": {
            "face_truth_support": _round_or_none(face_truth_support),
            "body_truth_support": _round_or_none(body_truth_support),
            "truth_center_score": _round_or_none(truth_center_score),
            "support_only_score": _round_or_none(support_only_score),
            "absolute_truth_support": _round_or_none(absolute_truth_support),
            "canonical_invariant_score": _round_or_none(canonical_invariant_score),
            "outfit_invariant_score": _round_or_none(outfit_invariant_score),
            "batch_relative_score": _round_or_none(batch_relative_score),
            "lane_validity": _round_or_none(lane_validity),
            "face_support_score": _round_or_none(face_support_score),
            "evidence_agreement_score": _round_or_none(evidence_agreement_score),
            "truth_proxy": _round_or_none(truth_proxy),
            "truth_proxy_confidence": _round_or_none(truth_proxy_confidence),
        },
        "review_only_hard_vetoes_v2": list(dict.fromkeys(hard_vetoes)),
        "review_only_soft_flags_v2": list(dict.fromkeys(soft_flags)),
        "review_only_policy_note_v2": policy_note,
        "why_not_high_confidence_v2": list(dict.fromkeys(why_not_high))[:4],
        "truth_proxy_v2": _round_or_none(truth_proxy),
        "truth_proxy_confidence_v2": _round_or_none(truth_proxy_confidence),
    }


def _append_unique(values: Sequence[Any], *extras: Any, limit: Optional[int] = None) -> List[str]:
    merged: List[str] = []
    for raw in list(values or []) + list(extras):
        text = str(raw or "").strip()
        if not text or text in merged:
            continue
        merged.append(text)
        if limit is not None and len(merged) >= int(limit):
            break
    return merged


def _review_only_pass_ratio(lane_family: str) -> float:
    mapping = {
        "front": 0.10,
        "three_quarter": 0.08,
        "side": 0.08,
        "back": 0.06,
    }
    return float(mapping.get(lane_family, 0.08))


def _review_only_pass_guard_reasons(candidate_row: Dict[str, Any]) -> List[str]:
    lane_family = str(candidate_row.get("lane_family") or "").strip().lower()
    breakdown = candidate_row.get("review_only_breakdown_v2") or {}
    reasons: List[str] = []
    face_truth = _safe_float(breakdown.get("face_truth_support"))
    body_truth = _safe_float(breakdown.get("body_truth_support"))

    if lane_family == "front":
        if face_truth is not None and face_truth < 0.72:
            reasons.append("FRONT_FACE_TRUTH_BELOW_PASS_FLOOR")
        if body_truth is not None and body_truth < 0.70:
            reasons.append("FRONT_BODY_TRUTH_BELOW_PASS_FLOOR")
    elif lane_family == "three_quarter":
        if face_truth is not None and face_truth < 0.64:
            reasons.append("THREE_QUARTER_FACE_TRUTH_BELOW_PASS_FLOOR")
        if body_truth is not None and body_truth < 0.68:
            reasons.append("THREE_QUARTER_BODY_TRUTH_BELOW_PASS_FLOOR")
    elif lane_family == "side":
        if body_truth is not None and body_truth < 0.66:
            reasons.append("SIDE_BODY_TRUTH_BELOW_PASS_FLOOR")
    elif lane_family == "back":
        if body_truth is not None and body_truth < 0.66:
            reasons.append("BACK_BODY_TRUTH_BELOW_PASS_FLOOR")
    return reasons


def _review_only_fail_guard_reasons(candidate_row: Dict[str, Any]) -> List[str]:
    lane_family = str(candidate_row.get("lane_family") or "").strip().lower()
    breakdown = candidate_row.get("review_only_breakdown_v2") or {}
    face_truth = _safe_float(breakdown.get("face_truth_support"))
    body_truth = _safe_float(breakdown.get("body_truth_support"))
    truth_center = _safe_float(breakdown.get("truth_center_score"))
    review_score = _safe_float(candidate_row.get("review_only_score_v2"))
    review_confidence = _safe_float(candidate_row.get("review_only_confidence_v2"))
    reasons: List[str] = []

    if (
        lane_family == "three_quarter"
        and face_truth is not None
        and truth_center is not None
        and review_score is not None
        and review_confidence is not None
        and face_truth < 0.60
        and truth_center < 0.72
        and review_score < 0.76
        and review_confidence < 0.82
    ):
        reasons.append("THREE_QUARTER_REVIEW_FAIL_GUARD")

    if (
        lane_family == "front"
        and face_truth is not None
        and truth_center is not None
        and review_score is not None
        and review_confidence is not None
        and face_truth < 0.70
        and truth_center < 0.75
        and review_score < 0.79
        and review_confidence < 0.83
    ):
        reasons.append("FRONT_REVIEW_FAIL_GUARD")

    if (
        lane_family == "side"
        and body_truth is not None
        and truth_center is not None
        and review_score is not None
        and review_confidence is not None
        and body_truth < 0.60
        and truth_center < 0.66
        and review_score < 0.74
        and review_confidence < 0.78
    ):
        reasons.append("SIDE_REVIEW_FAIL_GUARD")

    if (
        lane_family == "back"
        and body_truth is not None
        and truth_center is not None
        and review_score is not None
        and review_confidence is not None
        and body_truth < 0.60
        and truth_center < 0.64
        and review_score < 0.74
        and review_confidence < 0.76
    ):
        reasons.append("BACK_REVIEW_FAIL_GUARD")

    return reasons


def _refresh_review_only_policy_note(candidate_row: Dict[str, Any]) -> None:
    lane_family = str(candidate_row.get("lane_family") or "").strip().lower()
    status = str(candidate_row.get("review_only_status_v2") or "").strip().upper()
    if lane_family in {"side", "back"} and status == "PASS":
        candidate_row["review_only_policy_note_v2"] = (
            "shadow lane PASS means priority review only; release and admission remain governed outside review_only"
        )
    else:
        candidate_row["review_only_policy_note_v2"] = None


def _rebalance_review_only_group_statuses(group: Dict[str, Any]) -> None:
    candidates = list(group.get("candidates") or [])
    if len(candidates) == 0:
        return

    lanes: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidates:
        lane_family = str(row.get("lane_family") or "").strip().lower() or "three_quarter"
        lanes.setdefault(lane_family, []).append(row)

    for lane_family, lane_rows in lanes.items():
        ranked = sorted(
            lane_rows,
            key=lambda row: (
                1 if _safe_float(row.get("review_only_score_v2")) is None else 0,
                0.0 if _safe_float(row.get("review_only_score_v2")) is None else -float(row.get("review_only_score_v2")),
                0.0
                if _safe_float(row.get("review_only_confidence_v2")) is None
                else -float(row.get("review_only_confidence_v2")),
                str(row.get("image") or ""),
            ),
        )
        if len(ranked) < 8:
            pass_quota = len(ranked)
        else:
            pass_quota = max(1, int(math.ceil(len(ranked) * _review_only_pass_ratio(lane_family))))
        pass_used = 0

        for row in ranked:
            status = str(row.get("review_only_status_v2") or "").strip().upper()
            fail_guard_reasons = _review_only_fail_guard_reasons(row)
            if len(fail_guard_reasons) > 0:
                row["review_only_status_v2"] = "FAIL"
                row["review_only_hard_vetoes_v2"] = _append_unique(row.get("review_only_hard_vetoes_v2") or [], *fail_guard_reasons)
                row["why_not_high_confidence_v2"] = _append_unique(
                    row.get("why_not_high_confidence_v2") or [],
                    *fail_guard_reasons,
                    limit=4,
                )
                _refresh_review_only_policy_note(row)
                continue

            if status != "PASS":
                _refresh_review_only_policy_note(row)
                continue

            guard_reasons = _review_only_pass_guard_reasons(row)
            if len(guard_reasons) > 0:
                row["review_only_status_v2"] = "WARN"
                row["review_only_soft_flags_v2"] = _append_unique(row.get("review_only_soft_flags_v2") or [], *guard_reasons)
                row["why_not_high_confidence_v2"] = _append_unique(
                    row.get("why_not_high_confidence_v2") or [],
                    *guard_reasons,
                    limit=4,
                )
                _refresh_review_only_policy_note(row)
                continue

            if pass_used >= pass_quota:
                quota_reason = "GROUP_PASS_QUOTA_EXCEEDED"
                row["review_only_status_v2"] = "WARN"
                row["review_only_soft_flags_v2"] = _append_unique(row.get("review_only_soft_flags_v2") or [], quota_reason)
                row["why_not_high_confidence_v2"] = _append_unique(
                    row.get("why_not_high_confidence_v2") or [],
                    quota_reason,
                    limit=4,
                )
            else:
                pass_used += 1
            _refresh_review_only_policy_note(row)


def _sync_review_only_debug(item: Dict[str, Any], candidate_row: Dict[str, Any]) -> None:
    item_debug = item.setdefault("debug", {})
    item_debug["review_only_score_v2"] = {
        "score": candidate_row.get("review_only_score_v2"),
        "confidence": candidate_row.get("review_only_confidence_v2"),
        "status": candidate_row.get("review_only_status_v2"),
        "breakdown": candidate_row.get("review_only_breakdown_v2"),
        "hard_vetoes": candidate_row.get("review_only_hard_vetoes_v2"),
        "soft_flags": candidate_row.get("review_only_soft_flags_v2"),
        "policy_note": candidate_row.get("review_only_policy_note_v2"),
    }


def _strip_legacy_candidate_fields(row: Dict[str, Any]) -> None:
    for key in (
        "legacy_review_only_status",
        "legacy_review_only_confidence",
        "selection_score_legacy",
        "rank_legacy",
        "review_bucket_legacy",
    ):
        row.pop(key, None)
    delta_vs_top = row.get("delta_vs_top")
    if isinstance(delta_vs_top, dict):
        delta_vs_top.pop("selection_score_legacy", None)


def _priority_selection_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    strong_th: Optional[float],
    weak_th: Optional[float],
) -> Dict[str, Any]:
    by_lane: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lane_family = str(row.get("lane_family") or "").strip().lower() or "three_quarter"
        by_lane.setdefault(lane_family, []).append(row)

    selected_rows: List[Dict[str, Any]] = []
    for lane_family, lane_rows in by_lane.items():
        ranked = sorted(
            lane_rows,
            key=lambda row: (
                1 if _safe_float(row.get(score_key)) is None else 0,
                0.0 if _safe_float(row.get(score_key)) is None else -float(row.get(score_key)),
                str(row.get("image") or ""),
            ),
        )
        if len(ranked) == 0:
            continue
        quota = max(1, int(math.ceil(len(ranked) * _review_only_pass_ratio(lane_family))))
        selected_rows.extend(ranked[:quota])

    selected_truth_values = [_safe_float(row.get("truth_proxy_v2")) for row in selected_rows]
    selected_truth_values = [value for value in selected_truth_values if value is not None]
    bad_selected = 0
    strong_selected = 0
    for row in selected_rows:
        truth_value = _safe_float(row.get("truth_proxy_v2"))
        if truth_value is None:
            continue
        if weak_th is not None and truth_value <= weak_th:
            bad_selected += 1
        if strong_th is not None and truth_value >= strong_th:
            strong_selected += 1

    return {
        "selected_count": len(selected_rows),
        "selected_truth_mean": _round_or_none(_mean(selected_truth_values)),
        "bad_selected_ratio": _round_or_none(float(bad_selected / max(1, len(selected_rows)))) if len(selected_rows) > 0 else None,
        "strong_selected_count": strong_selected,
        "selected_images": [str(row.get("image") or "") for row in selected_rows[:8]],
    }


def _method_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    status_key: str,
    confidence_key: Optional[str],
) -> Dict[str, Any]:
    truth_values = [row.get("truth_proxy_v2") for row in rows]
    strong_th = _percentile(truth_values, 0.75)
    weak_th = _percentile(truth_values, 0.25)

    pass_truth_values: List[Optional[float]] = []
    blocked_truth_values: List[Optional[float]] = []
    strong_rows = 0
    strong_blocked = 0
    weak_rows = 0
    weak_pass = 0
    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for row in rows:
        truth_value = _safe_float(row.get("truth_proxy_v2"))
        raw_status = str(row.get(status_key) or "").strip().upper()
        status = raw_status if raw_status in {"PASS", "WARN", "FAIL"} else "FAIL"
        lane_family = str(row.get("lane_family") or "").strip().lower()
        kept_statuses = {"PASS", "WARN"} if lane_family in {"side", "back"} else {"PASS"}
        is_kept = status in kept_statuses
        status_counts[status] = status_counts.get(status, 0) + 1
        if is_kept:
            pass_truth_values.append(truth_value)
        else:
            blocked_truth_values.append(truth_value)
        if strong_th is not None and truth_value is not None and truth_value >= strong_th:
            strong_rows += 1
            if not is_kept:
                strong_blocked += 1
        if weak_th is not None and truth_value is not None and truth_value <= weak_th:
            weak_rows += 1
            if is_kept:
                weak_pass += 1

    ranked = sorted(
        list(rows),
        key=lambda row: (
            1 if _safe_float(row.get(score_key)) is None else 0,
            0.0 if _safe_float(row.get(score_key)) is None else -float(row.get(score_key)),
            str(row.get("image") or ""),
        ),
    )

    pass_truth_mean = _mean(pass_truth_values)
    blocked_truth_mean = _mean(blocked_truth_values)
    return {
        "candidate_count": len(rows),
        "pass_rate": _round_or_none(float(status_counts.get("PASS", 0) / max(1, len(rows)))),
        "status_counts": status_counts,
        "score_truth_corr": _round_or_none(_correlation([row.get(score_key) for row in rows], truth_values)),
        "confidence_truth_corr": _round_or_none(
            _correlation([row.get(confidence_key) for row in rows], truth_values) if confidence_key else None
        ),
        "top3_truth_mean": _round_or_none(_mean([row.get("truth_proxy_v2") for row in ranked[:3]])),
        "pass_truth_mean": _round_or_none(pass_truth_mean),
        "blocked_truth_mean": _round_or_none(blocked_truth_mean),
        "status_separation": _round_or_none(
            float(pass_truth_mean - blocked_truth_mean) if pass_truth_mean is not None and blocked_truth_mean is not None else None
        ),
        "false_block_proxy": _round_or_none(float(strong_blocked / max(1, strong_rows))) if strong_rows > 0 else None,
        "bad_pass_proxy": _round_or_none(float(weak_pass / max(1, weak_rows))) if weak_rows > 0 else None,
        "truth_proxy_strong_threshold": _round_or_none(strong_th),
        "truth_proxy_weak_threshold": _round_or_none(weak_th),
        "top_image": ranked[0].get("image") if len(ranked) > 0 else None,
        "top_score": _round_or_none(_safe_float(ranked[0].get(score_key))) if len(ranked) > 0 else None,
        "priority_selection": _priority_selection_summary(
            rows,
            score_key=score_key,
            strong_th=strong_th,
            weak_th=weak_th,
        ),
    }


def _prefer_v2(summary_legacy: Dict[str, Any], summary_v2: Dict[str, Any], truth_proxy_coverage: Optional[float]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if truth_proxy_coverage is None or truth_proxy_coverage < 0.55:
        return "selection_score_legacy", ["truth proxy coverage is too low for an unlabeled replacement decision"]

    score = 0
    legacy_corr = _safe_float(summary_legacy.get("score_truth_corr"))
    v2_corr = _safe_float(summary_v2.get("score_truth_corr"))
    if v2_corr is not None and (legacy_corr is None or v2_corr >= legacy_corr + 0.03):
        score += 1
        reasons.append("v2 score tracks truth proxy more tightly")

    legacy_conf_corr = _safe_float(summary_legacy.get("confidence_truth_corr"))
    v2_conf_corr = _safe_float(summary_v2.get("confidence_truth_corr"))
    confidence_gate_passed = False
    if v2_conf_corr is not None and (legacy_conf_corr is None or v2_conf_corr >= legacy_conf_corr + 0.02):
        score += 1
        confidence_gate_passed = True
        reasons.append("v2 confidence is better aligned with truth proxy")
    elif v2_conf_corr is not None and legacy_conf_corr is not None and v2_conf_corr < legacy_conf_corr:
        reasons.append("v2 confidence is still weaker than legacy confidence proxy")

    legacy_top3 = _safe_float(summary_legacy.get("top3_truth_mean"))
    v2_top3 = _safe_float(summary_v2.get("top3_truth_mean"))
    if v2_top3 is not None and (legacy_top3 is None or v2_top3 >= legacy_top3 + 0.02):
        score += 1
        reasons.append("v2 top ranks keep more truth-consistent candidates")

    legacy_priority = summary_legacy.get("priority_selection") or {}
    v2_priority = summary_v2.get("priority_selection") or {}
    legacy_priority_mean = _safe_float(legacy_priority.get("selected_truth_mean"))
    v2_priority_mean = _safe_float(v2_priority.get("selected_truth_mean"))
    if v2_priority_mean is not None and (legacy_priority_mean is None or v2_priority_mean >= legacy_priority_mean + 0.01):
        score += 2
        reasons.append("v2 picks a cleaner truth-centered priority shortlist under the same PASS quota")

    legacy_priority_bad = _safe_float(legacy_priority.get("bad_selected_ratio"))
    v2_priority_bad = _safe_float(v2_priority.get("bad_selected_ratio"))
    if v2_priority_bad is not None and (legacy_priority_bad is None or v2_priority_bad <= legacy_priority_bad - 0.05):
        score += 2
        reasons.append("v2 admits fewer weak truth-proxy candidates under quota-controlled PASS selection")

    legacy_priority_strong = _safe_float(legacy_priority.get("strong_selected_count"))
    v2_priority_strong = _safe_float(v2_priority.get("strong_selected_count"))
    if v2_priority_strong is not None and (legacy_priority_strong is None or v2_priority_strong >= legacy_priority_strong + 1):
        score += 1
        reasons.append("v2 keeps more strong truth-aligned samples inside the capped PASS bucket")

    legacy_false_block = _safe_float(summary_legacy.get("false_block_proxy"))
    v2_false_block = _safe_float(summary_v2.get("false_block_proxy"))
    if (
        legacy_priority_mean is None
        or v2_priority_mean is None
    ) and v2_false_block is not None and (legacy_false_block is None or v2_false_block <= legacy_false_block - 0.05):
        score += 1
        reasons.append("v2 blocks fewer strong truth-proxy candidates")

    legacy_sep = _safe_float(summary_legacy.get("status_separation"))
    v2_sep = _safe_float(summary_v2.get("status_separation"))
    if v2_sep is not None and (legacy_sep is None or v2_sep >= legacy_sep + 0.03):
        score += 1
        reasons.append("v2 PASS/WARN/FAIL separation is cleaner")

    legacy_bad_pass = _safe_float(summary_legacy.get("bad_pass_proxy"))
    v2_bad_pass = _safe_float(summary_v2.get("bad_pass_proxy"))
    if legacy_bad_pass is not None and v2_bad_pass is not None and v2_bad_pass > legacy_bad_pass + 0.05:
        score -= 2
        reasons.append("v2 passes too many weak truth-proxy candidates")
    if legacy_priority_bad is not None and v2_priority_bad is not None and v2_priority_bad > legacy_priority_bad + 0.05:
        score -= 3
        reasons.append("v2 priority shortlist quality is weaker than legacy under the same capped PASS regime")

    preferred = "review_only_score_v2" if score >= 2 and confidence_gate_passed else "selection_score_legacy"
    if len(reasons) == 0:
        reasons.append("legacy remains the conservative default under current proxy evidence")
    return preferred, reasons[:4]


def _apply_preferred_ranking(group: Dict[str, Any], *, preferred_method: str, comparison_summary: Dict[str, Any]) -> None:
    candidates = list(group.get("candidates") or [])
    if len(candidates) == 0:
        group["selection_method"] = preferred_method
        group["selection_comparison"] = comparison_summary
        return
    for row in candidates:
        row.setdefault("selection_score_legacy", row.get("selection_score"))
        row.setdefault("rank_legacy", row.get("rank"))
        row.setdefault("review_bucket_legacy", row.get("review_bucket"))
        row["selection_score_method"] = preferred_method

    if preferred_method == "review_only_score_v2":
        candidates.sort(
            key=lambda row: (
                1 if _safe_float(row.get("review_only_score_v2")) is None else 0,
                0.0 if _safe_float(row.get("review_only_score_v2")) is None else -float(row.get("review_only_score_v2")),
                0.0 if _safe_float(row.get("review_only_confidence_v2")) is None else -float(row.get("review_only_confidence_v2")),
                str(row.get("image") or ""),
            )
        )
        for row in candidates:
            row["selection_score"] = row.get("review_only_score_v2")
            row["selection_status"] = row.get("review_only_status_v2")
    else:
        candidates.sort(
            key=lambda row: (
                1 if _safe_float(row.get("selection_score_legacy")) is None else 0,
                0.0 if _safe_float(row.get("selection_score_legacy")) is None else -float(row.get("selection_score_legacy")),
                str(row.get("image") or ""),
            )
        )
        for row in candidates:
            row["selection_score"] = row.get("selection_score_legacy")
            row["selection_status"] = row.get("legacy_review_only_status")

    shortlist_size = max(1, min(int(group.get("shortlist_size") or 5), len(candidates)))
    top_components = candidates[0].get("component_scores") or {}
    top_score = _safe_float(candidates[0].get("selection_score"))
    legacy_top_score = _safe_float(candidates[0].get("selection_score_legacy"))
    v2_top_score = _safe_float(candidates[0].get("review_only_score_v2"))
    second_score = _safe_float(candidates[1].get("selection_score")) if len(candidates) > 1 else None
    gap_top2 = None if top_score is None or second_score is None else float(top_score - second_score)

    for index, row in enumerate(candidates, start=1):
        row["rank"] = index
        row["review_bucket"] = "shortlist" if index <= shortlist_size else "review"
        row["delta_vs_top"] = {
            "selection_score": _round_or_none(
                float(_safe_float(row.get("selection_score")) - top_score)
                if _safe_float(row.get("selection_score")) is not None and top_score is not None
                else None
            ),
            "selection_score_legacy": _round_or_none(
                float(_safe_float(row.get("selection_score_legacy")) - legacy_top_score)
                if _safe_float(row.get("selection_score_legacy")) is not None and legacy_top_score is not None
                else None
            ),
            "review_only_score_v2": _round_or_none(
                float(_safe_float(row.get("review_only_score_v2")) - v2_top_score)
                if _safe_float(row.get("review_only_score_v2")) is not None and v2_top_score is not None
                else None
            ),
            "component_scores": _component_deltas(top_components, row.get("component_scores", {})),
        }

    manual_review_window = 1
    gap_numeric = _safe_float(gap_top2)
    if gap_numeric is not None:
        if gap_numeric < 0.008:
            manual_review_window = min(3, shortlist_size)
        elif gap_numeric < 0.018:
            manual_review_window = min(2, shortlist_size)
    group["candidates"] = candidates
    group["shortlist"] = candidates[:shortlist_size]
    group["shortlist_size"] = shortlist_size
    group["top_ranked_image"] = candidates[0].get("image") if len(candidates) > 0 else None
    group["selection_gap_top2"] = _round_or_none(gap_top2)
    group["manual_review_window"] = max(int(group.get("manual_review_window", 1) or 1), manual_review_window)
    group["selection_method"] = preferred_method
    group["selection_comparison"] = comparison_summary


def apply_review_only_score_v2(
    report_items: Sequence[Dict[str, Any]],
    shot_selection: Dict[str, Any],
    *,
    target_profile: Optional[str] = None,
) -> Dict[str, Any]:
    del target_profile
    item_by_key: Dict[str, Dict[str, Any]] = {}
    for item in report_items:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        if record_key:
            item_by_key[record_key] = item

    comparison_groups: List[Dict[str, Any]] = []

    for group in shot_selection.get("groups") or []:
        candidates = list(group.get("candidates") or [])
        truth_proxy_available = 0
        candidate_items: Dict[str, Dict[str, Any]] = {}
        for row in candidates:
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            item = item_by_key.get(record_key)
            if item is None:
                continue
            candidate_items[record_key] = item
            lane_family = _lane_family(item, row)
            legacy_status = _status_for_score(
                row.get("selection_score"),
                pass_th=_legacy_thresholds(lane_family)["pass_score"],
                warn_th=_legacy_thresholds(lane_family)["warn_score"],
            )
            legacy_confidence = _legacy_confidence_proxy(item, lane_family)
            v2 = _compute_review_only_v2(item, row)
            row["legacy_review_only_status"] = legacy_status
            row["legacy_review_only_confidence"] = _round_or_none(legacy_confidence)
            row.update(v2)
            if row.get("truth_proxy_v2") is not None:
                truth_proxy_available += 1

        _rebalance_review_only_group_statuses(group)

        comparison_rows: List[Dict[str, Any]] = []
        for row in candidates:
            record_key = str(row.get("record_key") or row.get("image") or "").strip()
            item = candidate_items.get(record_key)
            if item is not None:
                _sync_review_only_debug(item, row)
            comparison_rows.append(
                {
                    "image": row.get("image"),
                    "selection_score_legacy": row.get("selection_score"),
                    "legacy_review_only_status": legacy_status,
                    "legacy_review_only_confidence": row.get("legacy_review_only_confidence"),
                    "lane_family": row.get("lane_family"),
                    "review_only_score_v2": row.get("review_only_score_v2"),
                    "review_only_status_v2": row.get("review_only_status_v2"),
                    "review_only_confidence_v2": row.get("review_only_confidence_v2"),
                    "truth_proxy_v2": row.get("truth_proxy_v2"),
                }
            )

        truth_proxy_coverage = float(truth_proxy_available / max(1, len(candidates))) if len(candidates) > 0 else None
        legacy_summary = _method_summary(
            comparison_rows,
            score_key="selection_score_legacy",
            status_key="legacy_review_only_status",
            confidence_key="legacy_review_only_confidence",
        )
        v2_summary = _method_summary(
            comparison_rows,
            score_key="review_only_score_v2",
            status_key="review_only_status_v2",
            confidence_key="review_only_confidence_v2",
        )
        legacy_monitor_preferred_method, decision_reasons = _prefer_v2(legacy_summary, v2_summary, truth_proxy_coverage)
        preferred_method = "review_only_score_v2"
        comparison_summary = {
            "comparison_mode": "legacy_monitor_only",
            "truth_proxy_coverage": _round_or_none(truth_proxy_coverage),
            "production_method": preferred_method,
            "legacy_retired_for_production": True,
            "legacy_monitor_preferred_method": legacy_monitor_preferred_method,
            "decision_reasons": decision_reasons,
            "legacy": legacy_summary,
            "v2": v2_summary,
        }
        _apply_preferred_ranking(group, preferred_method=preferred_method, comparison_summary=comparison_summary)
        for row in group.get("candidates") or []:
            _strip_legacy_candidate_fields(row)
        comparison_groups.append(
            {
                "group_key": group.get("group_key"),
                "production_method": preferred_method,
                "legacy_retired_for_production": True,
                "legacy_monitor_preferred_method": legacy_monitor_preferred_method,
                "truth_proxy_coverage": _round_or_none(truth_proxy_coverage),
                "decision_reasons": decision_reasons,
            }
        )
    selection_method = "review_only_score_v2"

    shot_selection["selection_method"] = selection_method
    shot_selection["review_only_score_v2_summary"] = {
        "comparison_mode": "legacy_monitor_only",
        "group_count": len(comparison_groups),
        "groups": comparison_groups,
        "production_method": selection_method,
        "legacy_retired_for_production": True,
    }
    return shot_selection
