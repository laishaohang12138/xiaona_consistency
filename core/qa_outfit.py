from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


_LAYER_TAGS: Sequence[str] = ("BODY_GOLD", "BRIDGE", "NECKLINE", "OUTER", "FACE_LOCK")
_VIEW_ALIASES: Sequence[tuple[str, Sequence[str]]] = (
    ("strict_side_90_left", ("strict_side_90_left", "strict-side-90-left")),
    ("strict_side_90_right", ("strict_side_90_right", "strict-side-90-right")),
    ("side_like_left", ("side_like_left", "side-like-left")),
    ("side_like_right", ("side_like_right", "side-like-right")),
    ("strict_back_180", ("strict_back_180", "strict-back-180")),
    ("back_like", ("back_like", "back-like")),
    ("three_quarter", ("three_quarter", "three-quarter", "3q", "threequarter")),
    ("front", ("front",)),
    ("side_90", ("side_90", "side-90")),
    ("back_180", ("back_180", "back-180")),
)
_NECKLINE_SLOTS: Dict[str, Sequence[str]] = {
    "turtleneck": ("turtleneck", "highneck", "high_neck"),
    "mock_neck": ("mockneck", "mock_neck"),
    "crew_neck": ("crewneck", "crew_neck"),
    "u_neck": ("u_neck", "u-neck"),
    "v_neck": ("v_neck", "v-neck"),
    "shirt_collar": ("shirtcollar", "shirt_collar", "open_collar", "open-collar"),
    "off_shoulder": ("offshoulder", "off_shoulder", "off-shoulder"),
    "halter": ("halter",),
}
_OUTER_SLOTS: Dict[str, Sequence[str]] = {
    "blazer": ("blazer",),
    "trench": ("trench", "trenchcoat", "trench_coat"),
    "coat": ("coat", "overcoat"),
    "jacket": ("jacket",),
    "cardigan": ("cardigan",),
    "knit_outer": ("knitouter", "knit_outer"),
}
_BRIDGE_SLOTS: Dict[str, Sequence[str]] = {
    "fitted_top": ("fittedtop", "fitted_top", "basic_top", "basictop"),
    "tank": ("tank", "tanktop", "tank_top"),
    "tee": ("tee", "tshirt", "t_shirt"),
    "camisole": ("camisole", "cami"),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _safe_relative_path(path: Path, base_dir: Optional[Path]) -> Path:
    resolved = path.resolve()
    if base_dir is None:
        return Path(path.name)
    try:
        return resolved.relative_to(base_dir.resolve())
    except Exception:
        return Path(path.name)


def _tokenize_path(rel_path: Path) -> List[str]:
    tokens: List[str] = []
    for part in rel_path.parts:
        normalized = _normalize_token(part)
        if normalized:
            tokens.append(normalized)
    return tokens


def _detect_layer_tag(tokens: Sequence[str]) -> Optional[str]:
    normalized_layers = {layer.lower(): layer for layer in _LAYER_TAGS}
    compact_layers = {layer.lower().replace("_", ""): layer for layer in _LAYER_TAGS}
    for token in tokens:
        if token in normalized_layers:
            return normalized_layers[token]
        compact = token.replace("_", "")
        if compact in compact_layers:
            return compact_layers[compact]
    return None


def _extract_named_key(tokens: Sequence[str], prefix: str) -> Optional[str]:
    compact_prefix = prefix.replace("_", "")
    for token in tokens:
        compact = token.replace("_", "")
        if compact.startswith(compact_prefix) and len(compact) > len(compact_prefix):
            return token
    return None


def _is_view_token(token: str) -> bool:
    for _, aliases in _VIEW_ALIASES:
        for alias in aliases:
            if token == _normalize_token(alias):
                return True
    return False


def _detect_view_expected(tokens: Sequence[str]) -> Optional[str]:
    normalized = set(tokens)
    for canonical, aliases in _VIEW_ALIASES:
        for alias in aliases:
            if _normalize_token(alias) in normalized:
                return canonical
    return None


def _detect_slot_from_keywords(layer_tag: Optional[str], tokens: Sequence[str]) -> Optional[str]:
    if layer_tag == "NECKLINE":
        mapping = _NECKLINE_SLOTS
    elif layer_tag == "OUTER":
        mapping = _OUTER_SLOTS
    elif layer_tag == "BRIDGE":
        mapping = _BRIDGE_SLOTS
    else:
        return None
    normalized = set(tokens)
    for slot_key, aliases in mapping.items():
        for alias in aliases:
            if _normalize_token(alias) in normalized:
                return slot_key
    return None


def parse_collection_metadata(image_path: Path, input_dir: Optional[Path]) -> Dict[str, Any]:
    rel_path = _safe_relative_path(image_path, input_dir)
    tokens = _tokenize_path(rel_path)
    layer_tag = _detect_layer_tag(tokens)
    look_key = _extract_named_key(tokens, "look")
    outfit_key = _extract_named_key(tokens, "outfit")
    slot_key = _extract_named_key(tokens, "slot")
    view_expected = _detect_view_expected(tokens)

    naming_source = "none"
    if look_key is None and layer_tag is not None:
        normalized_parts = [_normalize_token(part) for part in rel_path.parts[:-1]]
        try:
            layer_index = next(
                index
                for index, token in enumerate(normalized_parts)
                if token in {_normalize_token(layer_tag), _normalize_token(layer_tag).replace("_", "")}
            )
        except StopIteration:
            layer_index = -1
        if layer_index >= 0 and layer_index + 1 < len(normalized_parts):
            candidate = normalized_parts[layer_index + 1]
            if candidate and not _is_view_token(candidate) and not candidate.startswith("slot_"):
                if candidate.isdigit():
                    look_key = f"look_{candidate}"
                    naming_source = "layer_structure_numeric"
                elif candidate.startswith("look_") or candidate.startswith("outfit_"):
                    look_key = candidate
                    naming_source = "layer_structure_named"

    if slot_key is None:
        slot_key = _detect_slot_from_keywords(layer_tag, tokens)
        if slot_key is not None and naming_source == "none":
            naming_source = "keyword_slot"

    if outfit_key is None and look_key is not None and look_key.startswith("outfit_"):
        outfit_key = look_key
    if look_key is None and outfit_key is not None:
        look_key = outfit_key

    if naming_source == "none":
        if look_key is not None or outfit_key is not None or slot_key is not None:
            naming_source = "filename_pattern"
        elif layer_tag is not None:
            naming_source = "layer_only"

    groupable = layer_tag is not None and look_key is not None
    return {
        "input_relative_path": rel_path.as_posix(),
        "layer_tag": layer_tag,
        "look_key": look_key,
        "outfit_key": outfit_key,
        "slot_key": slot_key,
        "view_expected": view_expected,
        "groupable": bool(groupable),
        "naming_source": naming_source,
    }


def infer_layer_tag_from_profile(profile_name: Optional[str]) -> Optional[str]:
    normalized = _normalize_token(str(profile_name or ""))
    if normalized.startswith("body_gold"):
        return "BODY_GOLD"
    if normalized in {"full_body_outfit", "upper_body_product"}:
        return "BRIDGE"
    if "bridge" in normalized:
        return "BRIDGE"
    if "neckline" in normalized:
        return "NECKLINE"
    if "outer" in normalized:
        return "OUTER"
    if "outfit" in normalized:
        return "BRIDGE"
    if "face_lock" in normalized or "identity_lock" in normalized:
        return "FACE_LOCK"
    return None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _stability_score(values: Sequence[Optional[float]], scale: float, single_default: float = 0.58) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return float(single_default)
    spread = float(statistics.pstdev(valid))
    return _clamp(1.0 - (spread / max(1e-6, scale)), 0.0, 1.0)


def _weighted_mean(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _weighted_geometric_mean(items: Sequence[tuple[Optional[float], float]], floor: float = 1e-4) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        clipped = _clamp(float(value), floor, 1.0)
        numerator += float(weight) * float(np.log(clipped))
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(np.exp(numerator / denominator))


def _weighted_sum(items: Sequence[tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
    numerator: Optional[np.ndarray] = None
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        if numerator is None:
            numerator = np.zeros_like(value, dtype=np.float32)
        numerator += value.astype(np.float32) * float(weight)
        denominator += float(weight)
    if numerator is None or denominator <= 0.0:
        return None
    return numerator / float(denominator)


def _lane_family(view_lane_detail: Optional[str], fallback_lane: Optional[str]) -> Optional[str]:
    detail = str(view_lane_detail or "").strip()
    if detail.startswith("strict_side_90") or detail.startswith("side_like"):
        return "side"
    if detail in {"strict_back_180", "back_like"}:
        return "back"
    if detail in {"front", "three_quarter"}:
        return detail
    lane = str(fallback_lane or "").strip()
    if lane == "side_90":
        return "side"
    if lane == "back_180":
        return "back"
    if lane in {"front", "three_quarter"}:
        return lane
    return None


def _expected_view_families(layer_tag: Optional[str]) -> Set[str]:
    if layer_tag == "BODY_GOLD":
        return {"front", "side", "back"}
    if layer_tag == "BRIDGE":
        return {"front", "side", "back"}
    if layer_tag == "NECKLINE":
        return {"front", "three_quarter", "side"}
    if layer_tag == "OUTER":
        return {"front", "side", "back"}
    if layer_tag == "FACE_LOCK":
        return {"front", "three_quarter"}
    return {"front", "side", "back"}


def _coverage_score(layer_tag: Optional[str], families: Iterable[str]) -> float:
    observed = {family for family in families if family}
    expected = _expected_view_families(layer_tag)
    if len(expected) == 0:
        return 0.0
    return _clamp(len(observed & expected) / float(len(expected)), 0.0, 1.0)


def _dominant_family_ratio(families: Sequence[str]) -> Optional[float]:
    valid = [family for family in families if family]
    if len(valid) == 0:
        return None
    counts: Dict[str, int] = {}
    for family in valid:
        counts[family] = counts.get(family, 0) + 1
    return float(max(counts.values()) / max(1, len(valid)))


def _normalize_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        emb = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return None
    return emb / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    return float(statistics.median(valid))


def _mad(values: Sequence[Optional[float]], center: Optional[float] = None) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    if center is None:
        center = float(statistics.median(valid))
    deviations = [abs(value - center) for value in valid]
    return float(statistics.median(deviations))


def _collect_look_groups(
    report_items: Sequence[Dict[str, Any]],
    target_profile: Optional[str] = None,
) -> tuple[Dict[str, List[Dict[str, Any]]], bool]:
    grouped_items = [item for item in report_items if bool((item.get("collection") or {}).get("groupable", False))]
    look_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in grouped_items:
        collection = item.get("collection") or {}
        layer_tag = str(collection.get("layer_tag", "") or "").strip()
        look_key = str(collection.get("look_key", "") or "").strip()
        if not layer_tag or not look_key:
            continue
        group_key = f"{layer_tag}::{look_key}"
        look_groups.setdefault(group_key, []).append(item)

    implicit_batch_used = False
    if len(look_groups) == 0 and len(report_items) > 0:
        normalized_profile = _normalize_token(str(target_profile or "batch")) or "batch"
        group_key = f"BATCH::{normalized_profile}"
        look_groups[group_key] = list(report_items)
        implicit_batch_used = True
    return look_groups, implicit_batch_used


def build_collection_aggregates(
    report_items: Sequence[Dict[str, Any]],
    target_profile: Optional[str] = None,
) -> Dict[str, Any]:
    grouped_items = [item for item in report_items if bool((item.get("collection") or {}).get("groupable", False))]
    look_groups, implicit_batch_used = _collect_look_groups(report_items, target_profile=target_profile)

    look_aggregates: List[Dict[str, Any]] = []
    for group_key, items in sorted(look_groups.items()):
        collection = items[0].get("collection") or {}
        if implicit_batch_used and group_key.startswith("BATCH::"):
            layer_tag = infer_layer_tag_from_profile(target_profile)
            look_key = f"batch_{_normalize_token(str(target_profile or 'run')) or 'run'}"
            group_source = "active_profile_batch"
        else:
            layer_tag = collection.get("layer_tag")
            look_key = collection.get("look_key")
            group_source = "path_group"
        status_counts: Dict[str, int] = {}
        lane_families: List[str] = []
        slot_keys = sorted(
            {
                str((item.get("collection") or {}).get("slot_key"))
                for item in items
                if (item.get("collection") or {}).get("slot_key")
            }
        )
        for item in items:
            status = str(item.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1
            debug = item.get("debug") or {}
            lane_family = _lane_family(debug.get("view_lane_detail"), debug.get("view_lane"))
            if lane_family is not None:
                lane_families.append(lane_family)

        face_scores = [((item.get("scores") or {}).get("face")) for item in items]
        upper_scores = [((item.get("scores") or {}).get("upper")) for item in items]
        full_scores = [((item.get("scores") or {}).get("full")) for item in items]
        overall_scores = [((item.get("scores") or {}).get("overall")) for item in items]
        constitution_scores = [((item.get("scores") or {}).get("constitution")) for item in items]
        depth_scores = [((item.get("scores") or {}).get("depth_3d")) for item in items]
        garment_rows = [
            ((item.get("debug") or {}).get("garment_metrics") or {})
            for item in items
        ]
        clothing_coverage_values = [row.get("clothing_coverage_ratio") for row in garment_rows]
        upper_cloth_values = [row.get("upper_cloth_coverage") for row in garment_rows]
        lower_cloth_values = [row.get("lower_cloth_coverage") for row in garment_rows]
        neckline_values = [row.get("neckline_openness") for row in garment_rows]
        shoulder_balance_values = [row.get("shoulder_exposure_balance") for row in garment_rows]
        garment_confidence_values = [row.get("confidence") for row in garment_rows]

        identity_continuity = _weighted_mean(
            [
                (_stability_score(overall_scores, 0.08), 0.35),
                (_stability_score(full_scores, 0.07), 0.30),
                (_stability_score(constitution_scores, 0.05), 0.20),
                (_stability_score(face_scores, 0.10), 0.15),
            ]
        )
        garment_profile_stability = _weighted_mean(
            [
                (_stability_score(clothing_coverage_values, 0.08, single_default=0.62), 0.22),
                (_stability_score(upper_cloth_values, 0.10, single_default=0.62), 0.30),
                (_stability_score(lower_cloth_values, 0.10, single_default=0.60), 0.12),
                (_stability_score(neckline_values, 0.12, single_default=0.58), 0.20),
                (_mean(shoulder_balance_values), 0.08),
                (_mean(garment_confidence_values), 0.08),
            ]
        )
        garment_boundary_stability = _weighted_mean(
            [
                (garment_profile_stability, 0.60),
                (_stability_score(upper_scores, 0.08), 0.16),
                (_stability_score(full_scores, 0.08), 0.12),
                (_stability_score(depth_scores, 0.08), 0.07),
                (_stability_score(constitution_scores, 0.06), 0.05),
            ]
        )
        body_under_clothes_continuity = _weighted_mean(
            [
                (_stability_score(constitution_scores, 0.05), 0.45),
                (_stability_score(depth_scores, 0.07), 0.30),
                (_stability_score(full_scores, 0.08), 0.25),
            ]
        )
        image_quality_mean = _mean(overall_scores)
        coverage_balance = _coverage_score(layer_tag, lane_families)
        routing_consistency = _dominant_family_ratio(lane_families)
        group_structure_score = (
            routing_consistency
            if group_source == "active_profile_batch"
            else coverage_balance
        )
        look_score = _weighted_mean(
            [
                (identity_continuity, 0.35),
                (garment_boundary_stability, 0.35),
                (body_under_clothes_continuity, 0.20),
                (group_structure_score, 0.10),
            ]
        )

        look_aggregates.append(
            {
                "group_key": group_key,
                "layer_tag": layer_tag,
                "look_key": look_key,
                "outfit_key": collection.get("outfit_key"),
                "group_source": group_source,
                "slot_keys": slot_keys,
                "image_count": len(items),
                "status_counts": status_counts,
                "view_families": sorted(set(lane_families)),
                "image_quality_mean": _round_or_none(image_quality_mean),
                "identity_continuity": _round_or_none(identity_continuity),
                "garment_profile_stability": _round_or_none(garment_profile_stability),
                "garment_boundary_stability": _round_or_none(garment_boundary_stability),
                "body_under_clothes_continuity": _round_or_none(body_under_clothes_continuity),
                "coverage_balance": _round_or_none(coverage_balance),
                "routing_consistency": _round_or_none(routing_consistency),
                "group_structure_score": _round_or_none(group_structure_score),
                "clothing_coverage_mean": _round_or_none(_mean(clothing_coverage_values)),
                "upper_cloth_coverage_mean": _round_or_none(_mean(upper_cloth_values)),
                "lower_cloth_coverage_mean": _round_or_none(_mean(lower_cloth_values)),
                "neckline_openness_mean": _round_or_none(_mean(neckline_values)),
                "shoulder_exposure_balance_mean": _round_or_none(_mean(shoulder_balance_values)),
                "garment_confidence_mean": _round_or_none(_mean(garment_confidence_values)),
                "look_score": _round_or_none(look_score),
            }
        )

    layer_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in look_aggregates:
        layer_groups.setdefault(str(row.get("layer_tag", "")), []).append(row)

    layer_aggregates: List[Dict[str, Any]] = []
    for layer_tag, rows in sorted(layer_groups.items()):
        coverage_mean = _mean([row.get("coverage_balance") for row in rows])
        structure_mean = _mean([row.get("group_structure_score") for row in rows])
        quality_mean = _mean([row.get("image_quality_mean") for row in rows])
        look_score_mean = _mean([row.get("look_score") for row in rows])
        identity_mean = _mean([row.get("identity_continuity") for row in rows])
        garment_profile_mean = _mean([row.get("garment_profile_stability") for row in rows])
        cross_look_identity_stability = _stability_score(
            [row.get("identity_continuity") for row in rows],
            0.12,
            single_default=0.60,
        )
        set_score = _weighted_mean(
            [
                (look_score_mean, 0.40),
                (cross_look_identity_stability, 0.25),
                (garment_profile_mean, 0.20),
                (structure_mean if structure_mean is not None else coverage_mean, 0.10),
                (quality_mean, 0.15),
            ]
        )
        layer_aggregates.append(
            {
                "layer_tag": layer_tag,
                "look_count": len(rows),
                "image_count": int(sum(int(row.get("image_count", 0) or 0) for row in rows)),
                "mean_look_score": _round_or_none(look_score_mean),
                "mean_identity_continuity": _round_or_none(identity_mean),
                "mean_garment_profile_stability": _round_or_none(garment_profile_mean),
                "mean_coverage_balance": _round_or_none(coverage_mean),
                "mean_group_structure_score": _round_or_none(structure_mean),
                "cross_look_identity_stability": _round_or_none(cross_look_identity_stability),
                "image_quality_mean": _round_or_none(quality_mean),
                "set_score": _round_or_none(set_score),
            }
        )

    parser_summary = {
        "items_total": len(report_items),
        "items_with_layer_tag": sum(1 for item in report_items if (item.get("collection") or {}).get("layer_tag")),
        "items_with_look_key": sum(1 for item in report_items if (item.get("collection") or {}).get("look_key")),
        "groupable_items": len(grouped_items),
        "look_group_count": len(look_aggregates),
        "layer_group_count": len(layer_aggregates),
        "implicit_batch_used": bool(implicit_batch_used),
        "target_profile": target_profile,
    }
    return {
        "summary": parser_summary,
        "look_aggregates": look_aggregates,
        "layer_aggregates": layer_aggregates,
    }


def apply_collection_diagnostics(
    report_items: Sequence[Dict[str, Any]],
    collection_aggregates: Dict[str, Any],
    target_profile: Optional[str] = None,
    identity_samples: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    look_groups, implicit_batch_used = _collect_look_groups(report_items, target_profile=target_profile)
    group_diagnostics: List[Dict[str, Any]] = []
    outlier_item_count = 0
    outlier_reason_count = 0
    identity_by_key: Dict[str, Dict[str, Any]] = {}
    for row in identity_samples or []:
        record_key = str(row.get("record_key", "") or "").strip()
        if record_key:
            identity_by_key[record_key] = row

    for group_key, items in sorted(look_groups.items()):
        if len(items) == 0:
            continue
        collection = items[0].get("collection") or {}
        if implicit_batch_used and group_key.startswith("BATCH::"):
            group_source = "active_profile_batch"
            layer_tag = infer_layer_tag_from_profile(target_profile)
            look_key = f"batch_{_normalize_token(str(target_profile or 'run')) or 'run'}"
        else:
            group_source = "path_group"
            layer_tag = collection.get("layer_tag")
            look_key = collection.get("look_key")

        metric_map: Dict[str, List[Optional[float]]] = {
            "face": [],
            "clothing_coverage_ratio": [],
            "upper_cloth_coverage": [],
            "lower_cloth_coverage": [],
            "neckline_openness": [],
            "shoulder_exposure_balance": [],
            "overall": [],
            "constitution": [],
            "depth_3d": [],
        }
        lane_families: List[str] = []
        for item in items:
            garment = ((item.get("debug") or {}).get("garment_metrics") or {})
            scores = item.get("scores") or {}
            metric_map["face"].append(scores.get("face"))
            metric_map["clothing_coverage_ratio"].append(garment.get("clothing_coverage_ratio"))
            metric_map["upper_cloth_coverage"].append(garment.get("upper_cloth_coverage"))
            metric_map["lower_cloth_coverage"].append(garment.get("lower_cloth_coverage"))
            metric_map["neckline_openness"].append(garment.get("neckline_openness"))
            metric_map["shoulder_exposure_balance"].append(garment.get("shoulder_exposure_balance"))
            metric_map["overall"].append(scores.get("overall"))
            metric_map["constitution"].append(scores.get("constitution"))
            metric_map["depth_3d"].append(scores.get("depth_3d"))
            debug = item.get("debug") or {}
            lane_family = _lane_family(debug.get("view_lane_detail"), debug.get("view_lane"))
            if lane_family is not None:
                lane_families.append(lane_family)

        dominant_family = None
        if len(lane_families) > 0:
            family_counts: Dict[str, int] = {}
            for family in lane_families:
                family_counts[family] = family_counts.get(family, 0) + 1
            dominant_family = max(family_counts.items(), key=lambda pair: pair[1])[0]

        centers: Dict[str, Optional[float]] = {}
        scales: Dict[str, Optional[float]] = {}
        for metric_name, values in metric_map.items():
            center = _median(values)
            mad = _mad(values, center=center)
            min_scale = 0.02 if metric_name != "shoulder_exposure_balance" else 0.01
            if metric_name in {"constitution", "depth_3d", "overall"}:
                min_scale = 0.025
            scale = None
            if center is not None and mad is not None:
                scale = max(min_scale, 1.4826 * float(mad))
            centers[metric_name] = center
            scales[metric_name] = scale

        identity_rows: List[Dict[str, Any]] = []
        for item in items:
            record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
            signal = identity_by_key.get(record_key, {})
            embedding = _normalize_embedding(signal.get("embedding"))
            if embedding is None:
                continue
            face_conf = float(signal.get("face_conf", 0.0) or 0.0)
            face_score = float(signal.get("face_score", 0.0) or 0.0)
            bbox_area_ratio = float(signal.get("bbox_area_ratio", 0.0) or 0.0)
            quality_weight = float(
                max(0.0, face_conf)
                * (0.60 + 0.40 * min(1.0, bbox_area_ratio / 0.02))
                * (0.50 + 0.50 * face_score)
            )
            if quality_weight <= 0.0:
                continue
            identity_rows.append(
                {
                    "record_key": record_key,
                    "embedding": embedding,
                    "weight": quality_weight,
                    "face_conf": face_conf,
                    "face_score": face_score,
                }
            )

        identity_centroid = _weighted_sum(
            [(row["embedding"], row["weight"]) for row in identity_rows]
        )
        identity_centroid = _normalize_embedding(identity_centroid)
        centroid_sims: List[float] = []
        pairwise_sims: List[float] = []
        if identity_centroid is not None:
            for row in identity_rows:
                sim = _cosine(row["embedding"], identity_centroid)
                if sim is not None:
                    row["centroid_sim"] = sim
                    centroid_sims.append(float(sim))
        for idx, row_i in enumerate(identity_rows):
            for row_j in identity_rows[idx + 1:]:
                sim = _cosine(row_i["embedding"], row_j["embedding"])
                if sim is not None:
                    pairwise_sims.append(float(sim))

        centroid_median = _median(centroid_sims)
        centroid_scale = _mad(centroid_sims, center=centroid_median)
        if centroid_scale is not None:
            centroid_scale = max(0.01, 1.4826 * float(centroid_scale))
        pairwise_median = _median(pairwise_sims)
        batch_identity_cohesion = _weighted_mean(
            [
                (_weighted_mean([(row.get("centroid_sim"), row["weight"]) for row in identity_rows]), 0.70),
                (pairwise_median, 0.30),
            ]
        )

        body_rows: List[Dict[str, Any]] = []
        for item in items:
            record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
            signal = identity_by_key.get(record_key, {})
            body_signature = _normalize_embedding(signal.get("body_signature"))
            body_weight = float(signal.get("body_weight", 0.0) or 0.0)
            if body_signature is None or body_weight <= 0.0:
                continue
            body_rows.append(
                {
                    "record_key": record_key,
                    "embedding": body_signature,
                    "weight": body_weight,
                    "ready_features": int(signal.get("body_ready_features", 0) or 0),
                }
            )

        body_centroid = _weighted_sum(
            [(row["embedding"], row["weight"]) for row in body_rows]
        )
        body_centroid = _normalize_embedding(body_centroid)
        body_centroid_sims: List[float] = []
        body_pairwise_sims: List[float] = []
        if body_centroid is not None:
            for row in body_rows:
                sim = _cosine(row["embedding"], body_centroid)
                if sim is not None:
                    row["centroid_sim"] = sim
                    body_centroid_sims.append(float(sim))
        for idx, row_i in enumerate(body_rows):
            for row_j in body_rows[idx + 1:]:
                sim = _cosine(row_i["embedding"], row_j["embedding"])
                if sim is not None:
                    body_pairwise_sims.append(float(sim))

        body_centroid_median = _median(body_centroid_sims)
        body_centroid_scale = _mad(body_centroid_sims, center=body_centroid_median)
        if body_centroid_scale is not None:
            body_centroid_scale = max(0.01, 1.4826 * float(body_centroid_scale))
        body_pairwise_median = _median(body_pairwise_sims)
        batch_body_identity_cohesion = _weighted_mean(
            [
                (_weighted_mean([(row.get("centroid_sim"), row["weight"]) for row in body_rows]), 0.65),
                (body_pairwise_median, 0.35),
            ]
        )
        depth_rows: List[Dict[str, Any]] = []
        for item in items:
            record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
            signal = identity_by_key.get(record_key, {})
            depth_signature = _normalize_embedding(signal.get("depth_signature"))
            depth_weight = float(signal.get("depth_weight", 0.0) or 0.0)
            if depth_signature is None or depth_weight <= 0.0:
                continue
            depth_rows.append(
                {
                    "record_key": record_key,
                    "embedding": depth_signature,
                    "weight": depth_weight,
                    "ready_features": int(signal.get("depth_ready_features", 0) or 0),
                }
            )

        depth_centroid = _weighted_sum(
            [(row["embedding"], row["weight"]) for row in depth_rows]
        )
        depth_centroid = _normalize_embedding(depth_centroid)
        depth_centroid_sims: List[float] = []
        depth_pairwise_sims: List[float] = []
        if depth_centroid is not None:
            for row in depth_rows:
                sim = _cosine(row["embedding"], depth_centroid)
                if sim is not None:
                    row["centroid_sim"] = sim
                    depth_centroid_sims.append(float(sim))
        for idx, row_i in enumerate(depth_rows):
            for row_j in depth_rows[idx + 1:]:
                sim = _cosine(row_i["embedding"], row_j["embedding"])
                if sim is not None:
                    depth_pairwise_sims.append(float(sim))

        depth_centroid_median = _median(depth_centroid_sims)
        depth_centroid_scale = _mad(depth_centroid_sims, center=depth_centroid_median)
        if depth_centroid_scale is not None:
            depth_centroid_scale = max(0.01, 1.4826 * float(depth_centroid_scale))
        depth_pairwise_median = _median(depth_pairwise_sims)
        batch_3d_cohesion = _weighted_mean(
            [
                (_weighted_mean([(row.get("centroid_sim"), row["weight"]) for row in depth_rows]), 0.65),
                (depth_pairwise_median, 0.35),
            ]
        )
        batch_clothfree_identity_cohesion = _weighted_geometric_mean(
            [
                (batch_body_identity_cohesion, 0.68),
                (batch_3d_cohesion, 0.32),
            ]
        )
        batch_hybrid_identity_cohesion = _weighted_geometric_mean(
            [
                (batch_identity_cohesion, 0.58),
                (batch_clothfree_identity_cohesion, 0.42),
            ]
        )

        group_outlier_items = 0
        for item in items:
            garment = ((item.get("debug") or {}).get("garment_metrics") or {})
            scores = item.get("scores") or {}
            debug = item.get("debug") or {}
            lane_family = _lane_family(debug.get("view_lane_detail"), debug.get("view_lane"))
            reasons: List[str] = []
            metric_zscores: Dict[str, float] = {}

            def add_metric_reason(metric_name: str, value: Optional[float], reason: str, z_th: float = 3.2) -> None:
                center = centers.get(metric_name)
                scale = scales.get(metric_name)
                if value is None or center is None or scale is None or scale <= 1e-6:
                    return
                z_score = abs(float(value) - float(center)) / float(scale)
                if z_score >= z_th:
                    metric_zscores[metric_name] = round(float(z_score), 4)
                    reasons.append(reason)

            add_metric_reason(
                "face",
                scores.get("face"),
                "IDENTITY_FACE_OUTLIER_IN_BATCH",
                z_th=2.8,
            )
            add_metric_reason(
                "clothing_coverage_ratio",
                garment.get("clothing_coverage_ratio"),
                "GARMENT_COVERAGE_OUTLIER",
            )
            add_metric_reason(
                "upper_cloth_coverage",
                garment.get("upper_cloth_coverage"),
                "GARMENT_UPPER_COVERAGE_OUTLIER",
            )
            add_metric_reason(
                "lower_cloth_coverage",
                garment.get("lower_cloth_coverage"),
                "GARMENT_LOWER_COVERAGE_OUTLIER",
            )
            add_metric_reason(
                "neckline_openness",
                garment.get("neckline_openness"),
                "GARMENT_NECKLINE_OUTLIER",
                z_th=3.0,
            )
            add_metric_reason(
                "shoulder_exposure_balance",
                garment.get("shoulder_exposure_balance"),
                "GARMENT_SHOULDER_BALANCE_OUTLIER",
                z_th=3.0,
            )
            add_metric_reason(
                "constitution",
                scores.get("constitution"),
                "GARMENT_BODY_CONSTITUTION_OUTLIER",
            )
            add_metric_reason(
                "depth_3d",
                scores.get("depth_3d"),
                "GARMENT_BODY_DEPTH_OUTLIER",
            )
            add_metric_reason(
                "overall",
                scores.get("overall"),
                "GARMENT_BATCH_QUALITY_OUTLIER",
                z_th=3.4,
            )

            if dominant_family is not None and group_source == "active_profile_batch" and lane_family is not None and lane_family != dominant_family:
                reasons.append("GARMENT_VIEW_ROUTING_OUTLIER")

            record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
            identity_row = next((row for row in identity_rows if row.get("record_key") == record_key), None)
            body_row = next((row for row in body_rows if row.get("record_key") == record_key), None)
            depth_row = next((row for row in depth_rows if row.get("record_key") == record_key), None)
            identity_z = None
            body_identity_z = None
            depth_identity_z = None
            clothfree_identity_alignment = _weighted_geometric_mean(
                [
                    (body_row.get("centroid_sim") if body_row else None, 0.70),
                    (depth_row.get("centroid_sim") if depth_row else None, 0.30),
                ]
            )
            hybrid_identity_alignment = _weighted_geometric_mean(
                [
                    (identity_row.get("centroid_sim") if identity_row else None, 0.50),
                    (clothfree_identity_alignment, 0.35),
                    (scores.get("face"), 0.15),
                ]
            )
            if identity_row is not None and centroid_median is not None and centroid_scale is not None:
                centroid_sim = identity_row.get("centroid_sim")
                if centroid_sim is not None:
                    identity_z = max(0.0, (float(centroid_median) - float(centroid_sim)) / float(centroid_scale))
                    if identity_z >= 2.8 or float(centroid_sim) < 0.74:
                        metric_zscores["identity_face_batch"] = round(float(identity_z), 4)
                        reasons.append("IDENTITY_FACE_OUTLIER_IN_BATCH")
            if body_row is not None and body_centroid_median is not None and body_centroid_scale is not None:
                centroid_sim = body_row.get("centroid_sim")
                if centroid_sim is not None:
                    body_identity_z = max(0.0, (float(body_centroid_median) - float(centroid_sim)) / float(body_centroid_scale))
                    if body_identity_z >= 2.8 or float(centroid_sim) < 0.72:
                        metric_zscores["identity_body_batch"] = round(float(body_identity_z), 4)
                        reasons.append("IDENTITY_BODY_OUTLIER_IN_BATCH")
            if depth_row is not None and depth_centroid_median is not None and depth_centroid_scale is not None:
                centroid_sim = depth_row.get("centroid_sim")
                if centroid_sim is not None:
                    depth_identity_z = max(0.0, (float(depth_centroid_median) - float(centroid_sim)) / float(depth_centroid_scale))
                    if depth_identity_z >= 2.8 or float(centroid_sim) < 0.74:
                        metric_zscores["identity_3d_batch"] = round(float(depth_identity_z), 4)
                        reasons.append("IDENTITY_3D_OUTLIER_IN_BATCH")

            diagnostics = {
                "group_key": group_key,
                "group_source": group_source,
                "layer_tag": layer_tag,
                "look_key": look_key,
                "dominant_view_family": dominant_family,
                "identity_centroid_similarity": _round_or_none(identity_row.get("centroid_sim") if identity_row else None),
                "identity_centroid_z": _round_or_none(identity_z),
                "body_identity_centroid_similarity": _round_or_none(body_row.get("centroid_sim") if body_row else None),
                "body_identity_centroid_z": _round_or_none(body_identity_z),
                "depth_identity_centroid_similarity": _round_or_none(depth_row.get("centroid_sim") if depth_row else None),
                "depth_identity_centroid_z": _round_or_none(depth_identity_z),
                "clothfree_identity_alignment": _round_or_none(clothfree_identity_alignment),
                "hybrid_identity_alignment": _round_or_none(hybrid_identity_alignment),
                "metric_reference": {
                    key: {
                        "median": _round_or_none(centers.get(key)),
                        "scale": _round_or_none(scales.get(key)),
                    }
                    for key in centers.keys()
                },
                "metric_zscores": metric_zscores,
                "outlier_reasons": reasons,
                "outlier_score": _round_or_none(max(metric_zscores.values()) if len(metric_zscores) > 0 else 0.0),
            }
            debug["collection_diagnostics"] = diagnostics
            if len(reasons) > 0:
                existing_reasons = list(item.get("reasons") or [])
                item["reasons"] = list(dict.fromkeys(existing_reasons + reasons))
                group_outlier_items += 1
                outlier_item_count += 1
                outlier_reason_count += len(reasons)

        group_diagnostics.append(
            {
                "group_key": group_key,
                "group_source": group_source,
                "layer_tag": layer_tag,
                "look_key": look_key,
                "image_count": len(items),
                "dominant_view_family": dominant_family,
                "identity_support_count": len(identity_rows),
                "batch_identity_cohesion": _round_or_none(batch_identity_cohesion),
                "identity_centroid_similarity_median": _round_or_none(centroid_median),
                "identity_pairwise_similarity_median": _round_or_none(pairwise_median),
                "body_identity_support_count": len(body_rows),
                "batch_body_identity_cohesion": _round_or_none(batch_body_identity_cohesion),
                "body_identity_centroid_similarity_median": _round_or_none(body_centroid_median),
                "body_identity_pairwise_similarity_median": _round_or_none(body_pairwise_median),
                "batch_clothfree_identity_cohesion": _round_or_none(batch_clothfree_identity_cohesion),
                "batch_hybrid_identity_cohesion": _round_or_none(batch_hybrid_identity_cohesion),
                "depth_identity_support_count": len(depth_rows),
                "batch_3d_cohesion": _round_or_none(batch_3d_cohesion),
                "depth_identity_centroid_similarity_median": _round_or_none(depth_centroid_median),
                "depth_identity_pairwise_similarity_median": _round_or_none(depth_pairwise_median),
                "outlier_item_count": group_outlier_items,
                "outlier_item_ratio": _round_or_none(group_outlier_items / max(1, len(items))),
                "metric_reference": {
                    key: {
                        "median": _round_or_none(centers.get(key)),
                        "scale": _round_or_none(scales.get(key)),
                    }
                    for key in centers.keys()
                },
            }
        )

    summary = collection_aggregates.setdefault("summary", {})
    summary["outlier_item_count"] = int(outlier_item_count)
    summary["outlier_reason_count"] = int(outlier_reason_count)
    collection_aggregates["group_diagnostics"] = group_diagnostics
    return collection_aggregates


def evaluate_active_batch_gate(
    collection_aggregates: Dict[str, Any],
    policy: Dict[str, Any],
    target_profile: Optional[str] = None,
) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "enabled": bool(policy.get("batch_identity_gate_enabled", False)),
        "applied": False,
        "status": "disabled",
        "mode": str(policy.get("batch_gate_pass_cap_mode", "warn_all_pass")),
        "reasons": [],
        "thresholds": {},
        "metrics": {},
    }
    if not gate["enabled"]:
        collection_aggregates["batch_gate"] = gate
        return gate

    summary = collection_aggregates.get("summary") or {}
    look_rows = collection_aggregates.get("look_aggregates") or []
    group_rows = collection_aggregates.get("group_diagnostics") or []
    if not bool(summary.get("implicit_batch_used")) or len(look_rows) != 1 or len(group_rows) != 1:
        gate["status"] = "skipped"
        gate["skip_reason"] = "non_implicit_batch"
        collection_aggregates["batch_gate"] = gate
        return gate

    look = look_rows[0]
    group_diag = group_rows[0]
    thresholds = {
        "batch_image_count_min": int(policy.get("batch_image_count_min", 0) or 0),
        "batch_identity_continuity_min": policy.get("batch_identity_continuity_min", None),
        "batch_identity_cohesion_min": policy.get("batch_identity_cohesion_min", None),
        "batch_clothfree_identity_cohesion_min": policy.get("batch_clothfree_identity_cohesion_min", None),
        "batch_hybrid_identity_cohesion_min": policy.get("batch_hybrid_identity_cohesion_min", None),
        "batch_body_under_clothes_continuity_min": policy.get("batch_body_under_clothes_continuity_min", None),
        "batch_3d_cohesion_min": policy.get("batch_3d_cohesion_min", None),
        "batch_garment_boundary_stability_min": policy.get("batch_garment_boundary_stability_min", None),
        "batch_routing_consistency_min": policy.get("batch_routing_consistency_min", None),
        "batch_garment_confidence_min": policy.get("batch_garment_confidence_min", None),
        "batch_outlier_item_ratio_max": policy.get("batch_outlier_item_ratio_max", None),
    }
    metrics = {
        "image_count": look.get("image_count"),
        "identity_continuity": look.get("identity_continuity"),
        "batch_identity_cohesion": group_diag.get("batch_identity_cohesion"),
        "batch_body_identity_cohesion": group_diag.get("batch_body_identity_cohesion"),
        "batch_clothfree_identity_cohesion": group_diag.get("batch_clothfree_identity_cohesion"),
        "batch_hybrid_identity_cohesion": group_diag.get("batch_hybrid_identity_cohesion"),
        "batch_3d_cohesion": group_diag.get("batch_3d_cohesion"),
        "body_under_clothes_continuity": look.get("body_under_clothes_continuity"),
        "garment_boundary_stability": look.get("garment_boundary_stability"),
        "routing_consistency": look.get("routing_consistency"),
        "garment_confidence_mean": look.get("garment_confidence_mean"),
        "outlier_item_ratio": group_diag.get("outlier_item_ratio"),
    }
    reasons: List[str] = []

    def _lt(metric_name: str, threshold_name: str, reason: str) -> None:
        metric_value = metrics.get(metric_name)
        threshold_value = thresholds.get(threshold_name)
        if metric_value is None or threshold_value is None:
            return
        if float(metric_value) < float(threshold_value):
            reasons.append(reason)

    image_count = metrics.get("image_count")
    if image_count is not None and int(image_count) < int(thresholds["batch_image_count_min"]):
        reasons.append("BATCH_IMAGE_COUNT_BELOW_MIN")
    _lt("identity_continuity", "batch_identity_continuity_min", "BATCH_IDENTITY_CONTINUITY_LOW")
    _lt("batch_identity_cohesion", "batch_identity_cohesion_min", "BATCH_IDENTITY_COHESION_LOW")
    _lt(
        "batch_clothfree_identity_cohesion",
        "batch_clothfree_identity_cohesion_min",
        "BATCH_CLOTHFREE_IDENTITY_COHESION_LOW",
    )
    _lt(
        "batch_hybrid_identity_cohesion",
        "batch_hybrid_identity_cohesion_min",
        "BATCH_HYBRID_IDENTITY_COHESION_LOW",
    )
    _lt(
        "body_under_clothes_continuity",
        "batch_body_under_clothes_continuity_min",
        "BATCH_BODY_UNDER_CLOTHES_CONTINUITY_LOW",
    )
    _lt("batch_3d_cohesion", "batch_3d_cohesion_min", "BATCH_3D_COHESION_LOW")
    _lt(
        "garment_boundary_stability",
        "batch_garment_boundary_stability_min",
        "BATCH_GARMENT_BOUNDARY_STABILITY_LOW",
    )
    _lt("routing_consistency", "batch_routing_consistency_min", "BATCH_ROUTING_CONSISTENCY_LOW")
    _lt("garment_confidence_mean", "batch_garment_confidence_min", "BATCH_GARMENT_CONFIDENCE_LOW")
    outlier_ratio = metrics.get("outlier_item_ratio")
    outlier_max = thresholds.get("batch_outlier_item_ratio_max")
    if outlier_ratio is not None and outlier_max is not None and float(outlier_ratio) > float(outlier_max):
        reasons.append("BATCH_OUTLIER_RATIO_TOO_HIGH")

    gate["applied"] = True
    gate["status"] = "pass" if len(reasons) == 0 else "warn"
    gate["reasons"] = reasons
    gate["thresholds"] = {key: _round_or_none(value) if isinstance(value, (int, float)) else value for key, value in thresholds.items()}
    gate["metrics"] = {key: _round_or_none(value) if isinstance(value, (int, float)) else value for key, value in metrics.items()}
    gate["target_profile"] = target_profile
    collection_aggregates["batch_gate"] = gate
    summary["batch_gate_status"] = gate["status"]
    summary["batch_gate_applied"] = True
    return gate


def _selection_penalty(status: Optional[str], reasons: Sequence[str]) -> tuple[float, List[str]]:
    penalty = 0.0
    notes: List[str] = []
    normalized = set(str(reason) for reason in reasons)

    if str(status or "").upper() == "FAIL":
        penalty += 0.18
        notes.append("当前 QA 主结论为 FAIL")
    elif str(status or "").upper() == "WARN":
        penalty += 0.04

    penalty_map = [
        ("FACE_NO_RELIABLE_SIGNAL", 0.16, "脸部信号不可靠"),
        ("IDENTITY_FACE_OUTLIER_IN_BATCH", 0.10, "脸部身份偏离当前批次中心"),
        ("IDENTITY_BODY_OUTLIER_IN_BATCH", 0.08, "衣下人体结构偏离当前批次中心"),
        ("IDENTITY_3D_OUTLIER_IN_BATCH", 0.10, "3D 几何偏离当前批次中心"),
        ("DEPTH_3D_LITE_STRONG_WARN", 0.10, "3D 一致性存在强警告"),
        ("DEPTH_3D_LITE_WARN", 0.06, "3D 一致性存在警告"),
        ("BODY_CONSTITUTION_STRONG_WARN", 0.08, "身材量纲存在强警告"),
        ("BODY_CONSTITUTION_WARN", 0.05, "身材量纲存在警告"),
        ("VIEW_LANE_NOT_ALLOWED_FOR_PROFILE", 0.08, "视角不在当前 profile 放行范围"),
        ("GARMENT_VIEW_ROUTING_OUTLIER", 0.06, "视角相对当前批次不一致"),
        ("GARMENT_BATCH_QUALITY_OUTLIER", 0.06, "整体质量偏离当前批次"),
    ]
    for reason, reason_penalty, note in penalty_map:
        if reason in normalized:
            penalty += reason_penalty
            notes.append(note)

    garment_outlier_flags = {
        "GARMENT_COVERAGE_OUTLIER",
        "GARMENT_UPPER_COVERAGE_OUTLIER",
        "GARMENT_LOWER_COVERAGE_OUTLIER",
        "GARMENT_NECKLINE_OUTLIER",
        "GARMENT_SHOULDER_BALANCE_OUTLIER",
    }
    if len(garment_outlier_flags & normalized) > 0:
        penalty += 0.04
        notes.append("穿搭边界相对当前批次有偏离")

    return _clamp(penalty, 0.0, 0.45), notes


def _selection_highlights(components: Dict[str, Optional[float]], caution_notes: Sequence[str]) -> List[str]:
    notes: List[str] = []
    if isinstance(components.get("absolute_face_identity"), (int, float)) and float(components["absolute_face_identity"]) >= 0.78:
        notes.append("绝对 face 身份锚保持稳定")
    if isinstance(components.get("hybrid_identity_alignment"), (int, float)) and float(components["hybrid_identity_alignment"]) >= 0.88:
        notes.append("多通道身份对齐稳定")
    if isinstance(components.get("batch_face_alignment"), (int, float)) and float(components["batch_face_alignment"]) >= 0.88:
        notes.append("脸部位于当前批次身份中心附近")
    if isinstance(components.get("clothfree_identity_alignment"), (int, float)) and float(components["clothfree_identity_alignment"]) >= 0.90:
        notes.append("衣物无关身份对齐稳定")
    if isinstance(components.get("clothfree_body_alignment"), (int, float)) and float(components["clothfree_body_alignment"]) >= 0.90:
        notes.append("衣物无关的人体结构稳定")
    if isinstance(components.get("depth_alignment"), (int, float)) and float(components["depth_alignment"]) >= 0.86:
        notes.append("3D 几何与当前批次高度一致")
    if isinstance(components.get("structure_stability"), (int, float)) and float(components["structure_stability"]) >= 0.84:
        notes.append("全身构图与体态较稳")
    if len(notes) == 0 and len(caution_notes) == 0:
        notes.append("当前图在本批次内整体稳定")
    return notes[:4]


def _component_deltas(
    top_components: Dict[str, Optional[float]],
    candidate_components: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    deltas: Dict[str, Optional[float]] = {}
    for key in sorted(set(top_components.keys()) | set(candidate_components.keys())):
        top_value = top_components.get(key)
        candidate_value = candidate_components.get(key)
        if isinstance(top_value, (int, float)) and isinstance(candidate_value, (int, float)):
            deltas[key] = _round_or_none(float(candidate_value) - float(top_value))
        else:
            deltas[key] = None
    return deltas


def build_shot_selection_report(
    report_items: Sequence[Dict[str, Any]],
    collection_aggregates: Dict[str, Any],
    target_profile: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    look_groups, implicit_batch_used = _collect_look_groups(report_items, target_profile=target_profile)
    look_rows = {
        str(row.get("group_key", "")): row
        for row in (collection_aggregates.get("look_aggregates") or [])
        if isinstance(row, dict)
    }
    group_rows = {
        str(row.get("group_key", "")): row
        for row in (collection_aggregates.get("group_diagnostics") or [])
        if isinstance(row, dict)
    }

    groups: List[Dict[str, Any]] = []
    for group_key, items in sorted(look_groups.items()):
        if len(items) == 0:
            continue
        look_row = look_rows.get(group_key, {})
        group_diag = group_rows.get(group_key, {})
        ranked_candidates: List[Dict[str, Any]] = []

        for item in items:
            scores = item.get("scores") or {}
            debug = item.get("debug") or {}
            diagnostics = debug.get("collection_diagnostics") or {}
            reasons = list(item.get("reasons") or [])

            components = {
                "absolute_face_identity": scores.get("face"),
                "batch_face_alignment": diagnostics.get("identity_centroid_similarity"),
                "clothfree_identity_alignment": diagnostics.get("clothfree_identity_alignment"),
                "hybrid_identity_alignment": diagnostics.get("hybrid_identity_alignment"),
                "clothfree_body_alignment": _weighted_mean(
                    [
                        (diagnostics.get("body_identity_centroid_similarity"), 0.75),
                        (scores.get("constitution"), 0.15),
                        (scores.get("full"), 0.10),
                    ]
                ),
                "depth_alignment": _weighted_mean(
                    [
                        (diagnostics.get("depth_identity_centroid_similarity"), 0.60),
                        (scores.get("depth_3d"), 0.40),
                    ]
                ),
                "structure_stability": _weighted_mean(
                    [
                        (scores.get("full"), 0.40),
                        (scores.get("constitution"), 0.35),
                        (scores.get("upper"), 0.25),
                    ]
                ),
            }

            base_score = _weighted_mean(
                [
                    (components["absolute_face_identity"], 0.24),
                    (components["hybrid_identity_alignment"], 0.28),
                    (components["clothfree_identity_alignment"], 0.16),
                    (components["depth_alignment"], 0.18),
                    (components["structure_stability"], 0.14),
                ]
            )
            penalty_value, penalty_notes = _selection_penalty(item.get("status"), reasons)
            selection_score = None
            if base_score is not None:
                selection_score = _clamp(float(base_score) - float(penalty_value), 0.0, 1.0)

            caution_notes = list(penalty_notes)
            if isinstance(components.get("batch_face_alignment"), (int, float)) and float(components["batch_face_alignment"]) < 0.82:
                caution_notes.append("脸部与批次中心仍有距离")
            if isinstance(components.get("hybrid_identity_alignment"), (int, float)) and float(components["hybrid_identity_alignment"]) < 0.84:
                caution_notes.append("多通道身份对齐仍偏松")
            if isinstance(components.get("clothfree_identity_alignment"), (int, float)) and float(components["clothfree_identity_alignment"]) < 0.88:
                caution_notes.append("衣物无关身份对齐仍偏弱")
            if isinstance(components.get("clothfree_body_alignment"), (int, float)) and float(components["clothfree_body_alignment"]) < 0.84:
                caution_notes.append("衣物无关的人体结构仍偏松")
            if isinstance(components.get("depth_alignment"), (int, float)) and float(components["depth_alignment"]) < 0.80:
                caution_notes.append("3D 几何一致性仍偏弱")

            ranked_candidates.append(
                {
                    "image": item.get("image"),
                    "record_key": (item.get("collection") or {}).get("input_relative_path") or item.get("image"),
                    "status": item.get("status"),
                    "selection_score": _round_or_none(selection_score),
                    "component_scores": {key: _round_or_none(value) for key, value in components.items()},
                    "penalty": {
                        "value": _round_or_none(penalty_value),
                        "notes": penalty_notes[:4],
                    },
                    "outlier_score": diagnostics.get("outlier_score"),
                    "primary_3d_bottleneck": ((debug.get("depth_3d_metrics") or {}).get("primary_bottleneck")),
                    "winner_reasons": _selection_highlights(components, caution_notes),
                    "caution_reasons": caution_notes[:4],
                    "top_reasons": reasons[:8],
                }
            )

        ranked_candidates.sort(
            key=lambda row: (
                1 if row.get("selection_score") is None else 0,
                0.0 if row.get("selection_score") is None else -float(row.get("selection_score")),
                str(row.get("image") or ""),
            )
        )
        for index, row in enumerate(ranked_candidates, start=1):
            row["rank"] = index
            row["review_bucket"] = "shortlist" if index <= max(1, top_k) else "review"

        shortlist = ranked_candidates[: max(1, min(top_k, len(ranked_candidates)))]
        top_ranked_image = shortlist[0].get("image") if len(shortlist) > 0 else None
        gap_top2 = None
        top_components = shortlist[0].get("component_scores", {}) if len(shortlist) > 0 else {}
        if len(ranked_candidates) >= 2:
            first_score = ranked_candidates[0].get("selection_score")
            second_score = ranked_candidates[1].get("selection_score")
            if isinstance(first_score, (int, float)) and isinstance(second_score, (int, float)):
                gap_top2 = float(first_score) - float(second_score)

        manual_review_window = 1
        review_guidance: List[str] = []
        if isinstance(gap_top2, (int, float)):
            if float(gap_top2) < 0.005:
                manual_review_window = min(3, len(shortlist))
                review_guidance.append("机器排序的前两名差距很小，建议至少复核 top 3。")
            elif float(gap_top2) < 0.012:
                manual_review_window = min(2, len(shortlist))
                review_guidance.append("机器排序的前两名接近，建议重点复核 top 2。")
            else:
                review_guidance.append("机器排序的第 1 名与第 2 名差距相对明确，可优先从 top 1 开始复核。")
        else:
            review_guidance.append("当前批次候选过少，需直接做人工复核。")

        if group_diag.get("batch_identity_cohesion") is not None and float(group_diag.get("batch_identity_cohesion")) < 0.88:
            review_guidance.append("本批次 face 身份凝聚度仍偏弱，最终判断应提高对脸部细节的一票否决权。")
        if group_diag.get("batch_clothfree_identity_cohesion") is not None and float(group_diag.get("batch_clothfree_identity_cohesion")) >= 0.90:
            review_guidance.append("衣物无关结构总体稳定，复核时可优先看脸部和局部神态，而不是衣服边界。")
        if group_diag.get("batch_3d_cohesion") is not None and float(group_diag.get("batch_3d_cohesion")) >= 0.98:
            review_guidance.append("当前批次 3D 一致性整体稳定，3D 只需重点排查尾部候选。")

        for row in shortlist:
            row["delta_vs_top"] = {
                "selection_score": _round_or_none(
                    (float(row.get("selection_score")) - float(shortlist[0].get("selection_score")))
                    if isinstance(row.get("selection_score"), (int, float))
                    and isinstance(shortlist[0].get("selection_score"), (int, float))
                    else None
                ),
                "component_scores": _component_deltas(top_components, row.get("component_scores", {})),
            }

        groups.append(
            {
                "group_key": group_key,
                "group_source": group_diag.get("group_source") or ("active_profile_batch" if implicit_batch_used else "path_group"),
                "layer_tag": group_diag.get("layer_tag") or look_row.get("layer_tag"),
                "look_key": group_diag.get("look_key") or look_row.get("look_key"),
                "image_count": len(items),
                "top_ranked_image": top_ranked_image,
                "selection_gap_top2": _round_or_none(gap_top2),
                "manual_review_window": int(manual_review_window),
                "shortlist_size": len(shortlist),
                "review_guidance": review_guidance[:4],
                "batch_reference": {
                    "identity_continuity": look_row.get("identity_continuity"),
                    "garment_boundary_stability": look_row.get("garment_boundary_stability"),
                        "body_under_clothes_continuity": look_row.get("body_under_clothes_continuity"),
                        "routing_consistency": look_row.get("routing_consistency"),
                        "batch_identity_cohesion": group_diag.get("batch_identity_cohesion"),
                        "batch_clothfree_identity_cohesion": group_diag.get("batch_clothfree_identity_cohesion"),
                        "batch_hybrid_identity_cohesion": group_diag.get("batch_hybrid_identity_cohesion"),
                        "batch_3d_cohesion": group_diag.get("batch_3d_cohesion"),
                        "outlier_item_ratio": group_diag.get("outlier_item_ratio"),
                    },
                "shortlist": shortlist,
                "candidates": ranked_candidates,
            }
        )

    return {
        "mode": "advisory_rank_only",
        "final_decision_owner": "custom_gpt_plus_human",
        "target_profile": target_profile,
        "group_count": len(groups),
        "groups": groups,
    }
