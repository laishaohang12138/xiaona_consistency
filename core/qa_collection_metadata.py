from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_LAYER_TAGS: Sequence[str] = ("BODY_GOLD", "BRIDGE", "NECKLINE", "OUTER", "FACE_LOCK")
_VIEW_ALIASES: Sequence[tuple[str, Sequence[str]]] = (
    ("strict_side_90_left", ("strict_side_90_left", "strict-side-90-left")),
    ("strict_side_90_right", ("strict_side_90_right", "strict-side-90-right")),
    ("side_like_left", ("side_like_left", "side-like-left")),
    ("side_like_right", ("side_like_right", "side-like-right")),
    ("strict_back_180", ("strict_back_180", "strict-back-180")),
    ("back_like", ("back_like", "back-like")),
    ("three_quarter", ("three_quarter", "three-quarter", "3q", "threequarter", "30deg", "30_deg", "60deg", "60_deg")),
    ("front", ("front",)),
    ("side_90", ("side_90", "side-90", "90deg", "90_deg")),
    ("back_180", ("back_180", "back-180", "180deg", "180_deg")),
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


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _safe_relative_path(path: Path, base_dir: Optional[Path]) -> Path:
    resolved = path.resolve()
    if base_dir is None:
        return Path(resolved.name)
    try:
        return resolved.relative_to(base_dir.resolve())
    except Exception:
        return Path(resolved.name)


def _tokenize_path(rel_path: Path) -> List[str]:
    tokens: List[str] = []
    for part in rel_path.parts:
        normalized = _normalize_token(part)
        if normalized:
            tokens.extend(chunk for chunk in normalized.split("_") if chunk)
    return tokens


def _detect_layer_tag(tokens: Sequence[str]) -> Optional[str]:
    normalized = set(tokens)
    for layer_tag in _LAYER_TAGS:
        if _normalize_token(layer_tag) in normalized:
            return layer_tag
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


def _view_family_from_expected(view_expected: Optional[str]) -> Optional[str]:
    value = str(view_expected or "").strip().lower()
    if not value:
        return None
    if "back" in value:
        return "back"
    if "side" in value:
        return "side"
    if "three" in value or "3q" in value:
        return "three_quarter"
    if "front" in value:
        return "front"
    return None


def _view_center_deg_from_expected(view_expected: Optional[str]) -> Optional[float]:
    family = _view_family_from_expected(view_expected)
    mapping = {
        "front": 0.0,
        "three_quarter": 45.0,
        "side": 90.0,
        "back": 180.0,
    }
    if family is None:
        return None
    return float(mapping.get(family, 0.0))


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


def parse_collection_metadata(
    image_path: Path,
    input_dir: Optional[Path],
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rel_path = _safe_relative_path(image_path, input_dir)
    tokens = _tokenize_path(rel_path)
    layer_tag = _detect_layer_tag(tokens)
    look_key = _extract_named_key(tokens, "look")
    outfit_key = _extract_named_key(tokens, "outfit")
    slot_key = _extract_named_key(tokens, "slot")
    view_expected_from_path = _detect_view_expected(tokens)
    view_expected_family_from_path = _view_family_from_expected(view_expected_from_path)
    view_expected_center_deg_from_path = _view_center_deg_from_expected(view_expected_from_path)
    view_expected = view_expected_from_path
    view_expected_family = view_expected_family_from_path
    view_expected_center_deg = view_expected_center_deg_from_path

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

    prompt_intent_metadata_source = "path_tokens" if view_expected_family is not None else "none"
    manifest_payload = dict(manifest_entry) if isinstance(manifest_entry, dict) else {}
    manifest_present = len(manifest_payload) > 0
    if manifest_present:
        if not layer_tag and manifest_payload.get("layer_tag"):
            layer_tag = str(manifest_payload.get("layer_tag")).strip() or layer_tag
        if look_key is None and manifest_payload.get("look_key"):
            look_key = str(manifest_payload.get("look_key")).strip() or look_key
        if outfit_key is None and manifest_payload.get("outfit_key"):
            outfit_key = str(manifest_payload.get("outfit_key")).strip() or outfit_key
        if slot_key is None and manifest_payload.get("slot_key"):
            slot_key = str(manifest_payload.get("slot_key")).strip() or slot_key

        manifest_view_expected = str(
            manifest_payload.get("view_expected")
            or manifest_payload.get("intended_view")
            or ""
        ).strip()
        manifest_view_family = str(
            manifest_payload.get("view_expected_family")
            or manifest_payload.get("intended_lane_family")
            or ""
        ).strip()
        manifest_view_center_deg = manifest_payload.get(
            "view_expected_center_deg",
            manifest_payload.get("intended_view_center_deg"),
        )
        if manifest_view_expected:
            view_expected = manifest_view_expected
            prompt_intent_metadata_source = "input_manifest"
        if manifest_view_family:
            view_expected_family = _view_family_from_expected(manifest_view_family) or manifest_view_family
            prompt_intent_metadata_source = "input_manifest"
        if manifest_view_center_deg is not None:
            view_expected_center_deg = manifest_view_center_deg
            prompt_intent_metadata_source = "input_manifest"
        if view_expected_family is None and view_expected is not None:
            view_expected_family = _view_family_from_expected(view_expected)
        if view_expected_center_deg is None and view_expected is not None:
            view_expected_center_deg = _view_center_deg_from_expected(view_expected)
        if naming_source == "none":
            naming_source = "input_manifest"

    groupable = layer_tag is not None and look_key is not None
    return {
        "input_relative_path": rel_path.as_posix(),
        "layer_tag": layer_tag,
        "look_key": look_key,
        "outfit_key": outfit_key,
        "slot_key": slot_key,
        "view_expected": view_expected,
        "view_expected_family": view_expected_family,
        "view_expected_center_deg": view_expected_center_deg,
        "view_expected_from_path": view_expected_from_path,
        "view_expected_family_from_path": view_expected_family_from_path,
        "view_expected_center_deg_from_path": view_expected_center_deg_from_path,
        "view_expected_is_weak_prior": view_expected_family is not None,
        "prompt_intent_metadata_source": prompt_intent_metadata_source,
        "manifest_entry_present": bool(manifest_present),
        "prompt_id": manifest_payload.get("prompt_id"),
        "seed": manifest_payload.get("seed"),
        "seed_unavailable_reason": manifest_payload.get("seed_unavailable_reason"),
        "anchor_source": manifest_payload.get("anchor_source"),
        "generator_name": manifest_payload.get("generator_name"),
        "generator_version": manifest_payload.get("generator_version"),
        "prompt_pack": manifest_payload.get("prompt_pack"),
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
