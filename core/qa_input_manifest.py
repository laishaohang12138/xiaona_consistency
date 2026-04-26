from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qa_collection_metadata import parse_collection_metadata

_DEFAULT_MANIFEST_NAMES = (
    "input_manifest.json",
    "_input_manifest.json",
    "manifest.json",
)
_REQUIRED_PROMPT_INTENT_FIELDS = (
    "prompt_id",
    "seed",
    "anchor_source",
    "intended_view",
)
_RESERVED_TOP_LEVEL_KEYS = {
    "schema_version",
    "version",
    "generator",
    "generator_name",
    "generator_version",
    "created_at",
    "generated_at",
    "items",
    "images",
    "entries",
    "meta",
    "summary",
    "notes",
}


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/").lower()


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _seed_or_unavailable_ready(payload: Dict[str, Any]) -> bool:
    return payload.get("seed") is not None or _first_text(payload.get("seed_unavailable_reason")) is not None


def _safe_relative(path: Path, base_dir: Optional[Path]) -> str:
    resolved = path.resolve()
    if base_dir is None:
        return resolved.name
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return resolved.name


def _entry_list_from_root(root: Dict[str, Any]) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    for key in ("items", "images", "entries"):
        node = root.get(key)
        if isinstance(node, list):
            return [(None, dict(item)) for item in node if isinstance(item, dict)]
        if isinstance(node, dict):
            rows: List[Tuple[Optional[str], Dict[str, Any]]] = []
            for image_key, payload in node.items():
                if isinstance(payload, dict):
                    rows.append((str(image_key), dict(payload)))
            return rows

    rows = []
    for image_key, payload in root.items():
        if image_key in _RESERVED_TOP_LEVEL_KEYS or not isinstance(payload, dict):
            continue
        rows.append((str(image_key), dict(payload)))
    return rows


def _normalize_entry(payload: Dict[str, Any], manifest_key: Optional[str]) -> Optional[Dict[str, Any]]:
    image_ref = _first_text(
        payload.get("input_relative_path"),
        payload.get("image"),
        payload.get("source_image"),
        payload.get("relative_path"),
        payload.get("path"),
        payload.get("file"),
        payload.get("image_name"),
        manifest_key,
    )
    if not image_ref:
        return None

    view_expected = _first_text(
        payload.get("view_expected"),
        payload.get("intended_view"),
        payload.get("prompt_view"),
        payload.get("view"),
    )
    view_expected_family = _first_text(
        payload.get("view_expected_family"),
        payload.get("intended_lane_family"),
        payload.get("lane_family"),
    )
    view_expected_center_deg = payload.get("view_expected_center_deg", payload.get("intended_view_center_deg"))

    normalized = dict(payload)
    normalized["input_relative_path"] = str(image_ref).replace("\\", "/")
    normalized["image_name"] = Path(str(image_ref)).name
    if view_expected:
        normalized["view_expected"] = view_expected
    if view_expected_family:
        normalized["view_expected_family"] = view_expected_family
    if view_expected_center_deg is not None:
        normalized["view_expected_center_deg"] = view_expected_center_deg
    normalized["_manifest_match_key"] = _normalize_key(image_ref)
    normalized["_manifest_match_name"] = _normalize_key(Path(str(image_ref)).name)
    normalized["_manifest_source"] = "input_manifest"
    normalized["_manifest_entry_id"] = _first_text(
        payload.get("entry_id"),
        payload.get("id"),
        manifest_key,
        image_ref,
    )
    return normalized


def resolve_input_manifest_path(input_dir: Path, manifest_path: Optional[Path] = None) -> Optional[Path]:
    if manifest_path is not None:
        if manifest_path.is_absolute():
            resolved = manifest_path.resolve()
        else:
            direct = manifest_path.resolve()
            if manifest_path.parent != Path("."):
                resolved = direct
            else:
                resolved = direct if direct.exists() else (input_dir / manifest_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"input manifest does not exist: {resolved}")
        return resolved

    for candidate_name in _DEFAULT_MANIFEST_NAMES:
        candidate = (input_dir / candidate_name).resolve()
        if candidate.exists():
            return candidate
    return None


def load_input_manifest_index(
    input_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved = resolve_input_manifest_path(input_dir, manifest_path)
    if resolved is None:
        return {
            "available": False,
            "path": None,
            "entries": [],
            "path_index": {},
            "name_index": {},
            "summary": {
                "available": False,
                "path": None,
                "schema_version": "",
                "entry_count": 0,
                "path_matchable_count": 0,
                "unique_name_count": 0,
            },
        }

    root = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError(f"input manifest must decode to a JSON object: {resolved}")

    raw_rows = _entry_list_from_root(root)
    entries: List[Dict[str, Any]] = []
    path_index: Dict[str, Dict[str, Any]] = {}
    name_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for manifest_key, payload in raw_rows:
        entry = _normalize_entry(payload, manifest_key)
        if entry is None:
            continue
        entries.append(entry)
        path_key = str(entry.get("_manifest_match_key") or "")
        if path_key:
            path_index[path_key] = entry
        name_key = str(entry.get("_manifest_match_name") or "")
        if name_key:
            name_candidates.setdefault(name_key, []).append(entry)

    name_index = {
        key: rows[0]
        for key, rows in name_candidates.items()
        if len(rows) == 1
    }
    return {
        "available": True,
        "path": str(resolved),
        "entries": entries,
        "path_index": path_index,
        "name_index": name_index,
        "summary": {
            "available": True,
            "path": str(resolved),
            "schema_version": str(root.get("schema_version") or root.get("version") or "").strip(),
            "entry_count": len(entries),
            "path_matchable_count": len(path_index),
            "unique_name_count": len(name_index),
        },
    }


def resolve_input_manifest_entry(
    image_path: Path,
    input_dir: Path,
    manifest_index: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(manifest_index, dict) or not manifest_index.get("available"):
        return None
    path_key = _normalize_key(_safe_relative(image_path, input_dir))
    entry = (manifest_index.get("path_index") or {}).get(path_key)
    if isinstance(entry, dict):
        return copy.deepcopy(entry)
    name_key = _normalize_key(image_path.name)
    entry = (manifest_index.get("name_index") or {}).get(name_key)
    if isinstance(entry, dict):
        return copy.deepcopy(entry)
    return None


def required_prompt_intent_fields() -> List[str]:
    return list(_REQUIRED_PROMPT_INTENT_FIELDS)


def _default_manifest_output_path(input_dir: Path) -> Path:
    return (input_dir / "input_manifest.json").resolve()


def _resolve_manifest_output_path(input_dir: Path, manifest_path: Optional[Path]) -> Path:
    if manifest_path is None:
        return _default_manifest_output_path(input_dir)
    if manifest_path.is_absolute():
        return manifest_path.resolve()
    direct = manifest_path.resolve()
    if manifest_path.parent != Path(".") or direct.exists():
        return direct
    return (input_dir / manifest_path).resolve()


def _entry_template_from_image(
    image_path: Path,
    input_dir: Path,
    existing_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed_entry = dict(existing_entry) if isinstance(existing_entry, dict) else {}
    meta = parse_collection_metadata(image_path, input_dir, manifest_entry=seed_entry if seed_entry else None)
    template = {
        "image": image_path.name,
        "input_relative_path": meta.get("input_relative_path"),
        "prompt_id": seed_entry.get("prompt_id", ""),
        "seed": seed_entry.get("seed"),
        "seed_unavailable_reason": seed_entry.get("seed_unavailable_reason", ""),
        "anchor_source": seed_entry.get("anchor_source", ""),
        "intended_view": seed_entry.get("view_expected") or meta.get("view_expected") or "",
        "intended_lane_family": seed_entry.get("view_expected_family") or meta.get("view_expected_family") or "",
        "intended_view_center_deg": seed_entry.get("view_expected_center_deg", meta.get("view_expected_center_deg")),
        "generator_name": seed_entry.get("generator_name", ""),
        "generator_version": seed_entry.get("generator_version", ""),
        "prompt_pack": seed_entry.get("prompt_pack", ""),
        "notes": seed_entry.get("notes", ""),
    }
    for key in (
        "prompt_id",
        "seed_unavailable_reason",
        "anchor_source",
        "generator_name",
        "generator_version",
        "prompt_pack",
        "notes",
    ):
        if template.get(key) is None:
            template[key] = ""
    return template


def create_or_update_input_manifest(
    input_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    output_path = _resolve_manifest_output_path(input_dir, manifest_path)
    existing_index = load_input_manifest_index(input_dir, output_path) if output_path.exists() else {
        "available": False,
        "entries": [],
        "path_index": {},
        "name_index": {},
        "summary": {
            "available": False,
        },
    }
    images = sorted(
        [
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
    ) if input_dir.exists() else []

    items: List[Dict[str, Any]] = []
    matched_existing = 0
    for image_path in images:
        existing_entry = resolve_input_manifest_entry(image_path, input_dir, existing_index)
        if existing_entry is not None:
            matched_existing += 1
        items.append(_entry_template_from_image(image_path, input_dir, existing_entry))

    payload = {
        "schema_version": "input_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_name": "",
        "generator_version": "",
        "notes": "Fill prompt_id, seed or seed_unavailable_reason, anchor_source, and intended_view for generated batches.",
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "ok",
        "path": str(output_path),
        "schema_version": payload["schema_version"],
        "item_count": len(items),
        "matched_existing_count": matched_existing,
        "required_fields": required_prompt_intent_fields(),
    }


def _coerce_seed_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return text


def _field_coverage(items: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(items))
    return {
        "prompt_id": float(sum(1 for item in items if _first_text(item.get("prompt_id")) is not None) / total),
        "seed": float(sum(1 for item in items if _seed_or_unavailable_ready(item)) / total),
        "seed_available": float(sum(1 for item in items if item.get("seed") is not None) / total),
        "seed_unavailable_reason": float(
            sum(1 for item in items if _first_text(item.get("seed_unavailable_reason")) is not None) / total
        ),
        "anchor_source": float(sum(1 for item in items if _first_text(item.get("anchor_source")) is not None) / total),
        "intended_view": float(sum(1 for item in items if _first_text(item.get("intended_view")) is not None) / total),
    }


def fill_input_manifest_defaults(
    input_dir: Path,
    manifest_path: Optional[Path] = None,
    *,
    prompt_id: Optional[str] = None,
    seed: Any = None,
    seed_unavailable_reason: Optional[str] = None,
    anchor_source: Optional[str] = None,
    generator_name: Optional[str] = None,
    generator_version: Optional[str] = None,
    prompt_pack: Optional[str] = None,
    note: Optional[str] = None,
    missing_only: bool = True,
) -> Dict[str, Any]:
    output_path = _resolve_manifest_output_path(input_dir, manifest_path)
    if not output_path.exists():
        create_or_update_input_manifest(input_dir=input_dir, manifest_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input manifest must decode to a JSON object: {output_path}")
    items_node = payload.get("items")
    if not isinstance(items_node, list):
        raise ValueError(f"input manifest must contain an 'items' list: {output_path}")

    defaults: Dict[str, Any] = {}
    if prompt_id is not None:
        defaults["prompt_id"] = str(prompt_id).strip()
    if seed is not None:
        defaults["seed"] = _coerce_seed_value(seed)
    if seed_unavailable_reason is not None:
        defaults["seed_unavailable_reason"] = str(seed_unavailable_reason).strip()
    if anchor_source is not None:
        defaults["anchor_source"] = str(anchor_source).strip()
    if generator_name is not None:
        defaults["generator_name"] = str(generator_name).strip()
    if generator_version is not None:
        defaults["generator_version"] = str(generator_version).strip()
    if prompt_pack is not None:
        defaults["prompt_pack"] = str(prompt_pack).strip()

    if not defaults and note is None:
        raise ValueError("fill_input_manifest_defaults requires at least one manifest field override or note")

    items: List[Dict[str, Any]] = []
    updated_item_count = 0
    updated_field_count = 0
    for raw_item in items_node:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item_changed = False
        for field_name, field_value in defaults.items():
            if missing_only:
                existing_value = item.get(field_name)
                has_value = existing_value is not None and str(existing_value).strip() != ""
                if has_value:
                    continue
            if item.get(field_name) != field_value:
                item[field_name] = field_value
                updated_field_count += 1
                item_changed = True
        if note is not None:
            note_text = str(note).strip()
            if note_text:
                current_note = str(item.get("notes") or "").strip()
                if current_note:
                    if note_text not in current_note:
                        item["notes"] = f"{current_note}; {note_text}"
                        updated_field_count += 1
                        item_changed = True
                else:
                    item["notes"] = note_text
                    updated_field_count += 1
                    item_changed = True
        if item_changed:
            updated_item_count += 1
        items.append(item)

    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["items"] = items
    if generator_name is not None and (not missing_only or not _first_text(payload.get("generator_name"))):
        payload["generator_name"] = str(generator_name).strip()
    if generator_version is not None and (not missing_only or not _first_text(payload.get("generator_version"))):
        payload["generator_version"] = str(generator_version).strip()
    if note is not None:
        note_text = str(note).strip()
        if note_text:
            top_note = str(payload.get("notes") or "").strip()
            if top_note:
                if note_text not in top_note:
                    payload["notes"] = f"{top_note}; {note_text}"
            else:
                payload["notes"] = note_text

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage = _field_coverage(items)
    return {
        "status": "ok",
        "path": str(output_path),
        "schema_version": str(payload.get("schema_version") or ""),
        "item_count": len(items),
        "updated_item_count": updated_item_count,
        "updated_field_count": updated_field_count,
        "missing_only": bool(missing_only),
        "applied_defaults": defaults,
        "required_fields": required_prompt_intent_fields(),
        "required_field_coverage": coverage,
    }


def _build_manifest_item_index(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for item in items:
        image_name = _first_text(item.get("image"), item.get("input_relative_path"))
        if image_name is None:
            continue
        index[_normalize_key(Path(image_name).name)] = item
        index[_normalize_key(image_name)] = item
    return index


def merge_input_manifest_item_metadata(
    input_dir: Path,
    metadata_file: Path,
    manifest_path: Optional[Path] = None,
    *,
    missing_only: bool = True,
) -> Dict[str, Any]:
    output_path = _resolve_manifest_output_path(input_dir, manifest_path)
    if not output_path.exists():
        create_or_update_input_manifest(input_dir=input_dir, manifest_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"input manifest must decode to a JSON object: {output_path}")
    items_node = payload.get("items")
    if not isinstance(items_node, list):
        raise ValueError(f"input manifest must contain an 'items' list: {output_path}")

    metadata_payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(metadata_payload, dict):
        raise ValueError(f"metadata file must decode to a JSON object: {metadata_file}")
    raw_items = metadata_payload.get("items", metadata_payload)
    if not isinstance(raw_items, dict):
        raise ValueError("metadata file must be an object keyed by image name, or contain an 'items' object")

    items: List[Dict[str, Any]] = [dict(item) for item in items_node if isinstance(item, dict)]
    item_index = _build_manifest_item_index(items)
    matched_item_count = 0
    updated_item_count = 0
    updated_field_count = 0
    unmatched_keys: List[str] = []
    for raw_key, raw_update in raw_items.items():
        if not isinstance(raw_update, dict):
            continue
        item = item_index.get(_normalize_key(raw_key))
        if item is None:
            unmatched_keys.append(str(raw_key))
            continue
        matched_item_count += 1
        item_changed = False
        for field_name, field_value in raw_update.items():
            if field_name in {"image", "input_relative_path"}:
                continue
            if missing_only:
                existing_value = item.get(field_name)
                has_value = existing_value is not None and str(existing_value).strip() != ""
                if has_value:
                    continue
            normalized_value = _coerce_seed_value(field_value) if field_name == "seed" else field_value
            if item.get(field_name) != normalized_value:
                item[field_name] = normalized_value
                updated_field_count += 1
                item_changed = True
        if item_changed:
            updated_item_count += 1

    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["items"] = items
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    coverage = _field_coverage(items)
    return {
        "status": "ok",
        "path": str(output_path),
        "schema_version": str(payload.get("schema_version") or ""),
        "item_count": len(items),
        "matched_item_count": matched_item_count,
        "updated_item_count": updated_item_count,
        "updated_field_count": updated_field_count,
        "missing_only": bool(missing_only),
        "metadata_file": str(metadata_file),
        "unmatched_keys": unmatched_keys,
        "required_fields": required_prompt_intent_fields(),
        "required_field_coverage": coverage,
    }
