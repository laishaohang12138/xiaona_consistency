from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_MANIFEST_NAMES = (
    "input_manifest.json",
    "_input_manifest.json",
    "manifest.json",
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
