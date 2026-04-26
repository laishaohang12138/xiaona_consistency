from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_input_manifest import create_or_update_input_manifest

_REPLAY_ROOT = Path("input_replay") / "outer"
_DOC_PATH = Path("docs") / "28_outer_replay_pack.md"
_PROMPT_PACK_ROOT = Path("prompts") / "outer"

_LANE_SPECS: Dict[str, Dict[str, Any]] = {
    "front": {
        "target_profile": "body_gold_fullbody",
        "intended_view": "front",
        "intended_lane_family": "front",
        "intended_view_center_deg": 0,
        "source_reference_dir": Path("input_split") / "front",
    },
    "three_quarter": {
        "target_profile": "body_gold_threequarter_review",
        "intended_view": "three_quarter",
        "intended_lane_family": "three_quarter",
        "intended_view_center_deg": 45,
        "source_reference_dir": Path("input_split") / "three_quarter",
    },
}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _append_note(existing: Any, note: str) -> str:
    current = _safe_text(existing)
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_summary(base_dir: Path) -> Dict[str, Any]:
    manifest_file = (base_dir / _PROMPT_PACK_ROOT / "manifest.yaml").resolve()
    if not manifest_file.exists():
        return {
            "manifest_available": False,
            "manifest_file": str(manifest_file),
            "asset_id": "",
            "status": "",
            "review_only_replay_allowed": False,
        }
    asset_id = ""
    status = ""
    review_only_replay_allowed = False
    for raw_line in manifest_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("id:") and not asset_id:
            asset_id = line.split(":", 1)[1].strip()
        elif line.startswith("status:") and not status:
            status = line.split(":", 1)[1].strip()
        elif line.startswith("review_only_replay_allowed:"):
            review_only_replay_allowed = line.split(":", 1)[1].strip().lower() == "true"
    return {
        "manifest_available": True,
        "manifest_file": str(manifest_file),
        "asset_id": asset_id,
        "status": status,
        "review_only_replay_allowed": review_only_replay_allowed,
    }


def _prompt_root(base_dir: Path) -> Optional[Path]:
    prompt_root = (base_dir / _PROMPT_PACK_ROOT).resolve()
    if not prompt_root.exists():
        return None
    candidates = sorted(
        path
        for path in prompt_root.iterdir()
        if path.is_dir() and path.name.startswith("assembled_shortlist")
    )
    return candidates[0] if candidates else None


def _prompt_row_from_file(prompt_file: Path, *, base_dir: Path, prompt_pack_id: str) -> Optional[Dict[str, Any]]:
    stem = prompt_file.stem.lower()
    if not stem.startswith("assembled_ot_"):
        return None
    remainder = stem[len("assembled_ot_") :]
    if "_" not in remainder:
        return None
    prompt_code, descriptor = remainder.split("_", 1)

    lane = ""
    family = ""
    scene = ""
    if "_three_quarter_" in descriptor:
        family, scene_tail = descriptor.split("_three_quarter_", 1)
        lane = "three_quarter"
        scene = f"three_quarter_{scene_tail}"
    elif "_front_" in descriptor:
        family, scene_tail = descriptor.split("_front_", 1)
        lane = "front"
        scene = f"front_{scene_tail}"
    else:
        return None

    lane_spec = _LANE_SPECS.get(lane)
    if lane_spec is None:
        return None

    directory_name = stem[len("assembled_") :]
    prompt_id = f"OT-{prompt_code.upper()}"
    return {
        "prompt_id": prompt_id,
        "prompt_code": prompt_code,
        "lane": lane,
        "family": family,
        "scene": scene,
        "directory_name": directory_name,
        "prompt_file": str(prompt_file.resolve()),
        "prompt_file_rel": str(prompt_file.resolve().relative_to(base_dir.resolve())).replace("\\", "/"),
        "prompt_pack_id": prompt_pack_id,
        "target_profile": lane_spec["target_profile"],
        "intended_view": lane_spec["intended_view"],
        "intended_lane_family": lane_spec["intended_lane_family"],
        "intended_view_center_deg": lane_spec["intended_view_center_deg"],
        "source_reference_dir": str((base_dir / lane_spec["source_reference_dir"]).resolve()),
    }


def _prompt_rows(base_dir: Path) -> List[Dict[str, Any]]:
    manifest_summary = _manifest_summary(base_dir)
    prompt_pack_id = _safe_text(manifest_summary.get("asset_id")) or "outer_shortlist"
    prompt_root = _prompt_root(base_dir)
    if prompt_root is None:
        return []
    rows: List[Dict[str, Any]] = []
    for prompt_file in sorted(prompt_root.rglob("*.txt")):
        row = _prompt_row_from_file(prompt_file, base_dir=base_dir, prompt_pack_id=prompt_pack_id)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: (_safe_text(item.get("lane")), _safe_text(item.get("family")), _safe_text(item.get("prompt_id"))))
    return rows


def _leaf_dir(base_dir: Path, row: Dict[str, Any]) -> Path:
    return (
        base_dir
        / _REPLAY_ROOT
        / _safe_text(row.get("lane"))
        / _safe_text(row.get("family"))
        / _safe_text(row.get("directory_name"))
    ).resolve()


def _metadata_template(manifest_path: Path, payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    items: Dict[str, Dict[str, Any]] = {}
    note = (
        f"outer_replay_lane={row['lane']}; outer_family={row['family']}; "
        f"outer_prompt_id={row['prompt_id']}"
    )
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        image_name = _safe_text(raw_item.get("image") or raw_item.get("input_relative_path"))
        if not image_name:
            continue
        items[Path(image_name).name] = {
            "prompt_id": row["prompt_id"],
            "seed": raw_item.get("seed"),
            "seed_unavailable_reason": _safe_text(raw_item.get("seed_unavailable_reason")),
            "anchor_source": _safe_text(raw_item.get("anchor_source")),
            "prompt_pack": row["prompt_pack_id"],
            "generator_name": _safe_text(raw_item.get("generator_name")),
            "notes": _append_note(raw_item.get("notes"), note),
        }
    return {
        "schema_version": "input_manifest_metadata_patch_v1",
        "source_manifest": str(manifest_path),
        "outer_replay_lane": row["lane"],
        "outer_family": row["family"],
        "outer_prompt_id": row["prompt_id"],
        "outer_prompt_file": row["prompt_file_rel"],
        "items": items,
    }


def _write_replay_root_readme(base_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    replay_root = (base_dir / _REPLAY_ROOT).resolve()
    readme_path = replay_root / "README.md"
    family_names = sorted({_safe_text(row.get("family")) for row in rows if _safe_text(row.get("family"))})
    text = (
        "# OUTER Replay Pack\n\n"
        "This folder is for review-only outerwear and occlusion replay.\n\n"
        "Rules:\n"
        "- Keep one prompt leaf for one outerwear prompt only.\n"
        "- Do not mix lane families between front and three_quarter.\n"
        "- Change outerwear / occlusion only. Do not intentionally change identity, body structure, lighting, or framing.\n"
        "- Do not mix these replay images back into input_split/ clean lanes.\n"
        "- Rerun `prepare_outer_replay_pack` after adding images so manifests stay current.\n\n"
        f"Families: {', '.join(family_names)}\n\n"
        f"Operator doc: `{_DOC_PATH.as_posix()}`\n"
    )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _refresh_prompt_manifest(leaf_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    manifest_result = create_or_update_input_manifest(leaf_dir)
    manifest_path = Path(str(manifest_result.get("path") or leaf_dir / "input_manifest.json")).resolve()
    payload = _load_json(manifest_path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []

    payload["schema_version"] = "input_manifest_v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["outer_replay_pack"] = "outer_replay_pack_v1"
    payload["outer_scope"] = "OUTER"
    payload["outer_family"] = row["family"]
    payload["outer_prompt_id"] = row["prompt_id"]
    payload["outer_prompt_file"] = row["prompt_file_rel"]
    payload["outer_prompt_pack"] = row["prompt_pack_id"]
    payload["intended_view"] = row["intended_view"]
    payload["intended_lane_family"] = row["intended_lane_family"]
    payload["intended_view_center_deg"] = row["intended_view_center_deg"]
    payload["notes"] = (
        "OUTER replay pack. This directory is for review-only clothing or occlusion replay, not training admission."
    )

    note = (
        f"outer_replay_lane={row['lane']}; outer_family={row['family']}; "
        f"outer_prompt_id={row['prompt_id']}"
    )
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        raw_item["prompt_id"] = row["prompt_id"]
        raw_item["prompt_pack"] = row["prompt_pack_id"]
        if not _safe_text(raw_item.get("intended_view")):
            raw_item["intended_view"] = row["intended_view"]
        if not _safe_text(raw_item.get("intended_lane_family")):
            raw_item["intended_lane_family"] = row["intended_lane_family"]
        if raw_item.get("intended_view_center_deg") is None:
            raw_item["intended_view_center_deg"] = row["intended_view_center_deg"]
        raw_item["outer_scope"] = "OUTER"
        raw_item["outer_family"] = row["family"]
        raw_item["outer_prompt_id"] = row["prompt_id"]
        raw_item["outer_prompt_file"] = row["prompt_file_rel"]
        raw_item["notes"] = _append_note(raw_item.get("notes"), note)

    payload["items"] = items
    _write_json(manifest_path, payload)

    metadata_template_path = (leaf_dir / "_input_manifest_metadata_template.json").resolve()
    _write_json(metadata_template_path, _metadata_template(manifest_path, payload, row))

    image_count = len(
        [
            path
            for path in leaf_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
    )
    return {
        "prompt_id": row["prompt_id"],
        "family": row["family"],
        "lane": row["lane"],
        "variant_dir": str(leaf_dir),
        "manifest_path": str(manifest_path),
        "metadata_template_path": str(metadata_template_path),
        "prompt_file": row["prompt_file_rel"],
        "image_count": image_count,
        "manifest_item_count": len(items),
        "recommended_min_images": 4,
        "recommended_target_images": 6,
        "recommended_max_images": 8,
    }


def build_outer_replay_pack(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    manifest_summary = _manifest_summary(base_dir)
    rows = _prompt_rows(base_dir)
    readme_path = _write_replay_root_readme(base_dir, rows)

    lanes: List[Dict[str, Any]] = []
    total_prompt_dirs = 0
    total_images = 0
    for lane_key, lane_spec in _LANE_SPECS.items():
        lane_rows = [row for row in rows if _safe_text(row.get("lane")) == lane_key]
        families: Dict[str, List[Dict[str, Any]]] = {}
        for row in lane_rows:
            families.setdefault(_safe_text(row.get("family")), []).append(row)

        family_payloads: List[Dict[str, Any]] = []
        for family_name in sorted(families):
            prompt_payloads: List[Dict[str, Any]] = []
            for row in families[family_name]:
                leaf_dir = _leaf_dir(base_dir, row)
                leaf_dir.mkdir(parents=True, exist_ok=True)
                prompt_payload = _refresh_prompt_manifest(leaf_dir, row)
                prompt_payloads.append(prompt_payload)
                total_prompt_dirs += 1
                total_images += int(prompt_payload.get("image_count") or 0)
            family_payloads.append(
                {
                    "family": family_name,
                    "prompt_count": len(prompt_payloads),
                    "prompts": prompt_payloads,
                }
            )

        lanes.append(
            {
                "lane": lane_key,
                "target_profile": lane_spec["target_profile"],
                "intended_view": lane_spec["intended_view"],
                "intended_lane_family": lane_spec["intended_lane_family"],
                "intended_view_center_deg": lane_spec["intended_view_center_deg"],
                "source_reference_dir": str((base_dir / lane_spec["source_reference_dir"]).resolve()),
                "family_count": len(family_payloads),
                "prompt_count": sum(int(family.get("prompt_count") or 0) for family in family_payloads),
                "families": family_payloads,
            }
        )

    payload = {
        "schema_version": "outer_replay_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prepare a governed review-only OUTER replay pack before promoting clothing invariance to release-grade evidence.",
        "replay_root": str((base_dir / _REPLAY_ROOT).resolve()),
        "replay_root_readme": str(readme_path),
        "operator_doc": str((base_dir / _DOC_PATH).resolve()),
        "prompt_manifest_state": manifest_summary,
        "lane_count": len(lanes),
        "prompt_dir_count": total_prompt_dirs,
        "total_current_images": total_images,
        "collection_rules": [
            "Keep one outerwear prompt per prompt leaf directory.",
            "Keep the lane family fixed inside each replay leaf.",
            "Change outerwear or occlusion only; do not intentionally change identity, body structure, lighting, or framing.",
            "Do not mix OUTER replay images back into input_split clean lanes.",
        ],
        "operator_defaults": {
            "shared_seed_unavailable_reason_example": "nano_banana_seed_not_exposed",
            "shared_anchor_source_example": "A-Core_01_0deg_MASTER + Task-63987060-116-1",
            "prompt_id_is_prebound_from_directory": True,
            "do_not_fabricate_anchor_source": True,
            "rerun_prepare_outer_replay_pack_after_adding_images": True,
        },
        "after_images_are_added": [
            "Rerun prepare_outer_replay_pack to refresh manifests and metadata templates.",
            "Fill seed or seed_unavailable_reason and anchor_source for each prompt leaf after the actual batch provenance is confirmed.",
            "Run preflight_batch and shot_review separately per prompt leaf directory when you want controlled OUTER replay evidence.",
            "Judge clothing invariance from these controlled OUTER replays, not from mixed production batches.",
        ],
        "lanes": lanes,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
