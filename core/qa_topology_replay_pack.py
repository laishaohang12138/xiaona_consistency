from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_input_manifest import create_or_update_input_manifest
from .qa_io import atomic_write_json

_REPLAY_ROOT = Path("input_replay") / "topology"
_DOC_PATH = Path("docs") / "31_topology_replay_pack.md"

_VARIANT_SPECS: List[Dict[str, Any]] = [
    {
        "key": "side_left_profile",
        "lane": "side",
        "target_profile": "body_gold_side90_shadow",
        "intended_view": "side_90",
        "intended_lane_family": "side",
        "intended_view_center_deg": 90,
        "intended_view_side": "left",
        "pose_control_bucket": "strict_side_left_profile",
        "recommended_min_images": 8,
        "recommended_target_images": 12,
        "capture_hint": "Collect lane-pure left profile full-body frames with feet visible and stable torso depth.",
    },
    {
        "key": "side_right_profile",
        "lane": "side",
        "target_profile": "body_gold_side90_shadow",
        "intended_view": "side_90",
        "intended_lane_family": "side",
        "intended_view_center_deg": 90,
        "intended_view_side": "right",
        "pose_control_bucket": "strict_side_right_profile",
        "recommended_min_images": 8,
        "recommended_target_images": 12,
        "capture_hint": "Collect lane-pure right profile full-body frames with feet visible and stable neck-to-torso connection.",
    },
    {
        "key": "back180_neutral",
        "lane": "back",
        "target_profile": "body_gold_back180_shadow",
        "intended_view": "back_180",
        "intended_lane_family": "back",
        "intended_view_center_deg": 180,
        "intended_view_side": "center",
        "pose_control_bucket": "strict_back_neutral",
        "recommended_min_images": 8,
        "recommended_target_images": 12,
        "capture_hint": "Collect straight back-view full-body frames with readable shoulder, waist, hip, and leg contour.",
    },
    {
        "key": "back180_subtle_gait_shift",
        "lane": "back",
        "target_profile": "body_gold_back180_shadow",
        "intended_view": "back_180",
        "intended_lane_family": "back",
        "intended_view_center_deg": 180,
        "intended_view_side": "center",
        "pose_control_bucket": "back_subtle_gait_shift",
        "recommended_min_images": 8,
        "recommended_target_images": 12,
        "capture_hint": "Collect mild back-view gait or stance shifts while keeping body structure and framing stable.",
    },
]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _append_note(existing: Any, note: str) -> str:
    current = _safe_text(existing)
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"


def _has_images(path: Path) -> bool:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return any(child.is_file() and child.suffix.lower() in suffixes for child in path.iterdir()) if path.exists() else False


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _variant_dir(base_dir: Path, variant: Dict[str, Any]) -> Path:
    return (base_dir / _REPLAY_ROOT / _safe_text(variant.get("lane")) / _safe_text(variant.get("key"))).resolve()


def _metadata_template(
    manifest_path: Path,
    manifest_payload: Dict[str, Any],
    *,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    items: Dict[str, Dict[str, Any]] = {}
    note = f"topology_replay_lane={variant['lane']}; topology_variant={variant['key']}"
    for raw_item in manifest_payload.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        image_name = _safe_text(raw_item.get("image") or raw_item.get("input_relative_path"))
        if not image_name:
            continue
        items[Path(image_name).name] = {
            "prompt_id": _safe_text(raw_item.get("prompt_id")),
            "seed": raw_item.get("seed"),
            "seed_unavailable_reason": _safe_text(raw_item.get("seed_unavailable_reason")),
            "anchor_source": _safe_text(raw_item.get("anchor_source")),
            "prompt_pack": _safe_text(raw_item.get("prompt_pack")),
            "generator_name": _safe_text(raw_item.get("generator_name")),
            "notes": _append_note(raw_item.get("notes"), note),
        }
    return {
        "schema_version": "input_manifest_metadata_patch_v1",
        "source_manifest": str(manifest_path),
        "topology_replay_lane": variant["lane"],
        "topology_variant": variant["key"],
        "pose_control_bucket": variant["pose_control_bucket"],
        "items": items,
    }


def _write_replay_root_readme(base_dir: Path) -> Path:
    replay_root = (base_dir / _REPLAY_ROOT).resolve()
    readme_path = replay_root / "README.md"
    variant_rows = "\n".join(
        f"- `{variant['lane']}/{variant['key']}`: {variant['capture_hint']}"
        for variant in _VARIANT_SPECS
    )
    text = (
        "# Topology Replay Pack\n\n"
        "This folder is for review-only side/back topology validation.\n\n"
        "Rules:\n"
        "- Keep one pose/view variant inside one directory.\n"
        "- Keep full body and feet visible whenever possible.\n"
        "- Change view or mild gait/stance only; do not intentionally change identity, body structure, outfit class, or lighting.\n"
        "- Read `Task-63987060-116-1.png` through pose/gait-aware metrics, not as a rigid overlay.\n"
        "- Do not mix these replay images into front or three-quarter clean lanes.\n\n"
        "Variants:\n"
        f"{variant_rows}\n\n"
        f"Operator doc: `{_DOC_PATH.as_posix()}`\n"
    )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _refresh_variant_manifest(base_dir: Path, variant: Dict[str, Any]) -> Dict[str, Any]:
    variant_dir = _variant_dir(base_dir, variant)
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = variant_dir / "input_manifest.json"
    previous_payload = _load_json(manifest_file)
    previous_generated_at = _safe_text(previous_payload.get("generated_at_utc"))
    if previous_payload and not _has_images(variant_dir):
        manifest_path = manifest_file.resolve()
        payload = dict(previous_payload)
    else:
        manifest_result = create_or_update_input_manifest(variant_dir)
        manifest_path = Path(str(manifest_result.get("path") or manifest_file)).resolve()
        payload = _load_json(manifest_path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []

    payload["schema_version"] = "input_manifest_v1"
    payload["generated_at_utc"] = (
        previous_generated_at
        if previous_generated_at and not items
        else datetime.now(timezone.utc).isoformat()
    )
    payload["topology_replay_pack"] = "topology_replay_pack_v1"
    payload["topology_replay_lane"] = variant["lane"]
    payload["topology_variant"] = variant["key"]
    payload["pose_control_bucket"] = variant["pose_control_bucket"]
    payload["target_profile"] = variant["target_profile"]
    payload["intended_view"] = variant["intended_view"]
    payload["intended_lane_family"] = variant["intended_lane_family"]
    payload["intended_view_center_deg"] = variant["intended_view_center_deg"]
    payload["intended_view_side"] = variant["intended_view_side"]
    payload["face_truth_anchor"] = "A-Core_01_0deg_MASTER.png"
    payload["body_truth_anchor"] = "Task-63987060-116-1.png"
    payload["body_truth_read"] = "pose_gait_aware_absolute_116_1"
    payload["notes"] = (
        "Topology replay pack. This directory is for review-only side/back topology validation, "
        "not winner-bank freeze, parameter fitting, or final training admission."
    )

    note = f"topology_replay_lane={variant['lane']}; topology_variant={variant['key']}"
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        raw_item["topology_replay_lane"] = variant["lane"]
        raw_item["topology_variant"] = variant["key"]
        raw_item["pose_control_bucket"] = variant["pose_control_bucket"]
        raw_item["target_profile"] = variant["target_profile"]
        if not _safe_text(raw_item.get("intended_view")):
            raw_item["intended_view"] = variant["intended_view"]
        if not _safe_text(raw_item.get("intended_lane_family")):
            raw_item["intended_lane_family"] = variant["intended_lane_family"]
        if raw_item.get("intended_view_center_deg") is None:
            raw_item["intended_view_center_deg"] = variant["intended_view_center_deg"]
        raw_item["intended_view_side"] = variant["intended_view_side"]
        raw_item["face_truth_anchor"] = "A-Core_01_0deg_MASTER.png"
        raw_item["body_truth_anchor"] = "Task-63987060-116-1.png"
        raw_item["body_truth_read"] = "pose_gait_aware_absolute_116_1"
        raw_item["notes"] = _append_note(raw_item.get("notes"), note)

    payload["items"] = items
    if payload != previous_payload:
        _write_json(manifest_path, payload)

    metadata_template_path = (variant_dir / "_input_manifest_metadata_template.json").resolve()
    _write_json(metadata_template_path, _metadata_template(manifest_path, payload, variant=variant))

    image_count = len(
        [
            path
            for path in variant_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
    )
    return {
        "variant_key": variant["key"],
        "lane": variant["lane"],
        "target_profile": variant["target_profile"],
        "intended_view": variant["intended_view"],
        "intended_lane_family": variant["intended_lane_family"],
        "intended_view_center_deg": variant["intended_view_center_deg"],
        "intended_view_side": variant["intended_view_side"],
        "pose_control_bucket": variant["pose_control_bucket"],
        "variant_dir": str(variant_dir),
        "manifest_path": str(manifest_path),
        "metadata_template_path": str(metadata_template_path),
        "image_count": image_count,
        "manifest_item_count": len(items),
        "recommended_min_images": int(variant["recommended_min_images"]),
        "recommended_target_images": int(variant["recommended_target_images"]),
        "recommended_max_images": int(variant["recommended_target_images"]) + 4,
        "capture_hint": variant["capture_hint"],
    }


def build_topology_replay_pack(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    readme_path = _write_replay_root_readme(base_dir)
    variants = [_refresh_variant_manifest(base_dir, variant) for variant in _VARIANT_SPECS]
    lanes: List[Dict[str, Any]] = []
    for lane_name in sorted({str(row.get("lane")) for row in variants}):
        lane_variants = [row for row in variants if str(row.get("lane")) == lane_name]
        lanes.append(
            {
                "lane": lane_name,
                "variant_count": len(lane_variants),
                "total_current_images": sum(int(row.get("image_count") or 0) for row in lane_variants),
                "variants": lane_variants,
            }
        )

    payload = {
        "schema_version": "topology_replay_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prepare controlled side/back replay inputs for 3D topology and pose/gait-aware body truth validation.",
        "replay_root": str((base_dir / _REPLAY_ROOT).resolve()),
        "replay_root_readme": str(readme_path),
        "operator_doc": str((base_dir / _DOC_PATH).resolve()),
        "lane_count": len(lanes),
        "variant_dir_count": len(variants),
        "total_current_images": sum(int(row.get("image_count") or 0) for row in variants),
        "truth_policy": {
            "face_truth": "A-Core_01_0deg_MASTER.png",
            "body_truth": "Task-63987060-116-1.png",
            "body_truth_read": "pose_gait_aware_absolute_116_1",
        },
        "collection_rules": [
            "Keep one side/back pose variant per directory.",
            "Use segformer_body_truth_fusion through the profile default; do not override to a weaker provider.",
            "Do not intentionally change identity, body structure, outfit class, or lighting.",
            "Do not interpret pose/gait-sensitive deltas as body drift without topology review.",
        ],
        "operator_defaults": {
            "shared_seed_unavailable_reason_example": "nano_banana_seed_not_exposed",
            "shared_anchor_source_example": "A-Core_01_0deg_MASTER + Task-63987060-116-1",
            "do_not_fabricate_prompt_id": True,
            "rerun_prepare_topology_replay_pack_after_adding_images": True,
        },
        "after_images_are_added": [
            "Rerun prepare_topology_replay_pack to refresh manifests and metadata templates.",
            "Fill prompt_id / seed-or-unavailable / anchor_source from confirmed batch provenance.",
            "Run preflight_batch and shot_review separately per variant directory.",
            "Review topology and pose/gait reads before calling body drift.",
        ],
        "lanes": lanes,
    }
    _write_json(output_file, payload)
    return payload
