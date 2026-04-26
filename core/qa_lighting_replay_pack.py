from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .qa_input_manifest import create_or_update_input_manifest

_REPLAY_ROOT = Path("input_replay") / "lighting"
_DOC_PATH = Path("docs") / "27_lighting_replay_pack.md"

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

_VARIANT_SPECS: List[Dict[str, Any]] = [
    {
        "key": "neutral_base",
        "label": "neutral base",
        "control_bucket": "neutral",
        "expected_delta": "baseline_even_lighting",
        "capture_hint": "Use normal even lighting as the control baseline under the same lane and outfit class.",
    },
    {
        "key": "bright_exposure",
        "label": "bright exposure",
        "control_bucket": "exposure_up",
        "expected_delta": "brighter_but_not_overexposed",
        "capture_hint": "Make the frame brighter without clipping face or leg detail.",
    },
    {
        "key": "dim_exposure",
        "label": "dim exposure",
        "control_bucket": "exposure_down",
        "expected_delta": "dimmer_but_detail_visible",
        "capture_hint": "Make the frame dimmer while keeping facial and leg structure visible.",
    },
    {
        "key": "warm_cast",
        "label": "warm cast",
        "control_bucket": "temperature_warm",
        "expected_delta": "warm_white_balance_shift",
        "capture_hint": "Allow a mild warm shift without damaging identity or garment boundaries.",
    },
    {
        "key": "cool_cast",
        "label": "cool cast",
        "control_bucket": "temperature_cool",
        "expected_delta": "cool_white_balance_shift",
        "capture_hint": "Allow a mild cool shift without making the person look different.",
    },
]


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


def _append_note(existing: Any, note: str) -> str:
    current = str(existing or "").strip()
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _variant_dir(base_dir: Path, lane_key: str, variant_key: str) -> Path:
    return (base_dir / _REPLAY_ROOT / lane_key / variant_key).resolve()


def _manifest_metadata_template(
    manifest_path: Path,
    manifest_payload: Dict[str, Any],
    *,
    lane_key: str,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    items: Dict[str, Dict[str, Any]] = {}
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
            "notes": _append_note(
                raw_item.get("notes"),
                f"lighting_replay_lane={lane_key}; lighting_variant={variant['key']}",
            ),
        }
    return {
        "schema_version": "input_manifest_metadata_patch_v1",
        "source_manifest": str(manifest_path),
        "lighting_replay_lane": lane_key,
        "lighting_variant": variant["key"],
        "items": items,
    }


def _write_replay_root_readme(base_dir: Path) -> Path:
    replay_root = (base_dir / _REPLAY_ROOT).resolve()
    readme_path = replay_root / "README.md"
    variant_rows = "\n".join(
        f"- `{variant['key']}`: {variant['label']} - {variant['capture_hint']}"
        for variant in _VARIANT_SPECS
    )
    text = (
        "# Lighting Replay Pack\n\n"
        "This folder is for controlled lighting replay only.\n\n"
        "Rules:\n"
        "- Keep the same lane inside one lane folder.\n"
        "- Do not mix lighting validation images into `input_split/`.\n"
        "- Change lighting only. Do not intentionally change identity, body structure, outfit class, or framing.\n"
        "- When more images are added, rerun `prepare_lighting_replay_pack` to refresh manifests and metadata templates.\n\n"
        "Variants:\n"
        f"{variant_rows}\n\n"
        f"Operator doc: `{_DOC_PATH.as_posix()}`\n"
    )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _refresh_variant_manifest(
    variant_dir: Path,
    *,
    lane_key: str,
    lane_spec: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    manifest_result = create_or_update_input_manifest(variant_dir)
    manifest_path = Path(str(manifest_result.get("path") or variant_dir / "input_manifest.json")).resolve()
    payload = _load_json(manifest_path)
    items = payload.get("items") if isinstance(payload.get("items"), list) else []

    payload["schema_version"] = "input_manifest_v1"
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["lighting_replay_pack"] = "lighting_replay_pack_v1"
    payload["lighting_replay_lane"] = lane_key
    payload["lighting_variant"] = variant["key"]
    payload["lighting_variant_label"] = variant["label"]
    payload["lighting_control_bucket"] = variant["control_bucket"]
    payload["lighting_replay_expected_delta"] = variant["expected_delta"]
    payload["notes"] = (
        "Lighting replay pack. Fill prompt_id / seed or seed_unavailable_reason / "
        "anchor_source after images are added."
    )

    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        if not _safe_text(raw_item.get("intended_view")):
            raw_item["intended_view"] = lane_spec["intended_view"]
        if not _safe_text(raw_item.get("intended_lane_family")):
            raw_item["intended_lane_family"] = lane_spec["intended_lane_family"]
        if raw_item.get("intended_view_center_deg") is None:
            raw_item["intended_view_center_deg"] = lane_spec["intended_view_center_deg"]
        raw_item["lighting_variant"] = variant["key"]
        raw_item["lighting_variant_label"] = variant["label"]
        raw_item["lighting_control_bucket"] = variant["control_bucket"]
        raw_item["lighting_replay_expected_delta"] = variant["expected_delta"]
        raw_item["notes"] = _append_note(
            raw_item.get("notes"),
            f"lighting_replay_lane={lane_key}; lighting_variant={variant['key']}",
        )

    payload["items"] = items
    _write_json(manifest_path, payload)

    metadata_template_path = (variant_dir / "_input_manifest_metadata_template.json").resolve()
    metadata_template = _manifest_metadata_template(
        manifest_path,
        payload,
        lane_key=lane_key,
        variant=variant,
    )
    _write_json(metadata_template_path, metadata_template)

    image_count = len(
        [
            path
            for path in variant_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
    )
    return {
        "variant_key": variant["key"],
        "label": variant["label"],
        "variant_dir": str(variant_dir),
        "manifest_path": str(manifest_path),
        "metadata_template_path": str(metadata_template_path),
        "image_count": image_count,
        "manifest_item_count": len(items),
        "recommended_min_images": 6,
        "recommended_target_images": 8,
        "recommended_max_images": 12,
        "capture_hint": variant["capture_hint"],
        "expected_delta": variant["expected_delta"],
    }


def build_lighting_replay_pack(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    replay_root = (base_dir / _REPLAY_ROOT).resolve()
    readme_path = _write_replay_root_readme(base_dir)

    lanes: List[Dict[str, Any]] = []
    total_variant_dirs = 0
    total_images = 0
    for lane_key, lane_spec in _LANE_SPECS.items():
        source_reference_dir = (base_dir / lane_spec["source_reference_dir"]).resolve()
        variant_rows: List[Dict[str, Any]] = []
        for variant in _VARIANT_SPECS:
            variant_dir = _variant_dir(base_dir, lane_key, variant["key"])
            variant_dir.mkdir(parents=True, exist_ok=True)
            row = _refresh_variant_manifest(
                variant_dir,
                lane_key=lane_key,
                lane_spec=lane_spec,
                variant=variant,
            )
            variant_rows.append(row)
            total_variant_dirs += 1
            total_images += int(row["image_count"])
        lanes.append(
            {
                "lane": lane_key,
                "target_profile": lane_spec["target_profile"],
                "intended_view": lane_spec["intended_view"],
                "intended_lane_family": lane_spec["intended_lane_family"],
                "intended_view_center_deg": lane_spec["intended_view_center_deg"],
                "source_reference_dir": str(source_reference_dir),
                "variant_count": len(variant_rows),
                "variants": variant_rows,
            }
        )

    payload = {
        "schema_version": "lighting_replay_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prepare a controlled lighting replay pack before promoting lighting invariance to release-grade evidence.",
        "replay_root": str(replay_root),
        "replay_root_readme": str(readme_path),
        "operator_doc": str((base_dir / _DOC_PATH).resolve()),
        "lane_count": len(lanes),
        "variant_dir_count": total_variant_dirs,
        "total_current_images": total_images,
        "collection_rules": [
            "Keep the same lane family inside one lane folder.",
            "Change lighting only; do not intentionally change identity, body structure, outfit class, or framing.",
            "Do not mix these replay images back into input_split/ front or three_quarter clean lanes.",
            "Fill prompt_id and anchor_source only after the actual batch provenance is confirmed.",
        ],
        "operator_defaults": {
            "shared_seed_unavailable_reason_example": "nano_banana_seed_not_exposed",
            "shared_anchor_source_example": "A-Core_01_0deg_MASTER + Task-63987060-116-1",
            "do_not_fabricate_prompt_id": True,
            "do_not_mix_angle_and_lighting_in_same_variant": True,
            "rerun_prepare_lighting_replay_pack_after_adding_images": True,
        },
        "after_images_are_added": [
            "Rerun prepare_lighting_replay_pack to refresh variant manifests and metadata templates.",
            "Use fill_input_manifest_defaults or merge_input_manifest_metadata to complete prompt_id / seed or seed_unavailable_reason / anchor_source.",
            "Run preflight_batch and shot_review separately per variant directory.",
            "Judge lighting invariance from controlled variant replays, not from mixed production batches.",
        ],
        "lanes": lanes,
    }
    _write_json(output_file, payload)
    return payload
