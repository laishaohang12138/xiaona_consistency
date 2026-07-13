from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_input_manifest import create_or_update_input_manifest
from .qa_io import atomic_write_json


_PROFILE_BY_LANE = {
    "front": "body_gold_fullbody",
    "three_quarter": "body_gold_threequarter_review",
    "side": "body_gold_side90_shadow",
    "back": "body_gold_back180_shadow",
}

_INTENDED_VIEW_BY_LANE = {
    "front": ("front", 0),
    "three_quarter": ("three_quarter", 45),
    "side": ("side_90", 90),
    "back": ("back_180", 180),
}


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


def _candidate_lane(item: Dict[str, Any]) -> Dict[str, Any]:
    lane = item.get("lane") or {}
    observed_lane = str(
        lane.get("observed_lane_family")
        or lane.get("view_lane_detail")
        or "unknown"
    ).strip() or "unknown"
    confidence = lane.get("view_lane_detail_confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None
    return {
        "image": str(item.get("image") or "").strip(),
        "record_key": str(item.get("record_key") or "").strip(),
        "observed_lane_family": observed_lane,
        "view_lane_detail": str(lane.get("view_lane_detail") or observed_lane).strip(),
        "view_lane_detail_confidence": confidence,
        "review_only_status_v2": str(item.get("review_only_status_v2") or "").strip(),
        "selection_score": item.get("selection_score"),
    }


def _items_from_review_packet(review_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = review_packet.get("items")
    if not isinstance(items, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = _candidate_lane(item)
        if row["image"]:
            rows.append(row)
    return rows


def _items_from_preflight(preflight: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = preflight.get("items")
    if not isinstance(items, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image = str(item.get("image") or "").strip()
        if not image:
            continue
        confidence = item.get("route_confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None
        rows.append(
            {
                "image": image,
                "record_key": image,
                "observed_lane_family": str(item.get("observed_lane_family") or "unknown").strip() or "unknown",
                "view_lane_detail": str(item.get("observed_lane_detail") or item.get("observed_lane_family") or "unknown").strip() or "unknown",
                "view_lane_detail_confidence": confidence,
                "review_only_status_v2": "",
                "selection_score": None,
            }
        )
    return rows


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lane = str(row.get("observed_lane_family") or "unknown").strip() or "unknown"
        groups.setdefault(lane, []).append(row)

    output: List[Dict[str, Any]] = []
    for lane, lane_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        lane_rows_sorted = sorted(
            lane_rows,
            key=lambda item: (
                -1.0 if item.get("view_lane_detail_confidence") is None else -float(item["view_lane_detail_confidence"]),
                str(item.get("image") or ""),
            ),
        )
        file_names = [str(item.get("image") or "").strip() for item in lane_rows_sorted if str(item.get("image") or "").strip()]
        output.append(
            {
                "lane_family": lane,
                "count": len(file_names),
                "share": round(len(file_names) / max(len(rows), 1), 4),
                "suggested_profile": _PROFILE_BY_LANE.get(lane, ""),
                "suggested_input_dir": f"input_split/{lane}",
                "copy_command_template": (
                    f'New-Item -ItemType Directory ".\\input_split\\{lane}" -Force; '
                    f'Copy-Item ".\\input\\<image>" ".\\input_split\\{lane}\\"'
                    if lane in _PROFILE_BY_LANE
                    else ""
                ),
                "files": file_names,
                "examples": lane_rows_sorted[:5],
            }
        )
    return output


def build_batch_split_plan(
    *,
    review_packet_file: Path,
    preflight_file: Path,
    output_file: Path,
) -> Dict[str, Any]:
    review_packet = _load_json(review_packet_file)
    preflight = _load_json(preflight_file)

    rows = _items_from_review_packet(review_packet)
    lane_source = "review_packet"
    if not rows:
        rows = _items_from_preflight(preflight)
        lane_source = "preflight_batch"

    groups = _group_rows(rows)
    counts = {group["lane_family"]: int(group["count"]) for group in groups}
    recommended = [group for group in groups if group["lane_family"] in _PROFILE_BY_LANE and group["count"] > 0]

    payload = {
        "schema_version": "batch_split_plan_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lane_source": lane_source,
        "source_files": {
            "review_packet": str(review_packet_file),
            "preflight_batch": str(preflight_file),
        },
        "input_count": len(rows),
        "lane_counts": counts,
        "split_required": len([name for name, count in counts.items() if name != "unknown" and count > 0]) > 1,
        "recommended_action": (
            "split_input_into_lane_families_before_next_shot_review"
            if len(recommended) > 1
            else "current_batch_already_single_lane_or_runtime_missing"
        ),
        "lane_groups": groups,
    }
    atomic_write_json(output_file, payload)
    return payload


def materialize_split_batches(
    *,
    plan_file: Path,
    input_dir: Path,
    output_root: Path,
) -> Dict[str, Any]:
    plan = _load_json(plan_file)
    lane_groups = list(plan.get("lane_groups") or [])
    output_root.mkdir(parents=True, exist_ok=True)

    summary_groups: List[Dict[str, Any]] = []
    total_copied = 0
    total_sidecars = 0
    for group in lane_groups:
        if not isinstance(group, dict):
            continue
        lane = str(group.get("lane_family") or "").strip()
        files = [str(item).strip() for item in (group.get("files") or []) if str(item).strip()]
        if not lane or not files:
            continue
        lane_dir = (output_root / lane).resolve()
        lane_dir.mkdir(parents=True, exist_ok=True)

        copied_images = 0
        copied_sidecars = 0
        missing_files: List[str] = []
        for image_name in files:
            src = (input_dir / image_name).resolve()
            if not src.exists():
                missing_files.append(image_name)
                continue
            shutil.copy2(src, lane_dir / src.name)
            copied_images += 1
            total_copied += 1

            sidecar_glob = f"{src.name}.*"
            for sidecar in input_dir.glob(sidecar_glob):
                if sidecar.resolve() == src:
                    continue
                shutil.copy2(sidecar, lane_dir / sidecar.name)
                copied_sidecars += 1
                total_sidecars += 1

        manifest_result = create_or_update_input_manifest(lane_dir)
        manifest_path = Path(str(manifest_result.get("path") or lane_dir / "input_manifest.json"))
        manifest_payload = _load_json(manifest_path)
        intended_view, intended_center = _INTENDED_VIEW_BY_LANE.get(lane, ("", None))
        if isinstance(manifest_payload.get("items"), list):
            for item in manifest_payload["items"]:
                if not isinstance(item, dict):
                    continue
                if intended_view and not str(item.get("intended_view") or "").strip():
                    item["intended_view"] = intended_view
                if lane and not str(item.get("intended_lane_family") or "").strip():
                    item["intended_lane_family"] = lane
                if intended_center is not None and item.get("intended_view_center_deg") is None:
                    item["intended_view_center_deg"] = intended_center
                notes = str(item.get("notes") or "").strip()
                if "derived_from_observed_lane_split" not in notes:
                    item["notes"] = (
                        f"{notes}; derived_from_observed_lane_split".strip("; ").strip()
                        if notes
                        else "derived_from_observed_lane_split"
                    )
            _write_json(manifest_path, manifest_payload)

        summary_groups.append(
            {
                "lane_family": lane,
                "output_dir": str(lane_dir),
                "copied_images": copied_images,
                "copied_sidecars": copied_sidecars,
                "missing_files": missing_files,
                "manifest_path": str(manifest_path),
                "manifest_entry_count": manifest_result.get("item_count"),
                "suggested_profile": group.get("suggested_profile"),
            }
        )

    return {
        "schema_version": "materialized_batch_split_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_file": str(plan_file),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "lane_group_count": len(summary_groups),
        "total_copied_images": total_copied,
        "total_copied_sidecars": total_sidecars,
        "lane_groups": summary_groups,
    }
