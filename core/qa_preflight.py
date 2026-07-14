from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .qa_admission import resolve_target_bucket
from .qa_collection_metadata import infer_layer_tag_from_profile, parse_collection_metadata
from .qa_governance import fail_closed_release_gate
from .qa_industrial_summary import build_batch_preflight_summary
from .qa_input_manifest import load_input_manifest_index, resolve_input_manifest_entry

_PREFLIGHT_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
try:
    from .qa_features import extract_face_feat, extract_pose_feat
    from .qa_utils import canonicalize_view_lane, estimate_view_bucket_and_side, image_read_bgr
    from .qa_view_router import route_view_lane
except ModuleNotFoundError as exc:
    _PREFLIGHT_IMPORT_ERROR = exc
    extract_face_feat = None
    extract_pose_feat = None
    canonicalize_view_lane = None
    estimate_view_bucket_and_side = None
    image_read_bgr = None
    route_view_lane = None


def _list_images_in_dir(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts]
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _lane_family(value: Any) -> str:
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


def _lane_center_deg(lane_family: str) -> Optional[float]:
    mapping = {
        "front": 0.0,
        "three_quarter": 45.0,
        "side": 90.0,
        "back": 180.0,
    }
    return mapping.get(lane_family)


def _angle_delta_deg(observed: Optional[float], center: Optional[float]) -> Optional[float]:
    if observed is None or center is None:
        return None
    delta = abs(float(observed) - float(center))
    if center >= 170.0:
        delta = min(delta, abs(360.0 - delta))
    return float(delta)


def _preflight_report_meta(config: Any, target_profile: str) -> Dict[str, Any]:
    release_gates = config.release_gates if isinstance(getattr(config, "release_gates", None), dict) else {}
    target_bucket = resolve_target_bucket(target_profile)
    active_release_gate = dict((release_gates.get("release_gates") or {}).get(target_bucket) or {})
    return {
        "release_gate": fail_closed_release_gate(
            active_release_gate,
            target_bucket=target_bucket,
            source_schema_version=str(release_gates.get("schema_version") or "").strip(),
        )
    }


def run_preflight_batch(
    runtime: Optional[Any],
    *,
    config: Any,
    input_dir: Path,
    target_profile: str,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    images = _list_images_in_dir(input_dir)
    manifest_index = load_input_manifest_index(input_dir, manifest_path)
    report_meta = _preflight_report_meta(config, target_profile)
    runtime_ready = (
        runtime is not None
        and _PREFLIGHT_IMPORT_ERROR is None
        and image_read_bgr is not None
        and extract_face_feat is not None
        and extract_pose_feat is not None
        and estimate_view_bucket_and_side is not None
        and canonicalize_view_lane is not None
        and route_view_lane is not None
    )
    items: List[Dict[str, Any]] = []
    compact_items: List[Dict[str, Any]] = []
    manifest_matched_count = 0

    for image_path in images:
        manifest_entry = resolve_input_manifest_entry(image_path, input_dir, manifest_index)
        if manifest_entry is not None:
            manifest_matched_count += 1
        collection_meta = parse_collection_metadata(image_path, input_dir, manifest_entry=manifest_entry)
        if not collection_meta.get("layer_tag"):
            inferred_layer = infer_layer_tag_from_profile(target_profile)
            if inferred_layer:
                collection_meta["layer_tag"] = inferred_layer
                if str(collection_meta.get("naming_source") or "none") == "none":
                    collection_meta["naming_source"] = "profile_fallback"

        observed_lane_detail = "unknown"
        observed_lane = "unknown"
        route_confidence = None
        observed_center_distance = None
        body_yaw_deg = None
        issues: List[str] = []
        if runtime_ready and runtime is not None:
            try:
                img = image_read_bgr(image_path, runtime.config.standardization)
                if img is None:
                    raise RuntimeError("IMAGE_READ_ERROR")
                face_feat = extract_face_feat(runtime, img, image_path)
                pose_feat = extract_pose_feat(runtime, img)
                face_bucket, _, _ = estimate_view_bucket_and_side(face_feat)
                legacy_lane = canonicalize_view_lane(face_feat, face_bucket)
                shadow_route = route_view_lane(runtime, None, face_feat, pose_feat)
                observed_lane = str(shadow_route.lane or legacy_lane or "unknown")
                if str(shadow_route.lane_detail or "").strip() and str(shadow_route.lane_detail or "").strip() != "unknown":
                    observed_lane_detail = str(shadow_route.lane_detail)
                else:
                    observed_lane_detail = observed_lane
                route_confidence = _safe_float(shadow_route.confidence)
                body_yaw_deg = _safe_float(shadow_route.body_yaw_deg)
                family = _lane_family(observed_lane_detail)
                observed_center_distance = _angle_delta_deg(body_yaw_deg, _lane_center_deg(family))
                if not face_feat.ok:
                    issues.extend(list(face_feat.reasons or []))
                if not pose_feat.ok:
                    issues.extend(list(pose_feat.reasons or []))
            except Exception as exc:
                issues.append(str(exc))
        else:
            issues.append(
                f"PREFLIGHT_RUNTIME_UNAVAILABLE:{getattr(_PREFLIGHT_IMPORT_ERROR, 'name', 'runtime')}"
            )

        item = {
            "image": image_path.name,
            "collection": collection_meta,
            "lane": {
                "view_lane": observed_lane,
                "view_lane_detail": observed_lane_detail,
            },
            "review_only_breakdown_v2": {
                "observed_lane_family": _lane_family(observed_lane_detail),
                "observed_lane_center_distance_deg": _round_or_none(observed_center_distance),
                "observed_lane_source": "preflight_face_pose_router",
                "body_yaw_deg": _round_or_none(body_yaw_deg),
                "lane_membership_confidence": _round_or_none(route_confidence),
            },
            "debug": {
                "collection_metadata": collection_meta,
                "preflight_issues": issues,
            },
        }
        items.append(item)
        compact_items.append(
            {
                "image": image_path.name,
                "input_relative_path": collection_meta.get("input_relative_path"),
                "intended_view": collection_meta.get("view_expected"),
                "intended_lane_family": collection_meta.get("view_expected_family"),
                "observed_lane_family": item["review_only_breakdown_v2"]["observed_lane_family"],
                "observed_lane_detail": observed_lane_detail,
                "observed_lane_center_distance_deg": item["review_only_breakdown_v2"]["observed_lane_center_distance_deg"],
                "route_confidence": _round_or_none(route_confidence),
                "manifest_entry_present": bool(collection_meta.get("manifest_entry_present")),
                "issues": issues[:6],
            }
        )

    batch_preflight = build_batch_preflight_summary(items, report_meta=report_meta)
    reasons = list(batch_preflight.get("reasons") or [])
    if not runtime_ready:
        if "OBSERVED_LANE_RUNTIME_UNAVAILABLE" not in reasons:
            reasons.append("OBSERVED_LANE_RUNTIME_UNAVAILABLE")
        intended_coverage = _safe_float(batch_preflight.get("intended_lane_coverage")) or 0.0
        if intended_coverage < 0.50:
            batch_preflight["status"] = "FAIL"
            batch_preflight["recommended_action"] = (
                "run_preflight_with_project_venv_or_add_input_manifest_before_training_or_benchmark_use"
            )
        elif str(batch_preflight.get("status") or "").upper() == "PASS":
            batch_preflight["status"] = "WARN"
            batch_preflight["recommended_action"] = (
                "metadata_only_preflight_is_not_enough_for_lane_governance_use_project_venv_before_promotion"
            )
        batch_preflight["reasons"] = reasons
    mismatch_items = [
        row
        for row in compact_items
        if row.get("intended_lane_family")
        and row.get("observed_lane_family")
        and row.get("observed_lane_family") != "unknown"
        and row.get("intended_lane_family") != row.get("observed_lane_family")
    ]
    return {
        "schema_version": "batch_preflight_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_profile": target_profile,
        "input_dir": str(input_dir),
        "input_count": len(images),
        "runtime_ready": bool(runtime_ready),
        "observation_mode": "light_router" if runtime_ready else "metadata_only",
        "manifest_summary": {
            **dict(manifest_index.get("summary") or {}),
            "matched_image_count": manifest_matched_count,
            "matched_image_share": _round_or_none(manifest_matched_count / max(1, len(images))),
        },
        "batch_preflight": batch_preflight,
        "mismatch_count": len(mismatch_items),
        "mismatch_examples": mismatch_items[:20],
        "items": compact_items,
    }


def create_lightweight_preflight_config(base_dir: Path) -> Any:
    release_gates: Dict[str, Any] = {}
    yaml_path = (base_dir / "configs" / "release_gates.yaml").resolve()
    try:
        import yaml as pyyaml  # type: ignore

        if yaml_path.exists():
            node = pyyaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(node, dict):
                release_gates = node
    except Exception:
        release_gates = {}

    return SimpleNamespace(
        paths=SimpleNamespace(
            dir_input=(base_dir / "input").resolve(),
            dir_output=(base_dir / "outputs").resolve(),
        ),
        review=SimpleNamespace(active_profile="body_gold_fullbody"),
        release_gates=release_gates,
    )
