from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .qa_face_pose_canonical import _load_json_object as _load_face_json_object
from .qa_face_pose_canonical import _normalize_artifact as _normalize_face_artifact
from .qa_face_pose_canonical import _topology_signature_similarity
from .qa_review_only_score import apply_review_only_score_v2
from .qa_industrial_summary import (
    build_batch_preflight_summary,
    build_evidence_completeness_summary,
)
from .qa_review_packet import build_review_packet


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _dedupe_keep_order(values: Sequence[str], *, limit: Optional[int] = None) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if limit is not None and len(out) >= max(0, int(limit)):
            break
    return out


def _load_json_dict(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return payload


def _resolve_source_path(item: Dict[str, Any]) -> Optional[Path]:
    debug = item.get("debug") or {}
    for raw_path in [debug.get("source_path"), item.get("source_path")]:
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        try:
            return Path(path_text).resolve()
        except Exception:
            continue
    return None


def _metric_value(bundle: Any, metric_name: str) -> Optional[float]:
    if not isinstance(bundle, dict):
        return None
    metrics = bundle.get("metrics")
    if isinstance(metrics, dict):
        node = metrics.get(metric_name)
        if isinstance(node, dict):
            value = node.get("value")
            if value is None:
                value = node.get("metric_value")
            return _safe_float(value)
        return _safe_float(node)
    if isinstance(metrics, list):
        for row in metrics:
            if not isinstance(row, dict):
                continue
            if str(row.get("metric_name") or "").strip() != metric_name:
                continue
            value = row.get("value")
            if value is None:
                value = row.get("metric_value")
            return _safe_float(value)
    return None


def _load_face_artifact(face_shadow: Dict[str, Any], source_path: Path) -> Optional[Dict[str, Any]]:
    candidate_paths = [
        face_shadow.get("cache_file"),
        face_shadow.get("candidate_artifact_path"),
    ]
    for raw_path in candidate_paths:
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        payload = _load_face_json_object(path)
        if not isinstance(payload, dict):
            continue
        artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else payload
        if not isinstance(artifact, dict):
            continue
        normalized = _normalize_face_artifact(
            artifact,
            source_path=source_path,
            source_role=str(artifact.get("source_role") or "candidate"),
        )
        if path == Path(str(face_shadow.get("cache_file") or path)):
            normalized["cache_file"] = str(path)
            normalized["cache_key"] = face_shadow.get("cache_key")
            normalized["cache_state"] = face_shadow.get("cache_state") or "hit"
        return normalized
    return None


def _load_face_master_artifact(face_shadow: Dict[str, Any], source_path: Path) -> Optional[Dict[str, Any]]:
    try:
        default_master = source_path.parents[1] / "master_truth" / "face_master_canonical.json"
    except Exception:
        default_master = Path("outputs") / "master_truth" / "face_master_canonical.json"
    candidate_paths = [
        face_shadow.get("master_artifact_path"),
        default_master,
    ]
    for raw_path in candidate_paths:
        path = Path(str(raw_path or "")).resolve()
        if not path.exists():
            continue
        payload = _load_face_json_object(path)
        if not isinstance(payload, dict):
            continue
        return _normalize_face_artifact(payload, source_path=path, source_role="master_truth")
    return None


def _lane_family_from_item(item: Dict[str, Any], master: Dict[str, Any]) -> str:
    explicit = str(master.get("lane_family") or "").strip().lower()
    if explicit:
        return explicit
    debug = item.get("debug") or {}
    view_lane = str(debug.get("view_lane") or "").strip().lower()
    if view_lane == "side_90":
        return "side"
    if view_lane == "back_180":
        return "back"
    if view_lane in {"front", "three_quarter"}:
        return view_lane
    return "three_quarter"


def _augment_master_consistency_card(item: Dict[str, Any]) -> bool:
    debug = item.setdefault("debug", {})
    master = debug.get("master_consistency_card")
    if not isinstance(master, dict) or len(master) == 0:
        return False

    face_shadow = debug.get("face_canonical_shadow") or {}
    heavy_bundle = debug.get("heavy_evidence") or {}
    lane_family = _lane_family_from_item(item, master)

    absolute_alignment = _safe_float(master.get("face_master_alignment"))
    canonical_alignment = _safe_float(face_shadow.get("canonical_face_identity_similarity"))
    topology_alignment = _safe_float(face_shadow.get("canonical_face_topology_similarity"))
    body_topology_alignment = _safe_float(_metric_value(heavy_bundle, "body_topology_signature_similarity"))

    updated = False
    if master.get("absolute_front_master_alignment") != _round_or_none(absolute_alignment):
        master["absolute_front_master_alignment"] = _round_or_none(absolute_alignment)
        updated = True
    if canonical_alignment is not None and master.get("canonical_pose_normalized_alignment") != _round_or_none(canonical_alignment):
        master["canonical_pose_normalized_alignment"] = _round_or_none(canonical_alignment)
        updated = True
    if topology_alignment is not None and master.get("topology_signature_alignment") != _round_or_none(topology_alignment):
        master["topology_signature_alignment"] = _round_or_none(topology_alignment)
        updated = True
    if body_topology_alignment is not None and master.get("body_topology_alignment") != _round_or_none(body_topology_alignment):
        master["body_topology_alignment"] = _round_or_none(body_topology_alignment)
        updated = True

    disagreement_reason = None
    if absolute_alignment is not None and canonical_alignment is not None:
        if absolute_alignment < 0.52 and canonical_alignment >= 0.62 and (topology_alignment is None or topology_alignment >= 0.70):
            disagreement_reason = "frontal_gap_but_canonical_supported"
        elif absolute_alignment >= 0.64 and canonical_alignment < 0.58 and topology_alignment is not None and topology_alignment < 0.62:
            disagreement_reason = "absolute_front_ok_but_canonical_topology_weak"
    if canonical_alignment is not None and topology_alignment is not None and disagreement_reason is None:
        if canonical_alignment >= 0.66 and topology_alignment < 0.62:
            disagreement_reason = "canonical_identity_ok_but_topology_weak"
        elif canonical_alignment < 0.58 and topology_alignment >= 0.70:
            disagreement_reason = "frontal_signal_weak_but_topology_supported"
    if disagreement_reason and master.get("identity_disagreement_reason") != disagreement_reason:
        master["identity_disagreement_reason"] = disagreement_reason
        updated = True

    focus = list(master.get("manual_focus") or [])
    prompts = list(master.get("manual_review_prompts") or [])
    face_topology_floor = 0.74 if lane_family == "front" else 0.70 if lane_family == "three_quarter" else 0.64
    body_topology_floor = 0.68 if lane_family == "side" else 0.72
    if topology_alignment is not None and topology_alignment < face_topology_floor:
        focus.extend(
            [
                "check forehead-nose-lip-chin contour before trusting frontal resemblance",
                "verify jawline turn and chin projection as one 3D head structure",
            ]
        )
        prompts.append(
            "Face topology is weak. Judge same-head structure first, then treat raw frontal similarity as support only."
        )
    if body_topology_alignment is not None and body_topology_alignment < body_topology_floor:
        focus.extend(
            [
                "check torso compactness and shoulder-hip balance against 116-1",
                "treat gait or pose as secondary after checking 3D body structure",
            ]
        )
        prompts.append(
            "Body topology support is weak. Compare 116-1 shape structure first, then treat pose or gait as secondary."
        )
    next_focus = _dedupe_keep_order(focus, limit=6)
    next_prompts = _dedupe_keep_order(prompts, limit=5)
    if next_focus != list(master.get("manual_focus") or []):
        master["manual_focus"] = next_focus
        updated = True
    if next_prompts != list(master.get("manual_review_prompts") or []):
        master["manual_review_prompts"] = next_prompts
        updated = True
    return updated


def refresh_review_topology_state(report_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_items = 0
    face_topology_backfilled = 0
    face_topology_available = 0
    master_cards_augmented = 0

    for item in report_items:
        if not isinstance(item, dict):
            continue
        total_items += 1
        debug = item.setdefault("debug", {})
        face_shadow = debug.get("face_canonical_shadow")
        source_path = _resolve_source_path(item)
        if isinstance(face_shadow, dict) and source_path is not None:
            if _safe_float(face_shadow.get("canonical_face_topology_similarity")) is None:
                candidate_artifact = _load_face_artifact(face_shadow, source_path)
                master_artifact = _load_face_master_artifact(face_shadow, source_path)
                if isinstance(candidate_artifact, dict):
                    topology_similarity = None
                    topology_delta = None
                    if isinstance(master_artifact, dict):
                        topology_similarity, topology_delta = _topology_signature_similarity(
                            master_artifact.get("canonical_face_topology_signature"),
                            candidate_artifact.get("canonical_face_topology_signature"),
                        )
                    merged = dict(face_shadow)
                    merged.update(
                        {
                            "source_path": candidate_artifact.get("source_path") or merged.get("source_path"),
                            "cache_file": candidate_artifact.get("cache_file") or merged.get("cache_file"),
                            "cache_key": candidate_artifact.get("cache_key") or merged.get("cache_key"),
                            "cache_state": candidate_artifact.get("cache_state") or merged.get("cache_state"),
                            "canonical_face_topology_similarity": _round_or_none(topology_similarity),
                            "canonical_face_topology_delta": _round_or_none(topology_delta),
                        }
                    )
                    debug["face_canonical_shadow"] = merged
                    face_shadow = merged
                    face_topology_backfilled += 1
            if _safe_float((debug.get("face_canonical_shadow") or {}).get("canonical_face_topology_similarity")) is not None:
                face_topology_available += 1
        if _augment_master_consistency_card(item):
            master_cards_augmented += 1

    return {
        "schema_version": "review_topology_refresh_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "item_count": total_items,
        "face_topology_backfilled": face_topology_backfilled,
        "face_topology_available": face_topology_available,
        "master_cards_augmented": master_cards_augmented,
    }


def rebuild_review_artifacts_from_report(
    report_file: Path,
    *,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_report = Path(report_file).resolve()
    payload = _load_json_dict(resolved_report, "qa report")
    resolved_output = Path(output_dir).resolve() if output_dir is not None else resolved_report.parent
    report_items = list(payload.get("items") or [])
    shot_selection = dict(payload.get("shot_selection") or {})
    report_meta = dict(payload.get("report_meta") or {})

    refresh_summary = refresh_review_topology_state(report_items)
    payload["items"] = report_items
    payload["shot_selection"] = apply_review_only_score_v2(
        report_items,
        shot_selection,
        target_profile=str(report_meta.get("active_profile") or "").strip() or None,
    )
    report_meta["review_artifact_refresh"] = refresh_summary
    report_meta["batch_preflight"] = build_batch_preflight_summary(report_items, report_meta)
    report_meta["evidence_completeness"] = build_evidence_completeness_summary(report_items, report_meta)
    payload["report_meta"] = report_meta

    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ranked_candidates_file = resolved_output / "ranked_candidates.json"
    ranked_candidates_file.write_text(
        json.dumps(payload.get("shot_selection") or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    review_packet = build_review_packet(
        payload,
        resolved_output,
        resolved_report,
        ranked_candidates_file,
    )
    groups = list(((payload.get("shot_selection") or {}).get("groups") or []))
    primary_group = groups[0] if len(groups) > 0 and isinstance(groups[0], dict) else {}
    shortlist = list(primary_group.get("shortlist") or [])
    return {
        "status": "ok",
        "report_file": str(resolved_report),
        "ranked_candidates_file": str(ranked_candidates_file),
        "review_packet_file": str(resolved_output / "review_packet.json"),
        "gpt_review_packet_file": str(resolved_output / "gpt_review_packet.json"),
        "review_artifacts_file": str(resolved_output / "review_artifacts.json"),
        "target_profile": report_meta.get("active_profile"),
        "selection_method": primary_group.get("selection_method"),
        "top_ranked_image": primary_group.get("top_ranked_image"),
        "shortlist_size": len(shortlist),
        "refresh_summary": refresh_summary,
        "batch_review_only_status_counts": dict(((review_packet.get("batch_summary") or {}).get("review_only_status_counts") or {})),
    }
