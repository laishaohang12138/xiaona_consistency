from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_io import atomic_write_json


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _artifact_run_brief(artifact_root: Path, *, kind: str) -> Optional[Dict[str, Any]]:
    gpt_packet = _load_json(artifact_root / "gpt_review_packet.json")
    if not gpt_packet:
        return None
    winner_review = _load_json(artifact_root / "winner_bank_review_packet.json")
    batch = gpt_packet.get("batch") if isinstance(gpt_packet.get("batch"), dict) else {}
    release_gate = batch.get("release_gate") if isinstance(batch.get("release_gate"), dict) else {}
    admission_advice = batch.get("admission_advice") if isinstance(batch.get("admission_advice"), dict) else {}
    batch_preflight = batch.get("batch_preflight") if isinstance(batch.get("batch_preflight"), dict) else {}
    evidence = batch.get("evidence_completeness") if isinstance(batch.get("evidence_completeness"), dict) else {}
    lane_counts = batch.get("lane_detail_counts") if isinstance(batch.get("lane_detail_counts"), dict) else {}
    dominant_lane = ""
    if lane_counts:
        dominant_lane = sorted(
            ((str(key), int(value)) for key, value in lane_counts.items()),
            key=lambda row: (-row[1], row[0]),
        )[0][0]
    review_only_counts = batch.get("review_only_status_counts") if isinstance(batch.get("review_only_status_counts"), dict) else {}
    main_counts = batch.get("main_status_counts") if isinstance(batch.get("main_status_counts"), dict) else {}
    blockers = [str(item).strip() for item in (admission_advice.get("blockers") or []) if str(item).strip()]
    release_state = str(release_gate.get("release_state") or "").strip()
    external_review_route = str(
        admission_advice.get("external_review_route")
        or release_gate.get("external_review_route")
        or ("PRIORITY_REVIEW" if release_state == "primary" else "STANDARD_REVIEW")
    ).strip().upper()
    brief = {
        "artifact_root": str(artifact_root.resolve()),
        "kind": kind,
        "run_name": artifact_root.name,
        "generated_at_utc": str(gpt_packet.get("generated_at_utc") or "").strip(),
        "target_profile": str(batch.get("target_profile") or "").strip(),
        "input_count": int(batch.get("input_count") or 0),
        "selection_method": str(batch.get("selection_method") or "").strip(),
        "top_ranked_image": str(batch.get("top_ranked_image") or "").strip(),
        "dominant_lane_family": dominant_lane,
        "review_only_pass_count": int(review_only_counts.get("PASS") or 0),
        "review_only_warn_count": int(review_only_counts.get("WARN") or 0),
        "main_status_counts": main_counts,
        "preflight_status": str(batch_preflight.get("status") or "").strip(),
        "evidence_status": str(evidence.get("status") or "").strip(),
        "completeness_score": _round_or_none(evidence.get("completeness_score")),
        "active_heavy_provider": _safe_text(evidence.get("active_heavy_provider")),
        "release_state": release_state,
        "local_decision_authority": "NONE",
        "external_review_route": external_review_route,
        "training_admission_allowed": False,
        "legacy_admission_fields_state": "DEPRECATED_FORCED_FALSE",
        "admission_suggested_action": str(admission_advice.get("suggested_action") or "").strip(),
        "admission_blockers": blockers,
        "winner_review_ready": bool(winner_review.get("promotion_ready")),
        "winner_review_bootstrap_required": bool(winner_review.get("bank_bootstrap_required")),
        "winner_review_blockers": [
            str(item).strip()
            for item in (winner_review.get("promotion_blockers") or [])
            if str(item).strip()
        ],
        "winner_review_recommended_candidate": (
            str((winner_review.get("recommended_candidate") or {}).get("image") or "").strip()
            if isinstance(winner_review.get("recommended_candidate"), dict)
            else ""
        ),
    }
    brief["is_clean_lane_run"] = bool(
        brief["preflight_status"].upper() == "PASS"
        and len(lane_counts) == 1
        and brief["dominant_lane_family"]
    )
    brief["is_front_priority_review"] = bool(
        brief["dominant_lane_family"] == "front"
        and brief["external_review_route"] == "PRIORITY_REVIEW"
        and brief["preflight_status"].upper() == "PASS"
    )
    brief["is_front_primary_candidate"] = False
    return brief


def _front_bootstrap_score(row: Dict[str, Any]) -> tuple:
    blockers = list(row.get("winner_review_blockers") or [])
    blocker_penalty = 1 if blockers else 0
    face_identity_penalty = 1 if "BATCH_FACE_IDENTITY_STILL_WEAK" in blockers else 0
    return (
        blocker_penalty,
        face_identity_penalty,
        -int(row.get("review_only_pass_count") or 0),
        -float(row.get("completeness_score") or 0.0),
        str(row.get("run_name") or ""),
    )


def _three_quarter_clean_score(row: Dict[str, Any]) -> tuple:
    active_heavy_provider = _safe_text(row.get("active_heavy_provider")).lower()
    prefers_truth_fusion = 0 if "truth_fusion" in active_heavy_provider else 1
    return (
        prefers_truth_fusion,
        -float(row.get("completeness_score") or 0.0),
        -int(row.get("review_only_pass_count") or 0),
        -int(row.get("input_count") or 0),
        -int(str(row.get("generated_at_utc") or "").replace("-", "").replace(":", "").replace("T", "").replace(".", "").replace("+", "" )[:14] or 0),
        str(row.get("run_name") or ""),
    )


def build_review_run_index(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    snapshots_dir = (base_dir / "outputs_snapshots").resolve()

    current_run = _artifact_run_brief(outputs_dir, kind="current_outputs")
    snapshot_runs: List[Dict[str, Any]] = []
    if snapshots_dir.exists():
        for child in sorted([path for path in snapshots_dir.iterdir() if path.is_dir()]):
            brief = _artifact_run_brief(child, kind="snapshot")
            if brief is not None:
                snapshot_runs.append(brief)

    clean_lane_runs = [row for row in snapshot_runs if bool(row.get("is_clean_lane_run"))]
    front_candidates = [row for row in snapshot_runs if bool(row.get("is_front_priority_review"))]
    three_quarter_candidates = [
        row for row in clean_lane_runs if str(row.get("dominant_lane_family") or "").strip() == "three_quarter"
    ]
    recommended_front_bootstrap = sorted(front_candidates, key=_front_bootstrap_score)[0] if front_candidates else None
    recommended_three_quarter_clean = (
        sorted(three_quarter_candidates, key=_three_quarter_clean_score)[0]
        if three_quarter_candidates
        else None
    )

    payload = {
        "schema_version": "review_run_index_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_outputs": current_run,
        "snapshot_run_count": len(snapshot_runs),
        "snapshot_runs": snapshot_runs,
        "clean_lane_runs": clean_lane_runs,
        "recommended_runs": {
            "default_gpt_packet": (
                str(outputs_dir / "gpt_review_packet.json")
                if current_run is not None
                else ""
            ),
            "front_bootstrap_snapshot": recommended_front_bootstrap,
            "three_quarter_clean_snapshot": recommended_three_quarter_clean,
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_file, payload)
    return payload
