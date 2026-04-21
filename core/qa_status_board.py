from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_input_manifest import load_input_manifest_index, required_prompt_intent_fields
from .qa_winner_bank_policy import winner_bank_bootstrap_policy


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


def _first_text(value: Any) -> str:
    return str(value or "").strip()


def _manifest_required_field_coverage(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    total = max(1, len(entries))
    return {
        "prompt_id": float(sum(1 for item in entries if _first_text(item.get("prompt_id"))) / total),
        "seed": float(sum(1 for item in entries if item.get("seed") is not None) / total),
        "anchor_source": float(sum(1 for item in entries if _first_text(item.get("anchor_source"))) / total),
        "intended_view": float(sum(1 for item in entries if _first_text(item.get("intended_view"))) / total),
    }


def _manifest_board_entry(input_dir: Path) -> Dict[str, Any]:
    manifest_index = load_input_manifest_index(input_dir)
    entries = manifest_index.get("entries") if isinstance(manifest_index.get("entries"), list) else []
    coverage = _manifest_required_field_coverage(entries)
    missing_fields = [field for field in required_prompt_intent_fields() if float(coverage.get(field) or 0.0) < 1.0]
    return {
        "input_dir": str(input_dir.resolve()),
        "available": bool(manifest_index.get("available")),
        "manifest_path": manifest_index.get("path"),
        "item_count": len(entries),
        "required_fields": required_prompt_intent_fields(),
        "required_field_coverage": {key: _round_or_none(value) for key, value in coverage.items()},
        "required_field_ready": not missing_fields and bool(entries),
        "missing_fields": missing_fields,
    }


def _training_admission_summary(manifest_file: Path) -> Dict[str, Any]:
    payload = _load_json(manifest_file)
    if not payload:
        return {
            "available": False,
            "manifest_file": str(manifest_file.resolve()),
            "entry_count": 0,
            "last_sealed_at_utc": None,
        }
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    recent_entry = entries[-1] if entries else {}
    return {
        "available": True,
        "manifest_file": str(manifest_file.resolve()),
        "entry_count": len(entries),
        "last_sealed_at_utc": recent_entry.get("sealed_at_utc") if isinstance(recent_entry, dict) else None,
    }


def build_review_status_board(
    *,
    base_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:
    outputs_dir = (base_dir / "outputs").resolve()
    run_index = _load_json(outputs_dir / "review_run_index.json")
    front_sheet = _load_json(outputs_dir / "front_bootstrap_review_sheet.json")
    winner_bank_report = _load_json(outputs_dir / "winner_bank_report.json")
    winner_policy = winner_bank_bootstrap_policy()
    training_admission = _training_admission_summary(outputs_dir / "training_admission_manifest.json")

    current_outputs = run_index.get("current_outputs") if isinstance(run_index.get("current_outputs"), dict) else {}
    recommended_runs = run_index.get("recommended_runs") if isinstance(run_index.get("recommended_runs"), dict) else {}
    front_bootstrap = (
        recommended_runs.get("front_bootstrap_snapshot")
        if isinstance(recommended_runs.get("front_bootstrap_snapshot"), dict)
        else {}
    )
    three_quarter_clean = (
        recommended_runs.get("three_quarter_clean_snapshot")
        if isinstance(recommended_runs.get("three_quarter_clean_snapshot"), dict)
        else {}
    )
    top_candidates = front_sheet.get("top_candidates") if isinstance(front_sheet.get("top_candidates"), list) else []
    front_top3 = [
        {
            "rank": item.get("rank"),
            "image": item.get("image"),
            "selection_score": _round_or_none(item.get("selection_score")),
            "face_master_alignment": _round_or_none(item.get("face_master_alignment")),
            "body_truth_alignment": _round_or_none(item.get("body_truth_alignment")),
        }
        for item in top_candidates[:3]
        if isinstance(item, dict)
    ]

    manifest_states = {
        "input_split_front": _manifest_board_entry(base_dir / "input_split" / "front"),
        "input_split_three_quarter": _manifest_board_entry(base_dir / "input_split" / "three_quarter"),
    }

    next_actions: List[str] = []
    if not manifest_states["input_split_front"]["required_field_ready"]:
        next_actions.append("fill front split manifest fields: prompt_id / seed / anchor_source")
    if not manifest_states["input_split_three_quarter"]["required_field_ready"]:
        next_actions.append("fill three_quarter split manifest fields: prompt_id / seed / anchor_source")
    if str(winner_policy.get("state") or "") == "deferred":
        next_actions.append("defer winner_bank bootstrap until review-only invariance and 3D topology consistency mature")
        next_actions.append("use front_bootstrap_review_sheet top-3 for diagnostic review only")
    elif bool(front_sheet) and not bool(front_sheet.get("promotion_ready")):
        next_actions.append("manually choose one front bootstrap winner from front_bootstrap_review_sheet top-3")
    if not bool(winner_bank_report.get("curated_bank_available")) and str(winner_policy.get("state") or "") != "deferred":
        next_actions.append("bootstrap winner_bank after manual front winner selection")
    if not bool(training_admission.get("available")):
        if str(winner_policy.get("state") or "") == "deferred":
            next_actions.append("no training_admission manifest exists yet; do not seal until invariance gates mature")
        else:
            next_actions.append("no training_admission manifest exists yet; do not seal until front winner is confirmed")

    payload = {
        "schema_version": "review_status_board_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_outputs": {
            "artifact_root": current_outputs.get("artifact_root"),
            "target_profile": current_outputs.get("target_profile"),
            "dominant_lane_family": current_outputs.get("dominant_lane_family"),
            "top_ranked_image": current_outputs.get("top_ranked_image"),
            "evidence_status": current_outputs.get("evidence_status"),
            "preflight_status": current_outputs.get("preflight_status"),
        },
        "recommended_runs": {
            "front_bootstrap_snapshot": {
                "artifact_root": front_bootstrap.get("artifact_root"),
                "top_ranked_image": front_bootstrap.get("top_ranked_image"),
                "evidence_status": front_bootstrap.get("evidence_status"),
                "completeness_score": front_bootstrap.get("completeness_score"),
                "admission_blockers": front_bootstrap.get("admission_blockers"),
            },
            "three_quarter_clean_snapshot": {
                "artifact_root": three_quarter_clean.get("artifact_root"),
                "top_ranked_image": three_quarter_clean.get("top_ranked_image"),
                "evidence_status": three_quarter_clean.get("evidence_status"),
                "completeness_score": three_quarter_clean.get("completeness_score"),
                "admission_blockers": three_quarter_clean.get("admission_blockers"),
            },
        },
        "front_bootstrap_review": {
            "promotion_ready": bool(front_sheet.get("promotion_ready")),
            "promotion_blockers": front_sheet.get("promotion_blockers") or [],
            "top_candidates": front_top3,
        },
        "winner_bank": {
            "available": bool(winner_bank_report.get("curated_bank_available")),
            "entry_count": int(winner_bank_report.get("curated_entry_count") or 0),
            "status": str(winner_bank_report.get("status") or "").strip(),
            "manual_next_step": (
                str(winner_policy.get("reason") or "").strip()
                if str(winner_policy.get("state") or "") == "deferred"
                else str(winner_bank_report.get("manual_next_step") or "").strip()
            ),
        },
        "winner_bank_bootstrap_policy": winner_policy,
        "training_admission": training_admission,
        "input_manifests": manifest_states,
        "next_actions": next_actions,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
