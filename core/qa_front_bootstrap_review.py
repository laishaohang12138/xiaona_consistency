from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .qa_winner_bank_policy import WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER, winner_bank_bootstrap_policy


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _round_or_none(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _pass_candidate_lookup(gpt_packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    queue = gpt_packet.get("priority_review_queue") if isinstance(gpt_packet.get("priority_review_queue"), dict) else {}
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in queue.get("pass_candidates") or []:
        if not isinstance(row, dict):
            continue
        image = str(row.get("image") or "").strip()
        if image:
            lookup[image] = row
    return lookup


def _review_axes(candidate: Dict[str, Any]) -> List[str]:
    axes = list(candidate.get("manual_focus") or [])
    if not axes:
        axes = [
            "check age impression and face spacing against A-Core_01",
            "check torso compactness and waist containment against 116-1",
            "check whether lighting/exposure is creating a false positive",
        ]
    return [str(item).strip() for item in axes if str(item).strip()][:4]


def _summary_signals(candidate: Dict[str, Any], truth_center: Dict[str, Any], clothing: Dict[str, Any]) -> List[str]:
    signals: List[str] = []
    face_master = _round_or_none(candidate.get("face_master_alignment"))
    body_truth = _round_or_none(candidate.get("body_truth_alignment"))
    face_topology = _round_or_none(truth_center.get("face_topology_support"))
    clothing_score = _round_or_none(clothing.get("clothing_invariant_score"))
    if face_master is not None and float(face_master) >= 0.95:
        signals.append("strong face-master alignment")
    if body_truth is not None and float(body_truth) >= 0.82:
        signals.append("strong body-truth alignment")
    if face_topology is not None and float(face_topology) >= 0.99:
        signals.append("stable face topology")
    if clothing_score is not None and float(clothing_score) >= 0.78:
        signals.append("stable clothing-invariant support")
    return signals[:4]


def _review_risks(candidate: Dict[str, Any], truth_center: Dict[str, Any], clothing: Dict[str, Any]) -> List[str]:
    risks: List[str] = []
    body_topology = _round_or_none(truth_center.get("body_topology_support"))
    body_truth = _round_or_none(truth_center.get("body_truth_support"))
    clothing_conf = _round_or_none(clothing.get("clothing_invariant_confidence"))
    if body_topology is not None and float(body_topology) < 0.72:
        risks.append("body topology is only moderate")
    if body_truth is not None and float(body_truth) < 0.74:
        risks.append("body truth support is not yet strong")
    if clothing_conf is not None and float(clothing_conf) < 0.90:
        risks.append("clothing-invariant confidence still needs manual confirmation")
    return risks[:4]


def build_front_bootstrap_review_sheet(
    *,
    run_index_file: Path,
    output_file: Path,
) -> Dict[str, Any]:
    run_index = _load_json(run_index_file)
    recommended = (run_index.get("recommended_runs") or {}).get("front_bootstrap_snapshot")
    if not isinstance(recommended, dict) or not str(recommended.get("artifact_root") or "").strip():
        raise ValueError("review_run_index does not contain a front_bootstrap_snapshot recommendation")

    artifact_root = Path(str(recommended.get("artifact_root"))).resolve()
    winner_review = _load_json(artifact_root / "winner_bank_review_packet.json")
    gpt_packet = _load_json(artifact_root / "gpt_review_packet.json")
    if not winner_review or not gpt_packet:
        raise ValueError(f"front bootstrap snapshot is missing review artifacts: {artifact_root}")

    pass_lookup = _pass_candidate_lookup(gpt_packet)
    top_candidates: List[Dict[str, Any]] = []
    for candidate in list(winner_review.get("top_candidates") or [])[:3]:
        if not isinstance(candidate, dict):
            continue
        image = str(candidate.get("image") or "").strip()
        pass_row = pass_lookup.get(image, {})
        truth_center = pass_row.get("truth_center") if isinstance(pass_row.get("truth_center"), dict) else {}
        clothing = pass_row.get("clothing_invariant") if isinstance(pass_row.get("clothing_invariant"), dict) else {}
        top_candidates.append(
            {
                "rank": candidate.get("rank"),
                "image": image,
                "status": candidate.get("status"),
                "selection_score": _round_or_none(candidate.get("selection_score")),
                "face_master_alignment": _round_or_none(candidate.get("face_master_alignment")),
                "body_truth_alignment": _round_or_none(candidate.get("body_truth_alignment")),
                "topology_signature_alignment": _round_or_none(candidate.get("topology_signature_alignment")),
                "hybrid_master_alignment": _round_or_none(candidate.get("hybrid_master_alignment")),
                "review_only_confidence": _round_or_none(pass_row.get("review_only_confidence")),
                "face_truth_support": _round_or_none(truth_center.get("face_truth_support")),
                "body_truth_support": _round_or_none(truth_center.get("body_truth_support")),
                "face_topology_support": _round_or_none(truth_center.get("face_topology_support")),
                "body_topology_support": _round_or_none(truth_center.get("body_topology_support")),
                "clothing_invariant_score": _round_or_none(clothing.get("clothing_invariant_score")),
                "clothing_invariant_confidence": _round_or_none(clothing.get("clothing_invariant_confidence")),
                "summary_signals": _summary_signals(candidate, truth_center, clothing),
                "review_risks": _review_risks(candidate, truth_center, clothing),
                "review_axes": _review_axes(candidate),
                "manual_decision": "",
                "manual_note": "",
            }
        )

    blockers = [str(item).strip() for item in (winner_review.get("promotion_blockers") or []) if str(item).strip()]
    bootstrap_policy = winner_bank_bootstrap_policy()
    if str(bootstrap_policy.get("state") or "") == "deferred" and WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER not in blockers:
        blockers.append(WINNER_BANK_BOOTSTRAP_DEFERRED_BLOCKER)
    policy_state = str(bootstrap_policy.get("state") or "")
    promotion_ready = bool(winner_review.get("promotion_ready")) and policy_state != "deferred"
    payload = {
        "schema_version": "front_bootstrap_review_sheet_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "target_profile": str(recommended.get("target_profile") or "").strip(),
        "top_ranked_image": str(recommended.get("top_ranked_image") or "").strip(),
        "bootstrap_required": bool(winner_review.get("bank_bootstrap_required")),
        "promotion_ready": promotion_ready,
        "promotion_blockers": blockers,
        "winner_bank_bootstrap_policy": bootstrap_policy,
        "manual_goal": "Use top-3 front-core candidates for diagnostic review; winner_bank entries remain mutable review memory.",
        "manual_rule": (
            "Do not freeze winner_bank or use it for training admission until review-only angle, clothing, "
            "lighting, topology, and pose-aware body-truth reads are mature."
        ),
        "top_candidates": top_candidates,
        "suggested_commands": {
            "inspect_review_packet": (
                "python check_consistency.py --workflow inspect_review_packet "
                f'--artifacts-dir "{artifact_root}"'
            ),
            "winner_bank_status": (
                "python check_consistency.py --workflow winner_bank_status "
                f'--artifacts-dir "{artifact_root}"'
            ),
            "promote_rank_1": (
                "python check_consistency.py --workflow promote_winner --winner-rank 1 "
                f'--artifacts-dir "{artifact_root}"'
                if promotion_ready
                else "blocked_by_review_policy_or_current_blockers"
            ),
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
