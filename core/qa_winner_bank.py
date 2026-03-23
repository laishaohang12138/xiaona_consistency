from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _dedupe_keep_order(values: Sequence[str], limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _normalize_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.dot(a, b))


def _weighted_mean(items: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _weighted_sum(items: Sequence[tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
    numerator: Optional[np.ndarray] = None
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        if numerator is None:
            numerator = np.zeros_like(value, dtype=np.float32)
        numerator += value.astype(np.float32) * float(weight)
        denominator += float(weight)
    if numerator is None or denominator <= 0.0:
        return None
    return numerator / float(denominator)


def _json_ready_vector(value: Any) -> Optional[List[float]]:
    vector = _normalize_embedding(value)
    if vector is None:
        return None
    return [round(float(item), 6) for item in vector.tolist()]


def _heavy_delta(top_row: Dict[str, Any], candidate_row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    top_heavy = top_row.get("heavy_review") or {}
    cand_heavy = candidate_row.get("heavy_review") or {}
    keys = [
        "parser_confidence",
        "parser_boundary_alignment",
        "parser_visible_body_alignment",
        "parser_consensus_score",
        "enhanced_selection_score",
        "garment_coverage_ratio",
        "upper_cloth_coverage",
        "lower_cloth_coverage",
        "neckline_depth_ratio",
        "shoulder_cloth_balance",
        "visible_body_ratio",
        "visible_face_ratio",
        "visible_arm_ratio",
        "visible_leg_ratio",
    ]
    deltas: Dict[str, Optional[float]] = {}
    for key in keys:
        top_value = top_heavy.get(key)
        cand_value = cand_heavy.get(key)
        if isinstance(top_value, (int, float)) and isinstance(cand_value, (int, float)):
            deltas[key] = _round_or_none(float(cand_value) - float(top_value))
        else:
            deltas[key] = None
    return deltas


def _compare_note(label: str, delta: Optional[float], bigger_is_better: bool = True, eps: float = 0.003) -> Optional[str]:
    if delta is None:
        return None
    if bigger_is_better:
        if delta <= -eps:
            return f"top1 在 {label} 上更强"
        if delta >= eps:
            return f"候选在 {label} 上更强"
    else:
        if delta >= eps:
            return f"top1 在 {label} 上更稳"
        if delta <= -eps:
            return f"候选在 {label} 上更稳"
    return None


def _build_focus_points(top_row: Dict[str, Any], candidate_row: Dict[str, Any]) -> List[str]:
    component_deltas = ((candidate_row.get("delta_vs_top") or {}).get("component_scores") or {})
    heavy_deltas = _heavy_delta(top_row, candidate_row)
    notes: List[str] = []
    for note in [
        _compare_note("绝对 face 身份", component_deltas.get("absolute_face_identity")),
        _compare_note("批内 face 对齐", component_deltas.get("batch_face_alignment")),
        _compare_note("hybrid 身份", component_deltas.get("hybrid_identity_alignment")),
        _compare_note("cloth-free 身份", component_deltas.get("clothfree_identity_alignment")),
        _compare_note("world3d 结构", component_deltas.get("world3d_alignment")),
        _compare_note("parser 边界对齐", heavy_deltas.get("parser_boundary_alignment")),
        _compare_note("parser 可见身体对齐", heavy_deltas.get("parser_visible_body_alignment")),
        _compare_note("parser 领口深度", heavy_deltas.get("neckline_depth_ratio"), bigger_is_better=False, eps=0.01),
    ]:
        if note and note not in notes:
            notes.append(note)
    if len(notes) == 0:
        notes.append("两者量纲差距很小，应优先依赖 GPT 与人眼看脸部神态和年龄感。")
    return notes[:4]


def _drift_severity(
    face_sim: Optional[float],
    body_sim: Optional[float],
    depth_sim: Optional[float],
    world3d_sim: Optional[float],
    hybrid_sim: Optional[float],
) -> str:
    worst = min(
        [
            float(v)
            for v in [face_sim, body_sim, depth_sim, world3d_sim, hybrid_sim]
            if isinstance(v, (int, float))
        ]
        or [1.0]
    )
    if worst < 0.74:
        return "high"
    if worst < 0.82:
        return "medium"
    return "low"


def _drift_manual_focus(
    flags: Sequence[str],
    entry: Dict[str, Any],
    face_sim: Optional[float],
    body_sim: Optional[float],
    depth_sim: Optional[float],
    world3d_sim: Optional[float],
) -> List[str]:
    focus: List[str] = []
    if "WINNER_BANK_FACE_DRIFT" in flags:
        focus.extend(
            [
                "compare this winner with the curated bank on face shape, age impression, and eye-mouth spacing",
                "treat face drift as the first veto axis before accepting a new winner",
            ]
        )
    if "WINNER_BANK_BODY_DRIFT" in flags:
        focus.extend(
            [
                "check head-body ratio, torso length feel, and shoulder-hip structure against the curated bank",
                "verify that body proportion has not shifted even if the face still looks plausible",
            ]
        )
    if "WINNER_BANK_3D_DRIFT" in flags or "WINNER_BANK_WORLD3D_DRIFT" in flags:
        focus.extend(
            [
                "check volume feel, shoulder depth, pelvis axis, and lower-limb length consistency",
                "use 3D/world3d drift as a structural veto when top candidates are otherwise close",
            ]
        )
    lane_family = str(((entry.get("master_consistency_card") or {}).get("lane_family")) or "").strip()
    if lane_family == "three_quarter":
        focus.append("for three-quarter winners, prioritize cheek shape and turn naturalness over small score gaps")
    elif lane_family == "side":
        focus.append("for side winners, prioritize profile contour and neck-head connection over frontal face similarity")
    elif lane_family == "back":
        focus.append("for back winners, treat posterior structure as the main evidence and keep identity interpretation conservative")
    if isinstance(face_sim, (int, float)) and isinstance(body_sim, (int, float)) and float(face_sim) < float(body_sim) - 0.08:
        focus.append("body stays closer than face; check whether facial rendering drift is the real blocker")
    if isinstance(depth_sim, (int, float)) and isinstance(world3d_sim, (int, float)) and min(float(depth_sim), float(world3d_sim)) < 0.82:
        focus.append("3D structure is the main drift axis; compare shoulders, torso depth, and leg length feel side by side")
    if len(focus) == 0:
        focus.append("drift signals are mild; confirm with GPT/human before promoting into the curated bank")
    return _dedupe_keep_order(focus, limit=6)


def _drift_review_prompts(flags: Sequence[str], severity: str) -> List[str]:
    prompts: List[str] = []
    if "WINNER_BANK_FACE_DRIFT" in flags:
        prompts.append("Face similarity to the curated bank is weak. Verify face shape and age impression before admitting this winner.")
    if "WINNER_BANK_BODY_DRIFT" in flags:
        prompts.append("Body signature drift is visible. Compare body proportion and shoulder-hip structure against the curated bank.")
    if "WINNER_BANK_3D_DRIFT" in flags or "WINNER_BANK_WORLD3D_DRIFT" in flags:
        prompts.append("3D drift is visible. Check torso depth, shoulder-pelvis axis, and lower-limb structure before promotion.")
    if severity == "high":
        prompts.append("Cross-batch drift is high enough that this winner should remain advisory until GPT/human explicitly confirms it.")
    elif severity == "medium":
        prompts.append("Cross-batch drift is moderate. Review against curated winners before promotion.")
    if len(prompts) == 0:
        prompts.append("Curated-bank drift is currently mild. A final GPT/human check is still required before promotion.")
    return _dedupe_keep_order(prompts, limit=5)


def enrich_shot_selection_pairwise_cards(shot_selection: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
    for group in shot_selection.get("groups") or []:
        shortlist = list(group.get("shortlist") or [])
        if len(shortlist) <= 1:
            group["pairwise_compare_cards"] = []
            continue
        top_row = shortlist[0]
        cards: List[Dict[str, Any]] = []
        for candidate_row in shortlist[1 : max(1, min(len(shortlist), top_n))]:
            card = {
                "top_image": top_row.get("image"),
                "candidate_image": candidate_row.get("image"),
                "candidate_rank": candidate_row.get("rank"),
                "selection_gap": ((candidate_row.get("delta_vs_top") or {}).get("selection_score")),
                "component_deltas": ((candidate_row.get("delta_vs_top") or {}).get("component_scores") or {}),
                "heavy_review_deltas": _heavy_delta(top_row, candidate_row),
                "manual_focus": _build_focus_points(top_row, candidate_row),
                "top_winner_reasons": list(top_row.get("winner_reasons") or [])[:4],
                "candidate_winner_reasons": list(candidate_row.get("winner_reasons") or [])[:4],
                "candidate_caution_reasons": list(candidate_row.get("caution_reasons") or [])[:4],
            }
            cards.append(card)
        group["pairwise_compare_cards"] = cards
    shot_selection["pairwise_compare_enabled"] = True
    return shot_selection


def _build_candidate_entry(
    group: Dict[str, Any],
    candidate_row: Dict[str, Any],
    item: Dict[str, Any],
    signal: Dict[str, Any],
    target_profile: Optional[str],
) -> Dict[str, Any]:
    debug = item.get("debug") or {}
    diagnostics = debug.get("collection_diagnostics") or {}
    return {
        "schema_version": "winner_bank_candidate_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_profile": target_profile,
        "group_key": group.get("group_key"),
        "group_source": group.get("group_source"),
        "layer_tag": group.get("layer_tag"),
        "look_key": group.get("look_key"),
        "image": candidate_row.get("image"),
        "record_key": candidate_row.get("record_key"),
        "rank": candidate_row.get("rank"),
        "review_bucket": candidate_row.get("review_bucket"),
        "status": candidate_row.get("status"),
        "selection_score": candidate_row.get("selection_score"),
        "enhanced_selection_score": ((candidate_row.get("heavy_review") or {}).get("enhanced_selection_score")),
        "scores": dict(item.get("scores") or {}),
        "component_scores": dict(candidate_row.get("component_scores") or {}),
        "batch_reference": dict(group.get("batch_reference") or {}),
        "winner_reasons": list(candidate_row.get("winner_reasons") or []),
        "caution_reasons": list(candidate_row.get("caution_reasons") or []),
        "top_reasons": list(candidate_row.get("top_reasons") or []),
        "master_consistency_card": dict(debug.get("master_consistency_card") or {}),
        "identity_centroid_similarity": diagnostics.get("identity_centroid_similarity"),
        "body_identity_centroid_similarity": diagnostics.get("body_identity_centroid_similarity"),
        "depth_identity_centroid_similarity": diagnostics.get("depth_identity_centroid_similarity"),
        "world3d_identity_centroid_similarity": diagnostics.get("world3d_identity_centroid_similarity"),
        "face_embedding": _json_ready_vector(signal.get("embedding")),
        "body_signature": _json_ready_vector(signal.get("body_signature")),
        "depth_signature": _json_ready_vector(signal.get("depth_signature")),
        "world3d_signature": _json_ready_vector(signal.get("world3d_signature")),
    }


def _load_curated_bank(bank_file: Path) -> Dict[str, Any]:
    if not bank_file.exists():
        return {"available": False, "entries": [], "reason": "missing_bank_file"}
    try:
        payload = json.loads(bank_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "entries": [], "reason": f"invalid_json:{exc}"}
    entries = list(payload.get("entries") or [])
    return {
        "available": len(entries) > 0,
        "entries": entries,
        "reason": None if len(entries) > 0 else "empty_bank",
    }


def load_winner_bank_candidates(candidate_file: Path) -> Dict[str, Any]:
    if not candidate_file.exists():
        return {"available": False, "entries": [], "reason": "missing_candidate_file"}
    try:
        payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "entries": [], "reason": f"invalid_json:{exc}"}
    entries = list(payload.get("entries") or [])
    return {
        "available": len(entries) > 0,
        "entries": entries,
        "reason": None if len(entries) > 0 else "empty_candidate_file",
        "payload": payload,
    }


def load_curated_winner_bank(bank_file: Path) -> Dict[str, Any]:
    payload = _load_curated_bank(bank_file)
    payload["file"] = str(bank_file)
    return payload


def promote_winner_entry(
    candidate_entry: Dict[str, Any],
    curated_bank_file: Path,
    manual_note: Optional[str] = None,
) -> Dict[str, Any]:
    curated = _load_curated_bank(curated_bank_file)
    entries = list(curated.get("entries") or [])
    promoted = dict(candidate_entry)
    promoted["manual_promoted_at_utc"] = datetime.now(timezone.utc).isoformat()
    promoted["manual_note"] = manual_note or ""

    candidate_id = str(promoted.get("record_key") or promoted.get("image") or "").strip()
    existing_index: Optional[int] = None
    for index, entry in enumerate(entries):
        entry_id = str(entry.get("record_key") or entry.get("image") or "").strip()
        if candidate_id and entry_id == candidate_id:
            existing_index = index
            break

    if existing_index is None:
        entries.append(promoted)
        action = "added"
    else:
        entries[existing_index] = promoted
        action = "updated"

    payload = {
        "schema_version": "winner_bank_v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
        "policy": {
            "manual_promotion_required": True,
            "auto_promote_machine_top1": False,
            "final_decision_owner": "custom_gpt_plus_human",
        },
    }
    curated_bank_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "ok",
        "action": action,
        "curated_bank_file": str(curated_bank_file),
        "entry_count": len(entries),
        "promoted_image": promoted.get("image"),
        "promoted_record_key": promoted.get("record_key"),
    }


def _centroid_from_entries(entries: Sequence[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
    rows: List[tuple[np.ndarray, float]] = []
    for entry in entries:
        vector = _normalize_embedding(entry.get(key))
        if vector is None:
            continue
        weight = 1.0
        rows.append((vector, weight))
    centroid = _weighted_sum(rows)
    return _normalize_embedding(centroid)


def build_winner_bank_governance(
    report_items: Sequence[Dict[str, Any]],
    shot_selection: Dict[str, Any],
    identity_samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    target_profile: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_file = output_dir / "winner_bank_candidate.json"
    drift_file = output_dir / "winner_bank_report.json"
    curated_bank_file = output_dir / "winner_bank.json"

    item_by_key: Dict[str, Dict[str, Any]] = {}
    for item in report_items:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        if record_key:
            item_by_key[record_key] = item
    signal_by_key: Dict[str, Dict[str, Any]] = {}
    for signal in identity_samples or []:
        record_key = str(signal.get("record_key") or "").strip()
        if record_key:
            signal_by_key[record_key] = signal

    candidate_entries: List[Dict[str, Any]] = []
    for group in shot_selection.get("groups") or []:
        shortlist = list(group.get("shortlist") or [])
        if len(shortlist) == 0:
            continue
        export_limit = int(group.get("manual_review_window") or len(shortlist) or 1)
        for row in shortlist[: max(1, export_limit)]:
            record_key = str(row.get("record_key") or "").strip()
            item = item_by_key.get(record_key)
            signal = signal_by_key.get(record_key)
            if item is None or signal is None:
                continue
            candidate_entries.append(_build_candidate_entry(group, row, item, signal, target_profile))

    candidate_payload = {
        "schema_version": "winner_bank_candidate_v1",
        "target_profile": target_profile,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(candidate_entries),
        "entries": candidate_entries,
        "policy": {
            "manual_promotion_required": True,
            "auto_promote_machine_top1": False,
            "final_decision_owner": "custom_gpt_plus_human",
            "source": "shortlist_candidates",
        },
    }
    candidate_file.write_text(json.dumps(candidate_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    curated = _load_curated_bank(curated_bank_file)
    drift_rows: List[Dict[str, Any]] = []
    if curated.get("available"):
        entries = list(curated.get("entries") or [])
        face_centroid = _centroid_from_entries(entries, "face_embedding")
        body_centroid = _centroid_from_entries(entries, "body_signature")
        depth_centroid = _centroid_from_entries(entries, "depth_signature")
        world3d_centroid = _centroid_from_entries(entries, "world3d_signature")
        for entry in candidate_entries:
            face_sim = _cosine(_normalize_embedding(entry.get("face_embedding")), face_centroid)
            body_sim = _cosine(_normalize_embedding(entry.get("body_signature")), body_centroid)
            depth_sim = _cosine(_normalize_embedding(entry.get("depth_signature")), depth_centroid)
            world3d_sim = _cosine(_normalize_embedding(entry.get("world3d_signature")), world3d_centroid)
            hybrid_drift = _weighted_mean(
                [
                    (face_sim, 0.42),
                    (body_sim, 0.24),
                    (depth_sim, 0.16),
                    (world3d_sim, 0.18),
                ]
            )
            flags: List[str] = []
            if isinstance(face_sim, (int, float)) and float(face_sim) < 0.78:
                flags.append("WINNER_BANK_FACE_DRIFT")
            if isinstance(body_sim, (int, float)) and float(body_sim) < 0.82:
                flags.append("WINNER_BANK_BODY_DRIFT")
            if isinstance(depth_sim, (int, float)) and float(depth_sim) < 0.84:
                flags.append("WINNER_BANK_3D_DRIFT")
            if isinstance(world3d_sim, (int, float)) and float(world3d_sim) < 0.84:
                flags.append("WINNER_BANK_WORLD3D_DRIFT")
            if isinstance(hybrid_drift, (int, float)) and float(hybrid_drift) < 0.82:
                flags.append("WINNER_BANK_HYBRID_DRIFT")
            severity = _drift_severity(face_sim, body_sim, depth_sim, world3d_sim, hybrid_drift)
            manual_focus = _drift_manual_focus(flags, entry, face_sim, body_sim, depth_sim, world3d_sim)
            manual_review_prompts = _drift_review_prompts(flags, severity)
            drift_rows.append(
                {
                    "image": entry.get("image"),
                    "record_key": entry.get("record_key"),
                    "face_similarity_to_bank": _round_or_none(face_sim),
                    "body_similarity_to_bank": _round_or_none(body_sim),
                    "depth_similarity_to_bank": _round_or_none(depth_sim),
                    "world3d_similarity_to_bank": _round_or_none(world3d_sim),
                    "hybrid_similarity_to_bank": _round_or_none(hybrid_drift),
                    "drift_flags": flags,
                    "drift_severity": severity,
                    "manual_focus": manual_focus,
                    "manual_review_prompts": manual_review_prompts,
                }
            )

    drift_flag_counts: Dict[str, int] = {}
    for row in drift_rows:
        for flag in row.get("drift_flags") or []:
            drift_flag_counts[flag] = drift_flag_counts.get(flag, 0) + 1
    top_drift_risks = [
        {"flag": flag, "count": count}
        for flag, count in sorted(drift_flag_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:6]
    ]

    drift_payload = {
        "schema_version": "winner_bank_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_profile": target_profile,
        "curated_bank_file": str(curated_bank_file),
        "curated_bank_available": bool(curated.get("available")),
        "curated_bank_reason": curated.get("reason"),
        "curated_entry_count": len(curated.get("entries") or []),
        "candidate_file": str(candidate_file),
        "candidate_entry_count": len(candidate_entries),
        "drift_row_count": len(drift_rows),
        "status": "drift_checked" if bool(curated.get("available")) else "candidate_export_only",
        "manual_next_step": (
            "promote a human-approved winner into outputs/winner_bank.json before using cross-batch drift checks"
            if not bool(curated.get("available"))
            else "review drift flags before admitting the winner into the training bank"
        ),
        "top_drift_risks": top_drift_risks,
        "drift_rows": drift_rows,
        "manual_promotion_required": True,
    }
    drift_file.write_text(json.dumps(drift_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "enabled": True,
        "mode": "manual_promotion_required",
        "candidate_file": str(candidate_file),
        "drift_report_file": str(drift_file),
        "curated_bank_file": str(curated_bank_file),
        "curated_bank_available": bool(curated.get("available")),
        "curated_entry_count": len(curated.get("entries") or []),
        "candidate_entry_count": len(candidate_entries),
        "drift_row_count": len(drift_rows),
    }
