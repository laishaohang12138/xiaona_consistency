from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _build_item_lookup(report_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in report_payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        image_name = str(item.get("image") or "").strip()
        if image_name:
            lookup[image_name] = item
    return lookup


def _auto_status(
    *,
    current_status: str,
    lane: str,
    body_truth_alignment: float | None,
    face_identity: float | None,
    face_normalize: float | None,
) -> str:
    status = str(current_status or "").strip().upper() or "WARN"
    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"

    # Side/back batches are currently review lanes. Keep auto-labeling conservative:
    # preserve explicit FAIL, cap everything else at WARN, and move truth drift into notes.
    if lane in {"side_90", "back_180"}:
        return "FAIL" if status == "FAIL" else "WARN"

    if body_truth_alignment is not None:
        if body_truth_alignment < 0.68:
            return "FAIL"
        if body_truth_alignment < 0.78:
            status = "WARN"

    if lane in {"front", "three_quarter", "front_like", "three_quarter_left", "three_quarter_right"}:
        if (
            face_identity is not None
            and face_normalize is not None
            and face_normalize >= 0.50
            and face_identity < 0.58
        ):
            return "FAIL"
        if face_identity is not None and face_identity < 0.68:
            status = "WARN"

    return status


def _auto_note(
    *,
    lane: str,
    body_truth_alignment: float | None,
    face_identity: float | None,
    face_coverage: float | None,
) -> str:
    parts: List[str] = []
    if lane:
        parts.append(f"自动预标 lane={lane}")
    if body_truth_alignment is not None:
        if body_truth_alignment < 0.68:
            parts.append(f"116-1 真相偏离明显 align={body_truth_alignment:.4f}")
        elif body_truth_alignment < 0.78:
            parts.append(f"116-1 真相存在可见漂移 align={body_truth_alignment:.4f}")
        else:
            parts.append(f"116-1 真相较稳 align={body_truth_alignment:.4f}")
    if face_identity is not None:
        parts.append(f"0号脸辅助 identity={face_identity:.4f}")
    if face_coverage is not None and face_coverage < 0.12:
        parts.append(f"可见脸面积偏小 coverage={face_coverage:.4f}")
    return "；".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-prefill benchmark labels from qa_report evidence.")
    parser.add_argument("--template", required=True, help="Benchmark template JSON")
    parser.add_argument("--report", required=True, help="QA report JSON")
    parser.add_argument("--output", required=True, help="Output auto-filled benchmark JSON")
    args = parser.parse_args()

    template_path = Path(args.template).resolve()
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()

    template = _read_json_object(template_path)
    report_payload = _read_json_object(report_path)
    template_items = template.get("items")
    if not isinstance(template_items, dict):
        raise ValueError("template must contain an 'items' object")

    item_lookup = _build_item_lookup(report_payload)
    status_counter: Counter[str] = Counter()

    for image_name, node in template_items.items():
        if not isinstance(node, dict):
            continue
        item = item_lookup.get(str(image_name).strip())
        if not item:
            continue

        debug = item.get("debug") or {}
        if not isinstance(debug, dict):
            debug = {}
        master = debug.get("master_consistency_card") or {}
        if not isinstance(master, dict):
            master = {}
        face_shadow = debug.get("face_canonical_shadow") or {}
        if not isinstance(face_shadow, dict):
            face_shadow = {}

        lane = str(debug.get("view_lane") or node.get("current_view_lane") or "").strip()
        lane_detail = str(debug.get("view_lane_detail") or node.get("current_view_lane_detail") or "").strip()
        current_status = str(item.get("status") or node.get("current_status") or "WARN").strip().upper()
        current_profile = str(item.get("task_profile") or node.get("current_task_profile") or "").strip()
        body_truth_alignment = _safe_float(master.get("body_truth_alignment"))
        face_identity = _safe_float(face_shadow.get("canonical_face_identity_similarity"))
        face_normalize = _safe_float(face_shadow.get("face_pose_normalization_confidence"))
        face_coverage = _safe_float(face_shadow.get("visible_face_coverage"))

        expected_status = _auto_status(
            current_status=current_status,
            lane=lane,
            body_truth_alignment=body_truth_alignment,
            face_identity=face_identity,
            face_normalize=face_normalize,
        )
        status_counter[expected_status] += 1

        node["expected_status"] = expected_status
        node["expected_task_profile"] = current_profile
        node["expected_view_lane"] = lane
        node["expected_view_lane_detail"] = lane_detail
        node["notes"] = _auto_note(
            lane=lane,
            body_truth_alignment=body_truth_alignment,
            face_identity=face_identity,
            face_coverage=face_coverage,
        )

    template["autofill_meta"] = {
        "label_origin": "auto_prefill_v1",
        "requires_human_review": True,
        "note": "This file is machine-prefilled from current QA evidence and must not be treated as frozen benchmark truth.",
        "status_counts": dict(status_counter),
        "report_file": str(report_path),
        "template_file": str(template_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "count": len(template_items),
                "status_counts": dict(status_counter),
                "requires_human_review": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
