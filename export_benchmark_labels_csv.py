from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


CSV_COLUMNS = [
    "image",
    "current_status",
    "current_task_profile",
    "current_view_lane",
    "current_view_lane_detail",
    "current_view_lane_detail_confidence",
    "current_view_lane_strictness_score",
    "current_shadow_view_lane",
    "current_shadow_view_lane_detail",
    "current_shadow_view_confidence",
    "expected_status",
    "expected_task_profile",
    "expected_view_lane",
    "expected_view_lane_detail",
    "must_have_reasons",
    "must_not_have_reasons",
    "weight",
    "notes",
]


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _join_reasons(values: Iterable[Any]) -> str:
    return " | ".join(str(v).strip() for v in values if str(v).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark label JSON into CSV for Excel/WPS editing.")
    parser.add_argument("--input", required=True, help="Path to benchmark label JSON template")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    payload = _read_json_object(input_path)
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError("benchmark label JSON must contain an 'items' object")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for image_name in sorted(items.keys()):
            node = items.get(image_name)
            if not isinstance(node, dict):
                continue
            writer.writerow(
                {
                    "image": image_name,
                    "current_status": node.get("current_status", ""),
                    "current_task_profile": node.get("current_task_profile", ""),
                    "current_view_lane": node.get("current_view_lane", ""),
                    "current_view_lane_detail": node.get("current_view_lane_detail", ""),
                    "current_view_lane_detail_confidence": node.get("current_view_lane_detail_confidence", ""),
                    "current_view_lane_strictness_score": node.get("current_view_lane_strictness_score", ""),
                    "current_shadow_view_lane": node.get("current_shadow_view_lane", ""),
                    "current_shadow_view_lane_detail": node.get("current_shadow_view_lane_detail", ""),
                    "current_shadow_view_confidence": node.get("current_shadow_view_confidence", ""),
                    "expected_status": node.get("expected_status", ""),
                    "expected_task_profile": node.get("expected_task_profile", ""),
                    "expected_view_lane": node.get("expected_view_lane", ""),
                    "expected_view_lane_detail": node.get("expected_view_lane_detail", ""),
                    "must_have_reasons": _join_reasons(node.get("must_have_reasons") or []),
                    "must_not_have_reasons": _join_reasons(node.get("must_not_have_reasons") or []),
                    "weight": node.get("weight", 1.0),
                    "notes": node.get("notes", ""),
                }
            )

    print(
        json.dumps(
            {
                "ok": True,
                "input": str(input_path),
                "output": str(output_path),
                "count": len(items),
                "columns": CSV_COLUMNS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
