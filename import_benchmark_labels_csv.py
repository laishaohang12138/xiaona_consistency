from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _parse_reason_list(text: str) -> List[str]:
    raw = str(text or "").replace("；", ";").replace("，", ",")
    tokens: List[str] = []
    for chunk in raw.replace("|", ";").replace(",", ";").split(";"):
        value = chunk.strip()
        if value:
            tokens.append(value)
    return tokens


def _parse_weight(text: Any, default: float = 1.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Import edited benchmark label CSV back into JSON.")
    parser.add_argument("--template", required=True, help="Source benchmark label JSON template")
    parser.add_argument("--csv", required=True, help="Edited CSV file")
    parser.add_argument("--output", required=True, help="Output benchmark label JSON")
    args = parser.parse_args()

    template_path = Path(args.template).resolve()
    csv_path = Path(args.csv).resolve()
    output_path = Path(args.output).resolve()

    payload = _read_json_object(template_path)
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError("benchmark label JSON must contain an 'items' object")

    seen = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = str(row.get("image") or "").strip()
            if not image_name:
                continue
            if image_name not in items:
                raise ValueError(f"CSV contains unknown image key: {image_name}")
            seen.add(image_name)
            node = items[image_name]
            if not isinstance(node, dict):
                node = {}
                items[image_name] = node
            node["expected_status"] = str(row.get("expected_status") or "").strip()
            node["expected_task_profile"] = str(row.get("expected_task_profile") or "").strip()
            node["expected_view_lane"] = str(row.get("expected_view_lane") or "").strip()
            node["expected_view_lane_detail"] = str(row.get("expected_view_lane_detail") or "").strip()
            node["must_have_reasons"] = _parse_reason_list(str(row.get("must_have_reasons") or ""))
            node["must_not_have_reasons"] = _parse_reason_list(str(row.get("must_not_have_reasons") or ""))
            node["weight"] = _parse_weight(row.get("weight"), default=1.0)
            node["notes"] = str(row.get("notes") or "")

    missing = sorted(set(items.keys()) - seen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "template": str(template_path),
                "csv": str(csv_path),
                "output": str(output_path),
                "csv_rows": len(seen),
                "missing_rows_from_csv": missing[:10],
                "missing_count": len(missing),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
