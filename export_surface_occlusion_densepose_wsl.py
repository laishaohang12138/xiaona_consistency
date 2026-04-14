import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

ARTIFACT_SCHEMA = "clothing_surface_occlusion_artifact_v1"
FINE_TO_COARSE_SEGMENTATION = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 8,
    14: 9,
    15: 10,
    16: 11,
    17: 10,
    18: 11,
    19: 12,
    20: 13,
    21: 12,
    22: 13,
    23: 14,
    24: 14,
}
ARM_COARSE_LABELS = {2, 3, 10, 11, 12, 13}
LEG_COARSE_LABELS = {4, 5, 6, 7, 8, 9}
TORSO_COARSE_LABELS = {1}
HEAD_COARSE_LABELS = {14}


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    numeric = _safe_float(value)
    return None if numeric is None else round(float(numeric), digits)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _weighted_mean(items: List[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        numeric = _safe_float(value)
        if numeric is None or float(weight) <= 0.0:
            continue
        numerator += numeric * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _range_score(value: Optional[float], low: float, high: float) -> Optional[float]:
    numeric = _safe_float(value)
    if numeric is None or high <= low:
        return None
    return _clamp((numeric - float(low)) / (float(high) - float(low)))


def _wsl_to_windows(path: str) -> str:
    normalized = str(path or "").strip()
    lower = normalized.lower()
    if not lower.startswith("/mnt/") or len(normalized) < 7:
        return normalized
    drive = normalized[5]
    rest = normalized[6:].replace("/", "\\")
    return f"{drive.upper()}:{rest}"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise ValueError("DensePose dump must contain a list")
    return [row for row in payload if isinstance(row, dict)]


def _resolve_image_size(source_image: Path) -> Tuple[int, int]:
    with Image.open(source_image) as image:
        width, height = image.size
    return int(width), int(height)


def _coarse_pixel_counts(labels: torch.Tensor) -> Dict[int, int]:
    values, counts = torch.unique(labels.to(dtype=torch.int64), return_counts=True)
    rows: Dict[int, int] = {}
    for fine_label, count in zip(values.tolist(), counts.tolist()):
        fine_id = int(fine_label)
        if fine_id <= 0:
            continue
        coarse_id = FINE_TO_COARSE_SEGMENTATION.get(fine_id)
        if coarse_id is None:
            continue
        rows[coarse_id] = rows.get(coarse_id, 0) + int(count)
    return rows


def _pick_detection(row: Dict[str, Any], *, min_score: float) -> Tuple[int, Dict[str, Any]]:
    detections = row.get("pred_densepose") or []
    scores = row.get("scores")
    boxes = row.get("pred_boxes_XYXY")
    if not isinstance(detections, list) or scores is None or boxes is None:
        raise ValueError("DensePose dump row is missing detections, scores, or boxes")
    best_index = -1
    best_score = -1.0
    best_payload: Dict[str, Any] = {}
    for index, densepose_result in enumerate(detections):
        detection_score = _safe_float(scores[index].item() if hasattr(scores[index], "item") else scores[index], 0.0) or 0.0
        if detection_score < float(min_score):
            continue
        labels = getattr(densepose_result, "labels", None)
        if labels is None:
            continue
        labels_tensor = labels.to(dtype=torch.int64).cpu()
        body_pixels = int((labels_tensor > 0).sum().item())
        bbox = [float(value) for value in boxes[index].tolist()]
        bbox_width = max(1.0, float(bbox[2] - bbox[0]))
        bbox_height = max(1.0, float(bbox[3] - bbox[1]))
        bbox_area = bbox_width * bbox_height
        fill_ratio = float(body_pixels / max(1.0, float(labels_tensor.numel())))
        ranking = 0.58 * detection_score + 0.27 * fill_ratio + 0.15 * _clamp(bbox_area / 2_400_000.0)
        if ranking > best_score:
            best_index = index
            best_score = ranking
            best_payload = {
                "labels": labels_tensor,
                "bbox_xyxy": bbox,
                "detection_score": detection_score,
                "bbox_fill_ratio": fill_ratio,
                "body_pixels": body_pixels,
            }
    if best_index < 0:
        raise ValueError("no DensePose detections met the minimum score")
    return best_index, best_payload


def _scaled_ratio(pixel_count: int, bbox_xyxy: List[float], labels_shape: Tuple[int, int], image_size: Tuple[int, int]) -> float:
    width, height = image_size
    bbox_width = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
    bbox_height = max(1.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
    mask_height, mask_width = labels_shape
    scale_x = bbox_width / max(1.0, float(mask_width))
    scale_y = bbox_height / max(1.0, float(mask_height))
    estimated_area = float(pixel_count) * scale_x * scale_y
    return _clamp(estimated_area / max(1.0, float(width * height)))


def _artifact_from_row(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    file_name = str(row.get("file_name") or "")
    source_image = Path(args.source_image) if args.source_image else Path(_wsl_to_windows(file_name))
    if not source_image.exists():
        raise FileNotFoundError(f"source image not found: {source_image}")
    image_size = _resolve_image_size(source_image)
    detection_index, selected = _pick_detection(row, min_score=float(args.min_score))
    labels = selected["labels"]
    bbox_xyxy = selected["bbox_xyxy"]
    coarse_counts = _coarse_pixel_counts(labels)
    body_pixels = int(selected["body_pixels"])
    arm_pixels = sum(coarse_counts.get(key, 0) for key in ARM_COARSE_LABELS)
    leg_pixels = sum(coarse_counts.get(key, 0) for key in LEG_COARSE_LABELS)
    torso_pixels = sum(coarse_counts.get(key, 0) for key in TORSO_COARSE_LABELS)
    head_pixels = sum(coarse_counts.get(key, 0) for key in HEAD_COARSE_LABELS)

    visible_body_ratio = _scaled_ratio(body_pixels, bbox_xyxy, tuple(labels.shape), image_size)
    visible_arm_ratio = _scaled_ratio(arm_pixels, bbox_xyxy, tuple(labels.shape), image_size)
    visible_leg_ratio = _scaled_ratio(leg_pixels, bbox_xyxy, tuple(labels.shape), image_size)
    visible_face_ratio = _scaled_ratio(head_pixels, bbox_xyxy, tuple(labels.shape), image_size)

    body_fraction = float(body_pixels) / max(1.0, float(labels.numel()))
    torso_fraction = float(torso_pixels) / max(1.0, float(body_pixels))
    arm_fraction = float(arm_pixels) / max(1.0, float(body_pixels))
    leg_fraction = float(leg_pixels) / max(1.0, float(body_pixels))
    head_fraction = float(head_pixels) / max(1.0, float(body_pixels))
    diversity_count = sum(
        1
        for value in (torso_pixels, arm_pixels, leg_pixels, head_pixels)
        if int(value) > max(32, int(body_pixels * 0.01))
    )
    diversity_score = float(diversity_count / 4.0)
    body_score = _range_score(visible_body_ratio, 0.18, 0.72)
    arm_score = _range_score(visible_arm_ratio, 0.03, 0.18)
    leg_score = _range_score(visible_leg_ratio, 0.05, 0.24)
    head_score = _range_score(visible_face_ratio, 0.01, 0.10)
    torso_score = _range_score(torso_fraction, 0.14, 0.42)
    detection_score = _safe_float(selected["detection_score"], 0.0) or 0.0
    visible_body_surface_alignment = _weighted_mean(
        [
            (body_score, 0.34),
            (diversity_score, 0.20),
            (torso_score, 0.18),
            (arm_score, 0.10),
            (leg_score, 0.12),
            (head_score, 0.02),
            (detection_score, 0.04),
        ]
    )
    garment_occlusion_index = _weighted_mean(
        [
            (1.0 - body_fraction, 0.62),
            (1.0 - diversity_score, 0.18),
            (1.0 - (body_score or 0.0), 0.12),
            (1.0 - (torso_score or 0.0), 0.08),
        ]
    )
    garment_boundary_risk = _weighted_mean(
        [
            (1.0 - detection_score, 0.30),
            (1.0 - (visible_body_surface_alignment or 0.0), 0.34),
            (1.0 - body_fraction, 0.20),
            (abs(arm_fraction - leg_fraction), 0.16),
        ]
    )
    clothing_surface_confidence = _weighted_mean(
        [
            (detection_score, 0.42),
            (body_fraction, 0.26),
            (diversity_score, 0.18),
            (visible_body_surface_alignment, 0.14),
        ]
    )
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "provider_name": args.provider_name,
        "provider_family": "clothing_invariant_surface",
        "provider_version": args.provider_version,
        "model_id": args.model_id,
        "device": args.device,
        "source_path": str(source_image.resolve()),
        "source_role": args.source_role,
        "track_id": args.track_id,
        "metrics": {
            "visible_body_surface_alignment": _round_or_none(visible_body_surface_alignment),
            "garment_occlusion_index": _round_or_none(garment_occlusion_index),
            "garment_boundary_risk": _round_or_none(garment_boundary_risk),
            "visible_body_ratio": _round_or_none(visible_body_ratio),
            "visible_face_ratio": _round_or_none(visible_face_ratio),
            "visible_arm_ratio": _round_or_none(visible_arm_ratio),
            "visible_leg_ratio": _round_or_none(visible_leg_ratio),
            "clothing_surface_confidence": _round_or_none(clothing_surface_confidence),
        },
        "conversion_meta": {
            "raw_input": str(Path(args.dump_file).resolve()),
            "raw_file_name": file_name,
            "selected_detection_index": detection_index,
            "selected_detection_score": _round_or_none(detection_score),
            "selected_bbox_xyxy": [round(float(value), 3) for value in bbox_xyxy],
            "selected_bbox_fill_ratio": _round_or_none(body_fraction),
            "body_pixels": body_pixels,
            "coarse_part_pixel_counts": {str(key): int(value) for key, value in coarse_counts.items()},
            "torso_fraction_of_body": _round_or_none(torso_fraction),
            "arm_fraction_of_body": _round_or_none(arm_fraction),
            "leg_fraction_of_body": _round_or_none(leg_fraction),
            "head_fraction_of_body": _round_or_none(head_fraction),
            "surface_diversity_score": _round_or_none(diversity_score),
            "notes": args.notes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official DensePose dump results into XiaoNa clothing-invariant sidecars.")
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-image", default="", help="Optional Windows source image path. Defaults to the dump row file_name if it is already under /mnt/<drive>/...")
    parser.add_argument("--provider-name", default="densepose_surface_occlusion")
    parser.add_argument("--provider-version", default="densepose_surface_occlusion_v1")
    parser.add_argument("--model-id", default="densepose_detectron2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-role", default="candidate", choices=["candidate", "master_truth"])
    parser.add_argument("--track-id", default="")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    rows = _load_rows(Path(args.dump_file))
    if len(rows) == 0:
        raise ValueError("DensePose dump does not contain any rows")
    artifact = _artifact_from_row(rows[0], args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "metrics": artifact.get("metrics") or {}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
