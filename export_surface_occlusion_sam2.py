import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ARTIFACT_SCHEMA = "clothing_surface_occlusion_artifact_v1"
DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-tiny"
DEFAULT_DEPLOY_DIR = Path(__file__).resolve().parent / "external" / "models" / "SAM2"


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


def _image_files(input_dir: Path) -> List[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _centrality_score(bbox: List[int], width: int, height: int) -> float:
    x, y, w, h = [float(v) for v in bbox]
    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = abs(cx - width / 2.0) / max(1.0, width / 2.0)
    dy = abs(cy - height / 2.0) / max(1.0, height / 2.0)
    return _clamp(1.0 - 0.65 * dx - 0.35 * dy)


def _shape_score(bbox: List[int], width: int, height: int) -> float:
    del width
    _, _, w, h = [float(v) for v in bbox]
    aspect = h / max(1.0, w)
    height_ratio = h / max(1.0, float(height))
    aspect_score = _clamp(1.0 - abs(aspect - 2.1) / 2.4, 0.20, 1.0)
    height_score = _clamp((height_ratio - 0.38) / 0.46, 0.20, 1.0)
    return _clamp(0.55 * aspect_score + 0.45 * height_score)


def _pick_subject_mask(masks: List[Dict[str, Any]], width: int, height: int) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for row in masks:
        segmentation = row.get("segmentation")
        if segmentation is None:
            continue
        mask = np.asarray(segmentation).astype(np.uint8)
        bbox = row.get("bbox") or _mask_bbox(mask)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        area = float(row.get("area") or np.count_nonzero(mask))
        area_ratio = area / max(1.0, float(width * height))
        if area_ratio < 0.03 or area_ratio > 0.92:
            continue
        pred_iou = _safe_float(row.get("predicted_iou"), 0.0) or 0.0
        stability = _safe_float(row.get("stability_score"), 0.0) or 0.0
        score = (
            0.34 * _shape_score(list(bbox), width, height)
            + 0.28 * _centrality_score(list(bbox), width, height)
            + 0.20 * _clamp(area_ratio / 0.55)
            + 0.10 * _clamp(pred_iou)
            + 0.08 * _clamp(stability)
        )
        if score > best_score:
            best_score = score
            best = {
                "mask": mask,
                "bbox": [int(v) for v in bbox],
                "area": area,
                "area_ratio": area_ratio,
                "predicted_iou": pred_iou,
                "stability_score": stability,
                "selection_score": score,
            }
    return best


def _default_checkpoint_path(model_id: str, deploy_dir: Path) -> Optional[Path]:
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    filenames = HF_MODEL_ID_TO_FILENAMES.get(model_id)
    if filenames is None:
        return None
    _, checkpoint_name = filenames
    return deploy_dir / checkpoint_name


def _build_model(
    model_id: str,
    device: str,
    *,
    checkpoint: Optional[Path] = None,
    deploy_dir: Optional[Path] = None,
    prefer_hf: bool = False,
):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2, build_sam2_hf

    deploy_dir = (deploy_dir or DEFAULT_DEPLOY_DIR).resolve()
    load_meta: Dict[str, Any] = {
        "model_id": model_id,
        "device": device,
        "load_mode": "huggingface_hub",
        "checkpoint_path": None,
        "config_name": None,
    }
    model = None
    if not prefer_hf:
        resolved_checkpoint = checkpoint.resolve() if checkpoint else _default_checkpoint_path(model_id, deploy_dir)
        filenames = HF_MODEL_ID_TO_FILENAMES.get(model_id)
        if resolved_checkpoint is not None and resolved_checkpoint.is_file() and filenames is not None:
            config_name, _ = filenames
            model = build_sam2(config_file=config_name, ckpt_path=str(resolved_checkpoint), device=device)
            load_meta = {
                "model_id": model_id,
                "device": device,
                "load_mode": "local_checkpoint",
                "checkpoint_path": str(resolved_checkpoint),
                "config_name": config_name,
            }
    if model is None:
        model = build_sam2_hf(model_id, device=device)
        filenames = HF_MODEL_ID_TO_FILENAMES.get(model_id)
        load_meta["config_name"] = filenames[0] if filenames is not None else None
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.88,
        min_mask_region_area=256,
    ), load_meta


def _artifact_for_image(
    image_path: Path,
    mask_generator: Any,
    *,
    model_id: str,
    device: str,
    load_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    masks = mask_generator.generate(img_rgb)
    subject = _pick_subject_mask(list(masks or []), width, height)
    if subject is None:
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "provider_name": "sam2_surface_occlusion",
            "provider_family": "clothing_invariant_surface",
            "provider_version": "sam2_surface_occlusion_v1",
            "model_id": model_id,
            "device": device,
            "source_path": str(image_path.resolve()),
            "source_role": "candidate",
            "metrics": {},
            "reasons": ["SAM2_SUBJECT_MASK_UNAVAILABLE"],
            "conversion_meta": {
                "mask_count": len(masks or []),
                "model_load_mode": (load_meta or {}).get("load_mode"),
                "checkpoint_path": (load_meta or {}).get("checkpoint_path"),
                "config_name": (load_meta or {}).get("config_name"),
            },
        }
    mask = subject["mask"]
    bbox = subject["bbox"]
    mask_area = float(np.count_nonzero(mask))
    bbox_area = max(1.0, float(bbox[2] * bbox[3]))
    bbox_fill_ratio = _clamp(mask_area / bbox_area)
    visible_body_ratio = _clamp(mask_area / max(1.0, float(width * height)))
    stability = _safe_float(subject.get("stability_score"), 0.0) or 0.0
    predicted_iou = _safe_float(subject.get("predicted_iou"), 0.0) or 0.0
    confidence = _clamp(0.45 * stability + 0.35 * predicted_iou + 0.20 * subject["selection_score"])
    # SAM2 is a mask model, not a clothing parser. Treat this as silhouette
    # occlusion/boundary evidence; garment semantics still come from parsing.
    garment_boundary_risk = _clamp(1.0 - stability)
    garment_occlusion_index = _clamp(1.0 - bbox_fill_ratio)
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "provider_name": "sam2_surface_occlusion",
        "provider_family": "clothing_invariant_surface",
        "provider_version": "sam2_surface_occlusion_v1",
        "model_id": model_id,
        "device": device,
        "source_path": str(image_path.resolve()),
        "source_role": "candidate",
        "metrics": {
            "garment_occlusion_index": _round_or_none(garment_occlusion_index),
            "garment_boundary_risk": _round_or_none(garment_boundary_risk),
            "visible_body_ratio": _round_or_none(visible_body_ratio),
            "clothing_surface_confidence": _round_or_none(confidence),
        },
        "conversion_meta": {
            "mask_count": len(masks or []),
            "selected_bbox_xywh": bbox,
            "selected_mask_area_ratio": _round_or_none(subject["area_ratio"]),
            "selected_bbox_fill_ratio": _round_or_none(bbox_fill_ratio),
            "selected_predicted_iou": _round_or_none(predicted_iou),
            "selected_stability_score": _round_or_none(stability),
            "selected_mask_score": _round_or_none(subject["selection_score"]),
            "model_load_mode": (load_meta or {}).get("load_mode"),
            "checkpoint_path": (load_meta or {}).get("checkpoint_path"),
            "config_name": (load_meta or {}).get("config_name"),
            "notes": "SAM2 provides silhouette/occlusion evidence only; it does not identify XiaoNa or classify clothing.",
        },
    }


def _output_for(image_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        return image_path.with_suffix(image_path.suffix + ".surface_occlusion.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{image_path.name}.surface_occlusion.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM2 silhouette/occlusion sidecars for XiaoNa clothing-invariant QA.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Single image to process.")
    group.add_argument("--input-dir", type=Path, help="Directory of images to process.")
    parser.add_argument("--output", type=Path, default=None, help="Output file for --image mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --input-dir mode.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local SAM2 checkpoint path. Defaults to external/models/SAM2/<checkpoint_name> if present.")
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR, help="Directory holding locally deployed SAM2 checkpoints.")
    parser.add_argument("--prefer-hf", action="store_true", help="Ignore local deployment and always use Hugging Face cache/download.")
    args = parser.parse_args()

    images = [args.image] if args.image else _image_files(args.input_dir)
    mask_generator, load_meta = _build_model(
        args.model_id,
        args.device,
        checkpoint=args.checkpoint,
        deploy_dir=args.deploy_dir,
        prefer_hf=args.prefer_hf,
    )
    rows = []
    for image_path in images:
        image_path = image_path.resolve()
        output_path = args.output if args.output and len(images) == 1 else _output_for(image_path, args.output_dir)
        artifact = _artifact_for_image(
            image_path,
            mask_generator,
            model_id=args.model_id,
            device=args.device,
            load_meta=load_meta,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"image": str(image_path), "output": str(output_path), "metrics": artifact.get("metrics") or {}})
    print(json.dumps({"status": "ok", "count": len(rows), "model_load": load_meta, "items": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
