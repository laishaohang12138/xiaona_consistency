from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .qa_utils import image_read_bgr

try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover - optional heavy dependency
    torch = None
    torch_f = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional heavy dependency
    Image = None

try:
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
except Exception:  # pragma: no cover - optional heavy dependency
    AutoModelForSemanticSegmentation = None
    SegformerImageProcessor = None


_HEAVY_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
_HEAVY_BUNDLE: Optional[Dict[str, Any]] = None

_GARMENT_LABELS = {
    "Upper-clothes",
    "Skirt",
    "Pants",
    "Dress",
    "Belt",
    "Scarf",
}
_UPPER_LABELS = {
    "Upper-clothes",
    "Dress",
    "Belt",
    "Scarf",
}
_LOWER_LABELS = {
    "Skirt",
    "Pants",
    "Dress",
}
_VISIBLE_BODY_LABELS = {
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
}
_FACE_LABELS = {"Face"}
_ARM_LABELS = {"Left-arm", "Right-arm"}
_LEG_LABELS = {"Left-leg", "Right-leg"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _normalize_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        emb = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-8:
        return None
    return emb / norm


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


def _weighted_geometric_mean(items: Sequence[tuple[Optional[float], float]], floor: float = 1e-4) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None:
            continue
        clipped = _clamp(float(value), floor, 1.0)
        numerator += float(weight) * float(np.log(clipped))
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(np.exp(numerator / denominator))


def _load_heavy_bundle() -> Dict[str, Any]:
    global _HEAVY_BUNDLE
    if _HEAVY_BUNDLE is not None:
        return _HEAVY_BUNDLE

    bundle: Dict[str, Any] = {
        "available": False,
        "model_id": _HEAVY_MODEL_ID,
        "reason": None,
        "device": "cpu",
        "processor": None,
        "model": None,
        "id2label": {},
    }
    if torch is None or torch_f is None or Image is None or SegformerImageProcessor is None or AutoModelForSemanticSegmentation is None:
        bundle["reason"] = "missing_dependencies"
        _HEAVY_BUNDLE = bundle
        return bundle

    try:
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        processor = SegformerImageProcessor.from_pretrained(_HEAVY_MODEL_ID, local_files_only=True)
        model = AutoModelForSemanticSegmentation.from_pretrained(_HEAVY_MODEL_ID, local_files_only=True)
        model.to(device)
        model.eval()
        bundle.update(
            {
                "available": True,
                "device": device,
                "processor": processor,
                "model": model,
                "id2label": dict(getattr(model.config, "id2label", {}) or {}),
            }
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        bundle["reason"] = f"model_load_failed:{exc}"
    _HEAVY_BUNDLE = bundle
    return bundle


def _mask_ratio(mask: np.ndarray, roi: Tuple[int, int, int, int], subject_mask: np.ndarray) -> Optional[float]:
    x1, y1, x2, y2 = roi
    subject_crop = subject_mask[y1:y2, x1:x2]
    if subject_crop.size == 0:
        return None
    subject_pixels = int(np.count_nonzero(subject_crop))
    if subject_pixels <= 0:
        return None
    mask_crop = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(mask_crop) / max(1, subject_pixels))


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w <= 1 or h <= 1:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def _clip_roi(roi: Tuple[float, float, float, float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    x1 = int(max(0, min(width - 1, round(float(x1)))))
    y1 = int(max(0, min(height - 1, round(float(y1)))))
    x2 = int(max(1, min(width, round(float(x2)))))
    y2 = int(max(1, min(height, round(float(y2)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _label_mask(seg_map: np.ndarray, id2label: Dict[int, str], label_names: set[str]) -> np.ndarray:
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for label_id, label_name in id2label.items():
        if str(label_name) in label_names:
            mask[seg_map == int(label_id)] = 1
    return mask


def _first_and_last_rows(mask: np.ndarray, x1: int, x2: int) -> tuple[Optional[int], Optional[int]]:
    if x2 <= x1:
        return None, None
    crop = mask[:, x1:x2]
    if crop.size == 0:
        return None, None
    rows = np.where(np.count_nonzero(crop, axis=1) > 0)[0]
    if rows.size == 0:
        return None, None
    return int(rows[0]), int(rows[-1])


def _extract_parser_metrics(img_bgr: np.ndarray) -> Dict[str, Any]:
    bundle = _load_heavy_bundle()
    out: Dict[str, Any] = {
        "ok": False,
        "model_id": bundle.get("model_id"),
        "device": bundle.get("device"),
        "confidence": 0.0,
        "parser_boundary_alignment": None,
        "parser_visible_body_alignment": None,
        "garment_coverage_ratio": None,
        "upper_cloth_coverage": None,
        "lower_cloth_coverage": None,
        "neckline_depth_ratio": None,
        "shoulder_cloth_balance": None,
        "visible_body_ratio": None,
        "visible_face_ratio": None,
        "visible_arm_ratio": None,
        "visible_leg_ratio": None,
        "hem_depth_ratio": None,
        "boundary_signature": None,
        "visible_body_signature": None,
        "reasons": [],
    }
    if not bundle.get("available"):
        out["reasons"] = [f"HEAVY_REVIEW_UNAVAILABLE:{bundle.get('reason') or 'disabled'}"]
        return out

    processor = bundle["processor"]
    model = bundle["model"]
    id2label = bundle["id2label"]
    height, width = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    with torch.inference_mode():
        inputs = processor(images=pil_image, return_tensors="pt")
        device = str(bundle.get("device") or "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = model(**inputs).logits
        logits = torch_f.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.softmax(logits, dim=1)
        conf_map, label_map = torch.max(probs, dim=1)

    seg_map = label_map[0].detach().cpu().numpy().astype(np.int32)
    conf_map_np = conf_map[0].detach().cpu().numpy().astype(np.float32)
    subject_mask = np.where(seg_map != 0, 1, 0).astype(np.uint8)
    subject_bbox = _bbox_from_mask(subject_mask)
    if subject_bbox is None:
        out["reasons"] = ["HEAVY_REVIEW_SUBJECT_MISSING"]
        return out

    garment_mask = _label_mask(seg_map, id2label, _GARMENT_LABELS)
    upper_mask = _label_mask(seg_map, id2label, _UPPER_LABELS)
    lower_mask = _label_mask(seg_map, id2label, _LOWER_LABELS)
    visible_body_mask = _label_mask(seg_map, id2label, _VISIBLE_BODY_LABELS)
    face_mask = _label_mask(seg_map, id2label, _FACE_LABELS)
    arm_mask = _label_mask(seg_map, id2label, _ARM_LABELS)
    leg_mask = _label_mask(seg_map, id2label, _LEG_LABELS)

    x1, y1, x2, y2 = subject_bbox
    box_w = float(x2 - x1)
    box_h = float(y2 - y1)
    subject_pixels = int(np.count_nonzero(subject_mask))
    if subject_pixels <= 0:
        out["reasons"] = ["HEAVY_REVIEW_SUBJECT_EMPTY"]
        return out

    torso_roi = _clip_roi((x1 + box_w * 0.18, y1 + box_h * 0.18, x2 - box_w * 0.18, y1 + box_h * 0.60), width, height)
    lower_roi = _clip_roi((x1 + box_w * 0.22, y1 + box_h * 0.58, x2 - box_w * 0.22, y2 - box_h * 0.04), width, height)
    shoulder_left_roi = _clip_roi((x1 + box_w * 0.06, y1 + box_h * 0.10, x1 + box_w * 0.38, y1 + box_h * 0.28), width, height)
    shoulder_right_roi = _clip_roi((x2 - box_w * 0.38, y1 + box_h * 0.10, x2 - box_w * 0.06, y1 + box_h * 0.28), width, height)

    garment_coverage_ratio = float(np.count_nonzero(garment_mask) / max(1, subject_pixels))
    upper_cloth_coverage = _mask_ratio(upper_mask, torso_roi, subject_mask) if torso_roi is not None else None
    lower_cloth_coverage = _mask_ratio(lower_mask, lower_roi, subject_mask) if lower_roi is not None else None
    visible_body_ratio = float(np.count_nonzero(visible_body_mask) / max(1, subject_pixels))
    visible_face_ratio = float(np.count_nonzero(face_mask) / max(1, subject_pixels))
    visible_arm_ratio = float(np.count_nonzero(arm_mask) / max(1, subject_pixels))
    visible_leg_ratio = float(np.count_nonzero(leg_mask) / max(1, subject_pixels))

    left_upper = _mask_ratio(upper_mask, shoulder_left_roi, subject_mask) if shoulder_left_roi is not None else None
    right_upper = _mask_ratio(upper_mask, shoulder_right_roi, subject_mask) if shoulder_right_roi is not None else None
    shoulder_cloth_balance = None
    if left_upper is not None and right_upper is not None:
        hi = max(float(left_upper), float(right_upper))
        if hi > 1e-6:
            shoulder_cloth_balance = float(min(float(left_upper), float(right_upper)) / hi)

    band_x1 = int(x1 + box_w * 0.38)
    band_x2 = int(x2 - box_w * 0.38)
    face_y1, face_y2 = None, None
    face_bbox = _bbox_from_mask(face_mask)
    if face_bbox is not None:
        _, face_y1, _, face_y2 = face_bbox
    first_garment_row, last_garment_row = _first_and_last_rows(upper_mask | lower_mask, band_x1, band_x2)
    neckline_depth_ratio = None
    if face_y2 is not None and first_garment_row is not None and box_h > 1e-6:
        neckline_gap = max(0.0, float(first_garment_row - face_y2))
        neckline_depth_ratio = _clamp(neckline_gap / max(1.0, box_h * 0.28), 0.0, 1.0)
    hem_depth_ratio = None
    if last_garment_row is not None and box_h > 1e-6:
        hem_depth_ratio = _clamp((float(last_garment_row) - float(y1)) / max(1.0, box_h), 0.0, 1.0)

    boundary_vector = np.asarray(
        [
            (garment_coverage_ratio - 0.82) / 0.18,
            ((upper_cloth_coverage or 0.0) - 0.80) / 0.18,
            ((lower_cloth_coverage or 0.0) - 0.94) / 0.12,
            ((neckline_depth_ratio or 0.0) - 0.06) / 0.12,
            ((shoulder_cloth_balance or 0.0) - 0.94) / 0.10,
            ((hem_depth_ratio or 0.0) - 0.88) / 0.14,
        ],
        dtype=np.float32,
    )
    visible_vector = np.asarray(
        [
            (visible_body_ratio - 0.16) / 0.16,
            (visible_face_ratio - 0.06) / 0.05,
            (visible_arm_ratio - 0.06) / 0.06,
            (visible_leg_ratio - 0.06) / 0.06,
            ((shoulder_cloth_balance or 0.0) - 0.94) / 0.10,
        ],
        dtype=np.float32,
    )
    boundary_signature = _normalize_embedding(boundary_vector)
    visible_body_signature = _normalize_embedding(visible_vector)

    subject_conf = float(np.mean(conf_map_np[subject_mask > 0])) if np.count_nonzero(subject_mask) > 0 else 0.0
    garment_conf = float(np.mean(conf_map_np[garment_mask > 0])) if np.count_nonzero(garment_mask) > 0 else subject_conf
    out.update(
        {
            "ok": True,
            "confidence": _clamp(_weighted_mean([(subject_conf, 0.45), (garment_conf, 0.55)]) or 0.0, 0.0, 1.0),
            "garment_coverage_ratio": garment_coverage_ratio,
            "upper_cloth_coverage": upper_cloth_coverage,
            "lower_cloth_coverage": lower_cloth_coverage,
            "neckline_depth_ratio": neckline_depth_ratio,
            "shoulder_cloth_balance": shoulder_cloth_balance,
            "visible_body_ratio": visible_body_ratio,
            "visible_face_ratio": visible_face_ratio,
            "visible_arm_ratio": visible_arm_ratio,
            "visible_leg_ratio": visible_leg_ratio,
            "hem_depth_ratio": hem_depth_ratio,
            "boundary_signature": boundary_signature,
            "visible_body_signature": visible_body_signature,
        }
    )
    if out["confidence"] < 0.58:
        out["reasons"].append("HEAVY_REVIEW_CONFIDENCE_LOW")
    return out


def _resolve_candidate_path(runtime: Any, row: Dict[str, Any]) -> Optional[Path]:
    input_dir = getattr(getattr(runtime, "config", None), "paths", None)
    input_dir = getattr(input_dir, "dir_input", None)
    record_key = str(row.get("record_key") or "").strip()
    image_name = str(row.get("image") or "").strip()
    candidates = []
    if input_dir is not None and record_key:
        candidates.append(Path(input_dir) / record_key)
    if input_dir is not None and image_name:
        candidates.append(Path(input_dir) / image_name)
    for path in candidates:
        if path.exists():
            return path
    return None


def apply_shortlist_heavy_review(
    runtime: Any,
    report_items: Sequence[Dict[str, Any]],
    shot_selection: Dict[str, Any],
    target_profile: Optional[str] = None,
    max_candidates: int = 5,
) -> Dict[str, Any]:
    del target_profile
    groups = shot_selection.get("groups") or []
    bundle = _load_heavy_bundle()
    summary: Dict[str, Any] = {
        "enabled": bool(bundle.get("available")),
        "advisory_only": True,
        "mode": "shortlist_only",
        "model_id": bundle.get("model_id"),
        "device": bundle.get("device"),
        "reason": bundle.get("reason"),
        "group_count": len(groups),
        "processed_group_count": 0,
        "processed_candidate_count": 0,
    }
    if not bool(bundle.get("available")):
        shot_selection["heavy_review_summary"] = summary
        for group in groups:
            group["heavy_review"] = {
                "enabled": False,
                "reason": bundle.get("reason") or "unavailable",
            }
        return shot_selection

    item_by_key: Dict[str, Dict[str, Any]] = {}
    for item in report_items:
        record_key = str(((item.get("collection") or {}).get("input_relative_path") or item.get("image") or "")).strip()
        if record_key:
            item_by_key[record_key] = item

    boundary_cohesions: List[float] = []
    visible_cohesions: List[float] = []
    consensus_matches = 0

    for group in groups:
        shortlist = list(group.get("shortlist") or [])
        if len(shortlist) == 0:
            group["heavy_review"] = {"enabled": False, "reason": "empty_shortlist"}
            continue
        process_count = min(len(shortlist), max(1, max_candidates))
        candidate_rows: List[Dict[str, Any]] = []
        for row in shortlist[:process_count]:
            image_path = _resolve_candidate_path(runtime, row)
            if image_path is None:
                candidate_rows.append(
                    {
                        "record_key": row.get("record_key"),
                        "image": row.get("image"),
                        "ok": False,
                        "reasons": ["HEAVY_REVIEW_IMAGE_MISSING"],
                    }
                )
                continue
            img = image_read_bgr(image_path, runtime.config.standardization)
            if img is None:
                candidate_rows.append(
                    {
                        "record_key": row.get("record_key"),
                        "image": row.get("image"),
                        "ok": False,
                        "reasons": ["HEAVY_REVIEW_IMAGE_READ_ERROR"],
                    }
                )
                continue
            metrics = _extract_parser_metrics(img)
            metrics["record_key"] = row.get("record_key")
            metrics["image"] = row.get("image")
            metrics["base_selection_score"] = row.get("selection_score")
            candidate_rows.append(metrics)
            item = item_by_key.get(str(row.get("record_key") or ""))
            if item is not None:
                item.setdefault("debug", {})["heavy_review"] = {
                    "parser_confidence": _round_or_none(metrics.get("confidence")),
                    "garment_coverage_ratio": _round_or_none(metrics.get("garment_coverage_ratio")),
                    "upper_cloth_coverage": _round_or_none(metrics.get("upper_cloth_coverage")),
                    "lower_cloth_coverage": _round_or_none(metrics.get("lower_cloth_coverage")),
                    "neckline_depth_ratio": _round_or_none(metrics.get("neckline_depth_ratio")),
                    "shoulder_cloth_balance": _round_or_none(metrics.get("shoulder_cloth_balance")),
                    "visible_body_ratio": _round_or_none(metrics.get("visible_body_ratio")),
                    "visible_face_ratio": _round_or_none(metrics.get("visible_face_ratio")),
                    "visible_arm_ratio": _round_or_none(metrics.get("visible_arm_ratio")),
                    "visible_leg_ratio": _round_or_none(metrics.get("visible_leg_ratio")),
                    "reasons": list(metrics.get("reasons") or []),
                }

        boundary_centroid = _normalize_embedding(
            _weighted_sum(
                [
                    (row["boundary_signature"], max(0.1, float(row.get("confidence", 0.0) or 0.0)))
                    for row in candidate_rows
                    if row.get("ok") and row.get("boundary_signature") is not None
                ]
            )
        )
        visible_centroid = _normalize_embedding(
            _weighted_sum(
                [
                    (row["visible_body_signature"], max(0.1, float(row.get("confidence", 0.0) or 0.0)))
                    for row in candidate_rows
                    if row.get("ok") and row.get("visible_body_signature") is not None
                ]
            )
        )

        boundary_sims: List[float] = []
        visible_sims: List[float] = []
        for row in candidate_rows:
            row["parser_boundary_alignment"] = _cosine(row.get("boundary_signature"), boundary_centroid)
            row["parser_visible_body_alignment"] = _cosine(row.get("visible_body_signature"), visible_centroid)
            if isinstance(row.get("parser_boundary_alignment"), (int, float)):
                boundary_sims.append(float(row["parser_boundary_alignment"]))
            if isinstance(row.get("parser_visible_body_alignment"), (int, float)):
                visible_sims.append(float(row["parser_visible_body_alignment"]))
            row["parser_consensus_score"] = _weighted_geometric_mean(
                [
                    (row.get("parser_boundary_alignment"), 0.55),
                    (row.get("parser_visible_body_alignment"), 0.25),
                    (row.get("confidence"), 0.20),
                ]
            )
            row["enhanced_selection_score"] = _weighted_mean(
                [
                    (row.get("base_selection_score"), 0.76),
                    (row.get("parser_consensus_score"), 0.16),
                    (row.get("confidence"), 0.08),
                ]
            )

        boundary_cohesion = _mean(boundary_sims)
        visible_cohesion = _mean(visible_sims)
        if isinstance(boundary_cohesion, (int, float)):
            boundary_cohesions.append(float(boundary_cohesion))
        if isinstance(visible_cohesion, (int, float)):
            visible_cohesions.append(float(visible_cohesion))

        advisory_rows = sorted(
            [row for row in candidate_rows if row.get("ok")],
            key=lambda row: (
                1 if row.get("enhanced_selection_score") is None else 0,
                0.0 if row.get("enhanced_selection_score") is None else -float(row.get("enhanced_selection_score")),
                str(row.get("image") or ""),
            ),
        )
        consensus_top = advisory_rows[0].get("image") if len(advisory_rows) > 0 else None
        if consensus_top and consensus_top == group.get("top_ranked_image"):
            consensus_matches += 1

        shortlist_index = {str(row.get("record_key") or ""): row for row in shortlist}
        for rank, advisory in enumerate(advisory_rows, start=1):
            shortlist_row = shortlist_index.get(str(advisory.get("record_key") or ""))
            heavy_node = {
                "parser_confidence": _round_or_none(advisory.get("confidence")),
                "parser_boundary_alignment": _round_or_none(advisory.get("parser_boundary_alignment")),
                "parser_visible_body_alignment": _round_or_none(advisory.get("parser_visible_body_alignment")),
                "parser_consensus_score": _round_or_none(advisory.get("parser_consensus_score")),
                "enhanced_selection_score": _round_or_none(advisory.get("enhanced_selection_score")),
                "garment_coverage_ratio": _round_or_none(advisory.get("garment_coverage_ratio")),
                "upper_cloth_coverage": _round_or_none(advisory.get("upper_cloth_coverage")),
                "lower_cloth_coverage": _round_or_none(advisory.get("lower_cloth_coverage")),
                "neckline_depth_ratio": _round_or_none(advisory.get("neckline_depth_ratio")),
                "shoulder_cloth_balance": _round_or_none(advisory.get("shoulder_cloth_balance")),
                "visible_body_ratio": _round_or_none(advisory.get("visible_body_ratio")),
                "visible_face_ratio": _round_or_none(advisory.get("visible_face_ratio")),
                "visible_arm_ratio": _round_or_none(advisory.get("visible_arm_ratio")),
                "visible_leg_ratio": _round_or_none(advisory.get("visible_leg_ratio")),
                "rank_in_heavy_review": rank,
                "reasons": list(advisory.get("reasons") or []),
            }
            if shortlist_row is not None:
                shortlist_row["heavy_review"] = heavy_node

        heavy_guidance: List[str] = []
        if consensus_top and consensus_top != group.get("top_ranked_image"):
            heavy_guidance.append("重解析复核的首选与基础排序不一致，建议至少人工对比前两名的领口、肩线和可见肢体比例。")
            group["manual_review_window"] = max(int(group.get("manual_review_window", 1) or 1), min(3, len(shortlist)))
        elif consensus_top:
            heavy_guidance.append("基础排序与重解析复核的首选一致，可优先从该候选开始人工复核。")
        if isinstance(boundary_cohesion, (int, float)) and float(boundary_cohesion) < 0.88:
            heavy_guidance.append("shortlist 的服装边界一致性仍偏松，人工复核时要重点看领口和肩线是否在漂。")
        if isinstance(visible_cohesion, (int, float)) and float(visible_cohesion) < 0.86:
            heavy_guidance.append("shortlist 的可见身体比例仍有波动，人工复核时要注意脸面积和露臂露腿比例是否突然变化。")

        group["review_guidance"] = list(dict.fromkeys(list(group.get("review_guidance") or []) + heavy_guidance))[:6]
        group["heavy_review"] = {
            "enabled": True,
            "advisory_only": True,
            "candidate_count": len(advisory_rows),
            "consensus_top_image": consensus_top,
            "parser_boundary_cohesion": _round_or_none(boundary_cohesion),
            "parser_visible_body_cohesion": _round_or_none(visible_cohesion),
            "parser_confidence_mean": _round_or_none(_mean([row.get("confidence") for row in advisory_rows])),
            "guidance": heavy_guidance[:4],
        }
        summary["processed_group_count"] = int(summary.get("processed_group_count", 0)) + 1
        summary["processed_candidate_count"] = int(summary.get("processed_candidate_count", 0)) + len(advisory_rows)

    summary["parser_boundary_cohesion_mean"] = _round_or_none(_mean(boundary_cohesions))
    summary["parser_visible_body_cohesion_mean"] = _round_or_none(_mean(visible_cohesions))
    summary["consensus_top_match_ratio"] = _round_or_none(
        float(consensus_matches / max(1, int(summary.get("processed_group_count", 0) or 0)))
    )
    shot_selection["heavy_review_summary"] = summary
    return shot_selection
