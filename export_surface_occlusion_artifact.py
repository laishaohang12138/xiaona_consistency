import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

ARTIFACT_SCHEMA = "clothing_surface_occlusion_artifact_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_payload(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("surface occlusion input must be a JSON object")
    return payload


def _iter_children(node: Any) -> Iterable[Any]:
    if isinstance(node, dict):
        return node.values()
    if isinstance(node, (list, tuple)):
        return node
    return []


def _resolve_path(node: Any, dotted_path: str) -> Any:
    current = node
    for raw_chunk in str(dotted_path).split("."):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if isinstance(current, dict):
            if chunk not in current:
                return None
            current = current.get(chunk)
            continue
        if isinstance(current, (list, tuple)):
            try:
                current = current[int(chunk)]
            except Exception:
                return None
            continue
        return None
    return current


def _search_by_aliases(node: Any, aliases: Sequence[str], max_depth: int = 5) -> Any:
    if max_depth < 0:
        return None
    if isinstance(node, dict):
        for alias in aliases:
            if alias in node:
                return node.get(alias)
    for child in _iter_children(node):
        if isinstance(child, (dict, list, tuple)):
            found = _search_by_aliases(child, aliases, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _pick_metric(payload: Dict[str, Any], *, override: str, aliases: Sequence[str], fallback: Optional[float]) -> Optional[float]:
    if fallback is not None:
        return _safe_float(fallback)
    value = _resolve_path(payload, override) if override else _search_by_aliases(payload, aliases)
    return _safe_float(value)


def _build_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_payload(args.input)
    metrics = {
        "visible_body_surface_alignment": _pick_metric(
            payload,
            override=args.visible_body_surface_alignment_key,
            aliases=["visible_body_surface_alignment", "densepose_surface_alignment", "body_surface_alignment"],
            fallback=args.visible_body_surface_alignment,
        ),
        "garment_occlusion_index": _pick_metric(
            payload,
            override=args.garment_occlusion_index_key,
            aliases=["garment_occlusion_index", "occlusion_index", "clothing_occlusion_index"],
            fallback=args.garment_occlusion_index,
        ),
        "garment_boundary_risk": _pick_metric(
            payload,
            override=args.garment_boundary_risk_key,
            aliases=["garment_boundary_risk", "silhouette_boundary_risk"],
            fallback=args.garment_boundary_risk,
        ),
        "visible_body_ratio": _pick_metric(
            payload,
            override=args.visible_body_ratio_key,
            aliases=["visible_body_ratio", "densepose_visible_body_ratio", "human_surface_visible_ratio"],
            fallback=args.visible_body_ratio,
        ),
        "visible_face_ratio": _pick_metric(
            payload,
            override=args.visible_face_ratio_key,
            aliases=["visible_face_ratio", "face_visible_ratio"],
            fallback=args.visible_face_ratio,
        ),
        "visible_arm_ratio": _pick_metric(
            payload,
            override=args.visible_arm_ratio_key,
            aliases=["visible_arm_ratio", "arm_visible_ratio"],
            fallback=args.visible_arm_ratio,
        ),
        "visible_leg_ratio": _pick_metric(
            payload,
            override=args.visible_leg_ratio_key,
            aliases=["visible_leg_ratio", "leg_visible_ratio"],
            fallback=args.visible_leg_ratio,
        ),
        "clothing_surface_confidence": _pick_metric(
            payload,
            override=args.confidence_key,
            aliases=["clothing_surface_confidence", "surface_confidence", "occlusion_confidence", "confidence"],
            fallback=args.confidence,
        ),
    }
    source_image = Path(args.source_image).resolve() if args.source_image else None
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "provider_name": args.provider_name,
        "provider_family": "clothing_invariant_surface",
        "provider_version": args.provider_version,
        "model_id": args.model_id,
        "source_path": str(source_image) if source_image is not None else "",
        "source_role": args.source_role,
        "track_id": args.track_id,
        "metrics": {key: value for key, value in metrics.items() if value is not None},
        "conversion_meta": {
            "raw_input": str(args.input.resolve()) if args.input else "",
            "notes": args.notes,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert external DensePose/SAM2 occlusion output into XiaoNa sidecar artifact.")
    parser.add_argument("--input", type=Path, default=None, help="Optional raw JSON exported by DensePose/SAM2/custom segmenter.")
    parser.add_argument("--output", type=Path, required=True, help="Output .surface_occlusion.json file.")
    parser.add_argument("--source-image", default="", help="Candidate image path this artifact belongs to.")
    parser.add_argument("--source-role", default="candidate", choices=["candidate", "master_truth"])
    parser.add_argument("--track-id", default="", help="Optional track id for future video or multi-person exports.")
    parser.add_argument("--provider-name", default="external_surface_occlusion")
    parser.add_argument("--provider-version", default="external_surface_occlusion_v1")
    parser.add_argument("--model-id", default="densepose_or_sam2")
    parser.add_argument("--notes", default="")

    parser.add_argument("--visible-body-surface-alignment", type=float, default=None)
    parser.add_argument("--garment-occlusion-index", type=float, default=None)
    parser.add_argument("--garment-boundary-risk", type=float, default=None)
    parser.add_argument("--visible-body-ratio", type=float, default=None)
    parser.add_argument("--visible-face-ratio", type=float, default=None)
    parser.add_argument("--visible-arm-ratio", type=float, default=None)
    parser.add_argument("--visible-leg-ratio", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=None)

    parser.add_argument("--visible-body-surface-alignment-key", default="")
    parser.add_argument("--garment-occlusion-index-key", default="")
    parser.add_argument("--garment-boundary-risk-key", default="")
    parser.add_argument("--visible-body-ratio-key", default="")
    parser.add_argument("--visible-face-ratio-key", default="")
    parser.add_argument("--visible-arm-ratio-key", default="")
    parser.add_argument("--visible-leg-ratio-key", default="")
    parser.add_argument("--confidence-key", default="")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    artifact = _build_artifact(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(artifact), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "metrics": artifact.get("metrics") or {}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
