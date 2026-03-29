from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.qa_artifact_manifest import register_artifact_manifest

ARTIFACT_SCHEMA = "body_canonical_artifact_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _to_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size == 0:
        return None
    return vector


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix == ".npy":
        payload = np.load(path, allow_pickle=True)
        try:
            return payload.item()
        except Exception:
            return payload
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"Unsupported input format: {path.suffix}")


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
        if isinstance(current, (list, tuple, np.ndarray)):
            try:
                index = int(chunk)
            except Exception:
                return None
            try:
                current = current[index]
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
                return node[alias]
    for child in _iter_children(node):
        if isinstance(child, (dict, list, tuple)):
            found = _search_by_aliases(child, aliases, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _select_record(payload: Any, *, index: int, track_id: str = "") -> Any:
    if track_id:
        if isinstance(payload, dict) and track_id in payload:
            return payload[track_id]
        for key in ["tracklets", "tracks", "results", "items"]:
            container = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(container, dict) and track_id in container:
                return container[track_id]
            if isinstance(container, list):
                for item in container:
                    if isinstance(item, dict) and str(item.get("track_id") or item.get("id") or "") == track_id:
                        return item
    if isinstance(payload, list):
        if len(payload) == 0:
            return payload
        return payload[max(0, min(index, len(payload) - 1))]
    if isinstance(payload, tuple):
        if len(payload) == 0:
            return payload
        return payload[max(0, min(index, len(payload) - 1))]
    if isinstance(payload, dict):
        for key in ["predictions", "results", "items", "detections", "humans"]:
            container = payload.get(key)
            if isinstance(container, (list, tuple)) and len(container) > 0:
                return container[max(0, min(index, len(container) - 1))]
            if isinstance(container, dict) and len(container) == 1:
                return next(iter(container.values()))
        return payload
    return payload


def _parse_key_value_pairs(values: Sequence[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for raw in values:
        text = str(raw).strip()
        if not text or "=" not in text:
            raise ValueError(f"Expected KEY=VALUE pair, got: {raw!r}")
        key, value = text.split("=", 1)
        name = key.strip()
        if not name:
            raise ValueError(f"Expected KEY=VALUE pair, got: {raw!r}")
        numeric = _safe_float(value.strip(), None)
        if numeric is None:
            raise ValueError(f"Measurement value must be numeric: {raw!r}")
        parsed[name] = float(numeric)
    return parsed


def _measurement_mapping(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        numeric = _safe_float(raw_value, None)
        if key and numeric is not None:
            out[key] = float(numeric)
    return out


def _pick_field(record: Any, *, override: str, aliases: Sequence[str]) -> Any:
    if override:
        return _resolve_path(record, override)
    return _search_by_aliases(record, aliases)


def _compose_pose_vector(record: Any, *, pose_key: str, global_orient_key: str) -> Optional[np.ndarray]:
    if pose_key:
        explicit_pose = _resolve_path(record, pose_key)
        return _to_vector(explicit_pose)
    body_pose = _pick_field(
        record,
        override="",
        aliases=["body_pose", "pred_body_pose", "body_pose_axis_angle", "pose_body"],
    )
    global_orient = _pick_field(
        record,
        override=global_orient_key,
        aliases=["global_orient", "pred_global_orient", "root_orient", "orient"],
    )
    pose_vector = _pick_field(
        record,
        override="",
        aliases=["pose_vector", "smpl_pose", "pred_pose", "pose"],
    )
    global_orient_vec = _to_vector(global_orient)
    body_pose_vec = _to_vector(body_pose)
    if global_orient_vec is not None and body_pose_vec is not None:
        return np.concatenate([global_orient_vec, body_pose_vec], axis=0)
    direct_pose_vec = _to_vector(pose_vector)
    if direct_pose_vec is not None:
        return direct_pose_vec
    return body_pose_vec


def _build_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_payload(args.input)
    record = _select_record(payload, index=args.index, track_id=args.track_id)

    betas = _pick_field(
        record,
        override=args.beta_key,
        aliases=["betas", "shape_beta", "pred_betas", "smpl_betas", "shape"],
    )
    shape_beta = _to_vector(betas)
    if shape_beta is None:
        raise ValueError(
            "Could not extract shape_beta from input. Use --beta-key to point at the correct field."
        )

    pose_vector = _compose_pose_vector(
        record,
        pose_key=args.pose_key,
        global_orient_key=args.global_orient_key,
    )
    fit_confidence = _safe_float(
        _pick_field(
            record,
            override=args.confidence_key,
            aliases=["fit_confidence", "confidence", "score", "pred_score"],
        ),
        None,
    )
    coverage = _safe_float(
        _pick_field(
            record,
            override=args.coverage_key,
            aliases=["coverage", "visible_ratio", "mask_coverage"],
        ),
        None,
    )
    measurements = _measurement_mapping(
        _pick_field(
            record,
            override=args.measurements_key,
            aliases=["canonical_measurements", "measurements", "body_measurements"],
        )
    )
    measurements.update(_parse_key_value_pairs(args.measurement))

    measurement_scales = _parse_key_value_pairs(args.measurement_scale)
    for key in measurements.keys():
        measurement_scales.setdefault(key, float(args.default_measurement_scale))

    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "provider_name": str(args.provider_name),
        "provider_family": "body_canonical",
        "provider_version": str(args.provider_version),
        "model_id": str(args.model_id),
        "source_path": str(args.source_image.resolve()) if args.source_image else "",
        "source_role": str(args.source_role),
        "shape_beta": shape_beta,
        "pose_vector": pose_vector,
        "canonical_measurements": measurements,
        "measurement_scales": measurement_scales,
        "fit_confidence": fit_confidence,
        "coverage": coverage,
        "notes": str(args.notes or "").strip(),
        "conversion_meta": {
            "input_file": str(args.input.resolve()),
            "selected_index": int(args.index),
            "track_id": str(args.track_id or ""),
            "beta_key": str(args.beta_key or ""),
            "pose_key": str(args.pose_key or ""),
            "global_orient_key": str(args.global_orient_key or ""),
            "confidence_key": str(args.confidence_key or ""),
            "coverage_key": str(args.coverage_key or ""),
            "measurements_key": str(args.measurements_key or ""),
        },
    }
    return artifact


def _write_artifact(output_path: Path, artifact: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(artifact), indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert HMR2/4D-Humans exports into body_canonical_artifact_v1 for xiaona_consistency."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw HMR2 export (.json/.npz/.npy/.pkl).")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output body_canonical artifact json.")
    parser.add_argument("--source-image", type=Path, help="Original source image path represented by this artifact.")
    parser.add_argument(
        "--source-role",
        choices=["master_truth", "candidate"],
        required=True,
        help="Whether the artifact is for the 116-1 master truth or a candidate image.",
    )
    parser.add_argument("--index", type=int, default=0, help="Record index to select from list-like exports.")
    parser.add_argument("--track-id", default="", help="Optional track id when converting a tracking export.")
    parser.add_argument("--beta-key", default="", help="Optional dotted path for betas, e.g. pred_smpl_params.betas")
    parser.add_argument("--pose-key", default="", help="Optional dotted path for a full pose vector.")
    parser.add_argument("--global-orient-key", default="", help="Optional dotted path for global orientation.")
    parser.add_argument("--confidence-key", default="", help="Optional dotted path for fit confidence.")
    parser.add_argument("--coverage-key", default="", help="Optional dotted path for coverage.")
    parser.add_argument("--measurements-key", default="", help="Optional dotted path for canonical measurements.")
    parser.add_argument(
        "--measurement",
        action="append",
        default=[],
        help="Append a canonical measurement in KEY=VALUE form. Repeat as needed.",
    )
    parser.add_argument(
        "--measurement-scale",
        action="append",
        default=[],
        help="Append a measurement scale in KEY=VALUE form. Repeat as needed.",
    )
    parser.add_argument(
        "--default-measurement-scale",
        type=float,
        default=0.08,
        help="Fallback scale used for any measurement without an explicit scale.",
    )
    parser.add_argument("--provider-name", default="hmr2", help="Stored provider_name in the artifact.")
    parser.add_argument("--provider-version", default="hmr2_export_v1", help="Stored provider_version in the artifact.")
    parser.add_argument("--model-id", default="4d_humans_hmr2", help="Stored model_id in the artifact.")
    parser.add_argument("--notes", default="", help="Optional free-form note stored in the artifact.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    artifact = _build_artifact(args)
    _write_artifact(args.output, artifact)
    register_artifact_manifest(
        artifact_path=args.output.resolve(),
        artifact_family="body_canonical",
        artifact_role=str(args.source_role),
        provider_name=str(artifact.get("provider_name") or args.provider_name),
        provider_family=str(artifact.get("provider_family") or "body_canonical"),
        provider_version=str(artifact.get("provider_version") or args.provider_version),
        model_id=str(artifact.get("model_id") or args.model_id),
        schema_version=str(artifact.get("schema_version") or ARTIFACT_SCHEMA),
        source_path=args.source_image.resolve() if args.source_image else None,
        entrypoint=str(Path(__file__).resolve()),
        conversion_meta=dict(artifact.get("conversion_meta") or {}),
        extra={"notes": artifact.get("notes")},
    )
    shape_beta = _to_vector(artifact.get("shape_beta"))
    pose_vector = _to_vector(artifact.get("pose_vector"))
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "source_role": args.source_role,
                "shape_beta_dim": int(shape_beta.shape[0]) if shape_beta is not None else 0,
                "pose_vector_dim": int(pose_vector.shape[0]) if pose_vector is not None else 0,
                "measurement_count": len(dict(artifact.get("canonical_measurements") or {})),
                "fit_confidence": artifact.get("fit_confidence"),
                "coverage": artifact.get("coverage"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
