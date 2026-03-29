from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from core.qa_artifact_manifest import register_artifact_manifest

ARTIFACT_SCHEMA = "face_pose_canonical_artifact_v1"


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


def _pick_field(record: Any, *, override: str, aliases: Sequence[str]) -> Any:
    if override:
        return _resolve_path(record, override)
    return _search_by_aliases(record, aliases)


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
    if isinstance(payload, (list, tuple)):
        if len(payload) == 0:
            return payload
        return payload[max(0, min(index, len(payload) - 1))]
    if isinstance(payload, dict):
        for key in ["predictions", "results", "items", "detections", "faces"]:
            container = payload.get(key)
            if isinstance(container, (list, tuple)) and len(container) > 0:
                return container[max(0, min(index, len(container) - 1))]
            if isinstance(container, dict) and len(container) == 1:
                return next(iter(container.values()))
        return payload
    return payload


def _normalize_pose(value: Any) -> Dict[str, Optional[float]]:
    if isinstance(value, dict):
        return {
            "yaw": _safe_float(value.get("yaw"), None),
            "pitch": _safe_float(value.get("pitch"), None),
            "roll": _safe_float(value.get("roll"), None),
        }
    vector = _to_vector(value)
    if vector is None or vector.shape[0] < 3:
        return {"yaw": None, "pitch": None, "roll": None}
    return {
        "yaw": float(vector[0]),
        "pitch": float(vector[1]),
        "roll": float(vector[2]),
    }


def _build_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_payload(args.input)
    record = _select_record(payload, index=args.index, track_id=args.track_id)

    landmarks = _pick_field(
        record,
        override=args.landmarks_key,
        aliases=["canonical_landmarks", "landmarks_2d", "landmarks", "pts68", "pts106"],
    )
    identity_vector = _pick_field(
        record,
        override=args.identity_key,
        aliases=["canonical_identity_vector", "identity_vector", "face_embedding", "embedding"],
    )
    pose_value = _pick_field(
        record,
        override=args.pose_key,
        aliases=["pose_euler_deg", "pose_euler", "pose", "head_pose"],
    )
    visible_face_coverage = _safe_float(
        _pick_field(
            record,
            override=args.visible_face_coverage_key,
            aliases=["visible_face_coverage", "face_visible_ratio", "coverage"],
        ),
        None,
    )
    frontalization_quality = _safe_float(
        _pick_field(
            record,
            override=args.frontalization_quality_key,
            aliases=["frontalization_quality", "frontal_quality", "canonical_quality"],
        ),
        None,
    )
    pose_fit_confidence = _safe_float(
        _pick_field(
            record,
            override=args.pose_fit_confidence_key,
            aliases=["pose_fit_confidence", "fit_confidence", "confidence", "score"],
        ),
        None,
    )

    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "provider_name": str(args.provider_name),
        "provider_family": "face_canonical_shadow",
        "provider_version": str(args.provider_version),
        "model_id": str(args.model_id),
        "source_path": str(args.source_image.resolve()) if args.source_image else "",
        "source_role": str(args.source_role),
        "canonical_landmarks": _to_vector(landmarks),
        "canonical_identity_vector": _to_vector(identity_vector),
        "pose_euler_deg": _normalize_pose(pose_value),
        "visible_face_coverage": visible_face_coverage,
        "frontalization_quality": frontalization_quality,
        "pose_fit_confidence": pose_fit_confidence,
        "notes": str(args.notes or "").strip(),
        "conversion_meta": {
            "input_file": str(args.input.resolve()),
            "selected_index": int(args.index),
            "track_id": str(args.track_id or ""),
            "landmarks_key": str(args.landmarks_key or ""),
            "identity_key": str(args.identity_key or ""),
            "pose_key": str(args.pose_key or ""),
            "visible_face_coverage_key": str(args.visible_face_coverage_key or ""),
            "frontalization_quality_key": str(args.frontalization_quality_key or ""),
            "pose_fit_confidence_key": str(args.pose_fit_confidence_key or ""),
        },
    }
    return artifact


def _write_artifact(output_path: Path, artifact: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(artifact), indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert 3DDFA-V3-style exports into face_pose_canonical_artifact_v1 for xiaona_consistency."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw face canonical export (.json/.npz/.npy/.pkl).")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output face canonical artifact json.")
    parser.add_argument("--source-image", type=Path, help="Original source image path represented by this artifact.")
    parser.add_argument(
        "--source-role",
        choices=["master_truth", "candidate"],
        required=True,
        help="Whether the artifact is for the 0-degree face truth or a candidate image.",
    )
    parser.add_argument("--index", type=int, default=0, help="Record index to select from list-like exports.")
    parser.add_argument("--track-id", default="", help="Optional track id when converting a tracking export.")
    parser.add_argument("--landmarks-key", default="", help="Optional dotted path for canonical landmarks.")
    parser.add_argument("--identity-key", default="", help="Optional dotted path for canonical identity vector.")
    parser.add_argument("--pose-key", default="", help="Optional dotted path for pose euler or pose vector.")
    parser.add_argument("--visible-face-coverage-key", default="", help="Optional dotted path for visible face coverage.")
    parser.add_argument("--frontalization-quality-key", default="", help="Optional dotted path for frontalization quality.")
    parser.add_argument("--pose-fit-confidence-key", default="", help="Optional dotted path for pose fit confidence.")
    parser.add_argument("--provider-name", default="3ddfa_v3", help="Stored provider_name in the artifact.")
    parser.add_argument("--provider-version", default="3ddfa_v3_export_v1", help="Stored provider_version in the artifact.")
    parser.add_argument("--model-id", default="3ddfa_v3", help="Stored model_id in the artifact.")
    parser.add_argument("--notes", default="", help="Optional free-form note stored in the artifact.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    artifact = _build_artifact(args)
    _write_artifact(args.output, artifact)
    register_artifact_manifest(
        artifact_path=args.output.resolve(),
        artifact_family="face_canonical",
        artifact_role=str(args.source_role),
        provider_name=str(artifact.get("provider_name") or args.provider_name),
        provider_family=str(artifact.get("provider_family") or "face_canonical_shadow"),
        provider_version=str(artifact.get("provider_version") or args.provider_version),
        model_id=str(artifact.get("model_id") or args.model_id),
        schema_version=str(artifact.get("schema_version") or ARTIFACT_SCHEMA),
        source_path=args.source_image.resolve() if args.source_image else None,
        entrypoint=str(Path(__file__).resolve()),
        conversion_meta=dict(artifact.get("conversion_meta") or {}),
        extra={"notes": artifact.get("notes")},
    )
    landmarks = _to_vector(artifact.get("canonical_landmarks"))
    identity = _to_vector(artifact.get("canonical_identity_vector"))
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "source_role": args.source_role,
                "landmark_dim": int(landmarks.shape[0]) if landmarks is not None else 0,
                "identity_dim": int(identity.shape[0]) if identity is not None else 0,
                "visible_face_coverage": artifact.get("visible_face_coverage"),
                "frontalization_quality": artifact.get("frontalization_quality"),
                "pose_fit_confidence": artifact.get("pose_fit_confidence"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
