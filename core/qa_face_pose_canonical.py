from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .providers import FaceCanonicalProvider
from .qa_io import atomic_write_json
from .qa_utils import dedupe_keep_order

_PROVIDER_NAME = "face_pose_canonical_bridge"
_PROVIDER_FAMILY = "face_canonical_shadow"
_PROVIDER_VERSION = "face_pose_canonical_bridge_v1"
_MODEL_ID = "external_3ddfa_v3_artifact_bridge_v1"
_ARTIFACT_SCHEMA = "face_pose_canonical_artifact_v1"
_CACHE_SCHEMA = "face_pose_canonical_cache_v1"
_MASTER_ARTIFACT_NAME = "face_master_canonical.json"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _normalize_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if vector.size == 0:
        return None
    return vector


def _unit_vector(value: Any) -> Optional[np.ndarray]:
    vector = _normalize_vector(value)
    if vector is None or not bool(np.all(np.isfinite(vector))):
        return None
    norm = float(np.linalg.norm(vector, ord=2))
    if norm <= 1e-12:
        return None
    return vector / norm


def _landmark_points(value: Any) -> Optional[np.ndarray]:
    vector = _normalize_vector(value)
    if vector is None or vector.size < 6 or vector.size % 2 != 0:
        return None
    try:
        return vector.reshape(-1, 2)
    except Exception:
        return None


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


def _normalize_pose_euler(value: Any) -> Dict[str, Optional[float]]:
    if isinstance(value, dict):
        return {
            "yaw": _safe_float(value.get("yaw"), None),
            "pitch": _safe_float(value.get("pitch"), None),
            "roll": _safe_float(value.get("roll"), None),
        }
    vector = _normalize_vector(value)
    if vector is None or vector.shape[0] < 3:
        return {"yaw": None, "pitch": None, "roll": None}
    return {
        "yaw": float(vector[0]),
        "pitch": float(vector[1]),
        "roll": float(vector[2]),
    }


def _landmark_topology_signature(value: Any) -> Optional[np.ndarray]:
    points = _landmark_points(value)
    if points is None or points.shape[0] < 5:
        return None

    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered))
    if scale <= 1e-8:
        return None
    normalized = centered / scale

    xs = normalized[:, 0]
    ys = normalized[:, 1]
    abs_x = np.abs(xs)
    radii = np.linalg.norm(normalized, axis=1)
    pairwise = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    tri_upper = pairwise[np.triu_indices(normalized.shape[0], k=1)]
    if tri_upper.size == 0:
        return None

    try:
        covariance = np.cov(normalized.T)
        eigenvalues = np.sort(np.asarray(np.linalg.eigvalsh(covariance), dtype=np.float32))[::-1]
    except Exception:
        eigenvalues = np.asarray([], dtype=np.float32)

    eig_ratio = None
    if eigenvalues.size >= 2 and float(eigenvalues[0]) > 1e-8:
        eig_ratio = float(eigenvalues[1] / eigenvalues[0])

    y_q25, y_q50, y_q75 = np.quantile(ys, [0.25, 0.50, 0.75]).tolist()

    def _band_width(mask: np.ndarray) -> Optional[float]:
        if int(np.count_nonzero(mask)) == 0:
            return None
        return float(np.mean(abs_x[mask]))

    upper_width = _band_width(ys <= y_q25)
    mid_width = _band_width((ys > y_q25) & (ys < y_q75))
    lower_width = _band_width(ys >= y_q75)
    upper_radius = _band_width(ys <= y_q50)
    lower_radius = _band_width(ys > y_q50)

    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    width_height_ratio = width / max(height, 1e-8)

    signature_parts: List[float] = [
        width,
        height,
        width_height_ratio,
        float(np.std(xs)),
        float(np.std(ys)),
        float(eig_ratio if eig_ratio is not None else 0.0),
        float(upper_width if upper_width is not None else 0.0),
        float(mid_width if mid_width is not None else 0.0),
        float(lower_width if lower_width is not None else 0.0),
        float(upper_radius if upper_radius is not None else 0.0),
        float(lower_radius if lower_radius is not None else 0.0),
    ]
    signature_parts.extend(float(node) for node in np.quantile(radii, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(tri_upper, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(abs_x, [0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(ys, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
    return np.asarray(signature_parts, dtype=np.float32)


def _normalized_landmark_points(value: Any) -> Optional[np.ndarray]:
    points = _landmark_points(value)
    if points is None or points.shape[0] < 5:
        return None
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered))
    if scale <= 1e-8:
        return None
    return centered / scale


def _landmark_region_signature(points: np.ndarray) -> Optional[np.ndarray]:
    if points is None or points.shape[0] < 2:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    abs_x = np.abs(xs)
    radii = np.linalg.norm(points, axis=1)
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    tri_upper = pairwise[np.triu_indices(points.shape[0], k=1)]
    if tri_upper.size == 0:
        return None
    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    signature_parts: List[float] = [
        width,
        height,
        width / max(height, 1e-8),
        float(np.mean(xs)),
        float(np.mean(ys)),
        float(np.std(xs)),
        float(np.std(ys)),
    ]
    signature_parts.extend(float(node) for node in np.quantile(xs, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(ys, [0.10, 0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(abs_x, [0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(radii, [0.25, 0.50, 0.75, 0.90]).tolist())
    signature_parts.extend(float(node) for node in np.quantile(tri_upper, [0.10, 0.35, 0.50, 0.65, 0.90]).tolist())
    return np.asarray(signature_parts, dtype=np.float32)


def _masked_region_similarity(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    if int(np.count_nonzero(mask)) < 2:
        return None
    ref_signature = _landmark_region_signature(reference_points[mask])
    cand_signature = _landmark_region_signature(candidate_points[mask])
    similarity, _ = _vector_similarity(ref_signature, cand_signature)
    return similarity


def _lateral_balance_signature(points: np.ndarray) -> Optional[np.ndarray]:
    left = points[points[:, 0] < 0.0]
    right = points[points[:, 0] > 0.0]
    if left.shape[0] < 2 or right.shape[0] < 2:
        return None

    def _side_features(side: np.ndarray) -> List[float]:
        xs = side[:, 0]
        ys = side[:, 1]
        return [
            float(np.mean(np.abs(xs))),
            float(np.std(np.abs(xs))),
            float(np.mean(ys)),
            float(np.std(ys)),
            float(np.max(ys) - np.min(ys)),
        ]

    total = float(points.shape[0])
    parts: List[float] = [
        float(left.shape[0] / max(total, 1.0)),
        float(right.shape[0] / max(total, 1.0)),
    ]
    parts.extend(_side_features(left))
    parts.extend(_side_features(right))
    return np.asarray(parts, dtype=np.float32)


def _head_topology_partition_similarity(reference: Any, candidate: Any) -> Dict[str, Any]:
    ref_points = _normalized_landmark_points(reference)
    cand_points = _normalized_landmark_points(candidate)
    if ref_points is None or cand_points is None or ref_points.shape != cand_points.shape:
        return {
            "available": False,
            "schema_version": "head_topology_partition_v1",
            "partition_method": "canonical_landmark_quantile_bands_v1",
            "reason": "landmark_shape_unavailable",
        }

    ys = ref_points[:, 1]
    abs_x = np.abs(ref_points[:, 0])
    y_q33, y_q66 = np.quantile(ys, [0.33, 0.66]).tolist()
    contour_q75 = float(np.quantile(abs_x, 0.75))
    center_q35 = float(np.quantile(abs_x, 0.35))

    partition_values: Dict[str, Optional[float]] = {
        "upper_face_similarity": _masked_region_similarity(ref_points, cand_points, ys <= y_q33),
        "mid_face_similarity": _masked_region_similarity(ref_points, cand_points, (ys > y_q33) & (ys < y_q66)),
        "lower_face_similarity": _masked_region_similarity(ref_points, cand_points, ys >= y_q66),
        "contour_similarity": _masked_region_similarity(ref_points, cand_points, abs_x >= contour_q75),
        "center_axis_similarity": _masked_region_similarity(ref_points, cand_points, abs_x <= center_q35),
    }
    lateral_ref = _lateral_balance_signature(ref_points)
    lateral_cand = _lateral_balance_signature(cand_points)
    lateral_similarity, _ = _vector_similarity(lateral_ref, lateral_cand)
    partition_values["lateral_balance_similarity"] = lateral_similarity

    mean_similarity = _weighted_mean(
        [
            (partition_values.get("upper_face_similarity"), 0.16),
            (partition_values.get("mid_face_similarity"), 0.18),
            (partition_values.get("lower_face_similarity"), 0.18),
            (partition_values.get("contour_similarity"), 0.22),
            (partition_values.get("center_axis_similarity"), 0.18),
            (partition_values.get("lateral_balance_similarity"), 0.08),
        ]
    )
    available_parts = {
        key: value for key, value in partition_values.items() if value is not None
    }
    weakest_part = None
    weakest_part_similarity = None
    if available_parts:
        weakest_part, weakest_part_similarity = min(available_parts.items(), key=lambda row: float(row[1]))

    return {
        "available": True,
        "schema_version": "head_topology_partition_v1",
        "partition_method": "canonical_landmark_quantile_bands_v1",
        "landmark_count": int(ref_points.shape[0]),
        "mean_similarity": mean_similarity,
        "weakest_part": weakest_part,
        "weakest_part_similarity": weakest_part_similarity,
        **partition_values,
    }


def _cache_dir(runtime: Any) -> Path:
    cache_root = getattr(getattr(runtime.config, "paths", None), "dir_heavy_cache", Path("outputs") / "heavy_evidence_cache")
    cache_dir = Path(cache_root) / _PROVIDER_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _master_truth_dir(runtime: Any) -> Path:
    return Path(getattr(runtime.config.paths, "dir_master_truth", Path("outputs") / "master_truth"))


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _build_cache_key(runtime: Any, image_path: Path) -> tuple[str, Dict[str, Any]]:
    resolved = image_path.resolve()
    stat = resolved.stat()
    payload = {
        "provider_name": _PROVIDER_NAME,
        "provider_version": _PROVIDER_VERSION,
        "source_path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "master_artifact": str((_master_truth_dir(runtime) / _MASTER_ARTIFACT_NAME).resolve()),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest, payload


def _sidecar_candidates(image_path: Path) -> List[Path]:
    return [
        image_path.with_suffix(image_path.suffix + ".face_pose_canonical.json"),
        image_path.with_name(f"{image_path.stem}.face_pose_canonical.json"),
        image_path.with_suffix(image_path.suffix + ".face_canonical.json"),
        image_path.with_name(f"{image_path.stem}.face_canonical.json"),
    ]


def _normalize_artifact(raw: Dict[str, Any], *, source_path: Path, source_role: str) -> Dict[str, Any]:
    canonical_landmarks = _normalize_vector(raw.get("canonical_landmarks") or raw.get("landmarks_2d") or raw.get("landmarks"))
    landmark_visibility_weights = _normalize_vector(
        raw.get("landmark_visibility_weights")
        or raw.get("landmark_weights")
        or raw.get("landmark_confidence")
    )
    runtime_identity_value = raw.get("runtime_face_embedding_raw")
    if runtime_identity_value is None:
        runtime_identity_value = raw.get("canonical_identity_vector")
    if runtime_identity_value is None:
        runtime_identity_value = raw.get("identity_vector")
    if runtime_identity_value is None:
        runtime_identity_value = raw.get("face_embedding")
    runtime_identity_vector = _normalize_vector(runtime_identity_value)
    runtime_identity_unit_value = raw.get("runtime_face_embedding_unit")
    if runtime_identity_unit_value is None:
        runtime_identity_unit_value = runtime_identity_vector
    runtime_identity_unit = _unit_vector(runtime_identity_unit_value)
    topology_signature = _normalize_vector(
        raw.get("canonical_face_topology_signature") or raw.get("face_topology_signature")
    )
    if topology_signature is None:
        topology_signature = _landmark_topology_signature(canonical_landmarks)
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": str(raw.get("provider_name") or _PROVIDER_NAME),
        "provider_family": str(raw.get("provider_family") or _PROVIDER_FAMILY),
        "provider_version": str(raw.get("provider_version") or _PROVIDER_VERSION),
        "model_id": str(raw.get("model_id") or _MODEL_ID),
        "model_sha256": str(raw.get("model_sha256") or "").strip() or None,
        "provider_implementation_sha256": str(
            raw.get("provider_implementation_sha256") or ""
        ).strip()
        or None,
        "provider_execution_backend": str(raw.get("provider_execution_backend") or "").strip() or None,
        "source_path": str(raw.get("source_path") or source_path),
        "source_role": str(raw.get("source_role") or source_role),
        "canonical_landmarks": canonical_landmarks,
        "landmark_visibility_weights": landmark_visibility_weights,
        "landmark_schema_id": str(raw.get("landmark_schema_id") or "").strip() or None,
        "landmark_source_field": str(raw.get("landmark_source_field") or "").strip() or None,
        "landmark_coordinate_convention": str(
            raw.get("landmark_coordinate_convention") or ""
        ).strip()
        or None,
        "canonical_preprocessing_contract_id": str(
            raw.get("canonical_preprocessing_contract_id") or ""
        ).strip()
        or None,
        "canonical_landmark_contract": (
            dict(raw.get("canonical_landmark_contract"))
            if isinstance(raw.get("canonical_landmark_contract"), dict)
            else {}
        ),
        "canonical_face_topology_signature": topology_signature,
        "runtime_face_embedding_raw": runtime_identity_vector,
        "runtime_face_embedding_unit": runtime_identity_unit,
        "runtime_face_embedding_contract": (
            dict(raw.get("runtime_face_embedding_contract"))
            if isinstance(raw.get("runtime_face_embedding_contract"), dict)
            else {}
        ),
        "canonical_identity_vector": runtime_identity_vector,
        "pose_euler_deg": _normalize_pose_euler(raw.get("pose_euler_deg") or raw.get("pose_euler") or raw.get("pose")),
        "visible_face_coverage": _safe_float(raw.get("visible_face_coverage"), None),
        "frontalization_quality": _safe_float(raw.get("frontalization_quality"), None),
        "pose_fit_confidence": _safe_float(raw.get("pose_fit_confidence") or raw.get("fit_confidence"), None),
        "notes": str(raw.get("notes") or "").strip(),
    }


def _load_master_artifact(runtime: Any) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    master_path = _master_truth_dir(runtime) / _MASTER_ARTIFACT_NAME
    if not master_path.exists():
        return None, master_path
    payload = _load_json_object(master_path)
    if payload is None:
        return None, master_path
    return _normalize_artifact(payload, source_path=master_path, source_role="master_truth"), master_path


def _load_cached_candidate(runtime: Any, image_path: Path) -> tuple[Optional[Dict[str, Any]], str, Path]:
    cache_key, _ = _build_cache_key(runtime, image_path)
    cache_file = _cache_dir(runtime) / f"{cache_key}.json"
    if not cache_file.exists():
        return None, cache_key, cache_file
    payload = _load_json_object(cache_file)
    if payload is None or str(payload.get("schema_version") or "") != _CACHE_SCHEMA:
        return None, cache_key, cache_file
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        return None, cache_key, cache_file
    normalized = _normalize_artifact(artifact, source_path=image_path.resolve(), source_role="candidate")
    normalized["cache_key"] = cache_key
    normalized["cache_file"] = str(cache_file)
    normalized["cache_state"] = "hit"
    return normalized, cache_key, cache_file


def _write_cached_candidate(cache_file: Path, cache_key: str, artifact: Dict[str, Any]) -> bool:
    payload = {
        "schema_version": _CACHE_SCHEMA,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "cache_key": cache_key,
        "artifact": _json_ready(artifact),
    }
    try:
        atomic_write_json(cache_file, payload)
        return True
    except Exception:
        return False


def _load_candidate_artifact(runtime: Any, image_path: Path) -> tuple[Optional[Dict[str, Any]], str, Path]:
    cached, cache_key, cache_file = _load_cached_candidate(runtime, image_path)
    if cached is not None:
        return cached, cache_key, cache_file
    resolved = image_path.resolve()
    for sidecar in _sidecar_candidates(resolved):
        if not sidecar.exists():
            continue
        payload = _load_json_object(sidecar)
        if payload is None:
            continue
        normalized = _normalize_artifact(payload, source_path=resolved, source_role="candidate")
        normalized["sidecar_file"] = str(sidecar)
        normalized["cache_key"] = cache_key
        normalized["cache_file"] = str(cache_file)
        cache_written = _write_cached_candidate(cache_file, cache_key, normalized)
        normalized["cache_state"] = "write" if cache_written else "miss"
        return normalized, cache_key, cache_file
    return None, cache_key, cache_file


def _vector_similarity(reference: Any, candidate: Any) -> tuple[Optional[float], Optional[float]]:
    ref_vector = _normalize_vector(reference)
    cand_vector = _normalize_vector(candidate)
    if ref_vector is None or cand_vector is None or ref_vector.shape != cand_vector.shape:
        return None, None
    delta = float(np.mean(np.abs(ref_vector - cand_vector)))
    return float(np.exp(-delta)), delta


def _landmark_similarity(reference: Any, candidate: Any) -> tuple[Optional[float], Optional[float]]:
    ref_points = _landmark_points(reference)
    cand_points = _landmark_points(candidate)
    if ref_points is None or cand_points is None or ref_points.shape != cand_points.shape:
        return None, None

    ref_centered = ref_points - np.mean(ref_points, axis=0, keepdims=True)
    cand_centered = cand_points - np.mean(cand_points, axis=0, keepdims=True)

    ref_norm = float(np.linalg.norm(ref_centered))
    cand_norm = float(np.linalg.norm(cand_centered))
    if ref_norm <= 1e-8 or cand_norm <= 1e-8:
        return None, None

    ref_normalized = ref_centered / ref_norm
    cand_normalized = cand_centered / cand_norm

    delta = float(np.mean(np.abs(ref_normalized - cand_normalized)))
    similarity = float(np.exp(-(delta * 8.0)))
    return similarity, delta


def _topology_signature_similarity(reference: Any, candidate: Any) -> tuple[Optional[float], Optional[float]]:
    return _vector_similarity(reference, candidate)


def _pose_delta_similarity(master_pose: Dict[str, Optional[float]], candidate_pose: Dict[str, Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    keys = ["yaw", "pitch", "roll"]
    deltas: List[float] = []
    for key in keys:
        master_value = master_pose.get(key)
        candidate_value = candidate_pose.get(key)
        if master_value is None or candidate_value is None:
            continue
        deltas.append(abs(float(candidate_value) - float(master_value)))
    if len(deltas) == 0:
        return None, None
    mean_delta = float(sum(deltas) / len(deltas))
    return float(np.exp(-(mean_delta / 18.0))), mean_delta


def _weighted_mean(items: List[Tuple[Optional[float], float]]) -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for value, weight in items:
        if value is None or weight <= 0.0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 1e-8:
        return None
    return float(numerator / denominator)


class FacePoseCanonicalProvider(FaceCanonicalProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "artifact_bridge",
            "mode": "shadow_only",
            "master_artifact_path": str(_MASTER_ARTIFACT_NAME),
        }

    def analyze_face_canonical(
        self,
        runtime: Any,
        image_path: Path,
        *,
        img_bgr: Optional[np.ndarray] = None,
        face_feat: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del img_bgr
        resolved_path = Path(image_path).resolve()
        master_artifact, master_path = _load_master_artifact(runtime)
        candidate_artifact, cache_key, cache_file = _load_candidate_artifact(runtime, resolved_path)
        reasons: List[str] = []
        guidance: List[str] = []

        if master_artifact is None:
            reasons.append("FACE_CANONICAL_MASTER_MISSING")
            guidance.append("先准备 0 号脸的 face_master_canonical.json，再启用脸部 canonical 对照。")
        if candidate_artifact is None:
            reasons.append("FACE_CANONICAL_CANDIDATE_ARTIFACT_MISSING")
            guidance.append("为候选图补齐 .face_pose_canonical.json sidecar，才能进入脸部 canonical shadow 对照。")

        visible_face_coverage = None
        frontalization_quality = None
        pose_fit_confidence = None
        canonical_face_landmark_similarity = None
        canonical_face_identity_similarity = None
        canonical_face_topology_similarity = None
        canonical_face_topology_delta = None
        head_topology_partition: Dict[str, Any] = {
            "available": False,
            "schema_version": "head_topology_partition_v1",
            "partition_method": "canonical_landmark_quantile_bands_v1",
        }
        pose_delta_similarity = None
        pose_delta_deg = None
        pose_euler = {"yaw": None, "pitch": None, "roll": None}
        if candidate_artifact is not None:
            visible_face_coverage = candidate_artifact.get("visible_face_coverage")
            frontalization_quality = candidate_artifact.get("frontalization_quality")
            pose_fit_confidence = candidate_artifact.get("pose_fit_confidence")
            pose_euler = dict(candidate_artifact.get("pose_euler_deg") or pose_euler)

        if master_artifact is not None and candidate_artifact is not None:
            master_identity_vector = master_artifact.get("runtime_face_embedding_raw")
            if master_identity_vector is None:
                master_identity_vector = master_artifact.get("canonical_identity_vector")
            candidate_identity_vector = candidate_artifact.get("runtime_face_embedding_raw")
            if candidate_identity_vector is None:
                candidate_identity_vector = candidate_artifact.get("canonical_identity_vector")
            canonical_face_landmark_similarity, _ = _landmark_similarity(
                master_artifact.get("canonical_landmarks"),
                candidate_artifact.get("canonical_landmarks"),
            )
            canonical_face_identity_similarity, _ = _vector_similarity(
                master_identity_vector,
                candidate_identity_vector,
            )
            canonical_face_topology_similarity, canonical_face_topology_delta = _topology_signature_similarity(
                master_artifact.get("canonical_face_topology_signature"),
                candidate_artifact.get("canonical_face_topology_signature"),
            )
            head_topology_partition = _head_topology_partition_similarity(
                master_artifact.get("canonical_landmarks"),
                candidate_artifact.get("canonical_landmarks"),
            )
            pose_delta_similarity, pose_delta_deg = _pose_delta_similarity(
                dict(master_artifact.get("pose_euler_deg") or {}),
                dict(candidate_artifact.get("pose_euler_deg") or {}),
            )
            if canonical_face_landmark_similarity is None:
                reasons.append("FACE_CANONICAL_LANDMARK_ALIGNMENT_UNAVAILABLE")
            if canonical_face_identity_similarity is None:
                reasons.append("FACE_CANONICAL_IDENTITY_ALIGNMENT_UNAVAILABLE")
            if canonical_face_topology_similarity is None:
                reasons.append("FACE_CANONICAL_TOPOLOGY_ALIGNMENT_UNAVAILABLE")
            if not bool(head_topology_partition.get("available")):
                reasons.append("FACE_HEAD_TOPOLOGY_PARTITION_UNAVAILABLE")
            if pose_delta_similarity is None:
                reasons.append("FACE_CANONICAL_POSE_DELTA_UNAVAILABLE")

        face_bbox_ratio = _safe_float(getattr(face_feat, "bbox_area_ratio", None), None)
        if face_bbox_ratio is not None and face_bbox_ratio < 0.006:
            reasons.append("FACE_CANONICAL_LOW_SIGNAL_INPUT")
            guidance.append("候选脸部过小时，canonical shadow 只能作辅助证据，不应直接解释为身份漂移。")

        normalization_confidence = _weighted_mean(
            [
                (visible_face_coverage, 0.30),
                (frontalization_quality, 0.35),
                (pose_fit_confidence, 0.35),
            ]
        )

        available = bool(master_artifact is not None and candidate_artifact is not None)
        if available:
            guidance.append("这是一条 shadow-only 脸部 canonical 证据，不会改写当前主 face 分数。")
            if canonical_face_topology_similarity is not None and canonical_face_topology_similarity < 0.74:
                guidance.append("canonical 脸部 3D 拓扑支撑偏弱，人工复核时优先看鼻梁-嘴-下巴关系与下颌线走势。")
            if canonical_face_landmark_similarity is not None and canonical_face_landmark_similarity < 0.72:
                guidance.append("canonical 脸部拓扑相似度偏低，人工复核时优先看眼鼻口与下颌线。")
            weakest_part_similarity = _safe_float(head_topology_partition.get("weakest_part_similarity"), None)
            if weakest_part_similarity is not None and weakest_part_similarity < 0.70:
                weakest_part = str(head_topology_partition.get("weakest_part") or "unknown")
                guidance.append(f"head topology 分区弱项={weakest_part}，人工复核时优先确认该区域是否身份漂移。")
            if normalization_confidence is not None and normalization_confidence < 0.65:
                guidance.append("这张图的 frontalization/可见脸质量偏弱，canonical 结论只能作辅助参考。")

        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "artifact_bridge",
            "mode": "shadow_only",
            "available": available,
            "source_path": str(resolved_path),
            "master_artifact_path": str(master_path.resolve()) if master_path is not None else None,
            "candidate_artifact_path": (
                str(candidate_artifact.get("sidecar_file"))
                if isinstance(candidate_artifact, dict) and candidate_artifact.get("sidecar_file")
                else None
            ),
            "cache_key": cache_key,
            "cache_file": str(cache_file),
            "cache_state": (
                str(candidate_artifact.get("cache_state"))
                if isinstance(candidate_artifact, dict) and candidate_artifact.get("cache_state")
                else "miss"
            ),
            "canonical_truth_available": master_artifact is not None,
            "visible_face_coverage": _round_or_none(visible_face_coverage),
            "frontalization_quality": _round_or_none(frontalization_quality),
            "pose_fit_confidence": _round_or_none(pose_fit_confidence),
            "face_pose_normalization_confidence": _round_or_none(normalization_confidence),
            "canonical_face_landmark_similarity": _round_or_none(canonical_face_landmark_similarity),
            "canonical_face_identity_similarity": _round_or_none(canonical_face_identity_similarity),
            "canonical_face_topology_similarity": _round_or_none(canonical_face_topology_similarity),
            "canonical_face_topology_delta": _round_or_none(canonical_face_topology_delta),
            "head_topology_partition": _json_ready(head_topology_partition),
            "head_topology_mean_similarity": _round_or_none(head_topology_partition.get("mean_similarity")),
            "head_topology_weakest_part": head_topology_partition.get("weakest_part"),
            "head_topology_weakest_part_similarity": _round_or_none(
                head_topology_partition.get("weakest_part_similarity")
            ),
            "head_topology_upper_face_similarity": _round_or_none(
                head_topology_partition.get("upper_face_similarity")
            ),
            "head_topology_mid_face_similarity": _round_or_none(
                head_topology_partition.get("mid_face_similarity")
            ),
            "head_topology_lower_face_similarity": _round_or_none(
                head_topology_partition.get("lower_face_similarity")
            ),
            "head_topology_contour_similarity": _round_or_none(
                head_topology_partition.get("contour_similarity")
            ),
            "head_topology_center_axis_similarity": _round_or_none(
                head_topology_partition.get("center_axis_similarity")
            ),
            "head_topology_lateral_balance_similarity": _round_or_none(
                head_topology_partition.get("lateral_balance_similarity")
            ),
            "pose_delta_similarity": _round_or_none(pose_delta_similarity),
            "pose_delta_deg": _round_or_none(pose_delta_deg),
            "yaw_deg": _round_or_none(pose_euler.get("yaw")),
            "pitch_deg": _round_or_none(pose_euler.get("pitch")),
            "roll_deg": _round_or_none(pose_euler.get("roll")),
            "reasons": dedupe_keep_order(reasons),
            "guidance": dedupe_keep_order(guidance),
        }
