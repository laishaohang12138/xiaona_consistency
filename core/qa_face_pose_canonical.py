from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .providers import FaceCanonicalProvider
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
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": str(raw.get("provider_name") or _PROVIDER_NAME),
        "provider_family": str(raw.get("provider_family") or _PROVIDER_FAMILY),
        "provider_version": str(raw.get("provider_version") or _PROVIDER_VERSION),
        "source_path": str(raw.get("source_path") or source_path),
        "source_role": str(raw.get("source_role") or source_role),
        "canonical_landmarks": _normalize_vector(raw.get("canonical_landmarks") or raw.get("landmarks_2d") or raw.get("landmarks")),
        "canonical_identity_vector": _normalize_vector(raw.get("canonical_identity_vector") or raw.get("identity_vector") or raw.get("face_embedding")),
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
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
    ref_vector = _normalize_vector(reference)
    cand_vector = _normalize_vector(candidate)
    if ref_vector is None or cand_vector is None or ref_vector.shape != cand_vector.shape:
        return None, None
    if ref_vector.size % 2 != 0 or cand_vector.size % 2 != 0:
        return None, None

    ref_points = ref_vector.reshape(-1, 2)
    cand_points = cand_vector.reshape(-1, 2)

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
        pose_delta_similarity = None
        pose_delta_deg = None
        pose_euler = {"yaw": None, "pitch": None, "roll": None}
        if candidate_artifact is not None:
            visible_face_coverage = candidate_artifact.get("visible_face_coverage")
            frontalization_quality = candidate_artifact.get("frontalization_quality")
            pose_fit_confidence = candidate_artifact.get("pose_fit_confidence")
            pose_euler = dict(candidate_artifact.get("pose_euler_deg") or pose_euler)

        if master_artifact is not None and candidate_artifact is not None:
            canonical_face_landmark_similarity, _ = _landmark_similarity(
                master_artifact.get("canonical_landmarks"),
                candidate_artifact.get("canonical_landmarks"),
            )
            canonical_face_identity_similarity, _ = _vector_similarity(
                master_artifact.get("canonical_identity_vector"),
                candidate_artifact.get("canonical_identity_vector"),
            )
            pose_delta_similarity, pose_delta_deg = _pose_delta_similarity(
                dict(master_artifact.get("pose_euler_deg") or {}),
                dict(candidate_artifact.get("pose_euler_deg") or {}),
            )
            if canonical_face_landmark_similarity is None:
                reasons.append("FACE_CANONICAL_LANDMARK_ALIGNMENT_UNAVAILABLE")
            if canonical_face_identity_similarity is None:
                reasons.append("FACE_CANONICAL_IDENTITY_ALIGNMENT_UNAVAILABLE")
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
            if canonical_face_landmark_similarity is not None and canonical_face_landmark_similarity < 0.72:
                guidance.append("canonical 脸部拓扑相似度偏低，人工复核时优先看眼鼻口与下颌线。")
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
            "pose_delta_similarity": _round_or_none(pose_delta_similarity),
            "pose_delta_deg": _round_or_none(pose_delta_deg),
            "yaw_deg": _round_or_none(pose_euler.get("yaw")),
            "pitch_deg": _round_or_none(pose_euler.get("pitch")),
            "roll_deg": _round_or_none(pose_euler.get("roll")),
            "reasons": dedupe_keep_order(reasons),
            "guidance": dedupe_keep_order(guidance),
        }
