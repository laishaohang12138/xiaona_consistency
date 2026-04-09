from __future__ import annotations

import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .qa_artifact_manifest import register_artifact_manifest
from .providers import HeavyEvidenceProvider
from .qa_utils import dedupe_keep_order

_PROVIDER_NAME = "body_canonical_hmr2"
_PROVIDER_FAMILY = "body_canonical"
_PROVIDER_VERSION = "body_canonical_hmr2_direct_bridge_v2"
_MODEL_ID = "hmr2_direct_bridge_v2"
_ARTIFACT_SCHEMA = "body_canonical_artifact_v1"
_CACHE_SCHEMA = "body_canonical_cache_v1"
_MASTER_ARTIFACT_NAME = "body_master_shape_only.json"
_DEFAULT_MEASUREMENT_SCALE = 0.08
_MIN_CANONICAL_MEASUREMENTS = 6
_BODY_TOPOLOGY_MEASUREMENT_ORDER = [
    "shoulder_width_to_torso",
    "hip_width_to_torso",
    "shoulder_to_hip_ratio",
    "leg_length_to_torso",
    "upper_to_lower_leg_ratio",
    "left_right_leg_balance",
    "foot_length_to_leg",
    "left_right_foot_balance",
]
_DEFAULT_SMPL_MODEL = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
_ALT_SMPL_MODELS = [
    "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
]


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


def _vector_signature_ref(name: str, value: Any) -> Dict[str, Any]:
    vector = _normalize_vector(value)
    return {
        "kind": "signature",
        "name": str(name),
        "available": vector is not None and vector.size > 0,
        "dimension": int(vector.shape[0]) if vector is not None else 0,
    }


def _topology_signature_ref(name: str, value: Any) -> Dict[str, Any]:
    vector = _normalize_vector(value)
    return {
        "kind": "signature",
        "name": str(name),
        "available": vector is not None and vector.size > 0,
        "dimension": int(vector.shape[0]) if vector is not None else 0,
    }


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


def _measurement_vector(measurements: Dict[str, float]) -> Optional[np.ndarray]:
    if len(measurements) == 0:
        return None
    preferred = [key for key in _BODY_TOPOLOGY_MEASUREMENT_ORDER if key in measurements]
    if len(preferred) >= _MIN_CANONICAL_MEASUREMENTS:
        keys = preferred
    else:
        keys = sorted(measurements.keys())
        if len(keys) < _MIN_CANONICAL_MEASUREMENTS:
            return None
    values = [float(measurements[key]) for key in keys]
    try:
        return np.asarray(values, dtype=np.float32)
    except Exception:
        return None


def _body_topology_signature(shape_beta: Any, measurements: Dict[str, float]) -> Optional[np.ndarray]:
    beta = _normalize_vector(shape_beta)
    measurement_vector = _measurement_vector(measurements)
    if beta is None and measurement_vector is None:
        return None

    parts: List[np.ndarray] = []
    if beta is not None:
        beta_centered = beta - np.mean(beta)
        beta_norm = float(np.linalg.norm(beta_centered))
        if beta_norm > 1e-8:
            beta_centered = beta_centered / beta_norm
        parts.append(beta_centered.astype(np.float32))
        parts.append(np.asarray(np.quantile(beta, [0.1, 0.5, 0.9]), dtype=np.float32))

    if measurement_vector is not None:
        measure_scale = max(float(np.mean(np.abs(measurement_vector))), 1e-6)
        measurement_normalized = measurement_vector / measure_scale
        parts.append(measurement_normalized.astype(np.float32))
        parts.append(np.asarray(np.quantile(measurement_normalized, [0.1, 0.25, 0.5, 0.75, 0.9]), dtype=np.float32))
        parts.append(
            np.asarray(
                [
                    float(np.max(measurement_normalized) - np.min(measurement_normalized)),
                    float(np.std(measurement_normalized)),
                ],
                dtype=np.float32,
            )
        )

    if len(parts) == 0:
        return None
    return np.concatenate(parts, axis=0)


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


def _pick_field(record: Any, *, aliases: Sequence[str], override: str = "") -> Any:
    if override:
        return _resolve_path(record, override)
    return _search_by_aliases(record, aliases)


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
    raise ValueError(f"Unsupported HMR2 export format: {path.suffix}")


def _select_record(payload: Any, *, index: int = 0) -> Any:
    if isinstance(payload, (list, tuple)):
        if len(payload) == 0:
            return payload
        return payload[max(0, min(index, len(payload) - 1))]
    if isinstance(payload, dict):
        for key in ["predictions", "results", "items", "detections", "humans", "tracklets", "tracks"]:
            container = payload.get(key)
            if isinstance(container, (list, tuple)) and len(container) > 0:
                return container[max(0, min(index, len(container) - 1))]
            if isinstance(container, dict) and len(container) == 1:
                return next(iter(container.values()))
        return payload
    return payload


def _compose_pose_vector(record: Any) -> Optional[np.ndarray]:
    body_pose = _pick_field(
        record,
        aliases=["body_pose", "pred_body_pose", "body_pose_axis_angle", "pose_body"],
    )
    global_orient = _pick_field(
        record,
        aliases=["global_orient", "pred_global_orient", "root_orient", "orient"],
    )
    pose_vector = _pick_field(
        record,
        aliases=["pose_vector", "smpl_pose", "pred_pose", "pose"],
    )
    global_orient_vec = _normalize_vector(global_orient)
    body_pose_vec = _normalize_vector(body_pose)
    if global_orient_vec is not None and body_pose_vec is not None:
        return np.concatenate([global_orient_vec, body_pose_vec], axis=0)
    direct_pose_vec = _normalize_vector(pose_vector)
    if direct_pose_vec is not None:
        return direct_pose_vec
    return body_pose_vec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolved_device(device_name: str) -> str:
    normalized = str(device_name or "auto").strip().lower()
    if normalized in {"cpu", "cuda"}:
        return normalized
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_master_body_path(runtime: Any) -> Path:
    registry = getattr(runtime.config, "anchor_registry", {}) or {}
    anchors = registry.get("anchors", {}) if isinstance(registry, dict) else {}
    rules = registry.get("rules", {}) if isinstance(registry, dict) else {}
    body_truth_id = str(rules.get("body_truth_anchor") or "ref_2_full_body_master")
    body_truth = anchors.get(body_truth_id) if isinstance(anchors, dict) else None
    raw_path = ""
    if isinstance(body_truth, dict):
        raw_path = str(body_truth.get("path") or "").strip()
    if not raw_path:
        return (runtime.config.paths.base_dir / "anchors" / "full" / "Task-63987060-116-1.png").resolve()
    expanded = raw_path.replace("${PROJECT_ROOT}", str(runtime.config.paths.base_dir))
    return Path(expanded).resolve()


def _resolve_settings() -> Dict[str, Any]:
    repo_root = _repo_root()
    repo_dir = Path(os.getenv("XIAONA_HMR2_REPO", str(repo_root / "external" / "4D-Humans"))).resolve()
    python_exec = os.getenv("XIAONA_HMR2_PYTHON", sys.executable)
    default_entrypoint = repo_root / "export_body_canonical_direct_hmr2.py"
    if not default_entrypoint.exists():
        default_entrypoint = repo_dir / "demo.py"
    entrypoint = Path(os.getenv("XIAONA_HMR2_ENTRYPOINT", str(default_entrypoint))).resolve()
    custom_command_template = str(os.getenv("XIAONA_HMR2_CMD_TEMPLATE", "") or "").strip()
    extra_args = str(os.getenv("XIAONA_HMR2_EXTRA_ARGS", "") or "").strip()
    requested_device = str(os.getenv("XIAONA_HMR2_DEVICE", "auto") or "auto").strip()
    resolved_device = _resolved_device(requested_device)
    smpl_candidates = [
        repo_dir / "data" / _DEFAULT_SMPL_MODEL,
        repo_root / "data" / _DEFAULT_SMPL_MODEL,
        Path.home() / ".cache" / "4DHumans" / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
    ]
    for alt_name in _ALT_SMPL_MODELS:
        smpl_candidates.append(repo_dir / "data" / alt_name)
        smpl_candidates.append(repo_root / "data" / alt_name)
    smpl_model_path = next((path for path in smpl_candidates if path.exists()), None)
    built_in_ready = repo_dir.exists() and entrypoint.exists() and smpl_model_path is not None
    if custom_command_template:
        integration_state = "custom_template"
        integration_reason = "custom_command_template"
    elif not repo_dir.exists():
        integration_state = "missing_repo"
        integration_reason = "4D-Humans repository is missing"
    elif not entrypoint.exists():
        integration_state = "missing_entrypoint"
        integration_reason = "HMR2 export entrypoint is missing"
    elif smpl_model_path is None:
        integration_state = "missing_smpl_model"
        integration_reason = "SMPL neutral model is missing"
    else:
        integration_state = "direct_ready"
        integration_reason = "local HMR2 export is ready"
    command_template = (
        custom_command_template
        or "{python} {entrypoint} --image_path {image_path} --output_dir {output_dir} --device {device}"
    )
    direct_ready = bool(custom_command_template) or built_in_ready
    return {
        "repo_dir": repo_dir,
        "python_exec": python_exec,
        "entrypoint": entrypoint,
        "command_template": command_template,
        "custom_command_template": custom_command_template,
        "extra_args": extra_args,
        "device": requested_device,
        "resolved_device": resolved_device,
        "direct_ready": direct_ready,
        "integration_state": integration_state,
        "integration_reason": integration_reason,
        "smpl_model_path": str(smpl_model_path) if smpl_model_path is not None else None,
        "smpl_candidates": [str(path) for path in smpl_candidates],
    }


def _find_export_file(output_dir: Path, image_stem: str) -> Optional[Path]:
    candidates: List[Path] = []
    for suffix in [".json", ".npz", ".npy", ".pkl", ".pickle"]:
        candidates.extend(path for path in output_dir.rglob(f"{image_stem}*{suffix}") if path.is_file())
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.name), str(path)))
    return candidates[0]


def _build_direct_artifact(
    *,
    source_path: Path,
    source_role: str,
    export_path: Path,
) -> Dict[str, Any]:
    payload = _load_payload(export_path)
    if isinstance(payload, dict) and str(payload.get("schema_version") or "") == _ARTIFACT_SCHEMA:
        artifact = dict(payload)
        artifact["source_path"] = str(source_path)
        artifact["source_role"] = source_role
        artifact.setdefault("provider_name", _PROVIDER_NAME)
        artifact.setdefault("provider_family", _PROVIDER_FAMILY)
        artifact.setdefault("provider_version", _PROVIDER_VERSION)
        artifact.setdefault("model_id", _MODEL_ID)
        return artifact

    record = _select_record(payload)
    measurements = _measurement_mapping(
        _pick_field(record, aliases=["canonical_measurements", "measurements", "body_measurements"]) or {}
    )
    measurement_scales = _measurement_mapping(_pick_field(record, aliases=["measurement_scales"]) or {})
    fit_confidence = _safe_float(
        _pick_field(record, aliases=["fit_confidence", "confidence", "score", "pred_score"]),
        None,
    )
    coverage = _safe_float(
        _pick_field(record, aliases=["coverage", "visible_ratio", "mask_coverage"]),
        None,
    )
    topology_signature = _body_topology_signature(
        _pick_field(record, aliases=["betas", "shape_beta", "pred_betas", "smpl_betas", "shape"]),
        measurements,
    )
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "model_id": _MODEL_ID,
        "source_path": str(source_path),
        "source_role": source_role,
        "shape_beta": _normalize_vector(
            _pick_field(record, aliases=["betas", "shape_beta", "pred_betas", "smpl_betas", "shape"])
        ),
        "body_topology_signature": topology_signature,
        "pose_vector": _compose_pose_vector(record),
        "canonical_measurements": measurements,
        "measurement_scales": measurement_scales,
        "fit_confidence": fit_confidence,
        "coverage": coverage,
        "notes": "direct HMR2 export bridged into body canonical evidence",
        "conversion_meta": {
            "export_path": str(export_path),
        },
    }


def _cache_dir(runtime: Any) -> Path:
    cache_root = getattr(getattr(runtime.config, "paths", None), "dir_heavy_cache", Path("outputs") / "heavy_evidence_cache")
    cache_dir = Path(cache_root) / _PROVIDER_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _master_truth_dir(runtime: Any) -> Path:
    return Path(getattr(runtime.config.paths, "dir_master_truth", Path("outputs") / "master_truth"))


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
        image_path.with_suffix(image_path.suffix + ".body_canonical.json"),
        image_path.with_name(f"{image_path.stem}.body_canonical.json"),
    ]


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_artifact(raw: Dict[str, Any], *, source_path: Path, source_role: str) -> Dict[str, Any]:
    measurements = _measurement_mapping(
        raw.get("canonical_measurements")
        or raw.get("measurements")
        or raw.get("body_measurements")
        or {}
    )
    measurement_scales = _measurement_mapping(raw.get("measurement_scales") or {})
    artifact = {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": str(raw.get("provider_name") or _PROVIDER_NAME),
        "provider_family": str(raw.get("provider_family") or _PROVIDER_FAMILY),
        "provider_version": str(raw.get("provider_version") or _PROVIDER_VERSION),
        "source_path": str(raw.get("source_path") or source_path),
        "source_role": str(raw.get("source_role") or source_role),
        "shape_beta": _normalize_vector(raw.get("shape_beta") or raw.get("betas") or raw.get("shape")),
        "body_topology_signature": _normalize_vector(raw.get("body_topology_signature")),
        "pose_vector": _normalize_vector(raw.get("pose_vector") or raw.get("pose_theta") or raw.get("theta")),
        "canonical_measurements": measurements,
        "measurement_scales": measurement_scales,
        "fit_confidence": _safe_float(raw.get("fit_confidence"), None),
        "coverage": _safe_float(raw.get("coverage"), None),
        "notes": str(raw.get("notes") or "").strip(),
    }
    if artifact["body_topology_signature"] is None:
        artifact["body_topology_signature"] = _body_topology_signature(artifact.get("shape_beta"), measurements)
    return artifact


def _load_master_artifact(runtime: Any) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    master_path = _master_truth_dir(runtime) / _MASTER_ARTIFACT_NAME
    if not master_path.exists():
        return None, master_path
    payload = _load_json_object(master_path)
    if payload is None:
        return None, master_path
    return _normalize_artifact(payload, source_path=master_path, source_role="master_truth"), master_path


def _artifact_is_current(artifact: Optional[Dict[str, Any]], *, min_measurements: int = 0) -> bool:
    if not isinstance(artifact, dict):
        return False
    if str(artifact.get("schema_version") or "") != _ARTIFACT_SCHEMA:
        return False
    if str(artifact.get("provider_name") or "") != _PROVIDER_NAME:
        return False
    if str(artifact.get("provider_version") or "") != _PROVIDER_VERSION:
        return False
    if _normalize_vector(artifact.get("shape_beta")) is None:
        return False
    if min_measurements > 0:
        measurements = _measurement_mapping(artifact.get("canonical_measurements") or {})
        if len(measurements) < int(min_measurements):
            return False
    return True


def _load_cached_candidate(runtime: Any, image_path: Path) -> tuple[Optional[Dict[str, Any]], str, Path]:
    cache_key, _ = _build_cache_key(runtime, image_path)
    cache_file = _cache_dir(runtime) / f"{cache_key}.json"
    if not cache_file.exists():
        return None, cache_key, cache_file
    payload = _load_json_object(cache_file)
    if payload is None:
        return None, cache_key, cache_file
    if str(payload.get("schema_version") or "") != _CACHE_SCHEMA:
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


def _shape_beta_similarity(master_beta: Any, candidate_beta: Any) -> tuple[Optional[float], Optional[float]]:
    master = _normalize_vector(master_beta)
    candidate = _normalize_vector(candidate_beta)
    if master is None or candidate is None or master.shape != candidate.shape:
        return None, None
    delta = float(np.mean(np.abs(master - candidate)))
    similarity = float(np.exp(-delta))
    return similarity, delta


def _signature_similarity(master_signature: Any, candidate_signature: Any) -> tuple[Optional[float], Optional[float]]:
    master = _normalize_vector(master_signature)
    candidate = _normalize_vector(candidate_signature)
    if master is None or candidate is None or master.shape != candidate.shape:
        return None, None
    delta = float(np.mean(np.abs(master - candidate)))
    similarity = float(np.exp(-(delta * 3.0)))
    return similarity, delta


def _measurement_similarity(
    master_values: Dict[str, float],
    candidate_values: Dict[str, float],
    scales: Dict[str, float],
) -> Dict[str, Any]:
    keys = [key for key in master_values.keys() if key in candidate_values]
    if len(keys) == 0:
        return {"score": None, "coverage": 0.0, "top_drifts": []}
    scores: List[float] = []
    drifts: List[Dict[str, Any]] = []
    for key in keys:
        master_value = float(master_values[key])
        candidate_value = float(candidate_values[key])
        delta = abs(candidate_value - master_value)
        scale = max(abs(float(scales.get(key, _DEFAULT_MEASUREMENT_SCALE))), 1e-4)
        score = 1.0 / (1.0 + (delta / scale))
        scores.append(float(score))
        drifts.append(
            {
                "feature": key,
                "reference": _round_or_none(master_value),
                "candidate": _round_or_none(candidate_value),
                "abs_delta": _round_or_none(delta),
            }
        )
    drifts.sort(key=lambda row: float(row.get("abs_delta") or 0.0), reverse=True)
    return {
        "score": float(sum(scores) / max(1, len(scores))),
        "coverage": float(len(keys) / max(1, len(master_values))),
        "top_drifts": drifts[:4],
    }


def _pose_delta_similarity(master_pose: Any, candidate_pose: Any) -> tuple[Optional[float], Optional[float]]:
    master = _normalize_vector(master_pose)
    candidate = _normalize_vector(candidate_pose)
    if master is None or candidate is None or master.shape != candidate.shape:
        return None, None
    delta = float(np.mean(np.abs(master - candidate)))
    similarity = float(np.exp(-delta / 0.35))
    return similarity, delta


def _metric_spec(
    metric_name: str,
    metric_value: Optional[float],
    *,
    confidence: Optional[float],
    coverage: Optional[float],
    signature_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "metric_name": str(metric_name),
        "metric_value": _round_or_none(metric_value),
        "confidence": _round_or_none(confidence),
        "coverage": _round_or_none(coverage),
        "lane_scope": "report_item",
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "failure_reason": None,
    }
    if isinstance(signature_ref, dict):
        spec["signature_ref"] = signature_ref
    return spec


class BodyCanonicalHeavyEvidenceProvider(HeavyEvidenceProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        settings = _resolve_settings()
        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": settings["resolved_device"],
            "requested_device": settings["device"],
            "reason": settings["integration_reason"],
            "integration_state": settings["integration_state"],
            "requires_master_truth_artifact": True,
            "requires_candidate_artifact": True,
            "master_truth_artifact_name": _MASTER_ARTIFACT_NAME,
            "evidence_schema_version": "heavy_evidence_v1",
            "integration_ready": bool(settings["direct_ready"]),
            "repo_dir": str(settings["repo_dir"]),
            "entrypoint": str(settings["entrypoint"]),
            "smpl_model_path": settings["smpl_model_path"],
            "smpl_candidates": list(settings["smpl_candidates"]),
        }

    def _build_run_key(self, image_path: Path) -> str:
        stat = image_path.stat()
        payload = {
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "source_path": str(image_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _run_direct_export(self, image_path: Path) -> Path:
        settings = _resolve_settings()
        if not settings["direct_ready"]:
            raise RuntimeError("HMR2 direct export is not configured")

        run_root = (
            _repo_root()
            / "outputs"
            / "heavy_evidence_cache"
            / self.provider_name
            / "direct_runs"
            / self._build_run_key(image_path)
        )
        input_dir = run_root / "input"
        output_dir = run_root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_copy = input_dir / image_path.name
        if not input_copy.exists():
            shutil.copy2(image_path, input_copy)

        command_template = str(settings["command_template"] or "").strip()
        if command_template:
            command_text = command_template.format(
                python=settings["python_exec"],
                entrypoint=str(settings["entrypoint"]),
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                image_path=str(input_copy),
                repo=str(settings["repo_dir"]),
                device=str(settings["resolved_device"]),
            )
            completed = subprocess.run(
                command_text,
                cwd=str(settings["repo_dir"]) if Path(settings["repo_dir"]).exists() else str(_repo_root()),
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        else:
            command = [
                str(settings["python_exec"]),
                str(settings["entrypoint"]),
                "--image_path",
                str(input_copy),
                "--output_dir",
                str(output_dir),
                "--device",
                str(settings["resolved_device"]),
            ]
            extra_args = str(settings["extra_args"] or "").strip()
            if extra_args:
                command.extend(shlex.split(extra_args, posix=False))
            completed = subprocess.run(
                command,
                cwd=str(settings["repo_dir"]),
                capture_output=True,
                text=True,
                timeout=300,
            )

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"HMR2 export failed: returncode={completed.returncode} stderr={stderr[:400]}"
            )
        export_path = _find_export_file(output_dir, input_copy.stem)
        if export_path is None:
            raise FileNotFoundError(
                f"No raw HMR2 export found under {output_dir}; configure XIAONA_HMR2_CMD_TEMPLATE to write .pkl/.json/.npz."
            )
        return export_path

    def _ensure_master_artifact(self, runtime: Any) -> bool:
        master_artifact, _ = _load_master_artifact(runtime)
        if _artifact_is_current(master_artifact, min_measurements=_MIN_CANONICAL_MEASUREMENTS):
            return False
        settings = _resolve_settings()
        master_body_path = _resolve_master_body_path(runtime)
        if not master_body_path.exists():
            raise FileNotFoundError(f"Body truth anchor not found: {master_body_path}")
        export_path = self._run_direct_export(master_body_path)
        artifact = _build_direct_artifact(
            source_path=master_body_path,
            source_role="master_truth",
            export_path=export_path,
        )
        if _normalize_vector(artifact.get("shape_beta")) is None:
            raise ValueError("HMR2 master export did not produce shape_beta")
        master_path = _master_truth_dir(runtime) / _MASTER_ARTIFACT_NAME
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_text(json.dumps(_json_ready(artifact), indent=2, ensure_ascii=False), encoding="utf-8")
        register_artifact_manifest(
            artifact_path=master_path,
            manifest_root=_master_truth_dir(runtime),
            artifact_family="body_canonical",
            artifact_role="master_truth",
            provider_name=str(artifact.get("provider_name") or self.provider_name),
            provider_family=str(artifact.get("provider_family") or self.provider_family),
            provider_version=str(artifact.get("provider_version") or self.provider_version),
            model_id=str(artifact.get("model_id") or _MODEL_ID),
            schema_version=str(artifact.get("schema_version") or _ARTIFACT_SCHEMA),
            source_path=master_body_path,
            device=str(settings["resolved_device"]),
            repo_dir=Path(settings["repo_dir"]),
            entrypoint=str(settings["entrypoint"]),
            conversion_meta=dict(artifact.get("conversion_meta") or {}),
            extra={"notes": artifact.get("notes"), "direct_export_path": str(export_path)},
        )
        return True

    def _ensure_candidate_artifact(self, runtime: Any, image_path: Path) -> bool:
        candidate_artifact, cache_key, cache_file = _load_candidate_artifact(runtime, image_path)
        if candidate_artifact is not None:
            return False
        settings = _resolve_settings()
        export_path = self._run_direct_export(image_path)
        artifact = _build_direct_artifact(
            source_path=image_path,
            source_role="candidate",
            export_path=export_path,
        )
        if _normalize_vector(artifact.get("shape_beta")) is None:
            raise ValueError("HMR2 candidate export did not produce shape_beta")
        cache_written = bool(_write_cached_candidate(cache_file, cache_key, artifact))
        if cache_written:
            register_artifact_manifest(
                artifact_path=cache_file,
                manifest_root=_master_truth_dir(runtime),
                artifact_family="body_canonical",
                artifact_role="candidate",
                provider_name=str(artifact.get("provider_name") or self.provider_name),
                provider_family=str(artifact.get("provider_family") or self.provider_family),
                provider_version=str(artifact.get("provider_version") or self.provider_version),
                model_id=str(artifact.get("model_id") or _MODEL_ID),
                schema_version=str(artifact.get("schema_version") or _ARTIFACT_SCHEMA),
                source_path=image_path,
                device=str(settings["resolved_device"]),
                repo_dir=Path(settings["repo_dir"]),
                entrypoint=str(settings["entrypoint"]),
                conversion_meta=dict(artifact.get("conversion_meta") or {}),
                extra={
                    "notes": artifact.get("notes"),
                    "cache_key": cache_key,
                    "direct_export_path": str(export_path),
                },
            )
        return cache_written

    def get_heavy_evidence_metrics(
        self,
        runtime: Any,
        image_path: Path,
    ) -> Dict[str, Any]:
        resolved_path = Path(image_path).resolve()
        settings = _resolve_settings()
        direct_guidance: List[str] = []
        direct_export_state = "not_configured"
        direct_generated_master = False
        direct_generated_candidate = False
        if settings["direct_ready"]:
            try:
                direct_generated_master = self._ensure_master_artifact(runtime)
                direct_generated_candidate = self._ensure_candidate_artifact(runtime, resolved_path)
                direct_export_state = "ok"
            except Exception as exc:
                direct_export_state = "failed"
                direct_guidance.extend(
                    [
                        "Direct HMR2 export failed; cached artifacts are still accepted if already prepared.",
                        str(exc),
                    ]
                )
        else:
            if settings["integration_state"] == "missing_smpl_model":
                direct_guidance.extend(
                    [
                        "Local HMR2 direct export is blocked because the SMPL neutral model is missing.",
                        f"Place {_DEFAULT_SMPL_MODEL} under data/ or external/4D-Humans/data/, then rerun QA.",
                    ]
                )
            elif settings["integration_state"] == "missing_repo":
                direct_guidance.append(
                    "Clone 4D-Humans into external/4D-Humans to enable local body canonical export."
                )
            elif settings["integration_state"] == "missing_entrypoint":
                direct_guidance.append(
                    "Provide demo_xiaona_export.py under external/4D-Humans or set XIAONA_HMR2_ENTRYPOINT."
                )
            else:
                direct_guidance.append(
                    "Set XIAONA_HMR2_CMD_TEMPLATE to enable direct body canonical export, or continue using pre-generated artifacts."
                )

        master_artifact, master_path = _load_master_artifact(runtime)
        candidate_artifact, cache_key, cache_file = _load_candidate_artifact(runtime, resolved_path)
        if master_artifact is not None and master_path is not None and Path(master_path).exists():
            register_artifact_manifest(
                artifact_path=Path(master_path),
                manifest_root=_master_truth_dir(runtime),
                artifact_family="body_canonical",
                artifact_role="master_truth",
                provider_name=str(master_artifact.get("provider_name") or self.provider_name),
                provider_family=str(master_artifact.get("provider_family") or self.provider_family),
                provider_version=str(master_artifact.get("provider_version") or self.provider_version),
                model_id=str(master_artifact.get("model_id") or _MODEL_ID),
                schema_version=str(master_artifact.get("schema_version") or _ARTIFACT_SCHEMA),
                source_path=_resolve_master_body_path(runtime),
                device=str(settings["resolved_device"]),
                repo_dir=Path(settings["repo_dir"]),
                entrypoint=str(settings["entrypoint"]),
                conversion_meta=dict(master_artifact.get("conversion_meta") or {}),
                extra={"notes": master_artifact.get("notes")},
            )
        if candidate_artifact is not None:
            candidate_artifact_path = candidate_artifact.get("sidecar_file") or candidate_artifact.get("cache_file")
            if candidate_artifact_path:
                register_artifact_manifest(
                    artifact_path=Path(str(candidate_artifact_path)),
                    manifest_root=_master_truth_dir(runtime),
                    artifact_family="body_canonical",
                    artifact_role="candidate",
                    provider_name=str(candidate_artifact.get("provider_name") or self.provider_name),
                    provider_family=str(candidate_artifact.get("provider_family") or self.provider_family),
                    provider_version=str(candidate_artifact.get("provider_version") or self.provider_version),
                    model_id=str(candidate_artifact.get("model_id") or _MODEL_ID),
                    schema_version=str(candidate_artifact.get("schema_version") or _ARTIFACT_SCHEMA),
                    source_path=resolved_path,
                    device=str(settings["resolved_device"]),
                    repo_dir=Path(settings["repo_dir"]),
                    entrypoint=str(settings["entrypoint"]),
                    conversion_meta=dict(candidate_artifact.get("conversion_meta") or {}),
                    extra={
                        "notes": candidate_artifact.get("notes"),
                        "cache_key": candidate_artifact.get("cache_key") or cache_key,
                    },
                )

        reasons: List[str] = []
        if master_artifact is None:
            reasons.append("BODY_CANONICAL_MASTER_MISSING")
        if candidate_artifact is None:
            reasons.append("BODY_CANONICAL_CANDIDATE_ARTIFACT_MISSING")
        if master_artifact is None or candidate_artifact is None:
            guidance = [
                "Provide outputs/master_truth/body_master_shape_only.json from the 116-1 canonical export.",
                "Provide per-image *.body_canonical.json sidecars or pre-fill the provider cache before QA.",
            ]
            for item in direct_guidance:
                if item and item not in guidance:
                    guidance.append(item)
            return {
                "ok": False,
                "provider_name": self.provider_name,
                "provider_family": self.provider_family,
                "provider_version": self.provider_version,
                "model_id": _MODEL_ID,
                "device": settings["resolved_device"],
                "source_path": str(resolved_path),
                "cache_key": cache_key,
                "cache_file": str(cache_file),
                "cache_state": "miss",
                "reasons": reasons,
                "metric_specs": [],
                "summary": {
                    "integration_state": settings["integration_state"],
                    "direct_export_state": direct_export_state,
                    "direct_generated_master": bool(direct_generated_master),
                    "direct_generated_candidate": bool(direct_generated_candidate),
                    "master_artifact_path": str(master_path) if master_path is not None else None,
                    "candidate_sidecar_candidates": [str(path) for path in _sidecar_candidates(resolved_path)],
                    "cache_hit_count": 0,
                    "cache_miss_count": 1,
                    "cache_write_count": 0,
                    "guidance": guidance,
                },
            }

        beta_similarity, beta_delta = _shape_beta_similarity(
            master_artifact.get("shape_beta"),
            candidate_artifact.get("shape_beta"),
        )
        topology_similarity, topology_delta = _signature_similarity(
            master_artifact.get("body_topology_signature"),
            candidate_artifact.get("body_topology_signature"),
        )
        measurement_diag = _measurement_similarity(
            dict(master_artifact.get("canonical_measurements") or {}),
            dict(candidate_artifact.get("canonical_measurements") or {}),
            dict(master_artifact.get("measurement_scales") or {}),
        )
        pose_similarity, pose_delta = _pose_delta_similarity(
            master_artifact.get("pose_vector"),
            candidate_artifact.get("pose_vector"),
        )

        confidence_terms = [
            _safe_float(master_artifact.get("fit_confidence"), None),
            _safe_float(candidate_artifact.get("fit_confidence"), None),
            _safe_float(candidate_artifact.get("coverage"), None),
        ]
        confidence_values = [float(value) for value in confidence_terms if value is not None]
        confidence = float(sum(confidence_values) / max(1, len(confidence_values))) if confidence_values else None
        coverage = measurement_diag.get("coverage")
        body_shape_truth_alignment = None
        weighted_terms: List[Tuple[float, float]] = []
        if beta_similarity is not None:
            weighted_terms.append((float(beta_similarity), 0.64))
        if isinstance(measurement_diag.get("score"), (int, float)):
            weighted_terms.append((float(measurement_diag["score"]), 0.36))
        if len(weighted_terms) > 0:
            numerator = sum(value * weight for value, weight in weighted_terms)
            denominator = sum(weight for _, weight in weighted_terms)
            body_shape_truth_alignment = float(numerator / denominator) if denominator > 0.0 else None

        summary = {
            "integration_state": "artifact_compare_ready",
            "direct_export_state": direct_export_state,
            "direct_generated_master": bool(direct_generated_master),
            "direct_generated_candidate": bool(direct_generated_candidate),
            "master_artifact_path": str(master_path) if master_path is not None else None,
            "candidate_artifact_path": str(candidate_artifact.get("sidecar_file") or candidate_artifact.get("cache_file") or ""),
            "cache_hit_count": 1 if candidate_artifact.get("cache_state") == "hit" else 0,
            "cache_miss_count": 0 if candidate_artifact.get("cache_state") == "hit" else 1,
            "cache_write_count": 1 if candidate_artifact.get("cache_state") == "write" else 0,
            "top_drifts": list(measurement_diag.get("top_drifts") or []),
            "guidance": dedupe_keep_order(
                [
                    "Use this provider to separate 116-1 shape truth from gait pose before admission review.",
                    "Promote HMR2 inference only after the artifact contract is stable on frozen benchmarks.",
                    *direct_guidance,
                ]
            )[:4],
        }

        return {
            "ok": body_shape_truth_alignment is not None,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": settings["resolved_device"],
            "source_path": str(resolved_path),
            "cache_key": cache_key,
            "cache_file": str(cache_file),
            "cache_state": str(candidate_artifact.get("cache_state") or "miss"),
            "confidence": confidence,
            "coverage": _safe_float(coverage, 0.0),
            "reasons": [],
            "metric_specs": [
                _metric_spec(
                    "body_shape_truth_alignment",
                    body_shape_truth_alignment,
                    confidence=confidence,
                    coverage=coverage,
                    signature_ref=_vector_signature_ref("shape_beta", candidate_artifact.get("shape_beta")),
                ),
                _metric_spec(
                    "body_shape_beta_similarity",
                    beta_similarity,
                    confidence=confidence,
                    coverage=1.0 if beta_similarity is not None else 0.0,
                    signature_ref=_vector_signature_ref("shape_beta", candidate_artifact.get("shape_beta")),
                ),
                _metric_spec(
                    "body_topology_signature_similarity",
                    topology_similarity,
                    confidence=confidence,
                    coverage=coverage,
                    signature_ref=_topology_signature_ref("body_topology_signature", candidate_artifact.get("body_topology_signature")),
                ),
                _metric_spec(
                    "canonical_measurement_similarity",
                    _safe_float(measurement_diag.get("score"), None),
                    confidence=confidence,
                    coverage=coverage,
                ),
                _metric_spec(
                    "body_pose_delta_similarity",
                    pose_similarity,
                    confidence=confidence,
                    coverage=1.0 if pose_similarity is not None else 0.0,
                    signature_ref=_vector_signature_ref("pose_vector", candidate_artifact.get("pose_vector")),
                ),
                _metric_spec(
                    "body_mesh_fit_confidence",
                    _safe_float(candidate_artifact.get("fit_confidence"), None),
                    confidence=_safe_float(candidate_artifact.get("fit_confidence"), None),
                    coverage=coverage,
                ),
            ],
            "signature_refs": {
                "shape_beta": _vector_signature_ref("shape_beta", candidate_artifact.get("shape_beta")),
                "body_topology_signature": _topology_signature_ref("body_topology_signature", candidate_artifact.get("body_topology_signature")),
                "pose_vector": _vector_signature_ref("pose_vector", candidate_artifact.get("pose_vector")),
            },
            "summary": {
                **summary,
                "shape_beta_delta_l1": _round_or_none(beta_delta),
                "body_topology_delta_l1": _round_or_none(topology_delta),
                "pose_delta_l1": _round_or_none(pose_delta),
            },
        }
