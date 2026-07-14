from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .qa_artifact_manifest import register_artifact_manifest
from .providers import FaceCanonicalProvider
from .qa_features import extract_face_feat
from .qa_io import atomic_write_json

_PROVIDER_NAME = "face_pose_canonical_3ddfa"
_PROVIDER_FAMILY = "face_canonical_shadow"
_PROVIDER_VERSION = "face_pose_canonical_3ddfa_v1"
_MODEL_ID = "3ddfa_v3_direct_bridge_v1"
_ARTIFACT_SCHEMA = "face_pose_canonical_artifact_v1"
_DEFAULT_MASTER_FACE = Path("anchors") / "face" / "front" / "A-Core_01_0deg_MASTER.png"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _to_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if vector.size == 0:
        return None
    return vector.reshape(-1)


@lru_cache(maxsize=32)
def _sha256_file_cached(path_text: str) -> Optional[str]:
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _model_bundle_contract(bundle_id: str, path_texts: tuple[str, ...]) -> Dict[str, Any]:
    components = []
    complete = True
    for path_text in sorted(path_texts):
        path = Path(path_text)
        sha256 = _sha256_file_cached(str(path.resolve()))
        if sha256 is None:
            complete = False
        components.append({"name": path.name, "path": str(path.resolve()), "sha256": sha256})
    bundle_sha256 = None
    if complete and components:
        digest = hashlib.sha256()
        for component in components:
            digest.update(str(component["name"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(component["sha256"]).encode("ascii"))
            digest.update(b"\n")
        bundle_sha256 = digest.hexdigest()
    return {
        "bundle_id": str(bundle_id),
        "bundle_sha256": bundle_sha256,
        "components": components,
        "complete": bool(complete and components),
    }


def _insightface_model_contract() -> Dict[str, Any]:
    insightface_home = Path(os.getenv("INSIGHTFACE_HOME", str(Path.home() / ".insightface"))).expanduser()
    model_dir = insightface_home / "models" / "buffalo_l"
    expected_names = ["1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"]
    paths = tuple(str((model_dir / name).resolve()) for name in expected_names)
    return _model_bundle_contract("insightface_buffalo_l_onnx_bundle", paths)


def _shape_model_contract(repo_dir: Path, entrypoint: Path, execution_backend: str) -> Dict[str, Any]:
    asset_names = [
        "face_model.npy",
        "large_base_net.pth",
        "net_recon.pth",
        "retinaface_resnet50_2020-07-20_old_torch.pth",
        "similarity_Lm3D_all.mat",
    ]
    paths = tuple(str((repo_dir / "assets" / name).resolve()) for name in asset_names)
    bundle = _model_bundle_contract("3ddfa_v3_xiaona_required_asset_bundle", paths)
    return {
        "provider_name": _PROVIDER_NAME,
        "provider_version": _PROVIDER_VERSION,
        "model_id": _MODEL_ID,
        "model_sha256": bundle.get("bundle_sha256"),
        "model_bundle": bundle,
        "implementation_sha256": _sha256_file_cached(str(entrypoint.resolve())),
        "execution_backend": str(execution_backend or "").strip() or None,
    }


def _runtime_embedding_contract(
    identity_vector: Optional[np.ndarray],
    execution_backend: Optional[str],
) -> Dict[str, Any]:
    try:
        provider_version = importlib.metadata.version("insightface")
    except importlib.metadata.PackageNotFoundError:
        provider_version = None
    vector = _to_vector(identity_vector)
    model_contract = _insightface_model_contract()
    return {
        "provider_name": "insightface.FaceAnalysis",
        "provider_version": provider_version,
        "model_id": "buffalo_l",
        "model_sha256": model_contract.get("bundle_sha256"),
        "model_bundle": model_contract,
        "execution_backend": str(execution_backend or "").strip() or None,
        "detector_contract_id": "insightface_buffalo_l_largest_face_v1",
        "alignment_contract_id": "insightface_faceanalysis_norm_crop_buffalo_l_v1",
        "preprocessing_contract_id": "insightface_buffalo_l_det640_default_v1",
        "dimension": int(vector.size) if vector is not None else None,
        "source_field": "face_feat.embedding",
        "source_normalization": "raw_embedding",
    }


def _runtime_face_execution_backend(runtime: Any) -> Optional[str]:
    engines = getattr(runtime, "engines", None)
    face_app = getattr(engines, "face_app", None)
    models = getattr(face_app, "models", None)
    providers = set()
    if isinstance(models, dict):
        for model in models.values():
            session = getattr(model, "session", None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                try:
                    providers.update(str(value) for value in get_providers() if value)
                except Exception:
                    continue
    return "+".join(sorted(providers)) or None


def describe_face_measurement_runtime_contract(runtime: Any) -> Dict[str, Any]:
    """Describe model assets and backends before a repeatability run starts."""
    settings = _resolve_settings()
    execution_backend = _runtime_face_execution_backend(runtime)
    return {
        "schema_version": "face_measurement_runtime_contract_v0_1",
        "shape": _shape_model_contract(
            Path(settings["repo_dir"]),
            Path(settings["entrypoint"]),
            str(settings["resolved_device"]),
        ),
        "identity": _runtime_embedding_contract(None, execution_backend),
        "direct_ready": bool(settings["direct_ready"]),
        "requested_device": str(settings["device"]),
        "resolved_device": str(settings["resolved_device"]),
    }


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


def _select_record(payload: Any, *, index: int = 0) -> Any:
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_master_face_path(runtime: Any) -> Path:
    registry = getattr(runtime.config, "anchor_registry", {}) or {}
    anchors = registry.get("anchors", {}) if isinstance(registry, dict) else {}
    rules = registry.get("rules", {}) if isinstance(registry, dict) else {}
    face_truth_id = str(rules.get("face_truth_anchor") or "ref_1_face_master")
    face_truth = anchors.get(face_truth_id) if isinstance(anchors, dict) else None
    raw_path = ""
    if isinstance(face_truth, dict):
        raw_path = str(face_truth.get("path") or "").strip()
    if not raw_path:
        return (runtime.config.paths.base_dir / _DEFAULT_MASTER_FACE).resolve()
    expanded = raw_path.replace("${PROJECT_ROOT}", str(runtime.config.paths.base_dir))
    return Path(expanded).resolve()


def _resolve_settings() -> Dict[str, Any]:
    repo_root = _repo_root()
    repo_dir = Path(os.getenv("XIAONA_3DDFA_V3_REPO", str(repo_root / "external" / "3DDFA-V3"))).resolve()
    python_exec = os.getenv("XIAONA_3DDFA_V3_PYTHON", sys.executable)
    default_entrypoint = repo_dir / "demo_lite_export.py"
    if not default_entrypoint.exists():
        default_entrypoint = repo_dir / "demo.py"
    entrypoint = Path(os.getenv("XIAONA_3DDFA_V3_ENTRYPOINT", str(default_entrypoint))).resolve()
    command_template = str(os.getenv("XIAONA_3DDFA_V3_CMD_TEMPLATE", "") or "").strip()
    extra_args = str(os.getenv("XIAONA_3DDFA_V3_EXTRA_ARGS", "") or "").strip()
    device = str(os.getenv("XIAONA_3DDFA_V3_DEVICE", "auto") or "auto").strip()
    resolved_device = _resolved_device(device)
    direct_ready = bool(command_template) or (repo_dir.exists() and entrypoint.exists())
    return {
        "repo_dir": repo_dir,
        "python_exec": python_exec,
        "entrypoint": entrypoint,
        "command_template": command_template,
        "extra_args": extra_args,
        "device": device,
        "resolved_device": resolved_device,
        "direct_ready": direct_ready,
    }


def _resolved_device(device_name: str) -> str:
    device_name = str(device_name or "auto").strip().lower()
    require_gpu = str(os.getenv("XIAONA_REQUIRE_GPU", "")).strip().lower() in {"1", "true", "yes", "on"}
    if device_name == "cpu":
        return "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            if require_gpu:
                raise RuntimeError("CUDA requested for 3DDFA-V3 but torch.cuda.is_available() is False")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    raise ValueError(f"Unsupported 3DDFA export format: {path.suffix}")


def _find_export_file(output_dir: Path, image_stem: str) -> Optional[Path]:
    candidates: List[Path] = []
    for suffix in [".json", ".npz", ".npy"]:
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
    identity_vector: Optional[np.ndarray],
    face_confidence: Optional[float],
    shape_provider_contract: Optional[Dict[str, Any]] = None,
    identity_execution_backend: Optional[str] = None,
) -> Dict[str, Any]:
    from . import qa_face_pose_canonical as bridge_mod

    payload = _load_payload(export_path)
    if isinstance(payload, dict) and str(payload.get("schema_version") or "") == _ARTIFACT_SCHEMA:
        artifact = dict(payload)
        artifact["source_path"] = str(source_path)
        artifact["source_role"] = source_role
        if identity_vector is not None:
            artifact["canonical_identity_vector"] = _json_ready(identity_vector)
            artifact["runtime_face_embedding_raw"] = _json_ready(identity_vector)
            artifact["runtime_face_embedding_unit"] = _json_ready(bridge_mod._unit_vector(identity_vector))
            artifact["runtime_face_embedding_contract"] = _runtime_embedding_contract(
                identity_vector,
                identity_execution_backend,
            )
        if artifact.get("visible_face_coverage") is None and face_confidence is not None:
            artifact["visible_face_coverage"] = face_confidence
        if artifact.get("pose_fit_confidence") is None and face_confidence is not None:
            artifact["pose_fit_confidence"] = face_confidence
        artifact.setdefault("provider_name", _PROVIDER_NAME)
        artifact.setdefault("provider_family", _PROVIDER_FAMILY)
        artifact.setdefault("provider_version", _PROVIDER_VERSION)
        artifact.setdefault("model_id", _MODEL_ID)
        shape_contract = dict(shape_provider_contract or {})
        if shape_contract:
            artifact["model_sha256"] = shape_contract.get("model_sha256")
            artifact["provider_implementation_sha256"] = shape_contract.get("implementation_sha256")
            canonical_contract = dict(artifact.get("canonical_landmark_contract") or {})
            canonical_contract.update(shape_contract)
            canonical_contract.update(
                {
                    "landmark_schema_id": artifact.get("landmark_schema_id"),
                    "coordinate_convention": artifact.get("landmark_coordinate_convention"),
                    "preprocessing_contract_id": artifact.get("canonical_preprocessing_contract_id"),
                    "source_field": artifact.get("landmark_source_field"),
                }
            )
            artifact["canonical_landmark_contract"] = canonical_contract
        if artifact.get("canonical_face_topology_signature") is None:
            topology_signature = bridge_mod._landmark_topology_signature(artifact.get("canonical_landmarks"))
            if topology_signature is not None:
                artifact["canonical_face_topology_signature"] = _json_ready(topology_signature)
        return artifact

    record = _select_record(payload)
    landmarks = _pick_field(
        record,
        aliases=[
            "canonical_landmarks",
            "landmarks_2d",
            "landmarks",
            "ldm106_2d",
            "ldm106",
            "ldm68",
            "ldm134",
            "pts106",
            "pts68",
        ],
    )
    landmark_visibility_weights = _pick_field(
        record,
        aliases=[
            "landmark_visibility_weights",
            "landmark_weights",
            "landmark_confidence",
            "landmark_confidences",
            "landmark_scores",
        ],
    )
    landmark_schema_id = _pick_field(
        record,
        aliases=["landmark_schema_id", "landmark_topology_id", "landmark_layout_id"],
    )
    pose_value = _pick_field(
        record,
        aliases=[
            "pose_euler_deg",
            "pose_euler",
            "pose",
            "head_pose",
            "yaw_pitch_roll",
            "angles",
        ],
    )
    visible_face_coverage = _safe_float(
        _pick_field(record, aliases=["visible_face_coverage", "face_visible_ratio", "coverage"]),
        face_confidence,
    )
    frontalization_quality = _safe_float(
        _pick_field(record, aliases=["frontalization_quality", "frontal_quality", "canonical_quality"]),
        None,
    )
    pose_fit_confidence = _safe_float(
        _pick_field(record, aliases=["pose_fit_confidence", "fit_confidence", "confidence", "score"]),
        face_confidence,
    )
    topology_signature = bridge_mod._landmark_topology_signature(landmarks)
    normalized_shape_contract = dict(shape_provider_contract or {})
    normalized_shape_contract.update(
        {
            "landmark_schema_id": str(landmark_schema_id).strip() if landmark_schema_id is not None else None,
            "coordinate_convention": None,
            "preprocessing_contract_id": None,
            "source_field": None,
        }
    )
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "model_id": _MODEL_ID,
        "model_sha256": normalized_shape_contract.get("model_sha256"),
        "provider_implementation_sha256": normalized_shape_contract.get("implementation_sha256"),
        "source_path": str(source_path),
        "source_role": source_role,
        "canonical_landmarks": _to_vector(landmarks),
        "landmark_visibility_weights": _to_vector(landmark_visibility_weights),
        "landmark_schema_id": str(landmark_schema_id).strip() if landmark_schema_id is not None else None,
        "canonical_landmark_contract": normalized_shape_contract,
        "canonical_face_topology_signature": topology_signature,
        "canonical_identity_vector": identity_vector,
        "runtime_face_embedding_raw": identity_vector,
        "runtime_face_embedding_unit": bridge_mod._unit_vector(identity_vector),
        "runtime_face_embedding_contract": _runtime_embedding_contract(
            identity_vector,
            identity_execution_backend,
        ),
        "pose_euler_deg": _normalize_pose(pose_value),
        "visible_face_coverage": visible_face_coverage,
        "frontalization_quality": frontalization_quality,
        "pose_fit_confidence": pose_fit_confidence,
        "notes": "direct 3DDFA-V3 export bridged into face canonical shadow",
        "conversion_meta": {
            "export_path": str(export_path),
            "identity_source": "insightface_runtime_embedding",
        },
    }


class FacePoseCanonical3DDFAProvider(FaceCanonicalProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def __init__(self) -> None:
        from .qa_face_pose_canonical import FacePoseCanonicalProvider

        self._bridge = FacePoseCanonicalProvider()

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
            "mode": "shadow_only",
            "integration_ready": bool(settings["direct_ready"]),
            "repo_dir": str(settings["repo_dir"]),
            "entrypoint": str(settings["entrypoint"]),
            "bridge_fallback": "face_pose_canonical_bridge",
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
            raise RuntimeError("3DDFA-V3 integration is not configured")

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
                timeout=180,
            )
        else:
            device_name = str(settings["resolved_device"])
            command = [
                str(settings["python_exec"]),
                str(settings["entrypoint"]),
                "--inputpath",
                str(input_dir),
                "--savepath",
                str(output_dir),
                "--device",
                device_name,
                "--iscrop",
                "1",
                "--detector",
                "retinaface",
                "--ldm68",
                "1",
                "--ldm106",
                "1",
                "--ldm106_2d",
                "1",
                "--ldm134",
                "0",
                "--seg_visible",
                "0",
                "--seg",
                "0",
                "--useTex",
                "0",
                "--extractTex",
                "0",
                "--backbone",
                "resnet50",
            ]
            extra_args = str(settings["extra_args"] or "").strip()
            if extra_args:
                command.extend(shlex.split(extra_args, posix=False))
            completed = subprocess.run(
                command,
                cwd=str(settings["repo_dir"]),
                capture_output=True,
                text=True,
                timeout=180,
            )

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"3DDFA-V3 export failed: returncode={completed.returncode} stderr={stderr[:400]}"
            )
        export_path = _find_export_file(output_dir, input_copy.stem)
        if export_path is None:
            raise FileNotFoundError(f"No 3DDFA-V3 export found under {output_dir}")
        try:
            input_copy.unlink()
        except OSError:
            pass
        return export_path

    def _ensure_master_artifact(self, runtime: Any) -> bool:
        from . import qa_face_pose_canonical as bridge_mod

        master_artifact, _ = bridge_mod._load_master_artifact(runtime)
        if master_artifact is not None:
            return False
        settings = _resolve_settings()
        shape_provider_contract = _shape_model_contract(
            Path(settings["repo_dir"]),
            Path(settings["entrypoint"]),
            str(settings["resolved_device"]),
        )
        master_face_path = _resolve_master_face_path(runtime)
        if not master_face_path.exists():
            raise FileNotFoundError(f"Face truth anchor not found: {master_face_path}")
        master_img = cv2.imread(str(master_face_path), cv2.IMREAD_COLOR)
        master_face_feat = extract_face_feat(runtime, master_img, source_path=master_face_path) if master_img is not None else None
        export_path = self._run_direct_export(master_face_path)
        identity_vector = None
        face_confidence = None
        if master_face_feat is not None:
            identity_vector = getattr(master_face_feat, "embedding", None)
            face_confidence = _safe_float(getattr(master_face_feat, "confidence", None), None)
        artifact = _build_direct_artifact(
            source_path=master_face_path,
            source_role="master_truth",
            export_path=export_path,
            identity_vector=identity_vector,
            face_confidence=face_confidence,
            shape_provider_contract=shape_provider_contract,
            identity_execution_backend=_runtime_face_execution_backend(runtime),
        )
        master_path = bridge_mod._master_truth_dir(runtime) / bridge_mod._MASTER_ARTIFACT_NAME
        atomic_write_json(master_path, _json_ready(artifact))
        register_artifact_manifest(
            artifact_path=master_path,
            manifest_root=bridge_mod._master_truth_dir(runtime),
            artifact_family="face_canonical",
            artifact_role="master_truth",
            provider_name=str(artifact.get("provider_name") or self.provider_name),
            provider_family=str(artifact.get("provider_family") or self.provider_family),
            provider_version=str(artifact.get("provider_version") or self.provider_version),
            model_id=str(artifact.get("model_id") or _MODEL_ID),
            schema_version=str(artifact.get("schema_version") or _ARTIFACT_SCHEMA),
            source_path=master_face_path,
            device=str(settings["resolved_device"]),
            repo_dir=Path(settings["repo_dir"]),
            entrypoint=str(settings["entrypoint"]),
            conversion_meta=dict(artifact.get("conversion_meta") or {}),
            extra={"notes": artifact.get("notes"), "direct_export_path": str(export_path)},
        )
        return True

    def _ensure_candidate_artifact(
        self,
        runtime: Any,
        image_path: Path,
        *,
        img_bgr: Optional[np.ndarray],
        face_feat: Optional[Any],
    ) -> bool:
        from . import qa_face_pose_canonical as bridge_mod

        candidate_artifact, cache_key, cache_file = bridge_mod._load_candidate_artifact(runtime, image_path)
        if candidate_artifact is not None:
            return False
        candidate_face_feat = face_feat
        if candidate_face_feat is None and img_bgr is not None:
            candidate_face_feat = extract_face_feat(runtime, img_bgr, source_path=image_path)
        export_path = self._run_direct_export(image_path)
        settings = _resolve_settings()
        shape_provider_contract = _shape_model_contract(
            Path(settings["repo_dir"]),
            Path(settings["entrypoint"]),
            str(settings["resolved_device"]),
        )
        identity_vector = None
        face_confidence = None
        if candidate_face_feat is not None:
            identity_vector = getattr(candidate_face_feat, "embedding", None)
            face_confidence = _safe_float(getattr(candidate_face_feat, "confidence", None), None)
        artifact = _build_direct_artifact(
            source_path=image_path,
            source_role="candidate",
            export_path=export_path,
            identity_vector=identity_vector,
            face_confidence=face_confidence,
            shape_provider_contract=shape_provider_contract,
            identity_execution_backend=_runtime_face_execution_backend(runtime),
        )
        cache_written = bool(bridge_mod._write_cached_candidate(cache_file, cache_key, artifact))
        if cache_written:
            register_artifact_manifest(
                artifact_path=cache_file,
                manifest_root=bridge_mod._master_truth_dir(runtime),
                artifact_family="face_canonical",
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

    def analyze_face_canonical(
        self,
        runtime: Any,
        image_path: Path,
        *,
        img_bgr: Optional[np.ndarray] = None,
        face_feat: Optional[Any] = None,
    ) -> Dict[str, Any]:
        settings = _resolve_settings()
        direct_reasons: List[str] = []
        direct_guidance: List[str] = []
        generated_master = False
        generated_candidate = False

        if settings["direct_ready"]:
            try:
                generated_master = self._ensure_master_artifact(runtime)
                generated_candidate = self._ensure_candidate_artifact(
                    runtime,
                    Path(image_path).resolve(),
                    img_bgr=img_bgr,
                    face_feat=face_feat,
                )
            except Exception as exc:
                direct_reasons.append("FACE_CANONICAL_DIRECT_EXPORT_FAILED")
                direct_guidance.append(
                    "3DDFA-V3 direct export failed; bridge scoring remains available if cached sidecars already exist."
                )
                direct_guidance.append(str(exc))
        else:
            direct_reasons.append("FACE_CANONICAL_DIRECT_NOT_CONFIGURED")
            direct_guidance.append(
                "Clone 3DDFA-V3 into external/3DDFA-V3 or set XIAONA_3DDFA_V3_REPO to enable direct face canonical export."
            )

        result = self._bridge.analyze_face_canonical(
            runtime,
            Path(image_path),
            img_bgr=img_bgr,
            face_feat=face_feat,
        )
        manifest_root = Path(getattr(runtime.config.paths, "dir_master_truth"))
        master_artifact_path = str(result.get("master_artifact_path") or "").strip()
        candidate_artifact_path = str(result.get("candidate_artifact_path") or "").strip()
        cache_file = str(result.get("cache_file") or "").strip()
        if master_artifact_path:
            register_artifact_manifest(
                artifact_path=Path(master_artifact_path),
                manifest_root=manifest_root,
                artifact_family="face_canonical",
                artifact_role="master_truth",
                provider_name=self.provider_name,
                provider_family=self.provider_family,
                provider_version=self.provider_version,
                model_id=_MODEL_ID,
                schema_version=_ARTIFACT_SCHEMA,
                source_path=_resolve_master_face_path(runtime),
                device=str(settings["resolved_device"]),
                repo_dir=Path(settings["repo_dir"]),
                entrypoint=str(settings["entrypoint"]),
                extra={"bridge_fallback": "face_pose_canonical_bridge"},
            )
        candidate_manifest_path = candidate_artifact_path or cache_file
        if candidate_manifest_path:
            register_artifact_manifest(
                artifact_path=Path(candidate_manifest_path),
                manifest_root=manifest_root,
                artifact_family="face_canonical",
                artifact_role="candidate",
                provider_name=self.provider_name,
                provider_family=self.provider_family,
                provider_version=self.provider_version,
                model_id=_MODEL_ID,
                schema_version=_ARTIFACT_SCHEMA,
                source_path=Path(image_path).resolve(),
                device=str(settings["resolved_device"]),
                repo_dir=Path(settings["repo_dir"]),
                entrypoint=str(settings["entrypoint"]),
                extra={
                    "bridge_fallback": "face_pose_canonical_bridge",
                    "cache_key": result.get("cache_key"),
                },
            )
        merged_reasons = list(result.get("reasons") or [])
        merged_guidance = list(result.get("guidance") or [])
        for reason in direct_reasons:
            if reason not in merged_reasons:
                merged_reasons.append(reason)
        for item in direct_guidance:
            if item and item not in merged_guidance:
                merged_guidance.append(item)
        result.update(
            {
                "enabled": True,
                "provider_name": self.provider_name,
                "provider_family": self.provider_family,
                "provider_version": self.provider_version,
                "model_id": _MODEL_ID,
                "device": settings["resolved_device"],
                "integration_ready": bool(settings["direct_ready"]),
                "bridge_fallback": "face_pose_canonical_bridge",
                "direct_generated_master": bool(generated_master),
                "direct_generated_candidate": bool(generated_candidate),
                "reasons": merged_reasons,
                "guidance": merged_guidance,
            }
        )
        return result
