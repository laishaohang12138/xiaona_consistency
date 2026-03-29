from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .qa_artifact_manifest import register_artifact_manifest
from .providers import FaceCanonicalProvider
from .qa_features import extract_face_feat

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
    if device_name in {"cpu", "cuda"}:
        return device_name
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
) -> Dict[str, Any]:
    payload = _load_payload(export_path)
    if isinstance(payload, dict) and str(payload.get("schema_version") or "") == _ARTIFACT_SCHEMA:
        artifact = dict(payload)
        artifact["source_path"] = str(source_path)
        artifact["source_role"] = source_role
        if identity_vector is not None:
            artifact["canonical_identity_vector"] = _json_ready(identity_vector)
        if artifact.get("visible_face_coverage") is None and face_confidence is not None:
            artifact["visible_face_coverage"] = face_confidence
        if artifact.get("pose_fit_confidence") is None and face_confidence is not None:
            artifact["pose_fit_confidence"] = face_confidence
        artifact.setdefault("provider_name", _PROVIDER_NAME)
        artifact.setdefault("provider_family", _PROVIDER_FAMILY)
        artifact.setdefault("provider_version", _PROVIDER_VERSION)
        artifact.setdefault("model_id", _MODEL_ID)
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
    return {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "model_id": _MODEL_ID,
        "source_path": str(source_path),
        "source_role": source_role,
        "canonical_landmarks": _to_vector(landmarks),
        "canonical_identity_vector": identity_vector,
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
        return export_path

    def _ensure_master_artifact(self, runtime: Any) -> bool:
        from . import qa_face_pose_canonical as bridge_mod

        master_artifact, _ = bridge_mod._load_master_artifact(runtime)
        if master_artifact is not None:
            return False
        settings = _resolve_settings()
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
        )
        master_path = bridge_mod._master_truth_dir(runtime) / bridge_mod._MASTER_ARTIFACT_NAME
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_text(json.dumps(_json_ready(artifact), indent=2, ensure_ascii=False), encoding="utf-8")
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
