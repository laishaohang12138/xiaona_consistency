from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qa_io import atomic_write_json


_MANIFEST_SCHEMA = "artifact_manifest_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_manifest_path(manifest_root: Optional[Path] = None) -> Path:
    root = Path(manifest_root).resolve() if manifest_root is not None else (_repo_root() / "outputs" / "master_truth")
    root.mkdir(parents=True, exist_ok=True)
    return root / "artifact_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(node) for key, node in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(node) for node in value]
    return value


def _sha1_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_meta(path: Optional[Path]) -> Dict[str, Any]:
    resolved = Path(path).resolve() if path is not None else None
    if resolved is None or not resolved.exists():
        return {
            "path": str(resolved) if resolved is not None else "",
            "relpath": _safe_relpath(resolved),
            "exists": False,
            "size_bytes": None,
            "sha1": None,
            "mtime_utc": None,
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "relpath": _safe_relpath(resolved),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "sha1": _sha1_file(resolved),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _safe_relpath(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        return str(path.resolve())


def _detect_storage(artifact_path: Path, artifact_role: str, manifest_root: Optional[Path] = None) -> str:
    resolved = artifact_path.resolve()
    root = Path(manifest_root).resolve() if manifest_root is not None else artifact_manifest_path().parent
    path_text = str(resolved).replace("\\", "/").lower()
    if resolved.parent == root:
        return "master_truth"
    if "/heavy_evidence_cache/" in path_text:
        return "cache"
    if resolved.name.endswith(".body_canonical.json") or resolved.name.endswith(".face_pose_canonical.json"):
        return "sidecar"
    return "master_truth" if artifact_role == "master_truth" else "artifact"


def _git_meta(repo_dir: Optional[Path]) -> Dict[str, Any]:
    if repo_dir is None:
        return {"repo_dir": "", "git_commit": None, "git_dirty": None}
    resolved = Path(repo_dir).resolve()
    meta = {"repo_dir": str(resolved), "git_commit": None, "git_dirty": None}
    if not resolved.exists():
        return meta
    try:
        commit = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            meta["git_commit"] = commit.stdout.strip() or None
        dirty = subprocess.run(
            ["git", "-C", str(resolved), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0:
            meta["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        return meta
    return meta


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": _MANIFEST_SCHEMA,
            "generated_at_utc": _utc_now(),
            "manifest_root": str(path.parent.resolve()),
            "entries": [],
            "summary": {
                "total_entries": 0,
                "family_counts": {},
                "role_counts": {},
                "storage_counts": {},
            },
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": _MANIFEST_SCHEMA,
            "generated_at_utc": _utc_now(),
            "manifest_root": str(path.parent.resolve()),
            "entries": [],
            "summary": {
                "total_entries": 0,
                "family_counts": {},
                "role_counts": {},
                "storage_counts": {},
            },
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": _MANIFEST_SCHEMA,
            "generated_at_utc": _utc_now(),
            "manifest_root": str(path.parent.resolve()),
            "entries": [],
            "summary": {
                "total_entries": 0,
                "family_counts": {},
                "role_counts": {},
                "storage_counts": {},
            },
        }
    payload.setdefault("entries", [])
    return payload


def _rebuild_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    family_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    storage_counts: Dict[str, int] = {}
    for row in entries:
        family = str(row.get("artifact_family") or "").strip()
        role = str(row.get("artifact_role") or "").strip()
        storage = str(row.get("artifact_storage") or "").strip()
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        if storage:
            storage_counts[storage] = storage_counts.get(storage, 0) + 1
    return {
        "total_entries": len(entries),
        "family_counts": dict(sorted(family_counts.items(), key=lambda row: (-row[1], row[0]))),
        "role_counts": dict(sorted(role_counts.items(), key=lambda row: (-row[1], row[0]))),
        "storage_counts": dict(sorted(storage_counts.items(), key=lambda row: (-row[1], row[0]))),
    }


def register_artifact_manifest(
    *,
    artifact_path: Path,
    artifact_family: str,
    artifact_role: str,
    provider_name: str,
    provider_family: str,
    provider_version: str,
    model_id: str,
    schema_version: str,
    manifest_root: Optional[Path] = None,
    source_path: Optional[Path] = None,
    device: Optional[str] = None,
    repo_dir: Optional[Path] = None,
    entrypoint: Optional[str] = None,
    conversion_meta: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest_file = artifact_manifest_path(manifest_root)
    manifest = _load_manifest(manifest_file)
    artifact_meta = _path_meta(Path(artifact_path))
    source_meta = _path_meta(Path(source_path)) if source_path is not None else _path_meta(None)
    storage = _detect_storage(Path(artifact_path), artifact_role, manifest_file.parent)
    entry = {
        "artifact_key": f"{artifact_family}:{artifact_role}:{artifact_meta['relpath']}",
        "artifact_family": str(artifact_family),
        "artifact_role": str(artifact_role),
        "artifact_storage": storage,
        "artifact": artifact_meta,
        "source": source_meta,
        "provider_name": str(provider_name),
        "provider_family": str(provider_family),
        "provider_version": str(provider_version),
        "model_id": str(model_id),
        "schema_version": str(schema_version),
        "device": str(device or "").strip() or None,
        "entrypoint": str(entrypoint or "").strip() or None,
        "repo": _git_meta(repo_dir),
        "conversion_meta": _json_ready(conversion_meta or {}),
        "generated_at_utc": _utc_now(),
    }
    if isinstance(extra, dict):
        entry["extra"] = _json_ready(extra)

    entries = [row for row in list(manifest.get("entries") or []) if isinstance(row, dict)]
    replaced = False
    for index, row in enumerate(entries):
        if str(row.get("artifact_key") or "") == entry["artifact_key"]:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    entries.sort(key=lambda row: str(row.get("artifact_key") or ""))
    manifest["schema_version"] = _MANIFEST_SCHEMA
    manifest["generated_at_utc"] = _utc_now()
    manifest["manifest_root"] = str(manifest_file.parent.resolve())
    manifest["entries"] = entries
    manifest["summary"] = _rebuild_summary(entries)
    atomic_write_json(manifest_file, manifest)
    return entry


def load_artifact_manifest_summary(manifest_root: Optional[Path] = None) -> Dict[str, Any]:
    manifest_file = artifact_manifest_path(manifest_root)
    payload = _load_manifest(manifest_file)
    entries = [row for row in list(payload.get("entries") or []) if isinstance(row, dict)]
    summary = dict(payload.get("summary") or {})
    summary.setdefault("total_entries", len(entries))
    summary["manifest_file"] = str(manifest_file.resolve())
    summary["manifest_exists"] = manifest_file.exists()
    summary["generated_at_utc"] = payload.get("generated_at_utc")
    return summary
