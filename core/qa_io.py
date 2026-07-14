from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes by replacing the target only after the temp file is durable."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text by replacing the target only after the temp file is complete."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json_dict(path: Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class WorkflowFileLock:
    path: Path
    owner: str = ""
    timeout_s: float = 1800.0
    poll_s: float = 0.25
    stale_after_s: float = 21600.0
    _fd: Optional[int] = None

    def acquire(self) -> "WorkflowFileLock":
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + max(0.0, float(self.timeout_s))
        while True:
            try:
                self._fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = {
                    "pid": os.getpid(),
                    "owner": self.owner,
                    "created_at_epoch": time.time(),
                }
                os.write(self._fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                return self
            except FileExistsError:
                self._remove_stale_lock(target)
                if time.time() >= deadline:
                    raise TimeoutError(f"Workflow lock is busy: {target}")
                time.sleep(max(0.05, float(self.poll_s)))

    def _remove_stale_lock(self, target: Path) -> None:
        try:
            age_s = time.time() - target.stat().st_mtime
        except OSError:
            return
        if age_s < max(1.0, float(self.stale_after_s)):
            return
        try:
            target.unlink()
        except OSError:
            pass

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            Path(self.path).unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "WorkflowFileLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


def acquire_workflow_lock(path: Path, *, owner: str, timeout_s: float = 1800.0) -> WorkflowFileLock:
    return WorkflowFileLock(path=path, owner=owner, timeout_s=timeout_s).acquire()
