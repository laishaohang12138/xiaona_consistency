from __future__ import annotations

import os
import re
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .qa_gpu_device_policy import (
    detect_windows_nvidia_whea_risk,
    nvidia_whea_risk_blockers,
)
from .qa_io import WorkflowFileLock, acquire_workflow_lock, atomic_write_json


GPU_EXECUTION_GUARD_SCHEMA = "gpu_execution_guard_v1"
GPU_WHEA_DELTA_SCHEMA = "gpu_whea_delta_v1"


class GpuExecutionGuardError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return token[:80] or fallback


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _risk_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = payload.get("blockers")
    explicit = (
        [str(item) for item in blockers if str(item).strip()]
        if isinstance(blockers, list)
        else []
    )
    return list(dict.fromkeys([*explicit, *nvidia_whea_risk_blockers(payload)]))


def resolve_gpu_resource(policy: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    device = str(policy.get("device") or "auto").strip().lower()
    status = policy.get("gpu_status") if isinstance(policy.get("gpu_status"), dict) else {}
    cuda_selected = device.startswith("cuda") or (
        device == "auto"
        and (
            bool(status.get("gpu_ready_for_torch_models"))
            or bool(status.get("gpu_ready_for_insightface"))
        )
    )
    if not cuda_selected:
        return None

    logical_index = "0"
    if ":" in device:
        candidate = device.split(":", 1)[1].strip()
        if candidate:
            logical_index = candidate
    visible = str(os.getenv("CUDA_VISIBLE_DEVICES") or "").strip()
    physical_selector = logical_index
    if visible and visible != "-1":
        first_visible = visible.split(",", 1)[0].strip()
        if first_visible:
            physical_selector = first_visible
    resource_token = _safe_token(physical_selector, fallback=logical_index)
    return {
        "logical_device_id": f"cuda:{logical_index}",
        "physical_selector": physical_selector,
        "resource_id": f"gpu-{resource_token}",
    }


def build_whea_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    before_state = str(before.get("probe_state") or "UNKNOWN").upper()
    after_state = str(after.get("probe_state") or "UNKNOWN").upper()
    result: Dict[str, Any] = {
        "schema_version": GPU_WHEA_DELTA_SCHEMA,
        "status": "UNASSESSED",
        "same_boot": None,
        "nvidia_whea17_before": _int_or_none(before.get("nvidia_whea17_count_since_boot")),
        "nvidia_whea17_after": _int_or_none(after.get("nvidia_whea17_count_since_boot")),
        "nvidia_whea17_delta": None,
        "new_nvidia_whea_observed": None,
        "blockers": [],
    }
    if before_state == "NOT_APPLICABLE" and after_state == "NOT_APPLICABLE":
        result.update(
            {
                "status": "NOT_APPLICABLE",
                "same_boot": True,
                "new_nvidia_whea_observed": False,
            }
        )
        return result
    if before_state != "OBSERVED" or after_state != "OBSERVED":
        result["blockers"] = ["NVIDIA_WHEA_DELTA_UNASSESSED"]
        return result

    before_boot = str(before.get("boot_id") or before.get("boot_time") or "").strip()
    after_boot = str(after.get("boot_id") or after.get("boot_time") or "").strip()
    if not before_boot or not after_boot:
        result["blockers"] = ["NVIDIA_WHEA_BOOT_ID_UNAVAILABLE"]
        return result
    result["same_boot"] = before_boot == after_boot
    if not result["same_boot"]:
        result["blockers"] = ["NVIDIA_WHEA_BOOT_CHANGED_DURING_RUN"]
        return result

    before_count = result["nvidia_whea17_before"]
    after_count = result["nvidia_whea17_after"]
    if before_count is None or after_count is None or after_count < before_count:
        result["blockers"] = ["NVIDIA_WHEA_DELTA_UNASSESSED"]
        return result
    delta = int(after_count - before_count)
    result.update(
        {
            "status": "NEW_EVENTS" if delta > 0 else "NO_NEW_EVENTS",
            "nvidia_whea17_delta": delta,
            "new_nvidia_whea_observed": delta > 0,
            "blockers": ["NVIDIA_PCIE_WHEA17_DURING_RUN"] if delta > 0 else [],
        }
    )
    return result


@dataclass
class GpuExecutionGuard:
    base_dir: Path
    artifact_dir: Path
    policy: Mapping[str, Any]
    workload_class: str
    allow_hardware_risk: bool = False
    lock_timeout_s: float = 30.0
    lease_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _lock: Optional[WorkflowFileLock] = field(default=None, init=False, repr=False)
    _record: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        resource = resolve_gpu_resource(self.policy)
        if resource is None:
            raise GpuExecutionGuardError("GPU execution guard requires a CUDA-selected policy")
        self.base_dir = Path(self.base_dir).resolve()
        self.artifact_dir = Path(self.artifact_dir).resolve()
        self.resource = resource
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        workload = _safe_token(self.workload_class, fallback="gpu_workload")
        self.artifact_path = self.artifact_dir / f"{timestamp}_{workload}_{self.lease_id[:12]}.json"
        self.lock_path = (
            self.base_dir
            / "outputs"
            / ".gpu_device_locks"
            / f"{self.resource['resource_id']}.lock"
        )

    @property
    def record(self) -> Dict[str, Any]:
        return dict(self._record)

    @property
    def whea_before(self) -> Dict[str, Any]:
        payload = self._record.get("whea_before")
        return dict(payload) if isinstance(payload, dict) else {}

    def _write_record(self) -> None:
        atomic_write_json(self.artifact_path, self._record)

    def __enter__(self) -> "GpuExecutionGuard":
        started_at = _utc_now()
        self._record = {
            "schema_version": GPU_EXECUTION_GUARD_SCHEMA,
            "lease_id": self.lease_id,
            "status": "ACQUIRING",
            "workload_class": self.workload_class,
            "device": dict(self.resource),
            "process": {"pid": os.getpid(), "hostname": socket.gethostname()},
            "started_at_utc": started_at,
            "completed_at_utc": None,
            "lock_path": str(self.lock_path),
            "hardware_risk_override": bool(self.allow_hardware_risk),
            "whea_before": None,
            "whea_after": None,
            "whea_delta": None,
            "qualification_evidence_eligible": False,
            "decision_influence": "EXECUTION_SAFETY_ONLY",
        }
        try:
            self._lock = acquire_workflow_lock(
                self.lock_path,
                owner=f"gpu:{self.workload_class}",
                timeout_s=self.lock_timeout_s,
                metadata={
                    "lease_id": self.lease_id,
                    "device": dict(self.resource),
                    "workload_class": self.workload_class,
                    "artifact_path": str(self.artifact_path),
                },
            )
            before = detect_windows_nvidia_whea_risk()
            self._record["whea_before"] = before
            blockers = _risk_blockers(before)
            if blockers and not self.allow_hardware_risk:
                self._record["status"] = "BLOCKED_BEFORE_RUNTIME"
                self._record["blockers"] = blockers
                self._record["completed_at_utc"] = _utc_now()
                self._write_record()
                raise GpuExecutionGuardError(
                    "CUDA run blocked before runtime initialization: " + ", ".join(blockers)
                )
            self._record["status"] = "ACTIVE_NON_QUALIFYING_OVERRIDE" if blockers else "ACTIVE"
            self._record["blockers"] = blockers
            self._write_record()
            return self
        except GpuExecutionGuardError:
            if self._lock is not None:
                self._lock.release()
                self._lock = None
            raise
        except Exception as exc:
            if self._lock is not None:
                self._lock.release()
                self._lock = None
            self._record["status"] = "GUARD_SETUP_FAILED"
            self._record["completed_at_utc"] = _utc_now()
            self._record["guard_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            try:
                self._write_record()
            except Exception:
                pass
            raise GpuExecutionGuardError(f"GPU execution guard setup failed: {exc}") from exc

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        post_blockers: list[str] = []
        try:
            after = detect_windows_nvidia_whea_risk()
            delta = build_whea_delta(self.whea_before, after)
            post_blockers = [str(item) for item in delta.get("blockers") or []]
            self._record["whea_after"] = after
            self._record["whea_delta"] = delta
            self._record["completed_at_utc"] = _utc_now()
            if exc is not None:
                self._record["status"] = "WORKLOAD_FAILED"
                self._record["workload_error"] = {
                    "type": getattr(exc_type, "__name__", str(exc_type or "Exception")),
                    "message": str(exc),
                }
            elif post_blockers:
                self._record["status"] = "COMPLETED_WITH_HARDWARE_RISK"
            elif self.allow_hardware_risk or _risk_blockers(self.whea_before):
                self._record["status"] = "COMPLETED_NON_QUALIFYING_OVERRIDE"
            else:
                self._record["status"] = "COMPLETED"
            self._record["post_run_blockers"] = post_blockers
            self._record["qualification_evidence_eligible"] = bool(
                exc is None
                and not self.allow_hardware_risk
                and not _risk_blockers(self.whea_before)
                and not post_blockers
            )
            self._write_record()
        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None
        if exc is None and post_blockers:
            raise GpuExecutionGuardError(
                "CUDA run completed but hardware qualification failed: "
                + ", ".join(post_blockers)
            )


def create_gpu_execution_guard(
    *,
    base_dir: Path,
    artifact_dir: Path,
    policy: Optional[Mapping[str, Any]],
    workload_class: str,
    allow_hardware_risk: bool = False,
    lock_timeout_s: float = 30.0,
) -> Optional[GpuExecutionGuard]:
    if not isinstance(policy, Mapping) or resolve_gpu_resource(policy) is None:
        return None
    return GpuExecutionGuard(
        base_dir=base_dir,
        artifact_dir=artifact_dir,
        policy=policy,
        workload_class=workload_class,
        allow_hardware_risk=allow_hardware_risk,
        lock_timeout_s=lock_timeout_s,
    )
