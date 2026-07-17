from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


TRUTH_FUSION_GPU_POLICY_SCHEMA = "truth_fusion_gpu_device_policy_v1"
WINDOWS_NVIDIA_WHEA_RISK_SCHEMA = "windows_nvidia_whea_risk_v0_1"
GPU_RUNTIME_SAFETY_PREFLIGHT_SCHEMA = "gpu_runtime_safety_preflight_v1"
GPU_RUNTIME_POLICY_SCHEMA = "gpu_runtime_policy_v1"
NVIDIA_WHEA_GATE_EVALUATION_SCHEMA = "nvidia_whea_gate_evaluation_v1"
GPU_RUNTIME_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "gpu_runtime_policy.json"
).resolve()
DEFAULT_GPU_DEVICE = "cuda"
DEFAULT_SURFACE_AUTO_EXPORT = "densepose,sam2"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_env_default(key: str, value: str, *, force: bool) -> Dict[str, str]:
    previous = os.environ.get(key)
    if force or previous is None or str(previous).strip() == "":
        os.environ[key] = str(value)
        source = "forced" if force and previous not in {None, value} else "set"
    else:
        source = "kept"
    return {"key": key, "value": os.environ.get(key, ""), "previous": previous or "", "source": source}


def load_gpu_runtime_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    policy_path = Path(path or GPU_RUNTIME_POLICY_PATH).resolve()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"GPU runtime policy is unreadable: {policy_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"GPU runtime policy must be a JSON object: {policy_path}")
    if str(payload.get("schema_version") or "").strip() != GPU_RUNTIME_POLICY_SCHEMA:
        raise ValueError(
            f"GPU runtime policy schema must be {GPU_RUNTIME_POLICY_SCHEMA!r}: {policy_path}"
        )
    preference = str(payload.get("execution_preference") or "").strip().upper()
    if preference not in {"GPU_FIRST", "AUTO", "CPU_FIRST"}:
        raise ValueError(f"GPU runtime execution_preference is invalid: {preference!r}")
    whea_gate = payload.get("whea_gate")
    if not isinstance(whea_gate, dict):
        raise ValueError("GPU runtime policy requires a whea_gate object")
    mode = str(whea_gate.get("mode") or "").strip().upper()
    if mode not in {"BLOCK_ANY_SINCE_BOOT", "ACKNOWLEDGED_EVENT_WATERMARK"}:
        raise ValueError(f"GPU runtime WHEA mode is invalid: {mode!r}")
    return {**payload, "path": str(policy_path)}


def configured_default_device(policy: Optional[Mapping[str, Any]] = None) -> str:
    try:
        payload = dict(policy) if isinstance(policy, Mapping) else load_gpu_runtime_policy()
        preference = str(payload.get("execution_preference") or "GPU_FIRST").strip().upper()
    except Exception:
        return DEFAULT_GPU_DEVICE
    if preference == "CPU_FIRST":
        return "cpu"
    if preference == "AUTO":
        return "auto"
    return "cuda"


def _aware_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def evaluate_nvidia_whea_gate(
    status: Mapping[str, Any],
    policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    probe_state = str(status.get("probe_state") or "UNKNOWN").upper()
    result: Dict[str, Any] = {
        "schema_version": NVIDIA_WHEA_GATE_EVALUATION_SCHEMA,
        "status": "FAIL",
        "disposition": "UNASSESSED",
        "acknowledgement_applied": False,
        "raw_nvidia_whea17_count_since_boot": status.get("nvidia_whea17_count_since_boot"),
        "raw_latest_nvidia_whea_time": status.get("latest_nvidia_whea_time"),
        "blockers": [],
    }
    if probe_state == "NOT_APPLICABLE":
        result.update({"status": "NOT_APPLICABLE", "disposition": "PLATFORM_NOT_APPLICABLE"})
        return result
    if probe_state != "OBSERVED":
        result["blockers"] = ["NVIDIA_WHEA_RISK_UNASSESSED"]
        return result
    try:
        event_count = int(status.get("nvidia_whea17_count_since_boot") or 0)
    except (TypeError, ValueError):
        result["blockers"] = ["NVIDIA_WHEA_RISK_UNASSESSED"]
        return result
    if event_count <= 0:
        result.update({"status": "PASS", "disposition": "NO_NVIDIA_WHEA_OBSERVED"})
        return result

    try:
        policy_payload = dict(policy) if isinstance(policy, Mapping) else load_gpu_runtime_policy()
    except Exception as exc:
        result["disposition"] = "POLICY_UNAVAILABLE"
        result["policy_error"] = f"{type(exc).__name__}:{exc}"
        result["blockers"] = ["GPU_RUNTIME_POLICY_UNAVAILABLE"]
        return result
    whea_policy = (
        policy_payload.get("whea_gate")
        if isinstance(policy_payload.get("whea_gate"), Mapping)
        else {}
    )
    mode = str(whea_policy.get("mode") or "BLOCK_ANY_SINCE_BOOT").strip().upper()
    result["policy"] = {
        "schema_version": policy_payload.get("schema_version"),
        "path": policy_payload.get("path"),
        "mode": mode,
        "reason_code": whea_policy.get("reason_code"),
        "acknowledged_by": whea_policy.get("acknowledged_by"),
        "acknowledged_at_utc": whea_policy.get("acknowledged_at_utc"),
        "acknowledged_boot_id": whea_policy.get("acknowledged_boot_id"),
        "acknowledged_event_count_since_boot": whea_policy.get(
            "acknowledged_event_count_since_boot"
        ),
        "acknowledged_through": whea_policy.get("acknowledged_through"),
    }
    if mode != "ACKNOWLEDGED_EVENT_WATERMARK":
        result["disposition"] = "UNACKNOWLEDGED_EVENTS_SINCE_BOOT"
        result["blockers"] = ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"]
        return result

    latest = _aware_datetime(status.get("latest_nvidia_whea_time"))
    watermark = _aware_datetime(whea_policy.get("acknowledged_through"))
    if latest is None or watermark is None:
        result["disposition"] = "ACKNOWLEDGEMENT_TIMESTAMP_UNAVAILABLE"
        result["blockers"] = ["NVIDIA_WHEA_ACKNOWLEDGEMENT_UNUSABLE"]
        return result

    same_acknowledged_boot = (
        bool(str(status.get("boot_id") or "").strip())
        and str(status.get("boot_id")).strip()
        == str(whea_policy.get("acknowledged_boot_id") or "").strip()
    )
    acknowledged_count: Optional[int]
    try:
        acknowledged_count = int(whea_policy.get("acknowledged_event_count_since_boot"))
    except (TypeError, ValueError):
        acknowledged_count = None
    count_advanced = bool(
        same_acknowledged_boot
        and acknowledged_count is not None
        and event_count > acknowledged_count
    )
    timestamp_advanced = latest > watermark
    result["same_acknowledged_boot"] = same_acknowledged_boot
    result["event_count_advanced"] = count_advanced
    result["event_timestamp_advanced"] = timestamp_advanced
    if count_advanced or timestamp_advanced:
        result["disposition"] = "NEW_EVENTS_AFTER_ACKNOWLEDGED_BASELINE"
        result["blockers"] = ["NVIDIA_PCIE_WHEA17_AFTER_ACKNOWLEDGED_BASELINE"]
        return result

    result.update(
        {
            "status": "PASS",
            "disposition": "ACKNOWLEDGED_HISTORICAL_EVENTS",
            "acknowledgement_applied": True,
        }
    )
    return result


def detect_gpu_status() -> Dict[str, Any]:
    torch_status: Dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
        "reason": "",
    }
    if importlib.util.find_spec("torch") is None:
        torch_status["reason"] = "torch_module_missing"
    else:
        try:
            import torch  # type: ignore

            torch_status["available"] = True
            torch_status["cuda_available"] = bool(torch.cuda.is_available())
            torch_status["device_count"] = int(torch.cuda.device_count()) if torch_status["cuda_available"] else 0
            if torch_status["cuda_available"]:
                torch_status["device_names"] = [
                    str(torch.cuda.get_device_name(index)) for index in range(int(torch_status["device_count"]))
                ]
        except Exception as exc:  # pragma: no cover - environment dependent
            torch_status["reason"] = f"torch_probe_failed:{exc}"

    ort_status: Dict[str, Any] = {
        "available": False,
        "cuda_execution_provider": False,
        "providers": [],
        "reason": "",
    }
    if importlib.util.find_spec("onnxruntime") is None:
        ort_status["reason"] = "onnxruntime_module_missing"
    else:
        try:
            import onnxruntime as ort  # type: ignore

            providers = [str(item) for item in ort.get_available_providers()]
            ort_status["available"] = True
            ort_status["providers"] = providers
            ort_status["cuda_execution_provider"] = "CUDAExecutionProvider" in providers
        except Exception as exc:  # pragma: no cover - environment dependent
            ort_status["reason"] = f"onnxruntime_probe_failed:{exc}"

    return {
        "schema_version": TRUTH_FUSION_GPU_POLICY_SCHEMA,
        "torch": torch_status,
        "onnxruntime": ort_status,
        "gpu_ready_for_torch_models": bool(torch_status.get("cuda_available")),
        "gpu_ready_for_insightface": bool(ort_status.get("cuda_execution_provider")),
    }


def gpu_requirement_blockers(status: Mapping[str, Any]) -> List[str]:
    blockers: List[str] = []
    if not bool(status.get("gpu_ready_for_torch_models")):
        blockers.append("TORCH_CUDA_UNAVAILABLE")
    if not bool(status.get("gpu_ready_for_insightface")):
        blockers.append("ONNXRUNTIME_CUDA_EXECUTION_PROVIDER_UNAVAILABLE")
    return blockers


def detect_windows_nvidia_whea_risk() -> Dict[str, Any]:
    if os.name != "nt":
        return {
            "schema_version": WINDOWS_NVIDIA_WHEA_RISK_SCHEMA,
            "probe_state": "NOT_APPLICABLE",
            "risk_state": "UNASSESSED",
            "nvidia_whea17_count_since_boot": None,
            "blockers": [],
        }
    script = r"""
$ErrorActionPreference = 'Stop'
try {
  $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
  $bootSource = 'win32_operating_system'
} catch {
  $bootEvent = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-General'; Id=12} -MaxEvents 1 -ErrorAction Stop
  if ($null -eq $bootEvent -or $null -eq $bootEvent.TimeCreated) {
    throw 'Windows boot time could not be established from Kernel-General event 12'
  }
  $boot = $bootEvent.TimeCreated
  $bootSource = 'kernel_general_event_12'
}
$eventErrors = @()
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; Id=17; StartTime=$boot} -ErrorAction SilentlyContinue -ErrorVariable +eventErrors)
$fatalEventErrors = @($eventErrors | Where-Object {
  $_.FullyQualifiedErrorId -notmatch 'NoMatchingEventsFound' -and
  $_.Exception.Message -notmatch 'No events were found that match the specified selection criteria'
})
if ($fatalEventErrors.Count -gt 0) {
  throw "Get-WinEvent WHEA query failed: $($fatalEventErrors[0].Exception.Message)"
}
$nvidia = @($events | Where-Object { $_.Message -match 'VEN_10DE' })
$latest = $nvidia | Sort-Object TimeCreated -Descending | Select-Object -First 1
[ordered]@{
  probe_state = 'OBSERVED'
  risk_state = $(if ($nvidia.Count -gt 0) { 'NVIDIA_PCIE_WHEA_OBSERVED' } else { 'NO_NVIDIA_WHEA_OBSERVED' })
  boot_id = $boot.ToUniversalTime().ToString('o')
  boot_time = $boot.ToString('o')
  boot_time_source = $bootSource
  event_query_state = $(if ($eventErrors.Count -gt 0) { 'NO_MATCHING_EVENTS' } else { 'COMPLETE' })
  whea17_count_since_boot = $events.Count
  nvidia_whea17_count_since_boot = $nvidia.Count
  latest_nvidia_whea_time = $(if ($null -ne $latest) { $latest.TimeCreated.ToString('o') } else { $null })
} | ConvertTo-Json -Compress
"""
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "powershell probe failed").strip())
        payload = json.loads(completed.stdout.strip())
        if not isinstance(payload, dict):
            raise ValueError("PowerShell WHEA probe did not return an object")
        result = {"schema_version": WINDOWS_NVIDIA_WHEA_RISK_SCHEMA, **payload}
    except Exception as exc:
        result = {
            "schema_version": WINDOWS_NVIDIA_WHEA_RISK_SCHEMA,
            "probe_state": "FAILED",
            "risk_state": "UNASSESSED",
            "nvidia_whea17_count_since_boot": None,
            "error": f"{type(exc).__name__}:{exc}",
        }
    gate_evaluation = evaluate_nvidia_whea_gate(result)
    result["gate_evaluation"] = gate_evaluation
    result["blockers"] = list(gate_evaluation.get("blockers") or [])
    return result


def nvidia_whea_risk_blockers(status: Mapping[str, Any]) -> List[str]:
    gate_evaluation = status.get("gate_evaluation")
    if isinstance(gate_evaluation, Mapping):
        blockers = gate_evaluation.get("blockers")
        return (
            [str(item) for item in blockers if str(item).strip()]
            if isinstance(blockers, list)
            else ["NVIDIA_WHEA_RISK_UNASSESSED"]
        )
    probe_state = str(status.get("probe_state") or "UNKNOWN").upper()
    if probe_state == "NOT_APPLICABLE":
        return []
    if probe_state != "OBSERVED":
        return ["NVIDIA_WHEA_RISK_UNASSESSED"]
    try:
        nvidia_count = int(status.get("nvidia_whea17_count_since_boot") or 0)
    except (TypeError, ValueError):
        return ["NVIDIA_WHEA_RISK_UNASSESSED"]
    return ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"] if nvidia_count > 0 else []


def build_gpu_runtime_safety_preflight(
    risk_status: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    risk = dict(risk_status) if isinstance(risk_status, Mapping) else detect_windows_nvidia_whea_risk()
    gate_evaluation = (
        dict(risk.get("gate_evaluation"))
        if isinstance(risk.get("gate_evaluation"), Mapping)
        else evaluate_nvidia_whea_gate(risk)
    )
    risk["gate_evaluation"] = gate_evaluation
    blockers = [str(item) for item in gate_evaluation.get("blockers") or [] if str(item).strip()]
    probe_state = str(risk.get("probe_state") or "UNKNOWN").upper()
    if probe_state == "NOT_APPLICABLE":
        gate_status = "NOT_APPLICABLE"
        allowed = True
        next_action = "CONTINUE_WITH_PLATFORM_SPECIFIC_GPU_SAFETY_CHECKS"
    elif blockers:
        gate_status = "FAIL"
        allowed = False
        next_action = (
            "DIAGNOSE_NVIDIA_PCIE_PATH_AND_REBOOT_BEFORE_GPU_QA"
            if "NVIDIA_PCIE_WHEA17_SINCE_BOOT" in blockers
            else "RESTORE_WHEA_OBSERVABILITY_BEFORE_GPU_QA"
        )
    elif probe_state == "OBSERVED":
        gate_status = "PASS"
        allowed = True
        next_action = "PROCEED_TO_BATCH_PREFLIGHT_DEVICE_GUARD_WILL_RECHECK"
    else:
        gate_status = "FAIL"
        allowed = False
        blockers = list(dict.fromkeys([*blockers, "NVIDIA_WHEA_RISK_UNASSESSED"]))
        next_action = "RESTORE_WHEA_OBSERVABILITY_BEFORE_GPU_QA"

    return {
        "schema_version": GPU_RUNTIME_SAFETY_PREFLIGHT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate_status,
        "gpu_runtime_allowed_by_whea_gate": allowed,
        "runtime_or_provider_initialized_by_this_preflight": False,
        "risk_observation": risk,
        "gate_evaluation": gate_evaluation,
        "blockers": blockers,
        "next_action": next_action,
        "evidence_scope": "Windows WHEA-Logger event 17 entries linked to NVIDIA VEN_10DE since the current boot",
        "does_not_certify": [
            "full GPU stability",
            "VRAM integrity",
            "thermal or power stability",
            "driver correctness",
            "PCIe link stability under load",
        ],
        "decision_influence": "EXECUTION_SAFETY_ONLY",
    }


def apply_truth_fusion_gpu_env(
    *,
    device: str = DEFAULT_GPU_DEVICE,
    require_gpu: bool = False,
    surface_auto_export: str = DEFAULT_SURFACE_AUTO_EXPORT,
    force: bool = False,
) -> Dict[str, Any]:
    normalized_device = str(device or DEFAULT_GPU_DEVICE).strip().lower()
    if normalized_device not in {"auto", "cuda", "cpu"}:
        normalized_device = DEFAULT_GPU_DEVICE
    surface_auto = str(surface_auto_export or "").strip().lower()
    if surface_auto in {"off", "none", "false", "0"}:
        surface_auto = ""

    env_defaults = {
        "XIAONA_DEVICE_POLICY": normalized_device,
        "XIAONA_INSIGHTFACE_DEVICE": normalized_device,
        "XIAONA_HUMAN_PARSING_DEVICE": normalized_device,
        "XIAONA_SEGFORMER_DEVICE": normalized_device,
        "XIAONA_HMR2_DEVICE": normalized_device,
        "XIAONA_3DDFA_V3_DEVICE": normalized_device,
        "XIAONA_SURFACE_OCCLUSION_DEVICE": normalized_device,
        "XIAONA_SAM2_DEVICE": normalized_device,
        "XIAONA_DENSEPOSE_DEVICE": normalized_device,
    }
    if surface_auto or force:
        env_defaults["XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT"] = surface_auto
    if require_gpu:
        env_defaults["XIAONA_REQUIRE_GPU"] = "1"
    elif "XIAONA_REQUIRE_GPU" not in os.environ:
        env_defaults["XIAONA_REQUIRE_GPU"] = "0"
    env_defaults["XIAONA_TRUTH_FUSION_GPU_STACK"] = "1"

    applied = [_set_env_default(key, value, force=force) for key, value in env_defaults.items()]
    status = detect_gpu_status()
    return {
        "schema_version": TRUTH_FUSION_GPU_POLICY_SCHEMA,
        "device": normalized_device,
        "configured_execution_preference": configured_default_device(),
        "require_gpu": bool(require_gpu or _truthy(os.environ.get("XIAONA_REQUIRE_GPU"))),
        "surface_auto_export": os.environ.get("XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT", ""),
        "env": applied,
        "gpu_status": status,
        "requirement_blockers": gpu_requirement_blockers(status)
        if normalized_device == "cuda" and bool(require_gpu)
        else [],
    }
