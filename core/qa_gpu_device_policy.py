from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from typing import Any, Dict, List, Mapping


TRUTH_FUSION_GPU_POLICY_SCHEMA = "truth_fusion_gpu_device_policy_v1"
WINDOWS_NVIDIA_WHEA_RISK_SCHEMA = "windows_nvidia_whea_risk_v0_1"
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
$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; Id=17; StartTime=$boot} -ErrorAction SilentlyContinue)
$nvidia = @($events | Where-Object { $_.Message -match 'VEN_10DE' })
$latest = $nvidia | Sort-Object TimeCreated -Descending | Select-Object -First 1
[ordered]@{
  probe_state = 'OBSERVED'
  risk_state = $(if ($nvidia.Count -gt 0) { 'NVIDIA_PCIE_WHEA_OBSERVED' } else { 'NO_NVIDIA_WHEA_OBSERVED' })
  boot_time = $boot.ToString('o')
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
    result["blockers"] = nvidia_whea_risk_blockers(result)
    return result


def nvidia_whea_risk_blockers(status: Mapping[str, Any]) -> List[str]:
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
        "require_gpu": bool(require_gpu or _truthy(os.environ.get("XIAONA_REQUIRE_GPU"))),
        "surface_auto_export": os.environ.get("XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT", ""),
        "env": applied,
        "gpu_status": status,
        "requirement_blockers": gpu_requirement_blockers(status)
        if normalized_device == "cuda" and bool(require_gpu)
        else [],
    }
