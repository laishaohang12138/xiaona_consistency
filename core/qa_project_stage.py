from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_STAGE_SCHEMA = "project_stage_v1"
PROJECT_STAGE_CONFIG_NAME = "project_stage.json"
PROJECT_STAGE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / PROJECT_STAGE_CONFIG_NAME
).resolve()
KNOWN_PROJECT_PERMISSIONS = (
    "calibrate_quality_thresholds",
    "optuna_parameter_fitting",
    "winner_bank_freeze",
    "allow_preflight_fail_override",
    "allow_gpu_hardware_risk_override",
)


class ProjectStageError(RuntimeError):
    pass


class ProjectStagePermissionError(ProjectStageError):
    pass


def project_stage_config_path() -> Path:
    return PROJECT_STAGE_CONFIG_PATH


def load_project_stage() -> Dict[str, Any]:
    path = project_stage_config_path()
    if not path.is_file():
        raise ProjectStageError(f"project stage config is missing; sensitive actions are denied: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProjectStageError(f"project stage config is unreadable; sensitive actions are denied: {path}") from exc
    if not isinstance(payload, dict):
        raise ProjectStageError(f"project stage config must be a JSON object: {path}")

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != PROJECT_STAGE_SCHEMA:
        raise ProjectStageError(
            f"project stage schema must be {PROJECT_STAGE_SCHEMA!r}, got {schema_version!r}: {path}"
        )
    stage = str(payload.get("stage") or "").strip()
    if not stage:
        raise ProjectStageError(f"project stage name is missing: {path}")

    raw_permissions = payload.get("permissions")
    if not isinstance(raw_permissions, dict):
        raise ProjectStageError(f"project stage permissions must be a JSON object: {path}")
    permissions = {name: False for name in KNOWN_PROJECT_PERMISSIONS}
    for raw_name, value in raw_permissions.items():
        name = str(raw_name).strip()
        if not name:
            raise ProjectStageError(f"project stage contains an empty permission name: {path}")
        if not isinstance(value, bool):
            raise ProjectStageError(f"project stage permission {name!r} must be boolean: {path}")
        permissions[name] = value

    return {
        "schema_version": PROJECT_STAGE_SCHEMA,
        "stage": stage,
        "path": str(path),
        "permissions": permissions,
    }


def require_project_permission(
    permission: str,
    *,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    stage = load_project_stage()
    permission_name = str(permission).strip()
    if not bool(stage["permissions"].get(permission_name, False)):
        action_name = str(action or permission_name).strip()
        raise ProjectStagePermissionError(
            f"project stage {stage['stage']!r} denies {action_name!r} "
            f"(permission {permission_name!r}); an explicit tracked project-stage transition is required"
        )
    return stage
