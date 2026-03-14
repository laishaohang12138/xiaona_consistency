from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qa_benchmark import (
    DEFAULT_BENCHMARK_FROZEN_ROLE,
    DEFAULT_BENCHMARK_LABEL_ROLE,
    benchmark_report,
    load_benchmark_label_bundle,
)
from .qa_pipeline import print_runtime_config
from .qa_runtime import EngineState, RuntimeContext, anchor_registry_snapshot, create_runtime_config


OPTUNA_SEARCH_SPACE_SCHEMA = "qa_optuna_search_space_v1"
OPTUNA_GUARD_SCHEMA = "qa_optuna_guard_v1"
OPTUNA_PRESETS_SCHEMA = "qa_optuna_mode_presets_v1"
SUPPORTED_PARAM_TYPES = {"float", "int", "categorical"}
SUPPORTED_SAMPLERS = {"tpe", "random"}
SUPPORTED_DIRECTIONS = {"maximize", "minimize"}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_nested_value(root: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part.strip() for part in str(path).split(".") if str(part).strip()]
    if len(parts) == 0:
        raise ValueError("override path must not be empty")
    node = root
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        elif not isinstance(child, dict):
            raise ValueError(f"override path {path!r} collides with a non-object node at {part!r}")
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def _get_nested_value(root: Dict[str, Any], path: str) -> Any:
    node: Any = root
    for part in [part.strip() for part in str(path).split(".") if str(part).strip()]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _coerce_bool(node: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = node.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _read_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must decode to an object: {path}")
    return payload


def _normalize_text_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    out: List[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _normalize_objective_path(raw_value: Any) -> str:
    path = str(raw_value or "metrics.release_safety_score").strip()
    if not path:
        raise ValueError("objective metric_path must not be empty")
    if "." not in path:
        return f"metrics.{path}"
    return path


def _normalize_param_spec(index: int, node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"search_space.parameters[{index}] must be an object")

    name = str(node.get("name", "")).strip()
    path = str(node.get("path", "")).strip()
    param_type = str(node.get("type", "")).strip().lower()
    if not name:
        raise ValueError(f"search_space.parameters[{index}] is missing name")
    if not path:
        raise ValueError(f"search_space.parameters[{index}] is missing path")
    if param_type not in SUPPORTED_PARAM_TYPES:
        raise ValueError(
            f"search_space.parameters[{index}] type must be one of {sorted(SUPPORTED_PARAM_TYPES)}"
        )

    spec: Dict[str, Any] = {
        "name": name,
        "path": path,
        "type": param_type,
    }
    if param_type == "categorical":
        choices = node.get("choices", None)
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f"search_space.parameters[{index}] categorical choices must be a non-empty list")
        spec["choices"] = copy.deepcopy(choices)
        return spec

    if node.get("low", None) is None or node.get("high", None) is None:
        raise ValueError(f"search_space.parameters[{index}] must define low and high")

    low = node["low"]
    high = node["high"]
    try:
        if param_type == "int":
            spec["low"] = int(low)
            spec["high"] = int(high)
            spec["step"] = int(node.get("step", 1))
        else:
            spec["low"] = float(low)
            spec["high"] = float(high)
            if node.get("step", None) is not None:
                spec["step"] = float(node["step"])
            spec["log"] = _coerce_bool(node, "log", default=False)
    except Exception as exc:
        raise ValueError(f"search_space.parameters[{index}] has invalid numeric bounds") from exc

    if spec["low"] > spec["high"]:
        raise ValueError(f"search_space.parameters[{index}] low must be <= high")
    if param_type == "int" and spec["step"] <= 0:
        raise ValueError(f"search_space.parameters[{index}] int step must be > 0")
    if param_type == "float":
        if "step" in spec and spec["step"] <= 0:
            raise ValueError(f"search_space.parameters[{index}] float step must be > 0")
        if spec.get("log", False) and "step" in spec:
            raise ValueError(f"search_space.parameters[{index}] float params cannot use log=true with step")
    return spec


def _normalize_constraint_spec(index: int, node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"search_space.constraints[{index}] must be an object")
    kind = str(node.get("type", node.get("kind", "order"))).strip().lower()
    if kind != "order":
        raise ValueError("only search_space constraint type 'order' is currently supported")
    lower_path = str(node.get("lower_path", "")).strip()
    upper_path = str(node.get("upper_path", "")).strip()
    if not lower_path or not upper_path:
        raise ValueError(f"search_space.constraints[{index}] must define lower_path and upper_path")
    min_gap = float(node.get("min_gap", 0.0))
    return {
        "type": "order",
        "lower_path": lower_path,
        "upper_path": upper_path,
        "min_gap": min_gap,
    }


def _default_anchor_requirements() -> List[Dict[str, Any]]:
    return [
        {
            "role": "FACE_MASTER",
            "view_buckets": ["front", "three_quarter", "profile_like"],
            "min_count": 1,
        },
        {
            "role": "UPPER_SUPPORT",
            "view_buckets": ["front", "three_quarter", "side_90"],
            "min_count": 1,
        },
        {
            "role": "FULL_BODY_MASTER",
            "view_buckets": ["front", "side_90", "back_180"],
            "min_count": 1,
        },
    ]


def _normalize_guard_requirement(index: int, node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"optuna_guard.anchor_requirements[{index}] must be an object")
    role = str(node.get("role", "")).strip().upper()
    if not role:
        raise ValueError(f"optuna_guard.anchor_requirements[{index}] is missing role")
    raw_buckets = node.get("view_buckets", [])
    if not isinstance(raw_buckets, list) or len(raw_buckets) == 0:
        raise ValueError(f"optuna_guard.anchor_requirements[{index}] must define a non-empty view_buckets list")
    view_buckets = [str(bucket).strip() for bucket in raw_buckets if str(bucket).strip()]
    if len(view_buckets) == 0:
        raise ValueError(f"optuna_guard.anchor_requirements[{index}] has no valid view_buckets")
    min_count = int(node.get("min_count", 1))
    if min_count <= 0:
        raise ValueError(f"optuna_guard.anchor_requirements[{index}] min_count must be > 0")
    return {
        "role": role,
        "view_buckets": view_buckets,
        "min_count": min_count,
    }


def _resolve_project_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_preset_spec(name: str, node: Any) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"optuna preset {name!r} must be an object")
    fit_enabled = _coerce_bool(node, "fit_enabled", default=False)
    guard_path = str(node.get("guard_path", "")).strip()
    if not guard_path:
        raise ValueError(f"optuna preset {name!r} must define guard_path")
    search_space_path = str(node.get("search_space_path", "")).strip()
    if fit_enabled and not search_space_path:
        raise ValueError(f"optuna preset {name!r} must define search_space_path when fit_enabled=true")
    input_collection = node.get("recommended_input_collection", {})
    if input_collection is None:
        input_collection = {}
    if not isinstance(input_collection, dict):
        raise ValueError(f"optuna preset {name!r} recommended_input_collection must be an object")
    return {
        "label": str(node.get("label", name)).strip() or name,
        "description": str(node.get("description", "")).strip(),
        "fit_enabled": fit_enabled,
        "search_space_path": search_space_path,
        "guard_path": guard_path,
        "recommended_dataset_role": str(node.get("recommended_dataset_role", "")).strip(),
        "recommended_input_collection": {
            "include": _normalize_text_list(
                input_collection.get("include", []),
                f"optuna preset {name!r} recommended_input_collection.include",
            ),
            "exclude": _normalize_text_list(
                input_collection.get("exclude", []),
                f"optuna preset {name!r} recommended_input_collection.exclude",
            ),
        },
    }


def load_optuna_mode_presets(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    payload = _read_json_object(path)
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != OPTUNA_PRESETS_SCHEMA:
        raise ValueError(
            f"Optuna presets schema_version must be {OPTUNA_PRESETS_SCHEMA!r}, got {schema_version!r}"
        )
    presets_node = payload.get("presets", {})
    if not isinstance(presets_node, dict) or len(presets_node) == 0:
        raise ValueError("optuna presets file must define a non-empty presets object")
    presets: Dict[str, Dict[str, Any]] = {}
    for raw_name, node in presets_node.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("optuna presets file contains an empty preset key")
        presets[name] = _normalize_preset_spec(name, node)
    default_preset = str(payload.get("default_preset", "")).strip()
    if default_preset and default_preset not in presets:
        raise ValueError(f"optuna presets default_preset {default_preset!r} is not defined")
    return {
        "schema_version": OPTUNA_PRESETS_SCHEMA,
        "path": str(path),
        "default_preset": default_preset,
        "presets": presets,
    }


def resolve_optuna_mode_preset(
    *,
    base_dir: Path,
    preset_name: str,
    presets_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_presets_path = (
        presets_path.resolve()
        if presets_path is not None
        else (base_dir / "configs" / "optuna_mode_presets.json").resolve()
    )
    bundle = load_optuna_mode_presets(resolved_presets_path)
    name = str(preset_name).strip() or str(bundle.get("default_preset", "")).strip()
    presets = bundle["presets"]
    if name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown optuna preset {name!r}. Available presets: {available}")
    preset = copy.deepcopy(presets[name])
    preset["name"] = name
    preset["presets_file"] = str(resolved_presets_path)
    preset["guard_path"] = str(_resolve_project_path(base_dir, preset["guard_path"]))
    if preset.get("search_space_path", ""):
        preset["search_space_path"] = str(_resolve_project_path(base_dir, preset["search_space_path"]))
    return preset


def list_optuna_mode_presets(
    *,
    base_dir: Path,
    presets_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_presets_path = (
        presets_path.resolve()
        if presets_path is not None
        else (base_dir / "configs" / "optuna_mode_presets.json").resolve()
    )
    bundle = load_optuna_mode_presets(resolved_presets_path)
    preset_list: List[Dict[str, Any]] = []
    for name in sorted(bundle["presets"].keys()):
        preset_list.append(
            resolve_optuna_mode_preset(
                base_dir=base_dir,
                preset_name=name,
                presets_path=resolved_presets_path,
            )
        )
    return {
        "schema_version": OPTUNA_PRESETS_SCHEMA,
        "presets_file": str(resolved_presets_path),
        "default_preset": str(bundle.get("default_preset", "")),
        "presets": preset_list,
    }


def load_optuna_guard(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        return {
            "schema_version": OPTUNA_GUARD_SCHEMA,
            "path": str(path),
            "optuna_locked": True,
            "lock_reason": f"Optuna guard file is missing: {path}",
            "required_label_role": DEFAULT_BENCHMARK_FROZEN_ROLE,
            "require_optuna_ready": True,
            "anchor_requirements": _default_anchor_requirements(),
        }

    payload = _read_json_object(path)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != OPTUNA_GUARD_SCHEMA:
        raise ValueError(
            f"Optuna guard schema_version must be {OPTUNA_GUARD_SCHEMA!r}, got {schema_version!r}"
        )

    requirements_node = payload.get("anchor_requirements", _default_anchor_requirements())
    if requirements_node is None:
        requirements_node = []
    if not isinstance(requirements_node, list):
        raise ValueError("optuna_guard.anchor_requirements must be a list")

    return {
        "schema_version": OPTUNA_GUARD_SCHEMA,
        "path": str(path),
        "optuna_locked": _coerce_bool(payload, "optuna_locked", default=True),
        "lock_reason": str(payload.get("lock_reason", "")).strip(),
        "required_label_role": str(payload.get("required_label_role", DEFAULT_BENCHMARK_FROZEN_ROLE)).strip()
        or DEFAULT_BENCHMARK_FROZEN_ROLE,
        "require_optuna_ready": _coerce_bool(payload, "require_optuna_ready", default=True),
        "anchor_requirements": [
            _normalize_guard_requirement(index, node)
            for index, node in enumerate(requirements_node)
        ],
    }


def load_optuna_search_space(path: Path) -> Dict[str, Any]:
    payload = _read_json_object(path)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != OPTUNA_SEARCH_SPACE_SCHEMA:
        raise ValueError(
            f"Optuna search space schema_version must be {OPTUNA_SEARCH_SPACE_SCHEMA!r}, got {schema_version!r}"
        )

    objective_node = payload.get("objective", {})
    if objective_node is None:
        objective_node = {}
    if not isinstance(objective_node, dict):
        raise ValueError("search_space.objective must be an object")
    metric_path = _normalize_objective_path(
        objective_node.get("metric_path", objective_node.get("metric", "metrics.release_safety_score"))
    )
    direction = str(objective_node.get("direction", "maximize")).strip().lower()
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"search_space.objective.direction must be one of {sorted(SUPPORTED_DIRECTIONS)}")

    study_node = payload.get("study", {})
    if study_node is None:
        study_node = {}
    if not isinstance(study_node, dict):
        raise ValueError("search_space.study must be an object")
    sampler = str(study_node.get("sampler", "tpe")).strip().lower()
    if sampler not in SUPPORTED_SAMPLERS:
        raise ValueError(f"search_space.study.sampler must be one of {sorted(SUPPORTED_SAMPLERS)}")

    n_trials = int(study_node.get("n_trials", 40))
    if n_trials <= 0:
        raise ValueError("search_space.study.n_trials must be > 0")

    seed_raw = study_node.get("seed", None)
    seed = None if seed_raw is None else int(seed_raw)
    timeout_sec_raw = study_node.get("timeout_sec", None)
    timeout_sec = None if timeout_sec_raw is None else int(timeout_sec_raw)
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("search_space.study.timeout_sec must be > 0 when provided")

    fixed_override = payload.get("fixed_override", {})
    if fixed_override is None:
        fixed_override = {}
    if not isinstance(fixed_override, dict):
        raise ValueError("search_space.fixed_override must be an object")

    params_node = payload.get("parameters", None)
    if not isinstance(params_node, list) or len(params_node) == 0:
        raise ValueError("search_space.parameters must be a non-empty list")
    parameters = [_normalize_param_spec(index, node) for index, node in enumerate(params_node)]

    names = [node["name"] for node in parameters]
    if len(set(names)) != len(names):
        raise ValueError("search_space.parameters names must be unique")

    constraints_node = payload.get("constraints", [])
    if constraints_node is None:
        constraints_node = []
    if not isinstance(constraints_node, list):
        raise ValueError("search_space.constraints must be a list")
    constraints = [_normalize_constraint_spec(index, node) for index, node in enumerate(constraints_node)]

    produced_paths = {node["path"] for node in parameters}

    def _path_available(path_text: str) -> bool:
        if path_text in produced_paths:
            return True
        try:
            _get_nested_value(fixed_override, path_text)
            return True
        except KeyError:
            return False

    for constraint in constraints:
        for key in ("lower_path", "upper_path"):
            if not _path_available(constraint[key]):
                raise ValueError(
                    f"search_space constraint path {constraint[key]!r} must be provided by parameters or fixed_override"
                )

    return {
        "schema_version": OPTUNA_SEARCH_SPACE_SCHEMA,
        "objective": {
            "metric_path": metric_path,
            "direction": direction,
        },
        "study": {
            "name": str(study_node.get("name", "")).strip(),
            "sampler": sampler,
            "seed": seed,
            "n_trials": n_trials,
            "timeout_sec": timeout_sec,
        },
        "fixed_override": copy.deepcopy(fixed_override),
        "parameters": parameters,
        "constraints": constraints,
    }


def _build_override_from_trial_values(
    base_override: Dict[str, Any],
    parameters: List[Dict[str, Any]],
    trial_values: Dict[str, Any],
) -> Dict[str, Any]:
    merged = copy.deepcopy(base_override)
    for param in parameters:
        _set_nested_value(merged, param["path"], trial_values[param["name"]])
    return merged


def _constraint_violation(override: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Optional[str]:
    for constraint in constraints:
        if constraint["type"] != "order":
            continue
        lower = _get_nested_value(override, constraint["lower_path"])
        upper = _get_nested_value(override, constraint["upper_path"])
        lower_value = float(lower)
        upper_value = float(upper)
        if lower_value > (upper_value - float(constraint.get("min_gap", 0.0))):
            return (
                f"{constraint['lower_path']}={lower_value} must be <= "
                f"{constraint['upper_path']} - {float(constraint.get('min_gap', 0.0))}"
            )
    return None


def _extract_objective_value(result: Dict[str, Any], metric_path: str) -> float:
    node: Any = result
    for part in [part.strip() for part in metric_path.split(".") if part.strip()]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(metric_path)
        node = node[part]
    if node is None:
        raise ValueError(f"Objective metric {metric_path!r} resolved to null")
    return float(node)


def _make_sampler(optuna: Any, sampler_name: str, seed: Optional[int]) -> Any:
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def _storage_url(storage_path: Optional[Path]) -> Optional[str]:
    if storage_path is None:
        return None
    return "sqlite:///" + storage_path.resolve().as_posix()


def _build_runtime(base_dir: Optional[Path]) -> RuntimeContext:
    config = create_runtime_config(base_dir)
    runtime = RuntimeContext(
        config=config,
        providers=None,
        engines=EngineState(face_mode="disabled", pose_mode="disabled"),
    )
    runtime.config.run_mode = "benchmark"
    return runtime


def _evaluate_anchor_guard(runtime: RuntimeContext, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
    snapshot = anchor_registry_snapshot(runtime.config)
    entries = snapshot.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}

    available_by_role_bucket: Dict[Tuple[str, str], List[str]] = {}
    existing_anchor_count = 0
    for anchor_id, node in entries.items():
        if not isinstance(node, dict) or not bool(node.get("exists", False)):
            continue
        existing_anchor_count += 1
        role = str(node.get("role", "")).strip().upper()
        view_bucket = str(node.get("view_bucket", "")).strip()
        if not role or not view_bucket:
            continue
        available_by_role_bucket.setdefault((role, view_bucket), []).append(str(anchor_id))

    requirement_summaries: List[Dict[str, Any]] = []
    missing_requirements: List[str] = []
    for requirement in requirements:
        role = str(requirement["role"]).strip().upper()
        min_count = int(requirement.get("min_count", 1))
        per_bucket: Dict[str, Dict[str, Any]] = {}
        requirement_passed = True
        for view_bucket in requirement["view_buckets"]:
            matched_ids = sorted(available_by_role_bucket.get((role, view_bucket), []))
            bucket_passed = len(matched_ids) >= min_count
            per_bucket[view_bucket] = {
                "matched_anchor_ids": matched_ids,
                "count": len(matched_ids),
                "min_count": min_count,
                "passed": bucket_passed,
            }
            if not bucket_passed:
                requirement_passed = False
                missing_requirements.append(f"{role}:{view_bucket}<{min_count}")
        requirement_summaries.append(
            {
                "role": role,
                "passed": requirement_passed,
                "per_view_bucket": per_bucket,
            }
        )

    return {
        "anchor_source": str(snapshot.get("anchor_source", "")),
        "registered_entries": len(entries),
        "existing_entries": existing_anchor_count,
        "requirements": requirement_summaries,
        "missing_requirements": missing_requirements,
        "passed": len(missing_requirements) == 0,
    }


def _evaluate_optuna_guard(
    runtime: RuntimeContext,
    labels_bundle: Dict[str, Any],
    guard: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []

    if bool(guard.get("optuna_locked", True)):
        reason = str(guard.get("lock_reason", "")).strip() or "Optuna is locked by guard config"
        errors.append(reason)

    dataset_role = str(labels_bundle.get("dataset_role", DEFAULT_BENCHMARK_LABEL_ROLE)).strip() or DEFAULT_BENCHMARK_LABEL_ROLE
    required_role = str(guard.get("required_label_role", DEFAULT_BENCHMARK_FROZEN_ROLE)).strip()
    if required_role and dataset_role != required_role:
        errors.append(
            f"Benchmark label dataset_role must be {required_role!r}, got {dataset_role!r}"
        )

    optuna_ready = bool(labels_bundle.get("optuna_ready", False))
    if bool(guard.get("require_optuna_ready", True)) and not optuna_ready:
        errors.append("Benchmark labels must set optuna_ready=true before Optuna can run")

    anchor_guard = _evaluate_anchor_guard(runtime, guard.get("anchor_requirements", []))
    if not anchor_guard["passed"]:
        missing = ", ".join(anchor_guard["missing_requirements"][:8])
        extra = ""
        if len(anchor_guard["missing_requirements"]) > 8:
            extra = f" (+{len(anchor_guard['missing_requirements']) - 8} more)"
        errors.append(f"Anchor coverage is incomplete for Optuna guard: {missing}{extra}")

    return {
        "allowed": len(errors) == 0,
        "errors": errors,
        "guard_config": {
            "path": str(guard.get("path", "")),
            "optuna_locked": bool(guard.get("optuna_locked", True)),
            "lock_reason": str(guard.get("lock_reason", "")),
            "required_label_role": required_role,
            "require_optuna_ready": bool(guard.get("require_optuna_ready", True)),
        },
        "label_bundle": {
            "dataset_role": dataset_role,
            "optuna_ready": optuna_ready,
            "benchmark_id": str(labels_bundle.get("benchmark_id", "")),
            "freeze_tag": str(labels_bundle.get("freeze_tag", "")),
            "num_items": len(labels_bundle.get("items", {})),
        },
        "anchor_guard": anchor_guard,
    }


def run_optuna_search(
    *,
    base_dir: Optional[Path],
    report_path: Path,
    labels_path: Path,
    search_space_path: Path,
    cli_fixed_override: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    best_override_out: Optional[Path] = None,
    study_name_override: Optional[str] = None,
    storage_path: Optional[Path] = None,
    trials_override: Optional[int] = None,
    guard_path: Optional[Path] = None,
    preset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed in the project environment. Install it with "
            r"'.\.venv\Scripts\pip.exe install optuna'"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = load_optuna_search_space(search_space_path)
    runtime = _build_runtime(base_dir)
    print_runtime_config(runtime)

    report_path = report_path.resolve()
    labels_path = labels_path.resolve()
    guard_path = (
        guard_path.resolve()
        if guard_path is not None
        else (runtime.config.paths.config_dir / "optuna_guard.json").resolve()
    )
    labels_bundle = load_benchmark_label_bundle(labels_path)
    guard = load_optuna_guard(guard_path)
    guard_status = _evaluate_optuna_guard(runtime, labels_bundle, guard)
    if not guard_status["allowed"]:
        detail = "; ".join(str(message) for message in guard_status["errors"] if str(message).strip())
        raise RuntimeError(f"Optuna guard blocked this run. {detail}")

    base_override = copy.deepcopy(search_space["fixed_override"])
    if cli_fixed_override:
        base_override = _deep_merge_dict(base_override, cli_fixed_override)

    baseline_result = benchmark_report(
        runtime=runtime,
        report_path=report_path,
        labels_path=labels_path,
        threshold_override=base_override or None,
    )

    metric_path = search_space["objective"]["metric_path"]
    direction = search_space["objective"]["direction"]
    sampler_name = search_space["study"]["sampler"]
    seed = search_space["study"]["seed"]
    n_trials = int(trials_override if trials_override is not None else search_space["study"]["n_trials"])
    timeout_sec = search_space["study"]["timeout_sec"]
    study_name = (
        str(study_name_override).strip()
        if study_name_override is not None
        else (search_space["study"]["name"] or "xiaona_benchmark_optuna")
    )
    storage_url = _storage_url(storage_path)

    sampler = _make_sampler(optuna, sampler_name, seed)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    trial_param_specs = search_space["parameters"]
    constraints = search_space["constraints"]

    def objective(trial: Any) -> float:
        trial_values: Dict[str, Any] = {}
        for param in trial_param_specs:
            if param["type"] == "categorical":
                trial_values[param["name"]] = trial.suggest_categorical(param["name"], param["choices"])
            elif param["type"] == "int":
                trial_values[param["name"]] = trial.suggest_int(
                    param["name"],
                    int(param["low"]),
                    int(param["high"]),
                    step=int(param.get("step", 1)),
                )
            else:
                suggest_kwargs: Dict[str, Any] = {}
                if "step" in param:
                    suggest_kwargs["step"] = float(param["step"])
                else:
                    suggest_kwargs["log"] = bool(param.get("log", False))
                trial_values[param["name"]] = trial.suggest_float(
                    param["name"],
                    float(param["low"]),
                    float(param["high"]),
                    **suggest_kwargs,
                )

        override = _build_override_from_trial_values(base_override, trial_param_specs, trial_values)
        violation = _constraint_violation(override, constraints)
        if violation is not None:
            trial.set_user_attr("constraint_violation", violation)
            raise optuna.TrialPruned(violation)

        result = benchmark_report(
            runtime=runtime,
            report_path=report_path,
            labels_path=labels_path,
            threshold_override=override,
        )
        objective_value = _extract_objective_value(result, metric_path)
        trial.set_user_attr("threshold_override", override)
        trial.set_user_attr("metrics", result.get("metrics", {}))
        trial.set_user_attr("agreement_metrics", result.get("agreement_metrics", {}))
        trial.set_user_attr("objective_metric_path", metric_path)
        return objective_value

    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=False)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    if len(completed_trials) == 0:
        examples = [
            str(trial.user_attrs.get("constraint_violation", "pruned without a recorded reason"))
            for trial in pruned_trials[: min(3, len(pruned_trials))]
        ]
        detail = "; ".join(example for example in examples if example) or "no completed trial was produced"
        raise RuntimeError(
            f"Optuna search finished without any completed trials. "
            f"pruned_trials={len(pruned_trials)}. Examples: {detail}"
        )

    best_trial = study.best_trial
    best_override = copy.deepcopy(best_trial.user_attrs.get("threshold_override", base_override))
    best_result = benchmark_report(
        runtime=runtime,
        report_path=report_path,
        labels_path=labels_path,
        threshold_override=best_override,
    )
    baseline_value = _extract_objective_value(baseline_result, metric_path)
    best_value = float(best_trial.value)
    improvement = best_value - baseline_value if direction == "maximize" else baseline_value - best_value

    top_trials: List[Dict[str, Any]] = []
    sort_reverse = direction == "maximize"
    for trial in sorted(completed_trials, key=lambda item: float(item.value), reverse=sort_reverse)[: min(10, len(completed_trials))]:
        top_trials.append(
            {
                "number": int(trial.number),
                "value": float(trial.value),
                "params": copy.deepcopy(trial.params),
                "metrics": copy.deepcopy(trial.user_attrs.get("metrics", {})),
                "agreement_metrics": copy.deepcopy(trial.user_attrs.get("agreement_metrics", {})),
            }
        )

    result = {
        "schema_version": "qa_optuna_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": {
            "name": study.study_name,
            "direction": direction,
            "sampler": sampler_name,
            "seed": seed,
            "n_trials_requested": n_trials,
            "timeout_sec": timeout_sec,
            "storage": storage_url,
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
        },
        "report_file": str(report_path),
        "labels_file": str(labels_path),
        "search_space_file": str(search_space_path.resolve()),
        "guard": guard_status,
        "preset": copy.deepcopy(preset) if isinstance(preset, dict) else None,
        "objective_metric_path": metric_path,
        "baseline": {
            "objective_value": baseline_value,
            "threshold_override": base_override,
            "metrics": baseline_result.get("metrics", {}),
            "agreement_metrics": baseline_result.get("agreement_metrics", {}),
        },
        "best": {
            "trial_number": int(best_trial.number),
            "objective_value": best_value,
            "improvement": float(improvement),
            "params": copy.deepcopy(best_trial.params),
            "threshold_override": best_override,
            "metrics": best_result.get("metrics", {}),
            "agreement_metrics": best_result.get("agreement_metrics", {}),
        },
        "top_trials": top_trials,
    }

    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if best_override_out is not None:
        best_override_out = best_override_out.resolve()
        best_override_out.parent.mkdir(parents=True, exist_ok=True)
        best_override_out.write_text(
            json.dumps(best_override, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result
