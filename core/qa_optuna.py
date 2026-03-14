from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .qa_benchmark import benchmark_report
from .qa_pipeline import print_runtime_config
from .qa_runtime import EngineState, RuntimeContext, create_runtime_config


OPTUNA_SEARCH_SPACE_SCHEMA = "qa_optuna_search_space_v1"
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


def load_optuna_search_space(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optuna search space file must decode to a JSON object")

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
