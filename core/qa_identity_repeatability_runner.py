from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # Protocol inspection remains available without the vision runtime.
    cv2 = None  # type: ignore[assignment]

from .qa_identity_evidence_contract import angular_distance_radians
from .qa_io import atomic_write_bytes, atomic_write_json, load_json_dict
from .qa_procrustes_shape import weighted_irls_procrustes
from .qa_repeatability_shadow import (
    REPEATABILITY_DOMAINS,
    load_repeatability_protocol,
    summarize_repeatability_cohort,
    summarize_repeatability_trials,
)


REPEATABILITY_RUN_SCHEMA = "identity_repeatability_shadow_run_v0_1"
REPEATABILITY_TRIAL_SCHEMA = "identity_repeatability_shadow_trial_v0_1"
SUPPORTED_AXES = ("face_identity", "face_shape")
TERMINAL_STATES = {"COMPLETE", "MEASUREMENT_UNAVAILABLE", "FAILED"}


class RepeatabilityMeasurementAdapter(Protocol):
    def describe(self) -> Dict[str, Any]: ...

    def measure(self, image_path: Path) -> Dict[str, Any]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: str, *, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return token[:96] or fallback


def _normalized_axes(axes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(axis).strip().lower() for axis in axes))
    invalid = [axis for axis in normalized if axis not in SUPPORTED_AXES]
    if invalid or not normalized:
        raise ValueError(f"unsupported repeatability axes: {invalid or list(axes)!r}")
    return normalized


def build_repeatability_trial_plan(protocol: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    payload = protocol or load_repeatability_protocol()
    domains = payload.get("domains") if isinstance(payload.get("domains"), dict) else {}
    plan: List[Dict[str, Any]] = []
    for domain in REPEATABILITY_DOMAINS:
        node = domains.get(domain) if isinstance(domains.get(domain), dict) else {}
        transforms = node.get("transforms") if isinstance(node.get("transforms"), list) else []
        repeat_count = int(node.get("repeat_count") or 1)
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            repetitions = repeat_count if domain == "numerical_repeatability" else 1
            for repeat_index in range(1, repetitions + 1):
                base_id = str(transform.get("id") or "").strip()
                trial_id = f"{base_id}__repeat_{repeat_index:02d}" if repetitions > 1 else base_id
                spec = {
                    "domain": domain,
                    "trial_id": trial_id,
                    "transform_id": base_id,
                    "perturbation_family": str(transform.get("family") or "unclassified"),
                    "signed_strength": float(transform.get("signed_strength")),
                    "unit": str(transform.get("unit") or "none"),
                    "repeat_index": repeat_index,
                }
                spec["trial_spec_sha256"] = _canonical_sha256(spec)
                plan.append(spec)
    trial_ids = [str(spec["trial_id"]) for spec in plan]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("repeatability trial plan contains duplicate trial ids")
    return plan


def _decode_image(path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to execute identity repeatability image transforms")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"image decode failed: {path}")
    return image


def _encode_image(extension: str, image: np.ndarray, params: Iterable[int]) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to execute identity repeatability image transforms")
    ok, encoded = cv2.imencode(extension, image, list(params))
    if not ok:
        raise ValueError(f"image encode failed: {extension}")
    return encoded.tobytes()


def _transformed_bytes(
    source_path: Path,
    spec: Dict[str, Any],
    transform_contract: Dict[str, Any],
) -> tuple[bytes, str]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to execute identity repeatability image transforms")
    transform_id = str(spec.get("transform_id") or "")
    strength = float(spec.get("signed_strength") or 0.0)
    if transform_id == "identity_transform":
        return source_path.read_bytes(), source_path.suffix.lower() or ".png"

    image = _decode_image(source_path)
    height, width = image.shape[:2]
    if transform_id == "png_lossless_roundtrip":
        compression = int((transform_contract.get("png") or {}).get("compression", 3))
        return _encode_image(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression]), ".png"
    if transform_id == "jpeg_quality_95_roundtrip":
        quality = int((transform_contract.get("jpeg") or {}).get("quality", 95))
        return _encode_image(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]), ".jpg"
    if transform_id.startswith("resize_roundtrip_"):
        scale = 1.0 + strength
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        first_interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (target_width, target_height), interpolation=first_interpolation)
        transformed = cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR)
    elif transform_id.startswith("crop_translate_x_") or transform_id.startswith("crop_translate_y_"):
        dx = strength * width if "_x_" in transform_id else 0.0
        dy = strength * height if "_y_" in transform_id else 0.0
        matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        transformed = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    elif transform_id.startswith("gamma_"):
        exponent = 1.0 + strength
        normalized = image.astype(np.float64) / 255.0
        transformed = np.clip(255.0 * np.power(normalized, exponent), 0.0, 255.0).round().astype(np.uint8)
    else:
        raise ValueError(f"unsupported preregistered transform: {transform_id}")

    if transformed.shape[:2] != (height, width):
        raise ValueError(f"transform changed output dimensions: {transform_id}")
    compression = int((transform_contract.get("png") or {}).get("compression", 3))
    return _encode_image(".png", transformed, [cv2.IMWRITE_PNG_COMPRESSION, compression]), ".png"


def materialize_repeatability_image(
    source_path: Path,
    destination_stem: Path,
    spec: Dict[str, Any],
    transform_contract: Dict[str, Any],
) -> Path:
    payload, extension = _transformed_bytes(Path(source_path), spec, transform_contract)
    destination = Path(destination_stem).with_suffix(extension)
    atomic_write_bytes(destination, payload)
    decoded = _decode_image(destination)
    source_shape = _decode_image(Path(source_path)).shape[:2]
    if decoded.shape[:2] != source_shape:
        destination.unlink(missing_ok=True)
        raise ValueError("materialized repeatability image dimensions do not match source")
    return destination


def _axis_node(observation: Dict[str, Any], axis: str) -> Dict[str, Any]:
    node = observation.get(axis)
    return dict(node) if isinstance(node, dict) else {"available": False, "errors": ["AXIS_MISSING"]}


def _provider_contract_sha(node: Dict[str, Any]) -> Optional[str]:
    contract = node.get("provider_contract")
    if not isinstance(contract, dict) or not contract:
        return None
    return str(
        contract.get("comparable_contract_sha256")
        or contract.get("observed_contract_sha256")
        or _canonical_sha256(contract)
    )


def _landmark_weights(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Any:
    def normalize(value: Any, count: int) -> Optional[np.ndarray]:
        try:
            weights = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if weights.size != count or not bool(np.all(np.isfinite(weights))):
            return None
        weights = np.clip(weights, 0.0, None)
        return weights if int(np.count_nonzero(weights > 0.0)) >= 3 else None

    try:
        count = int(np.asarray(reference.get("value"), dtype=np.float64).reshape(-1, 2).shape[0])
    except Exception:
        return None
    reference_weights = normalize(reference.get("visibility_weights"), count)
    candidate_weights = normalize(candidate.get("visibility_weights"), count)
    if reference_weights is not None and candidate_weights is not None:
        return np.minimum(reference_weights, candidate_weights)
    return reference_weights if reference_weights is not None else candidate_weights


def _native_residual(
    baseline: Dict[str, Any],
    trial: Dict[str, Any],
    axis: str,
) -> Dict[str, Any]:
    reference = _axis_node(baseline, axis)
    candidate = _axis_node(trial, axis)
    errors: List[str] = []
    if not bool(reference.get("available")):
        errors.append("BASELINE_MEASUREMENT_UNAVAILABLE")
    if not bool(candidate.get("available")):
        errors.append("TRIAL_MEASUREMENT_UNAVAILABLE")
    reference_contract_sha = _provider_contract_sha(reference)
    candidate_contract_sha = _provider_contract_sha(candidate)
    if not reference_contract_sha or not candidate_contract_sha:
        errors.append("MEASUREMENT_PROVIDER_CONTRACT_UNAVAILABLE")
    elif reference_contract_sha != candidate_contract_sha:
        errors.append("MEASUREMENT_PROVIDER_CONTRACT_MISMATCH")
    if axis == "face_shape":
        reference_schema = str(reference.get("landmark_schema_id") or "").strip()
        candidate_schema = str(candidate.get("landmark_schema_id") or "").strip()
        if reference_schema and candidate_schema and reference_schema != candidate_schema:
            errors.append("LANDMARK_SCHEMA_MISMATCH")
    if errors:
        return {
            "available": False,
            "residual": None,
            "unit": "radian" if axis == "face_identity" else "normalized_shape_distance",
            "errors": errors,
            "baseline_provider_contract_sha256": reference_contract_sha,
            "trial_provider_contract_sha256": candidate_contract_sha,
        }

    if axis == "face_identity":
        result = angular_distance_radians(reference.get("value"), candidate.get("value"))
    else:
        result = weighted_irls_procrustes(
            reference.get("value"),
            candidate.get("value"),
            visibility_weights=_landmark_weights(reference, candidate),
        )
        if not bool(result.get("available")):
            result["errors"] = [str(result.get("error") or "PROCRUSTES_UNAVAILABLE")]
    return {
        **_json_ready(result),
        "baseline_provider_contract_sha256": reference_contract_sha,
        "trial_provider_contract_sha256": candidate_contract_sha,
    }


def _terminal_result(path: Path, expected_spec_sha256: str, *, retry_failed: bool) -> Optional[Dict[str, Any]]:
    payload = load_json_dict(path)
    if not payload:
        return None
    if str(payload.get("trial_spec_sha256") or "") != expected_spec_sha256:
        raise ValueError(f"REPEATABILITY_TRIAL_CONTRACT_MISMATCH:{path}")
    state = str(payload.get("state") or "")
    if state in TERMINAL_STATES and not (state == "FAILED" and retry_failed):
        return payload
    return None


def _measure_adapter(adapter: RepeatabilityMeasurementAdapter, image_path: Path) -> Dict[str, Any]:
    observation = adapter.measure(Path(image_path))
    if not isinstance(observation, dict):
        raise ValueError("repeatability adapter must return a JSON object")
    return _json_ready(observation)


def _compact_observation_provenance(
    observation: Dict[str, Any],
    axes: Sequence[str],
) -> Dict[str, Any]:
    axis_contracts: Dict[str, Any] = {}
    for axis in axes:
        node = _axis_node(observation, axis)
        axis_contracts[axis] = {
            "available": bool(node.get("available")),
            "provider_contract": _json_ready(node.get("provider_contract")),
            "landmark_schema_id": node.get("landmark_schema_id"),
            "measurement_order": _json_ready(node.get("measurement_order")),
            "used_components": _json_ready(node.get("used_components")),
            "prior_dependence": node.get("prior_dependence"),
            "errors": list(node.get("errors") or []),
        }
    return {
        "chain_observation": _json_ready(observation.get("chain_observation")),
        "axes": axis_contracts,
        "canonical_result_provenance": _json_ready(observation.get("canonical_result_provenance")),
        "errors": list(observation.get("errors") or []),
        "decision_influence": "NONE",
    }


def build_detector_chain_diagnostics(
    baseline_observation: Dict[str, Any],
    trial_observation: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_chain = (
        baseline_observation.get("chain_observation")
        if isinstance(baseline_observation.get("chain_observation"), dict)
        else {}
    )
    trial_chain = (
        trial_observation.get("chain_observation")
        if isinstance(trial_observation.get("chain_observation"), dict)
        else {}
    )
    diagnostics: Dict[str, Any] = {
        "schema_version": "detector_alignment_chain_diagnostics_v0_1",
        "bbox_iou": None,
        "bbox_center_displacement_image_fraction": None,
        "bbox_width_relative_delta": None,
        "bbox_height_relative_delta": None,
        "kps5_raw_rms_image_fraction": None,
        "kps5_similarity_shape_residual": None,
        "canonical_pose_l2_delta_deg": None,
        "canonical_pose_axis_delta_deg": {},
        "decision_influence": "NONE",
    }

    def vector(node: Dict[str, Any], key: str, size: int) -> Optional[np.ndarray]:
        try:
            value = np.asarray(node.get(key), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if value.size != size or not bool(np.all(np.isfinite(value))):
            return None
        return value

    reference_bbox = vector(baseline_chain, "face_bbox_normalized_xyxy", 4)
    trial_bbox = vector(trial_chain, "face_bbox_normalized_xyxy", 4)
    if reference_bbox is not None and trial_bbox is not None:
        intersection_width = max(0.0, min(reference_bbox[2], trial_bbox[2]) - max(reference_bbox[0], trial_bbox[0]))
        intersection_height = max(0.0, min(reference_bbox[3], trial_bbox[3]) - max(reference_bbox[1], trial_bbox[1]))
        intersection = intersection_width * intersection_height
        reference_width = max(0.0, reference_bbox[2] - reference_bbox[0])
        reference_height = max(0.0, reference_bbox[3] - reference_bbox[1])
        trial_width = max(0.0, trial_bbox[2] - trial_bbox[0])
        trial_height = max(0.0, trial_bbox[3] - trial_bbox[1])
        union = reference_width * reference_height + trial_width * trial_height - intersection
        if union > 0.0:
            diagnostics["bbox_iou"] = float(intersection / union)
        reference_center = np.asarray(
            [(reference_bbox[0] + reference_bbox[2]) / 2.0, (reference_bbox[1] + reference_bbox[3]) / 2.0]
        )
        trial_center = np.asarray(
            [(trial_bbox[0] + trial_bbox[2]) / 2.0, (trial_bbox[1] + trial_bbox[3]) / 2.0]
        )
        diagnostics["bbox_center_displacement_image_fraction"] = float(
            np.linalg.norm(trial_center - reference_center)
        )
        if reference_width > 0.0:
            diagnostics["bbox_width_relative_delta"] = float(trial_width / reference_width - 1.0)
        if reference_height > 0.0:
            diagnostics["bbox_height_relative_delta"] = float(trial_height / reference_height - 1.0)

    try:
        reference_kps = np.asarray(baseline_chain.get("kps5_normalized_xy"), dtype=np.float64).reshape(-1, 2)
        trial_kps = np.asarray(trial_chain.get("kps5_normalized_xy"), dtype=np.float64).reshape(-1, 2)
    except Exception:
        reference_kps = np.empty((0, 2), dtype=np.float64)
        trial_kps = np.empty((0, 2), dtype=np.float64)
    if (
        reference_kps.shape == (5, 2)
        and trial_kps.shape == (5, 2)
        and bool(np.all(np.isfinite(reference_kps)))
        and bool(np.all(np.isfinite(trial_kps)))
    ):
        diagnostics["kps5_raw_rms_image_fraction"] = float(
            np.sqrt(np.mean(np.sum((trial_kps - reference_kps) ** 2, axis=1)))
        )
        kps_shape = weighted_irls_procrustes(reference_kps, trial_kps)
        if bool(kps_shape.get("available")):
            diagnostics["kps5_similarity_shape_residual"] = kps_shape.get("residual")

    reference_pose = baseline_chain.get("canonical_pose_euler_deg")
    trial_pose = trial_chain.get("canonical_pose_euler_deg")
    if isinstance(reference_pose, dict) and isinstance(trial_pose, dict):
        pose_deltas: Dict[str, float] = {}
        for axis in ["yaw", "pitch", "roll"]:
            try:
                reference_value = float(reference_pose.get(axis))
                trial_value = float(trial_pose.get(axis))
            except (TypeError, ValueError):
                continue
            if np.isfinite(reference_value) and np.isfinite(trial_value):
                pose_deltas[axis] = trial_value - reference_value
        diagnostics["canonical_pose_axis_delta_deg"] = pose_deltas
        if pose_deltas:
            diagnostics["canonical_pose_l2_delta_deg"] = float(
                np.linalg.norm(np.asarray(list(pose_deltas.values()), dtype=np.float64))
            )
    return diagnostics


def _baseline_spec_sha(source_sha256: str) -> str:
    return _canonical_sha256({"kind": "baseline", "source_sha256": source_sha256})


def _record_materialized_image(
    result: Dict[str, Any],
    image_path: Path,
    *,
    retain: bool,
) -> None:
    result["materialized_image"] = str(image_path.resolve())
    result["materialized_sha256"] = _file_sha256(image_path)
    result["materialized_size_bytes"] = int(image_path.stat().st_size)
    result["materialized_image_retained"] = True
    if not retain:
        try:
            image_path.unlink()
            result["materialized_image_retained"] = False
        except OSError as exc:
            result["materialized_cleanup_error"] = f"{type(exc).__name__}:{exc}"


def _load_or_measure_baseline(
    *,
    adapter: RepeatabilityMeasurementAdapter,
    source_path: Path,
    source_sha256: str,
    item_dir: Path,
    retry_failed: bool,
    retain_completed_image: bool,
    axes: Sequence[str],
    trial_schema: str = REPEATABILITY_TRIAL_SCHEMA,
) -> Dict[str, Any]:
    baseline_dir = item_dir / "baseline"
    result_path = baseline_dir / "result.json"
    spec_sha = _baseline_spec_sha(source_sha256)
    existing = _terminal_result(result_path, spec_sha, retry_failed=retry_failed)
    if existing is not None:
        return existing
    attempt = int(load_json_dict(result_path).get("attempt") or 0) + 1
    result: Dict[str, Any] = {
        "schema_version": str(trial_schema),
        "kind": "baseline",
        "trial_spec_sha256": spec_sha,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "attempt": attempt,
        "started_at": _utc_now(),
        "calibration_state": "SHADOW_UNCALIBRATED",
        "decision_influence": "NONE",
    }
    try:
        if _file_sha256(source_path) != source_sha256:
            raise ValueError("REPEATABILITY_SOURCE_CHANGED_AFTER_RUN_CONTRACT")
        baseline_image = materialize_repeatability_image(
            source_path,
            baseline_dir / "input",
            {"transform_id": "identity_transform", "signed_strength": 0.0},
            {},
        )
        observation = _measure_adapter(adapter, baseline_image)
        available_count = sum(
            bool(_axis_node(observation, axis).get("available"))
            for axis in axes
        )
        result.update(
            {
                "state": "COMPLETE" if available_count else "MEASUREMENT_UNAVAILABLE",
                "observation": observation,
            }
        )
        _record_materialized_image(result, baseline_image, retain=retain_completed_image)
    except Exception as exc:
        result.update({"state": "FAILED", "error": f"{type(exc).__name__}:{exc}"})
        if "baseline_image" in locals() and baseline_image.exists():
            _record_materialized_image(result, baseline_image, retain=True)
    result["completed_at"] = _utc_now()
    atomic_write_json(result_path, result)
    return result


def _run_trial(
    *,
    adapter: RepeatabilityMeasurementAdapter,
    source_path: Path,
    source_sha256: str,
    baseline: Dict[str, Any],
    item_dir: Path,
    spec: Dict[str, Any],
    axes: Sequence[str],
    transform_contract: Dict[str, Any],
    retry_failed: bool,
    retain_completed_image: bool,
    native_residual_builder: Callable[[Dict[str, Any], Dict[str, Any], str], Dict[str, Any]],
    chain_diagnostics_builder: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    trial_schema: str = REPEATABILITY_TRIAL_SCHEMA,
) -> Dict[str, Any]:
    trial_dir = item_dir / "trials" / str(spec["domain"]) / str(spec["trial_id"])
    result_path = trial_dir / "result.json"
    spec_sha = str(spec["trial_spec_sha256"])
    existing = _terminal_result(result_path, spec_sha, retry_failed=retry_failed)
    if existing is not None:
        return existing
    attempt = int(load_json_dict(result_path).get("attempt") or 0) + 1
    result: Dict[str, Any] = {
        "schema_version": str(trial_schema),
        **dict(spec),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "attempt": attempt,
        "started_at": _utc_now(),
        "calibration_state": "SHADOW_UNCALIBRATED",
        "decision_influence": "NONE",
    }
    try:
        if _file_sha256(source_path) != source_sha256:
            raise ValueError("REPEATABILITY_SOURCE_CHANGED_AFTER_RUN_CONTRACT")
        trial_image = materialize_repeatability_image(
            source_path,
            trial_dir / "input",
            spec,
            transform_contract,
        )
        observation = _measure_adapter(adapter, trial_image)
        baseline_observation = baseline.get("observation") if isinstance(baseline.get("observation"), dict) else {}
        residuals = {
            axis: native_residual_builder(baseline_observation, observation, axis)
            for axis in axes
        }
        chain_diagnostics = chain_diagnostics_builder(baseline_observation, observation)
        available_count = sum(bool(node.get("available")) for node in residuals.values())
        result.update(
            {
                "state": "COMPLETE" if available_count else "MEASUREMENT_UNAVAILABLE",
                "chain_signature": observation.get("chain_signature"),
                "baseline_chain_signature": baseline_observation.get("chain_signature"),
                "observation_provenance": _compact_observation_provenance(observation, axes),
                "chain_diagnostics": chain_diagnostics,
                "residuals": residuals,
                "measurement_errors": list(observation.get("errors") or []),
            }
        )
        _record_materialized_image(result, trial_image, retain=retain_completed_image)
    except Exception as exc:
        result.update({"state": "FAILED", "error": f"{type(exc).__name__}:{exc}", "residuals": {}})
        if "trial_image" in locals() and trial_image.exists():
            _record_materialized_image(result, trial_image, retain=True)
    result["completed_at"] = _utc_now()
    atomic_write_json(result_path, result)
    return result


def _summary_rows(results: Iterable[Dict[str, Any]], axis: str, domain: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        if str(result.get("domain")) != domain:
            continue
        residuals = result.get("residuals") if isinstance(result.get("residuals"), dict) else {}
        axis_result = residuals.get(axis) if isinstance(residuals.get(axis), dict) else {}
        rows.append(
            {
                "trial_id": result.get("trial_id"),
                "perturbation_family": result.get("perturbation_family"),
                "signed_strength": result.get("signed_strength"),
                "native_residual": axis_result.get("residual") if axis_result.get("available") else None,
                "baseline_chain_signature": result.get("baseline_chain_signature"),
                "trial_chain_signature": result.get("chain_signature"),
                "chain_diagnostics": result.get("chain_diagnostics"),
            }
        )
    return rows


def _item_summary(
    source: Dict[str, Any],
    baseline: Dict[str, Any],
    results: List[Dict[str, Any]],
    axes: Sequence[str],
) -> Dict[str, Any]:
    state_counts: Dict[str, int] = {}
    for result in results:
        state = str(result.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1
    axes_summary: Dict[str, Any] = {}
    for axis in axes:
        unit = "radian" if axis == "face_identity" else "normalized_shape_distance"
        axes_summary[axis] = {
            domain: summarize_repeatability_trials(
                _summary_rows(results, axis, domain),
                domain=domain,
                residual_unit=unit,
            )
            for domain in REPEATABILITY_DOMAINS
        }
    return {
        "source": source,
        "baseline_state": baseline.get("state"),
        "trial_state_counts": state_counts,
        "axes": axes_summary,
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "decision_influence": "NONE",
    }


def run_repeatability_shadow_engine(
    *,
    image_paths: Sequence[Path],
    output_root: Path,
    adapter: RepeatabilityMeasurementAdapter,
    axes: Sequence[str],
    protocol: Dict[str, Any],
    run_schema: str,
    trial_schema: str,
    runner_implementation_path: Path,
    native_residual_builder: Callable[[Dict[str, Any], Dict[str, Any], str], Dict[str, Any]],
    chain_diagnostics_builder: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    item_summary_builder: Callable[
        [Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Sequence[str]],
        Dict[str, Any],
    ],
    cohort_summary_builder: Callable[[List[Dict[str, Any]], Sequence[str]], Dict[str, Any]],
    run_id: Optional[str] = None,
    retry_failed: bool = False,
) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to execute repeatability trials")
    normalized_axes = tuple(dict.fromkeys(str(axis).strip().lower() for axis in axes))
    if not normalized_axes or any(not axis for axis in normalized_axes):
        raise ValueError("at least one valid repeatability axis is required")
    protocol_sha = _canonical_sha256(protocol)
    plan = build_repeatability_trial_plan(protocol)
    execution_policy = protocol.get("execution") if isinstance(protocol.get("execution"), dict) else {}
    retain_completed_images = bool(execution_policy.get("retain_completed_materialized_images", False))
    stop_on_failed_trial = bool(execution_policy.get("stop_on_failed_trial", False))
    resolved_images = [Path(path).resolve() for path in image_paths]
    if not resolved_images:
        raise ValueError("at least one repeatability image is required")
    sources: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    for image_path in resolved_images:
        if not image_path.is_file():
            raise ValueError(f"repeatability image does not exist: {image_path}")
        normalized_path = str(image_path).lower()
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        source_sha = _file_sha256(image_path)
        path_sha = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()
        item_key = _safe_token(
            f"{image_path.stem}_{source_sha[:10]}_{path_sha[:8]}",
            fallback=f"{source_sha[:10]}_{path_sha[:8]}",
        )
        sources.append(
            {
                "item_key": item_key,
                "path": str(image_path),
                "sha256": source_sha,
                "size_bytes": image_path.stat().st_size,
            }
        )
    adapter_contract = _json_ready(adapter.describe())
    run_contract = {
        "schema_version": str(run_schema),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha,
        "axes": list(normalized_axes),
        "sources": sources,
        "adapter_contract": adapter_contract,
        "execution_implementation": {
            "execution_engine_sha256": _file_sha256(Path(__file__).resolve()),
            "workflow_runner_sha256": _file_sha256(Path(runner_implementation_path).resolve()),
            "opencv_version": str(getattr(cv2, "__version__", "unavailable")),
            "numpy_version": str(np.__version__),
        },
        "protocol": protocol,
        "trial_plan": plan,
        "trial_plan_sha256": _canonical_sha256(plan),
        "decision_influence": "NONE",
    }
    run_contract_sha = _canonical_sha256(run_contract)
    resolved_run_id = _safe_token(run_id or f"repeatability_{run_contract_sha[:16]}", fallback=run_contract_sha[:16])
    run_dir = Path(output_root).resolve() / resolved_run_id
    manifest_path = run_dir / "run_manifest.json"
    existing_manifest = load_json_dict(manifest_path)
    if existing_manifest:
        if str(existing_manifest.get("run_contract_sha256") or "") != run_contract_sha:
            raise ValueError(f"REPEATABILITY_RUN_CONTRACT_MISMATCH:{run_dir}")
    else:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": str(run_schema),
                "run_id": resolved_run_id,
                "run_contract_sha256": run_contract_sha,
                "run_contract": run_contract,
                "created_at": _utc_now(),
                "execution": {
                    "explicit_workflow_required": True,
                    "serialized": True,
                    "max_concurrent_gpu_jobs": 1,
                    "atomic_trial_writes": True,
                    "resume_per_trial": True,
                    "retain_completed_materialized_images": retain_completed_images,
                    "retain_failed_materialized_images": True,
                },
            },
        )

    item_summaries: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    baseline_state_counts: Dict[str, int] = {}
    for source in sources:
        source_path = Path(str(source["path"]))
        item_dir = run_dir / "items" / str(source["item_key"])
        baseline = _load_or_measure_baseline(
            adapter=adapter,
            source_path=source_path,
            source_sha256=str(source["sha256"]),
            item_dir=item_dir,
            retry_failed=retry_failed,
            retain_completed_image=retain_completed_images,
            axes=normalized_axes,
            trial_schema=trial_schema,
        )
        baseline_state = str(baseline.get("state") or "UNKNOWN")
        baseline_state_counts[baseline_state] = baseline_state_counts.get(baseline_state, 0) + 1
        results: List[Dict[str, Any]] = []
        if baseline_state != "COMPLETE":
            item_summaries.append(item_summary_builder(source, baseline, results, normalized_axes))
            continue
        for spec in plan:
            result = _run_trial(
                adapter=adapter,
                source_path=source_path,
                source_sha256=str(source["sha256"]),
                baseline=baseline,
                item_dir=item_dir,
                spec=spec,
                axes=normalized_axes,
                transform_contract=dict(protocol.get("transform_contract") or {}),
                retry_failed=retry_failed,
                retain_completed_image=retain_completed_images,
                native_residual_builder=native_residual_builder,
                chain_diagnostics_builder=chain_diagnostics_builder,
                trial_schema=trial_schema,
            )
            results.append(result)
            all_results.append(result)
            if str(result.get("state") or "") == "FAILED" and stop_on_failed_trial:
                break
        item_summaries.append(item_summary_builder(source, baseline, results, normalized_axes))

    state_counts: Dict[str, int] = {}
    for result in all_results:
        state = str(result.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1
    status = "PARTIAL" if state_counts.get("FAILED") or baseline_state_counts.get("FAILED") else "COMPLETE"
    if status == "COMPLETE" and (
        state_counts.get("MEASUREMENT_UNAVAILABLE")
        or baseline_state_counts.get("MEASUREMENT_UNAVAILABLE")
    ):
        status = "COMPLETE_WITH_UNAVAILABLE_MEASUREMENTS"
    summary = {
        "schema_version": str(run_schema),
        "run_id": resolved_run_id,
        "run_contract_sha256": run_contract_sha,
        "status": status,
        "completed_at": _utc_now(),
        "run_dir": str(run_dir),
        "source_count": len(sources),
        "trial_count_per_source": len(plan),
        "planned_trial_count": len(plan) * len(sources),
        "executed_or_resumed_trial_count": len(all_results),
        "baseline_state_counts": baseline_state_counts,
        "trial_state_counts": state_counts,
        "materialized_image_retention": {
            "completed_retained": retain_completed_images,
            "failed_retained": True,
        },
        "items": item_summaries,
        "cross_source_descriptors": cohort_summary_builder(item_summaries, normalized_axes),
        "combined_repeatability_score": None,
        "stable_unstable_classification": None,
        "parameter_fitting_allowed": False,
        "decision_influence": "NONE",
    }
    atomic_write_json(run_dir / "run_summary.json", summary)
    return summary


def run_identity_repeatability_shadow(
    *,
    image_paths: Sequence[Path],
    output_root: Path,
    adapter: RepeatabilityMeasurementAdapter,
    axes: Sequence[str] = SUPPORTED_AXES,
    run_id: Optional[str] = None,
    retry_failed: bool = False,
) -> Dict[str, Any]:
    normalized_axes = _normalized_axes(axes)
    return run_repeatability_shadow_engine(
        image_paths=image_paths,
        output_root=output_root,
        adapter=adapter,
        axes=normalized_axes,
        protocol=load_repeatability_protocol(),
        run_schema=REPEATABILITY_RUN_SCHEMA,
        trial_schema=REPEATABILITY_TRIAL_SCHEMA,
        runner_implementation_path=Path(__file__).resolve(),
        native_residual_builder=_native_residual,
        chain_diagnostics_builder=build_detector_chain_diagnostics,
        item_summary_builder=_item_summary,
        cohort_summary_builder=lambda items, selected_axes: summarize_repeatability_cohort(
            items,
            axes=selected_axes,
        ),
        run_id=run_id,
        retry_failed=retry_failed,
    )
