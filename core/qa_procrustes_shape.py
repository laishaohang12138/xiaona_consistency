from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


PROCRUSTES_SHAPE_SCHEMA = "weighted_irls_procrustes_v0_1"
DEFAULT_HUBER_DELTA = 0.05
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TOLERANCE = 1e-9


def _point_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if array.ndim == 1:
        if array.size < 6 or array.size % 2 != 0:
            return None
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 3:
        return None
    if not bool(np.all(np.isfinite(array))):
        return None
    return array


def _weight_array(value: Any, count: int) -> Optional[np.ndarray]:
    if value is None:
        return np.ones(count, dtype=np.float64)
    try:
        weights = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if weights.size != count or not bool(np.all(np.isfinite(weights))):
        return None
    weights = np.clip(weights, 0.0, None)
    if int(np.count_nonzero(weights > 0.0)) < 3:
        return None
    return weights


def _weighted_normalize(points: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None
    center = np.sum(points * weights[:, None], axis=0) / weight_sum
    centered = points - center
    scale_squared = float(np.sum(weights * np.sum(centered * centered, axis=1)) / weight_sum)
    if scale_squared <= 1e-16:
        return None
    return centered / float(np.sqrt(scale_squared))


def _optimal_rotation(candidate: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    covariance = candidate.T @ (weights[:, None] * reference)
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def _huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    weights = np.ones_like(residuals, dtype=np.float64)
    mask = residuals > delta
    weights[mask] = delta / np.maximum(residuals[mask], 1e-12)
    return weights


def _huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    quadratic = 0.5 * residuals * residuals
    linear = delta * (residuals - 0.5 * delta)
    return np.where(residuals <= delta, quadratic, linear)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> Optional[float]:
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None
    return float(np.sqrt(np.sum(weights * values * values) / weight_sum))


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(np.any(valid)):
        return np.full(len(quantiles), np.nan, dtype=np.float64)
    valid_values = values[valid]
    valid_weights = weights[valid]
    order = np.argsort(valid_values, kind="stable")
    sorted_values = valid_values[order]
    cumulative = np.cumsum(valid_weights[order])
    targets = np.asarray(quantiles, dtype=np.float64) * float(cumulative[-1])
    return np.interp(targets, cumulative, sorted_values)


def _partition_diagnostics(
    reference: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Optional[float]]:
    ys = reference[:, 1]
    abs_x = np.abs(reference[:, 0])
    visible = weights > 0.0
    y_q33, y_q66 = _weighted_quantiles(ys, weights, [0.33, 0.66]).tolist()
    contour_q75 = float(_weighted_quantiles(abs_x, weights, [0.75])[0])
    center_q35 = float(_weighted_quantiles(abs_x, weights, [0.35])[0])
    masks = {
        "low_y_band": visible & (ys <= y_q33),
        "mid_y_band": visible & (ys > y_q33) & (ys < y_q66),
        "high_y_band": visible & (ys >= y_q66),
        "lateral_band": visible & (abs_x >= contour_q75),
        "center_axis_band": visible & (abs_x <= center_q35),
    }
    output: Dict[str, Optional[float]] = {}
    for name, mask in masks.items():
        output[name] = _weighted_rms(residuals[mask], weights[mask]) if int(np.count_nonzero(mask)) else None
    return output


def weighted_irls_procrustes(
    reference: Any,
    candidate: Any,
    *,
    visibility_weights: Any = None,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    reference_points = _point_array(reference)
    candidate_points = _point_array(candidate)
    if reference_points is None or candidate_points is None:
        return {"available": False, "error": "LANDMARK_POINTS_INVALID"}
    if reference_points.shape != candidate_points.shape:
        return {"available": False, "error": "LANDMARK_SHAPE_MISMATCH"}
    base_weights = _weight_array(visibility_weights, reference_points.shape[0])
    if base_weights is None:
        return {"available": False, "error": "LANDMARK_VISIBILITY_WEIGHTS_INVALID"}
    if not (float(huber_delta) > 0.0 and int(max_iterations) > 0 and float(tolerance) > 0.0):
        return {"available": False, "error": "PROCRUSTES_SOLVER_CONTRACT_INVALID"}

    robust_weights = np.ones(reference_points.shape[0], dtype=np.float64)
    previous_rotation: Optional[np.ndarray] = None
    converged = False
    iterations = 0
    normalized_reference: Optional[np.ndarray] = None
    aligned_candidate: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        combined_weights = base_weights * robust_weights
        normalized_reference = _weighted_normalize(reference_points, combined_weights)
        normalized_candidate = _weighted_normalize(candidate_points, combined_weights)
        if normalized_reference is None or normalized_candidate is None:
            return {"available": False, "error": "LANDMARK_CONFIGURATION_DEGENERATE"}
        rotation = _optimal_rotation(normalized_candidate, normalized_reference, combined_weights)
        aligned_candidate = normalized_candidate @ rotation
        residuals = np.linalg.norm(aligned_candidate - normalized_reference, axis=1)
        next_robust_weights = _huber_weights(residuals, float(huber_delta))
        rotation_delta = (
            float("inf")
            if previous_rotation is None
            else float(np.linalg.norm(rotation - previous_rotation, ord="fro"))
        )
        weight_delta = float(np.max(np.abs(next_robust_weights - robust_weights)))
        robust_weights = next_robust_weights
        previous_rotation = rotation
        if rotation_delta <= float(tolerance) and weight_delta <= float(tolerance):
            converged = True
            break

    assert normalized_reference is not None and aligned_candidate is not None and residuals is not None
    final_weights = base_weights * robust_weights
    final_reference = _weighted_normalize(reference_points, final_weights)
    final_candidate = _weighted_normalize(candidate_points, final_weights)
    if final_reference is None or final_candidate is None:
        return {"available": False, "error": "LANDMARK_CONFIGURATION_DEGENERATE"}
    final_rotation = _optimal_rotation(final_candidate, final_reference, final_weights)
    final_aligned_candidate = final_candidate @ final_rotation
    final_residuals = np.linalg.norm(final_aligned_candidate - final_reference, axis=1)
    weight_sum = float(np.sum(base_weights))
    robust_loss = _huber_loss(final_residuals, float(huber_delta))
    robust_residual = float(np.sqrt(2.0 * np.sum(base_weights * robust_loss) / max(weight_sum, 1e-12)))
    raw_rms = _weighted_rms(final_residuals, base_weights)
    visibility_coverage = float(np.count_nonzero(base_weights > 0.0) / base_weights.size)
    effective_weight_share = float(np.sum(final_weights) / max(weight_sum, 1e-12))

    return {
        "schema_version": PROCRUSTES_SHAPE_SCHEMA,
        "available": True,
        "residual": robust_residual,
        "raw_rms_residual": raw_rms,
        "unit": "normalized_shape_distance",
        "landmark_count": int(reference_points.shape[0]),
        "visibility_coverage": visibility_coverage,
        "effective_weight_share": effective_weight_share,
        "huber_delta": float(huber_delta),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
        "iterations": iterations,
        "converged": converged,
        "rotation_matrix": final_rotation.tolist(),
        "partition_diagnostics": _partition_diagnostics(final_reference, final_residuals, final_weights),
        "error": None,
    }
