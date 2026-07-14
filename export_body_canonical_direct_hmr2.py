from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from core.qa_artifact_manifest import register_artifact_manifest

for _alias, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _value)

if not hasattr(inspect, "getargspec"):
    _ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _compat_getargspec(func):
        spec = inspect.getfullargspec(func)
        return _ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]

if not os.environ.get("HOME"):
    fallback_home = os.environ.get("USERPROFILE") or (
        f"{os.environ.get('HOMEDRIVE', '')}{os.environ.get('HOMEPATH', '')}"
    ).strip()
    if fallback_home:
        os.environ["HOME"] = fallback_home

_EXPECTED_SMPL_NAME = "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
_ALT_SMPL_NAMES = [
    "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
]
_BODY25 = {
    "neck": 1,
    "right_shoulder": 2,
    "left_shoulder": 5,
    "mid_hip": 8,
    "right_hip": 9,
    "right_knee": 10,
    "right_ankle": 11,
    "left_hip": 12,
    "left_knee": 13,
    "left_ankle": 14,
    "left_big_toe": 19,
    "left_small_toe": 20,
    "left_heel": 21,
    "right_big_toe": 22,
    "right_small_toe": 23,
    "right_heel": 24,
}
_MEASUREMENT_SCALES = {
    "shoulder_width_to_torso": 0.08,
    "hip_width_to_torso": 0.08,
    "shoulder_to_hip_ratio": 0.08,
    "leg_length_to_torso": 0.12,
    "upper_to_lower_leg_ratio": 0.08,
    "left_right_leg_balance": 0.05,
    "foot_length_to_leg": 0.04,
    "left_right_foot_balance": 0.04,
}
_ARTIFACT_SCHEMA = "body_canonical_artifact_v3"
_PROVIDER_VERSION = "body_canonical_hmr2_direct_bridge_v4"
_TOPOLOGY_SCHEMA_ID = "smpl_neutral_zero_pose_vertices_v1"
_TOPOLOGY_COORDINATE_CONVENTION = "smpl_neutral_zero_pose_model_space"
_TOPOLOGY_CANONICALIZATION_ID = "smpl_identity_global_and_body_rotations_v1"
_TOPOLOGY_ALIGNMENT_CONTRACT_ID = "centroid_translation_removal_only_v1"
_TOPOLOGY_REPRESENTATION = "dense_smpl_vertices_exact_index_correspondence"
_EXPECTED_SMPL_VERTEX_COUNT = 6890


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_dir() -> Path:
    repo_root = Path(__file__).resolve().parent
    return Path(os.getenv("XIAONA_HMR2_REPO", str(repo_root / "external" / "4D-Humans"))).resolve()


HMR2_REPO = _resolve_repo_dir()
if str(HMR2_REPO) not in sys.path:
    sys.path.insert(0, str(HMR2_REPO))

from hmr2.configs import CACHE_DIR_4DHUMANS, get_config  # type: ignore
from hmr2.datasets.vitdet_dataset import ViTDetDataset  # type: ignore
from hmr2.models import DEFAULT_CHECKPOINT, HMR2, check_smpl_exists, download_models  # type: ignore


def _recursive_to(x: Any, target: torch.device) -> Any:
    if isinstance(x, dict):
        return {k: _recursive_to(v, target) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(target)
    if isinstance(x, list):
        return [_recursive_to(i, target) for i in x]
    return x


def _resolved_device(device_name: str) -> torch.device:
    normalized = str(device_name or "auto").strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_hmr2_no_renderer(checkpoint_path: str):
    model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    current_cwd = Path.cwd()
    try:
        os.chdir(HMR2_REPO)
        check_smpl_exists()
    finally:
        os.chdir(current_cwd)
    original_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load  # type: ignore[assignment]
    try:
        model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]
    return model, model_cfg


def _ensure_smpl_neutral() -> Path:
    repo_root = Path(__file__).resolve().parent
    candidate_dirs = [
        HMR2_REPO / "data",
        repo_root / "data",
    ]
    expected_paths = [directory / _EXPECTED_SMPL_NAME for directory in candidate_dirs]
    for path in expected_paths:
        if path.exists():
            return path

    alt_candidates = []
    for directory in candidate_dirs:
        for name in _ALT_SMPL_NAMES:
            alt_candidates.append(directory / name)
    existing_alt = next((path for path in alt_candidates if path.exists()), None)
    if existing_alt is None:
        raise FileNotFoundError(
            "SMPL neutral model not found. Expected one of: "
            + ", ".join(str(path) for path in expected_paths + alt_candidates)
        )

    target = HMR2_REPO / "data" / _EXPECTED_SMPL_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(existing_alt, target)
    return target


def _single_bbox(image: np.ndarray, margin: float) -> np.ndarray:
    height, width = image.shape[:2]
    margin = max(0.0, min(float(margin), 0.2))
    x0 = width * margin
    y0 = height * margin
    x1 = width * (1.0 - margin)
    y1 = height * (1.0 - margin)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid bbox generated from image size")
    return np.asarray([[x0, y0, x1, y1]], dtype=np.float32)


def _as_list(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().reshape(-1).to(torch.float32).tolist()


def _as_vertex_rows(value: torch.Tensor) -> list[list[float]]:
    vertices = value.detach().cpu().to(torch.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"canonical SMPL vertices must have shape [V, 3], got {tuple(vertices.shape)}")
    return vertices.tolist()


def _zero_pose_canonical_vertices(
    model: Any,
    pred_smpl_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    betas = pred_smpl_params["betas"].reshape(pred_smpl_params["betas"].shape[0], -1).float()
    body_pose = pred_smpl_params["body_pose"].reshape(betas.shape[0], -1, 3, 3)
    identity = torch.eye(3, device=betas.device, dtype=torch.float32).reshape(1, 1, 3, 3)
    global_orient = identity.expand(betas.shape[0], 1, 3, 3).contiguous()
    zero_body_pose = identity.expand(betas.shape[0], body_pose.shape[1], 3, 3).contiguous()
    smpl_output = model.smpl(
        betas=betas,
        global_orient=global_orient,
        body_pose=zero_body_pose,
        pose2rot=False,
    )
    vertices = smpl_output.vertices.reshape(betas.shape[0], -1, 3)
    if vertices.shape[1] != _EXPECTED_SMPL_VERTEX_COUNT:
        raise RuntimeError(
            "zero-pose SMPL vertex count mismatch: "
            f"expected {_EXPECTED_SMPL_VERTEX_COUNT}, got {vertices.shape[1]}"
        )
    return vertices


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(torch.float32).numpy()


def _joint_point(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    index = _BODY25.get(str(name))
    if index is None or keypoints.ndim != 2 or keypoints.shape[0] <= index:
        return None
    point = np.asarray(keypoints[index], dtype=np.float32).reshape(-1)
    if point.shape[0] < 3 or not np.isfinite(point).all():
        return None
    return point[:3]


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-6:
        return None
    return float(float(numerator) / float(denominator))


def _pair_balance(left_value: float | None, right_value: float | None) -> float | None:
    if left_value is None or right_value is None:
        return None
    hi = max(float(left_value), float(right_value))
    lo = min(float(left_value), float(right_value))
    if hi < 1e-6:
        return None
    return float(lo / hi)


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _foot_length(heel: np.ndarray | None, toe_a: np.ndarray | None, toe_b: np.ndarray | None) -> float | None:
    candidates = [_distance(heel, toe_a), _distance(heel, toe_b), _distance(toe_a, toe_b)]
    clean = [float(value) for value in candidates if value is not None]
    if not clean:
        return None
    return float(max(clean))


def _measurement_payload(
    pred_keypoints_3d: torch.Tensor,
    pred_keypoints_2d: torch.Tensor,
    image_bgr: np.ndarray,
) -> tuple[Dict[str, float], Dict[str, float], float, Dict[str, Any]]:
    keypoints_3d = _as_numpy(pred_keypoints_3d)
    keypoints_2d = _as_numpy(pred_keypoints_2d)
    if keypoints_3d.ndim != 2 or keypoints_3d.shape[0] < 25:
        return {}, dict(_MEASUREMENT_SCALES), 0.0, {"measurement_count": 0, "measurement_basis": "hmr2_body25_3d_v2"}

    left_shoulder = _joint_point(keypoints_3d, "left_shoulder")
    right_shoulder = _joint_point(keypoints_3d, "right_shoulder")
    left_hip = _joint_point(keypoints_3d, "left_hip")
    right_hip = _joint_point(keypoints_3d, "right_hip")
    neck = _joint_point(keypoints_3d, "neck")
    mid_hip = _joint_point(keypoints_3d, "mid_hip")
    left_knee = _joint_point(keypoints_3d, "left_knee")
    right_knee = _joint_point(keypoints_3d, "right_knee")
    left_ankle = _joint_point(keypoints_3d, "left_ankle")
    right_ankle = _joint_point(keypoints_3d, "right_ankle")
    left_big_toe = _joint_point(keypoints_3d, "left_big_toe")
    left_small_toe = _joint_point(keypoints_3d, "left_small_toe")
    left_heel = _joint_point(keypoints_3d, "left_heel")
    right_big_toe = _joint_point(keypoints_3d, "right_big_toe")
    right_small_toe = _joint_point(keypoints_3d, "right_small_toe")
    right_heel = _joint_point(keypoints_3d, "right_heel")

    shoulder_width = _distance(left_shoulder, right_shoulder)
    hip_width = _distance(left_hip, right_hip)
    torso_length = _distance(neck, mid_hip)
    left_upper_leg = _distance(left_hip, left_knee)
    right_upper_leg = _distance(right_hip, right_knee)
    left_lower_leg = _distance(left_knee, left_ankle)
    right_lower_leg = _distance(right_knee, right_ankle)
    left_leg_length = None if left_upper_leg is None or left_lower_leg is None else float(left_upper_leg + left_lower_leg)
    right_leg_length = None if right_upper_leg is None or right_lower_leg is None else float(right_upper_leg + right_lower_leg)
    mean_leg_length = _mean([left_leg_length, right_leg_length])
    mean_upper_leg = _mean([left_upper_leg, right_upper_leg])
    mean_lower_leg = _mean([left_lower_leg, right_lower_leg])
    left_foot_length = _foot_length(left_heel, left_big_toe, left_small_toe)
    right_foot_length = _foot_length(right_heel, right_big_toe, right_small_toe)
    mean_foot_length = _mean([left_foot_length, right_foot_length])

    raw_measurements = {
        "shoulder_width_to_torso": _safe_ratio(shoulder_width, torso_length),
        "hip_width_to_torso": _safe_ratio(hip_width, torso_length),
        "shoulder_to_hip_ratio": _safe_ratio(shoulder_width, hip_width),
        "leg_length_to_torso": _safe_ratio(mean_leg_length, torso_length),
        "upper_to_lower_leg_ratio": _safe_ratio(mean_upper_leg, mean_lower_leg),
        "left_right_leg_balance": _pair_balance(left_leg_length, right_leg_length),
        "foot_length_to_leg": _safe_ratio(mean_foot_length, mean_leg_length),
        "left_right_foot_balance": _pair_balance(left_foot_length, right_foot_length),
    }
    measurements = {
        key: round(float(value), 6)
        for key, value in raw_measurements.items()
        if value is not None and np.isfinite(value)
    }

    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    body25_2d = keypoints_2d[:25] if keypoints_2d.ndim == 2 and keypoints_2d.shape[0] >= 25 else np.empty((0, 2), dtype=np.float32)
    valid_2d = body25_2d[np.isfinite(body25_2d).all(axis=1)] if body25_2d.size else np.empty((0, 2), dtype=np.float32)
    image_area = float(max(1, image_bgr.shape[0] * image_bgr.shape[1]))
    keypoint_bbox_coverage = 0.0
    if len(valid_2d) >= 3:
        min_xy = valid_2d.min(axis=0)
        max_xy = valid_2d.max(axis=0)
        bbox_area = float(max(0.0, max_xy[0] - min_xy[0]) * max(0.0, max_xy[1] - min_xy[1]))
        keypoint_bbox_coverage = min(1.0, bbox_area / image_area)

    measurement_coverage = float(len(measurements) / max(1, len(_MEASUREMENT_SCALES)))
    coverage = round(float((measurement_coverage + keypoint_bbox_coverage) / 2.0), 6)
    measurement_meta = {
        "measurement_basis": "hmr2_body25_3d_v2",
        "measurement_count": int(len(measurements)),
        "measurement_expected_count": int(len(_MEASUREMENT_SCALES)),
        "measurement_coverage": round(measurement_coverage, 6),
        "keypoint_bbox_coverage": round(keypoint_bbox_coverage, 6),
        "body25_joint_count": int(min(keypoints_3d.shape[0], 25)),
    }
    return measurements, dict(_MEASUREMENT_SCALES), coverage, measurement_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal 4D-Humans exporter for XiaoNa body canonical artifacts")
    parser.add_argument("--image_path", required=True, help="Input image path")
    parser.add_argument("--output_dir", required=True, help="Directory to store <stem>.body_canonical.json")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HMR2 checkpoint path")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--bbox_margin", type=float, default=0.04, help="Relative margin used for the single-person bbox")
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    smpl_source = _ensure_smpl_neutral()
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = _load_hmr2_no_renderer(args.checkpoint)
    device = _resolved_device(args.device)
    model = model.to(device)
    model.eval()

    boxes = _single_bbox(image_bgr, args.bbox_margin)
    dataset = ViTDetDataset(model_cfg, image_bgr, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch = _recursive_to(batch, device)

    with torch.inference_mode():
        output = model(batch)
        canonical_smpl_vertices = _zero_pose_canonical_vertices(
            model,
            output["pred_smpl_params"],
        )[0]

    pred_smpl_params: Dict[str, torch.Tensor] = output["pred_smpl_params"]
    betas = pred_smpl_params["betas"][0]
    global_orient = pred_smpl_params["global_orient"][0]
    body_pose = pred_smpl_params["body_pose"][0]
    pred_cam = output["pred_cam"][0]
    pred_keypoints_2d = output["pred_keypoints_2d"][0]
    pred_keypoints_3d = output["pred_keypoints_3d"][0]
    bbox = boxes[0]
    measurements, measurement_scales, coverage, measurement_meta = _measurement_payload(
        pred_keypoints_3d,
        pred_keypoints_2d,
        image_bgr,
    )
    all_finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in [
            betas,
            global_orient,
            body_pose,
            pred_cam,
            pred_keypoints_2d,
            pred_keypoints_3d,
            canonical_smpl_vertices,
        ]
    )
    artifact = {
        "schema_version": _ARTIFACT_SCHEMA,
        "provider_name": "body_canonical_hmr2",
        "provider_family": "body_canonical",
        "provider_version": _PROVIDER_VERSION,
        "model_id": "hmr2_direct_bridge_v2",
        "source_path": str(image_path),
        "source_role": "candidate",
        "shape_beta": _as_list(betas),
        "pose_vector": _as_list(torch.cat([global_orient.reshape(-1), body_pose.reshape(-1)], dim=0)),
        "canonical_smpl_vertices": _as_vertex_rows(canonical_smpl_vertices),
        "canonical_measurements": measurements,
        "measurement_scales": measurement_scales,
        "fit_confidence": 1.0 if all_finite else 0.0,
        "coverage": coverage,
        "notes": "direct 4D-Humans export for XiaoNa body canonical bridge",
        "body_canonical_contract": {
            "provider_name": "body_canonical_hmr2",
            "provider_version": _PROVIDER_VERSION,
            "model_id": str(Path(args.checkpoint).name),
            "model_sha256": _sha256_file(Path(args.checkpoint).resolve()),
            "implementation_sha256": _sha256_file(Path(__file__).resolve()),
            "execution_backend": str(device),
            "body_model_id": smpl_source.name,
            "body_model_sha256": _sha256_file(smpl_source),
            "preprocessing_contract_id": (
                f"hmr2_vitdet_single_bbox_margin_{float(args.bbox_margin):.6f}_v1"
            ),
            "measurement_schema_id": str(
                measurement_meta.get("measurement_basis") or "hmr2_body25_3d_v2"
            ),
            "measurement_order": list(_MEASUREMENT_SCALES),
            "shape_dimension": int(betas.numel()),
            "topology_schema_id": _TOPOLOGY_SCHEMA_ID,
            "topology_dimension": int(canonical_smpl_vertices.numel()),
            "topology_vertex_count": int(canonical_smpl_vertices.shape[0]),
            "topology_representation": _TOPOLOGY_REPRESENTATION,
            "topology_coordinate_convention": _TOPOLOGY_COORDINATE_CONVENTION,
            "canonicalization_contract_id": _TOPOLOGY_CANONICALIZATION_ID,
            "topology_alignment_contract_id": _TOPOLOGY_ALIGNMENT_CONTRACT_ID,
            "coordinate_convention": "hmr2_camera_relative_body25_3d",
            "source_field": "canonical_measurements",
            "topology_source_field": "canonical_smpl_vertices",
        },
        "conversion_meta": {
            "repo_dir": str(HMR2_REPO),
            "smpl_source": str(smpl_source),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "bbox_margin": float(args.bbox_margin),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "pred_cam": _as_list(pred_cam),
            "image_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            **measurement_meta,
        },
    }
    output_path = output_dir / f"{image_path.stem}.body_canonical.json"
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    register_artifact_manifest(
        artifact_path=output_path.resolve(),
        artifact_family="body_canonical",
        artifact_role=str(artifact.get("source_role") or "candidate"),
        provider_name=str(artifact.get("provider_name") or "body_canonical_hmr2"),
        provider_family=str(artifact.get("provider_family") or "body_canonical"),
        provider_version=str(artifact.get("provider_version") or "hmr2_xiaona_export_v1"),
        model_id=str(artifact.get("model_id") or Path(args.checkpoint).name),
        schema_version=str(artifact.get("schema_version") or _ARTIFACT_SCHEMA),
        source_path=image_path,
        device=str(device),
        repo_dir=HMR2_REPO,
        entrypoint=str(Path(__file__).resolve()),
        conversion_meta=dict(artifact.get("conversion_meta") or {}),
        extra={"notes": artifact.get("notes")},
    )
    print(json.dumps({"ok": True, "output_path": str(output_path), "device": str(device), "repo_dir": str(HMR2_REPO)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
