from __future__ import annotations

import argparse
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

    pred_smpl_params: Dict[str, torch.Tensor] = output["pred_smpl_params"]
    betas = pred_smpl_params["betas"][0]
    global_orient = pred_smpl_params["global_orient"][0]
    body_pose = pred_smpl_params["body_pose"][0]
    pred_cam = output["pred_cam"][0]
    bbox = boxes[0]
    bbox_area_ratio = float(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1.0, image_bgr.shape[0] * image_bgr.shape[1]))
    all_finite = all(
        torch.isfinite(tensor).all().item()
        for tensor in [betas, global_orient, body_pose, pred_cam]
    )
    artifact = {
        "schema_version": "body_canonical_artifact_v1",
        "provider_name": "body_canonical_hmr2",
        "provider_family": "body_canonical",
        "provider_version": "hmr2_xiaona_export_v1",
        "model_id": str(Path(args.checkpoint).name),
        "source_path": str(image_path),
        "source_role": "candidate",
        "shape_beta": _as_list(betas),
        "pose_vector": _as_list(torch.cat([global_orient.reshape(-1), body_pose.reshape(-1)], dim=0)),
        "canonical_measurements": {},
        "measurement_scales": {},
        "fit_confidence": 1.0 if all_finite else 0.0,
        "coverage": round(bbox_area_ratio, 6),
        "notes": "direct 4D-Humans export for XiaoNa body canonical bridge",
        "conversion_meta": {
            "repo_dir": str(HMR2_REPO),
            "smpl_source": str(smpl_source),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "bbox_margin": float(args.bbox_margin),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "pred_cam": _as_list(pred_cam),
            "image_size": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
        },
    }
    output_path = output_dir / f"{image_path.stem}.body_canonical.json"
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "output_path": str(output_path), "device": str(device), "repo_dir": str(HMR2_REPO)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
