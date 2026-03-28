import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _pose_quality(pose_deg):
    if pose_deg is None or len(pose_deg) < 3:
        return None
    yaw = abs(_safe_float(pose_deg[0], 0.0))
    pitch = abs(_safe_float(pose_deg[1], 0.0))
    # Simple heuristic for shadow evidence only.
    quality = 1.0 - min(1.0, (yaw / 120.0) * 0.7 + (pitch / 80.0) * 0.3)
    return round(max(0.0, quality), 6)


def _to_image_name(path_text):
    normalized = str(path_text).replace("\\", "/")
    return Path(normalized).stem


def main(args):
    args.skip_renderer = True
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    image_paths = get_data_path(args.inputpath)

    for index, image_path in enumerate(image_paths):
        print(index, image_path)
        image = Image.open(image_path).convert("RGB")
        trans_params, im_tensor = facebox_detector(image)
        recon_model.input_img = im_tensor.to(args.device)
        results = recon_model.forward()

        image_name = _to_image_name(image_path)
        save_dir = Path(args.savepath) / image_name
        save_dir.mkdir(parents=True, exist_ok=True)

        ldm106_2d = results.get("ldm106_2d")
        ldm106 = results.get("ldm106")
        ldm68 = results.get("ldm68")
        pose_deg = results.get("pose_euler_deg")
        pose_deg = pose_deg[0].tolist() if isinstance(pose_deg, np.ndarray) and len(pose_deg) > 0 else None

        payload = {
            "schema_version": "face_pose_canonical_artifact_v1",
            "provider_name": "face_pose_canonical_3ddfa",
            "provider_family": "face_canonical_shadow",
            "provider_version": "face_pose_canonical_3ddfa_v1",
            "model_id": "3ddfa_v3_direct_bridge_v1",
            "source_path": str(Path(image_path).resolve()),
            "source_role": "candidate",
            "canonical_landmarks": None,
            "pose_euler_deg": {
                "yaw": _safe_float(pose_deg[0], None) if pose_deg else None,
                "pitch": _safe_float(pose_deg[1], None) if pose_deg else None,
                "roll": _safe_float(pose_deg[2], None) if pose_deg else None,
            },
            "visible_face_coverage": None,
            "frontalization_quality": _pose_quality(pose_deg),
            "pose_fit_confidence": 1.0,
            "notes": "lite 3DDFA-V3 export without renderer",
            "conversion_meta": {
                "export_path": str((save_dir / f"{image_name}.json").resolve()),
                "skip_renderer": True,
                "trans_params": trans_params.tolist() if isinstance(trans_params, np.ndarray) else trans_params,
            },
        }

        if isinstance(ldm106_2d, np.ndarray) and len(ldm106_2d) > 0:
            payload["canonical_landmarks"] = ldm106_2d[0].reshape(-1).tolist()
        elif isinstance(ldm106, np.ndarray) and len(ldm106) > 0:
            payload["canonical_landmarks"] = ldm106[0].reshape(-1).tolist()
        elif isinstance(ldm68, np.ndarray) and len(ldm68) > 0:
            payload["canonical_landmarks"] = ldm68[0].reshape(-1).tolist()

        with open(save_dir / f"{image_name}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DDFA-V3 lite export")
    parser.add_argument("-i", "--inputpath", default="examples/", type=str)
    parser.add_argument("-s", "--savepath", default="examples/results", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--iscrop", default=True, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--detector", default="retinaface", type=str)
    parser.add_argument("--ldm68", default=True, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--ldm106", default=True, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--ldm106_2d", default=True, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--ldm134", default=False, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--seg", default=False, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--seg_visible", default=False, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--useTex", default=False, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--extractTex", default=False, type=lambda x: str(x).lower() in ["true", "1"])
    parser.add_argument("--backbone", default="resnet50", type=str)
    main(parser.parse_args())
