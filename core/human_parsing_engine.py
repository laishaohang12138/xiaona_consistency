from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import importlib.util
import os

import numpy as np

DEFAULT_HUMAN_PARSING_MODEL = "mattmdjaga/segformer_b2_clothes"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


TRANSFORMERS_AVAILABLE = _module_available("transformers")
TORCH_AVAILABLE = _module_available("torch")
PILLOW_AVAILABLE = _module_available("PIL")


def normalize_label_name(label: str) -> str:
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.replace("/", "_").replace(".", "_")
    alias_map = {
        "bg": "background",
        "upperclothes": "upper_clothes",
        "upper_clothes": "upper_clothes",
        "upper_cloth": "upper_clothes",
        "top": "upper_clothes",
        "coat": "upper_clothes",
        "hair": "hair",
        "face": "face",
        "leftarm": "left_arm",
        "left_arm": "left_arm",
        "rightarm": "right_arm",
        "right_arm": "right_arm",
        "leftleg": "left_leg",
        "left_leg": "left_leg",
        "rightleg": "right_leg",
        "right_leg": "right_leg",
        "pants": "pants",
        "trousers": "pants",
        "skirt": "skirt",
        "dress": "dress",
        "leftshoe": "left_shoe",
        "left_shoe": "left_shoe",
        "rightshoe": "right_shoe",
        "right_shoe": "right_shoe",
        "skin": "skin",
    }
    return alias_map.get(key, key)


@dataclass
class HumanParsingOutput:
    label_map: np.ndarray
    masks: Dict[str, np.ndarray]
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    model_name: str


class HumanParsingEngine:
    def __init__(self, model_name: str = DEFAULT_HUMAN_PARSING_MODEL, device: str = "auto") -> None:
        self.model_name = str(model_name)
        self.device_preference = str(device)
        self._processor = None
        self._model = None
        self._torch = None
        self._Image = None
        self._device = None
        self._load_source = None

    def is_available(self) -> bool:
        return TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and PILLOW_AVAILABLE

    def unavailable_reason(self) -> str:
        missing = []
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not PILLOW_AVAILABLE:
            missing.append("Pillow")
        return "missing dependencies: " + ", ".join(missing) if missing else ""

    def _resolve_device(self, torch_module) -> str:
        if self.device_preference != "auto":
            return self.device_preference
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _local_snapshot_path(self) -> Optional[Path]:
        candidate = Path(self.model_name)
        if candidate.exists():
            return candidate.resolve()

        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        repo_dir = hf_home / "hub" / f"models--{self.model_name.replace('/', '--')}"
        if not repo_dir.exists():
            return None

        ref_main = repo_dir / "refs" / "main"
        if ref_main.exists():
            try:
                revision = ref_main.read_text(encoding="utf-8").strip()
                snapshot_dir = repo_dir / "snapshots" / revision
                if snapshot_dir.exists():
                    return snapshot_dir.resolve()
            except Exception:
                pass

        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), reverse=True)
        return snapshots[0].resolve() if snapshots else None

    def _load_processor(self, source: str, local_files_only: bool):
        from transformers import AutoImageProcessor
        try:
            return AutoImageProcessor.from_pretrained(source, local_files_only=local_files_only)
        except Exception:
            from transformers import SegformerImageProcessor
            return SegformerImageProcessor.from_pretrained(source, local_files_only=local_files_only)

    def ensure_ready(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())

        import torch
        from PIL import Image
        try:
            from transformers import SegformerForSemanticSegmentation as ModelClass
        except Exception:
            from transformers import AutoModelForSemanticSegmentation as ModelClass

        self._torch = torch
        self._Image = Image
        self._device = torch.device(self._resolve_device(torch))
        local_snapshot = self._local_snapshot_path()
        load_attempts = []
        if local_snapshot is not None:
            load_attempts.append((str(local_snapshot), True, "local_snapshot"))
        load_attempts.append((self.model_name, True, "hub_cache"))
        load_attempts.append((self.model_name, False, "remote"))

        errors = []
        for source, local_files_only, source_name in load_attempts:
            try:
                self._processor = self._load_processor(source, local_files_only=local_files_only)
                self._model = ModelClass.from_pretrained(source, local_files_only=local_files_only)
                self._load_source = source_name
                break
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")
                self._processor = None
                self._model = None

        if self._processor is None or self._model is None:
            raise RuntimeError(" | ".join(errors))

        self._model.to(self._device)
        self._model.eval()

    def parse(self, img_bgr: np.ndarray) -> HumanParsingOutput:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("img_bgr is empty")

        self.ensure_ready()

        h, w = img_bgr.shape[:2]
        pil_image = self._Image.fromarray(img_bgr[:, :, ::-1])
        inputs = self._processor(images=pil_image, return_tensors="pt")
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            upsampled_logits = self._torch.nn.functional.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            label_map = (
                upsampled_logits.argmax(dim=1)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )

        raw_id2label = getattr(self._model.config, "id2label", {}) or {}
        id2label: Dict[int, str] = {}
        label2id: Dict[str, int] = {}
        for raw_idx, raw_label in raw_id2label.items():
            idx = int(raw_idx)
            label_name = normalize_label_name(str(raw_label))
            id2label[idx] = label_name
            label2id[label_name] = idx

        masks: Dict[str, np.ndarray] = {}
        for label_id, label_name in id2label.items():
            cur_mask = (label_map == label_id).astype(np.uint8) * 255
            if not np.any(cur_mask):
                continue
            if label_name in masks:
                masks[label_name] = np.maximum(masks[label_name], cur_mask)
            else:
                masks[label_name] = cur_mask

        return HumanParsingOutput(
            label_map=label_map,
            masks=masks,
            id2label=id2label,
            label2id=label2id,
            model_name=self.model_name,
        )
