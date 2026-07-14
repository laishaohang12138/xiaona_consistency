from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class ThreeDdfaExportInputDiscoveryTests(unittest.TestCase):
    def test_jpeg_extension_is_discovered_case_insensitively(self) -> None:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "external_patches"
            / "3DDFA-V3"
            / "demo_lite_export.py"
        )
        face_box_module = types.ModuleType("face_box")
        face_box_module.face_box = object
        model_module = types.ModuleType("model")
        recon_module = types.ModuleType("model.recon")
        recon_module.face_model = object
        with patch.dict(
            sys.modules,
            {
                "face_box": face_box_module,
                "model": model_module,
                "model.recon": recon_module,
            },
        ):
            spec = importlib.util.spec_from_file_location("xiaona_3ddfa_export_test", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("unable to load 3DDFA exporter patch")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            for name in ["a.jpg", "b.JPEG", "c.png", "ignored.txt"]:
                (root / name).write_bytes(b"fixture")

            discovered = [Path(path).name for path in module._get_image_paths(root)]

        self.assertEqual(discovered, ["a.jpg", "b.JPEG", "c.png"])


if __name__ == "__main__":
    unittest.main()
