from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.qa_truth_integrity import validate_truth_integrity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(root: Path, face_hash: str, body_hash: str) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(base_dir=root, config_dir=root),
        provider_policy={"anchor_source": "registry_only"},
        anchor_registry={
            "anchors": {
                "face": {
                    "role": "FACE_MASTER",
                    "anchor_tier": "absolute",
                    "authority": "ABSOLUTE_FROZEN",
                    "mutable": False,
                    "may_modify_truth": False,
                    "required_default": True,
                    "path": "face.png",
                    "expected_sha256": face_hash,
                },
                "body": {
                    "role": "FULL_BODY_MASTER",
                    "anchor_tier": "absolute",
                    "authority": "ABSOLUTE_FROZEN",
                    "mutable": False,
                    "may_modify_truth": False,
                    "required_default": True,
                    "path": "body.png",
                    "expected_sha256": body_hash,
                },
            },
            "rules": {
                "face_truth_anchor": "face",
                "body_truth_anchor": "body",
            },
        },
    )


class TruthIntegrityTests(unittest.TestCase):
    def test_valid_frozen_truths_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            face = root / "face.png"
            body = root / "body.png"
            face.write_bytes(b"face-truth")
            body.write_bytes(b"body-truth")

            result = validate_truth_integrity(_config(root, _sha256(face), _sha256(body)))
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["gpu_initialization_allowed"])

    def test_hash_mismatch_blocks_gpu_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            face = root / "face.png"
            body = root / "body.png"
            face.write_bytes(b"face-truth")
            body.write_bytes(b"body-truth")
            config = _config(root, _sha256(face), _sha256(body))
            face.write_bytes(b"tampered-face-truth")

            result = validate_truth_integrity(config)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gpu_initialization_allowed"])
            self.assertIn("FACE_IDENTITY_SHA256_MISMATCH", result["issues"])

    def test_runtime_does_not_construct_providers_or_engines_after_truth_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            face = root / "face.png"
            body = root / "body.png"
            face.write_bytes(b"face-truth")
            body.write_bytes(b"body-truth")
            config = _config(root, "0" * 64, _sha256(body))

            with (
                patch("core.qa_pipeline.create_runtime_config", return_value=config),
                patch("core.qa_pipeline.build_provider_bundle") as build_providers,
                patch("core.qa_pipeline.init_engines") as init_engines,
            ):
                from core.qa_pipeline import create_runtime

                with self.assertRaisesRegex(RuntimeError, "TRUTH_INTEGRITY_CHECK_FAILED"):
                    create_runtime(root)
                build_providers.assert_not_called()
                init_engines.assert_not_called()

    def test_truth_roles_cannot_be_swapped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            face = root / "face.png"
            body = root / "body.png"
            face.write_bytes(b"face-truth")
            body.write_bytes(b"body-truth")
            config = _config(root, _sha256(face), _sha256(body))
            config.anchor_registry["anchors"]["face"]["role"] = "FULL_BODY_MASTER"

            result = validate_truth_integrity(config)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("FACE_IDENTITY_ROLE_INVALID", result["issues"])


if __name__ == "__main__":
    unittest.main()
