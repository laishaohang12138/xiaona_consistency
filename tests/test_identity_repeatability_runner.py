from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.qa_identity_repeatability_runner import (
    build_detector_chain_diagnostics,
    build_repeatability_trial_plan,
    materialize_repeatability_image,
    run_identity_repeatability_shadow,
)
from core.qa_repeatability_shadow import load_repeatability_protocol


class SyntheticFaceAdapter:
    def __init__(self, *, contract_id: str = "synthetic_v1", fail_once_pattern: str = "") -> None:
        self.contract_id = contract_id
        self.fail_once_pattern = fail_once_pattern
        self.failed = False
        self.calls: list[str] = []

    def describe(self) -> dict:
        return {
            "schema_version": "synthetic_repeatability_adapter_v1",
            "contract_id": self.contract_id,
            "decision_influence": "NONE",
        }

    def measure(self, image_path: Path) -> dict:
        path_text = str(image_path)
        self.calls.append(path_text)
        if self.fail_once_pattern in path_text and self.fail_once_pattern and not self.failed:
            self.failed = True
            raise RuntimeError("synthetic interrupted trial")
        image = cv2.imdecode(np.fromfile(path_text, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("synthetic image decode failed")
        mean = float(np.mean(image)) / 255.0
        contract = {
            "observed_contract_sha256": "a" * 64,
            "comparable_contract_sha256": "a" * 64,
        }
        return {
            "chain_signature": "synthetic_chain_v1",
            "face_identity": {
                "available": True,
                "value": [1.0, mean, 0.25],
                "provider_contract": contract,
            },
            "face_shape": {
                "available": True,
                "value": [[0.0, 0.0], [1.0, 0.0], [0.1 + mean * 0.01, 1.0]],
                "visibility_weights": [1.0, 1.0, 1.0],
                "landmark_schema_id": "synthetic_triangle_v1",
                "provider_contract": contract,
            },
            "errors": [],
            "decision_influence": "NONE",
        }


def _write_source(path: Path) -> None:
    image = np.zeros((48, 36, 3), dtype=np.uint8)
    image[:, :18] = (30, 90, 170)
    image[:, 18:] = (210, 140, 50)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("test fixture encode failed")
    path.write_bytes(encoded.tobytes())


class IdentityRepeatabilityRunnerTests(unittest.TestCase):
    def test_chain_diagnostics_separate_raw_kps_translation_from_shape_change(self) -> None:
        baseline_kps = np.asarray(
            [[0.40, 0.40], [0.60, 0.40], [0.50, 0.50], [0.43, 0.62], [0.57, 0.62]],
            dtype=np.float64,
        )
        shift = np.asarray([0.01, 0.02], dtype=np.float64)
        baseline = {
            "chain_observation": {
                "face_bbox_normalized_xyxy": [0.30, 0.20, 0.70, 0.80],
                "kps5_normalized_xy": baseline_kps.tolist(),
                "canonical_pose_euler_deg": {"yaw": 1.0, "pitch": 2.0, "roll": 3.0},
            }
        }
        trial = {
            "chain_observation": {
                "face_bbox_normalized_xyxy": [0.31, 0.22, 0.71, 0.82],
                "kps5_normalized_xy": (baseline_kps + shift).tolist(),
                "canonical_pose_euler_deg": {"yaw": 2.0, "pitch": 2.0, "roll": 3.0},
            }
        }

        diagnostics = build_detector_chain_diagnostics(baseline, trial)

        self.assertLess(diagnostics["bbox_iou"], 1.0)
        self.assertGreater(diagnostics["bbox_center_displacement_image_fraction"], 0.0)
        self.assertGreater(diagnostics["kps5_raw_rms_image_fraction"], 0.0)
        self.assertLess(diagnostics["kps5_similarity_shape_residual"], 0.000000000001)
        self.assertAlmostEqual(diagnostics["canonical_pose_l2_delta_deg"], 1.0)
        self.assertEqual(diagnostics["decision_influence"], "NONE")

    def test_preregistered_plan_has_three_separate_domains_and_thirteen_trials(self) -> None:
        plan = build_repeatability_trial_plan()

        self.assertEqual(len(plan), 13)
        self.assertEqual(sum(row["domain"] == "numerical_repeatability" for row in plan), 3)
        self.assertEqual(sum(row["domain"] == "preprocessing_repeatability" for row in plan), 4)
        self.assertEqual(sum(row["domain"] == "admissible_perturbation_stability" for row in plan), 6)
        self.assertEqual(len({row["trial_id"] for row in plan}), 13)
        self.assertTrue(all(len(row["trial_spec_sha256"]) == 64 for row in plan))

    def test_materialized_transforms_preserve_dimensions_and_identity_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            protocol = load_repeatability_protocol()
            plan = build_repeatability_trial_plan(protocol)

            for index, spec in enumerate(plan):
                target = materialize_repeatability_image(
                    source,
                    root / f"trial_{index}" / "input",
                    spec,
                    dict(protocol["transform_contract"]),
                )
                image = cv2.imdecode(np.fromfile(str(target), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape[:2], (48, 36))
                if spec["transform_id"] == "identity_transform":
                    self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_completed_run_is_idempotent_and_has_zero_decision_influence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = SyntheticFaceAdapter()
            output_root = root / "runs"

            first = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="idempotent",
            )
            first_call_count = len(adapter.calls)
            second = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="idempotent",
            )

            self.assertEqual(first_call_count, 14)
            self.assertEqual(len(adapter.calls), first_call_count)
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(second["trial_state_counts"], {"COMPLETE": 13})
            self.assertIsNone(second["combined_repeatability_score"])
            self.assertIsNone(second["stable_unstable_classification"])
            self.assertEqual(second["decision_influence"], "NONE")
            self.assertEqual(second["cross_source_descriptors"]["source_count"], 1)
            self.assertIsNone(second["cross_source_descriptors"]["combined_repeatability_score"])
            self.assertNotIn("rank", json.dumps(second))
            self.assertNotIn("winner_bank", json.dumps(second))
            self.assertEqual(list(Path(first["run_dir"]).rglob("input.*")), [])

    def test_failed_trial_requires_explicit_retry_and_only_failed_trial_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = SyntheticFaceAdapter(fail_once_pattern="gamma_positive")
            output_root = root / "runs"

            first = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="retry",
            )
            calls_after_first = len(adapter.calls)
            without_retry = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="retry",
            )
            calls_without_retry = len(adapter.calls)
            self.assertEqual(len(list(Path(first["run_dir"]).rglob("input.*"))), 1)
            with_retry = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="retry",
                retry_failed=True,
            )

            self.assertEqual(first["status"], "PARTIAL")
            self.assertEqual(first["trial_state_counts"]["FAILED"], 1)
            self.assertEqual(calls_after_first, 14)
            self.assertEqual(calls_without_retry, calls_after_first)
            self.assertEqual(without_retry["status"], "PARTIAL")
            self.assertEqual(len(adapter.calls), calls_after_first + 1)
            self.assertEqual(with_retry["status"], "COMPLETE")
            self.assertEqual(with_retry["trial_state_counts"], {"COMPLETE": 13})
            self.assertEqual(list(Path(with_retry["run_dir"]).rglob("input.*")), [])

    def test_failed_baseline_pauses_trials_until_explicit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = SyntheticFaceAdapter(fail_once_pattern="baseline")
            output_root = root / "runs"

            first = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="baseline_retry",
            )
            without_retry = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="baseline_retry",
            )
            with_retry = run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="baseline_retry",
                retry_failed=True,
            )

            self.assertEqual(first["status"], "PARTIAL")
            self.assertEqual(first["baseline_state_counts"], {"FAILED": 1})
            self.assertEqual(first["executed_or_resumed_trial_count"], 0)
            self.assertEqual(without_retry["executed_or_resumed_trial_count"], 0)
            self.assertEqual(with_retry["status"], "COMPLETE")
            self.assertEqual(with_retry["baseline_state_counts"], {"COMPLETE": 1})
            self.assertEqual(with_retry["executed_or_resumed_trial_count"], 13)
            self.assertEqual(len(adapter.calls), 15)

    def test_run_contract_drift_is_rejected_before_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            output_root = root / "runs"
            run_identity_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=SyntheticFaceAdapter(contract_id="adapter_a"),
                run_id="drift",
            )
            changed_adapter = SyntheticFaceAdapter(contract_id="adapter_b")

            with self.assertRaisesRegex(ValueError, "REPEATABILITY_RUN_CONTRACT_MISMATCH"):
                run_identity_repeatability_shadow(
                    image_paths=[source],
                    output_root=output_root,
                    adapter=changed_adapter,
                    run_id="drift",
                )
            self.assertEqual(changed_adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
