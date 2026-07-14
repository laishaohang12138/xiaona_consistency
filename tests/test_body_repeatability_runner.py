from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.qa_body_repeatability_runner import (
    build_body_chain_diagnostics,
    build_body_native_repeatability_residual,
    run_body_repeatability_shadow,
)
from core.qa_body_repeatability_adapter import BodyCanonicalRepeatabilityAdapter
from core.qa_identity_repeatability_runner import build_repeatability_trial_plan
from core.qa_repeatability_shadow import (
    BODY_REPEATABILITY_PROTOCOL_ID,
    body_repeatability_protocol_snapshot,
    empty_body_repeatability_contract,
    load_body_repeatability_protocol,
)


class SyntheticBodyAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def describe(self) -> dict:
        return {
            "schema_version": "synthetic_body_repeatability_adapter_v1",
            "measurement_order": [
                "shoulder_width_to_torso",
                "hip_width_to_torso",
                "shoulder_to_hip_ratio",
                "upper_to_lower_leg_ratio",
                "foot_length_to_leg",
            ],
            "axes": ["body_core_shape", "body_topology"],
            "decision_influence": "NONE",
        }

    def measure(self, image_path: Path) -> dict:
        self.calls.append(str(image_path))
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("synthetic image decode failed")
        mean = float(np.mean(image)) / 255.0
        contract = {
            "observed_contract_sha256": "a" * 64,
            "comparable_contract_sha256": "a" * 64,
        }
        measurements = {
            "shoulder_width_to_torso": 0.50 + mean * 0.01,
            "hip_width_to_torso": 0.40 + mean * 0.01,
            "shoulder_to_hip_ratio": 1.25 + mean * 0.01,
            "upper_to_lower_leg_ratio": 1.00 + mean * 0.01,
            "foot_length_to_leg": 0.20 + mean * 0.01,
        }
        topology_vertices = _topology_vertices(scale=1.0 + mean * 0.001)
        return {
            "chain_signature": "synthetic_body_chain_v1",
            "chain_observation": {
                "body_bbox_normalized_xyxy": [0.2, 0.1, 0.8, 0.9],
                "body_canonical_coverage": 0.9,
                "body_fit_confidence": 1.0,
                "pose_vector": [0.0, mean],
            },
            "body_core_shape": {
                "available": True,
                "value": measurements,
                "provider_contract": contract,
                "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
            },
            "body_topology": {
                "available": True,
                "value": topology_vertices,
                "vertex_count": 6890,
                "coordinate_count": 20670,
                "coordinate_axis_order": ["x", "y", "z"],
                "provider_contract": contract,
                "prior_dependence": "HMR2_SMPL_RECONSTRUCTION",
            },
            "errors": [],
            "decision_influence": "NONE",
        }


class UnavailableBodyAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def describe(self) -> dict:
        return {
            "schema_version": "synthetic_unavailable_body_adapter_v1",
            "decision_influence": "NONE",
        }

    def measure(self, image_path: Path) -> dict:
        self.calls += 1
        return {
            "chain_signature": "body_unavailable",
            "body_core_shape": {
                "available": False,
                "value": {},
                "provider_contract": {},
                "errors": ["BODY_CANONICAL_UNAVAILABLE"],
            },
            "errors": ["BODY_CANONICAL_UNAVAILABLE"],
            "decision_influence": "NONE",
        }


class CoreOnlySyntheticBodyAdapter(SyntheticBodyAdapter):
    def measure(self, image_path: Path) -> dict:
        observation = super().measure(image_path)
        observation.pop("body_topology", None)
        observation["chain_observation"]["native_topology_available"] = False
        observation["chain_observation"]["canonical_smpl_vertex_count"] = None
        return observation


def _write_source(path: Path) -> None:
    image = np.zeros((48, 36, 3), dtype=np.uint8)
    image[:, :18] = (30, 90, 170)
    image[:, 18:] = (210, 140, 50)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("test fixture encode failed")
    path.write_bytes(encoded.tobytes())


def _topology_vertices(
    *,
    scale: float = 1.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    return [
        [
            float(index % 37) / 36.0 * scale + translation[0],
            float((index // 37) % 31) / 30.0 * scale + translation[1],
            float(index // (37 * 31)) / 6.0 * scale + translation[2],
        ]
        for index in range(6890)
    ]


def _observation(*, scale: float = 1.0, contract_sha: str = "a") -> dict:
    return {
        "body_core_shape": {
            "available": True,
            "value": {
                "shoulder_width_to_torso": 0.5 * scale,
                "hip_width_to_torso": 0.4 * scale,
                "shoulder_to_hip_ratio": 1.25 * scale,
                "upper_to_lower_leg_ratio": 1.0 * scale,
                "foot_length_to_leg": 0.2 * scale,
            },
            "provider_contract": {
                "comparable_contract_sha256": contract_sha * 64,
            },
        }
    }


def _topology_observation(
    *,
    scale: float = 1.0,
    contract_sha: str = "a",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return {
        "body_topology": {
            "available": True,
            "value": _topology_vertices(scale=scale, translation=translation),
            "provider_contract": {
                "comparable_contract_sha256": contract_sha * 64,
            },
        }
    }


class BodyRepeatabilityRunnerTests(unittest.TestCase):
    def test_body_adapter_reads_hmr2_artifact_and_exposes_chain_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            artifact_path = root / "candidate.body_canonical.json"
            _write_source(source)
            artifact_path.write_text(
                json.dumps(
                    {
                        "shape_beta": [0.1, 0.2, 0.3],
                        "canonical_smpl_vertices": _topology_vertices(),
                        "pose_vector": [0.0, 0.1],
                        "canonical_measurements": _observation()["body_core_shape"]["value"],
                        "fit_confidence": 0.9,
                        "coverage": 0.8,
                        "conversion_meta": {"bbox_xyxy": [3.6, 4.8, 32.4, 43.2]},
                        "body_canonical_contract": {
                            "provider_name": "body_canonical_hmr2",
                            "provider_version": "test",
                            "model_id": "hmr2.ckpt",
                            "model_sha256": "a" * 64,
                            "implementation_sha256": "b" * 64,
                            "execution_backend": "cuda",
                            "body_model_id": "SMPL_NEUTRAL.pkl",
                            "body_model_sha256": "c" * 64,
                            "preprocessing_contract_id": "preprocess_v1",
                            "measurement_schema_id": "body25_v2",
                            "measurement_order": list(_observation()["body_core_shape"]["value"]),
                            "shape_dimension": 3,
                            "coordinate_convention": "camera_relative",
                            "source_field": "canonical_measurements",
                            "topology_schema_id": "smpl_neutral_zero_pose_vertices_v1",
                            "topology_coordinate_convention": "smpl_neutral_zero_pose_model_space",
                            "canonicalization_contract_id": "smpl_identity_global_and_body_rotations_v1",
                            "topology_alignment_contract_id": "centroid_translation_removal_only_v1",
                            "topology_representation": "dense_smpl_vertices_exact_index_correspondence",
                            "topology_vertex_count": 6890,
                            "topology_dimension": 20670,
                            "topology_source_field": "canonical_smpl_vertices",
                        },
                    }
                ),
                encoding="utf-8",
            )

            class FakeProviders:
                @staticmethod
                def describe_heavy_evidence() -> dict:
                    return {"provider_name": "body_canonical_hmr2", "integration_ready": True}

                @staticmethod
                def get_heavy_evidence(runtime: object, image_path: Path) -> dict:
                    return {
                        "provider_name": "body_canonical_hmr2",
                        "provider_version": "test",
                        "summary": {"candidate_artifact_path": str(artifact_path)},
                    }

            adapter = BodyCanonicalRepeatabilityAdapter(
                SimpleNamespace(providers=FakeProviders()),
                execution_context={"device": "cuda"},
            )
            observation = adapter.measure(source)

            self.assertTrue(observation["body_core_shape"]["available"])
            self.assertTrue(observation["body_topology"]["available"])
            self.assertEqual(observation["body_topology"]["vertex_count"], 6890)
            self.assertEqual(
                observation["body_core_shape"]["provider_contract"]["completeness_state"],
                "COMPLETE",
            )
            for actual, expected in zip(
                observation["chain_observation"]["body_bbox_normalized_xyxy"],
                [0.1, 0.1, 0.9, 0.9],
            ):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(observation["decision_influence"], "NONE")

    def test_body_protocol_is_separate_preregistered_thirteen_trial_contract(self) -> None:
        snapshot = body_repeatability_protocol_snapshot()
        protocol = load_body_repeatability_protocol()
        plan = build_repeatability_trial_plan(protocol)
        empty = empty_body_repeatability_contract()

        self.assertEqual(snapshot["validation_status"], "VALID")
        self.assertEqual(protocol["protocol_id"], BODY_REPEATABILITY_PROTOCOL_ID)
        self.assertEqual(len(plan), 13)
        self.assertTrue(protocol["execution"]["stop_on_failed_trial"])
        self.assertGreaterEqual(protocol["execution"]["inter_execution_cooldown_seconds"], 0.0)
        self.assertEqual(empty["protocol_id"], BODY_REPEATABILITY_PROTOCOL_ID)
        self.assertFalse(empty["component_aggregation_allowed"])
        self.assertIsNone(empty["combined_repeatability_score"])
        self.assertEqual(set(protocol["measurement_axes"]), {"body_core_shape", "body_topology"})
        self.assertTrue(
            protocol["reporting"]["topology"]["raw_residual_vector_retained_per_trial"]
        )
        self.assertFalse(
            protocol["reporting"]["topology"]["vertex_norm_aggregation_allowed"]
        )

    def test_native_residual_preserves_components_and_withholds_contract_mismatch(self) -> None:
        result = build_body_native_repeatability_residual(
            _observation(),
            _observation(scale=1.1),
            "body_core_shape",
        )
        mismatch = build_body_native_repeatability_residual(
            _observation(contract_sha="a"),
            _observation(scale=1.1, contract_sha="b"),
            "body_core_shape",
        )

        self.assertTrue(result["available"])
        self.assertIsNone(result["residual"])
        self.assertEqual(len(result["residual_vector"]), 5)
        self.assertFalse(result["scalar_residual_authorized"])
        self.assertFalse(mismatch["available"])
        self.assertIsNone(mismatch["residual_vector"])
        self.assertIn("MEASUREMENT_PROVIDER_CONTRACT_MISMATCH", mismatch["errors"])

    def test_topology_repeatability_residual_is_translation_invariant_and_vector_only(self) -> None:
        translated = build_body_native_repeatability_residual(
            _topology_observation(),
            _topology_observation(translation=(5.0, -2.0, 1.0)),
            "body_topology",
        )
        scaled = build_body_native_repeatability_residual(
            _topology_observation(),
            _topology_observation(scale=1.01),
            "body_topology",
        )
        mismatch = build_body_native_repeatability_residual(
            _topology_observation(contract_sha="a"),
            _topology_observation(scale=1.01, contract_sha="b"),
            "body_topology",
        )

        self.assertTrue(translated["available"])
        self.assertEqual(len(translated["residual_vector"]), 20670)
        self.assertTrue(
            all(abs(value) < 0.0000001 for value in translated["residual_vector"])
        )
        self.assertTrue(scaled["available"])
        self.assertTrue(any(abs(value) > 0.0001 for value in scaled["residual_vector"]))
        self.assertIsNone(scaled["residual"])
        self.assertFalse(scaled["scalar_residual_authorized"])
        self.assertFalse(scaled["alignment_contract"]["scale_fit_applied"])
        self.assertFalse(mismatch["available"])
        self.assertIn("MEASUREMENT_PROVIDER_CONTRACT_MISMATCH", mismatch["errors"])

        truncated = {
            "body_topology": {
                "available": True,
                "value": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "provider_contract": {"comparable_contract_sha256": "a" * 64},
            }
        }
        truncated_result = build_body_native_repeatability_residual(
            truncated,
            truncated,
            "body_topology",
        )
        self.assertFalse(truncated_result["available"])
        self.assertIn(
            "BODY_TOPOLOGY_VERTEX_CONTRACT_MISMATCH",
            truncated_result["errors"],
        )

    def test_body_chain_diagnostics_separate_bbox_pose_and_quality(self) -> None:
        baseline = {
            "chain_observation": {
                "body_bbox_normalized_xyxy": [0.2, 0.1, 0.8, 0.9],
                "body_canonical_coverage": 0.8,
                "body_fit_confidence": 0.9,
                "pose_vector": [0.0, 0.1],
            }
        }
        trial = {
            "chain_observation": {
                "body_bbox_normalized_xyxy": [0.21, 0.11, 0.81, 0.91],
                "body_canonical_coverage": 0.7,
                "body_fit_confidence": 0.8,
                "pose_vector": [0.1, 0.1],
            }
        }

        diagnostics = build_body_chain_diagnostics(baseline, trial)

        self.assertLess(diagnostics["bbox_iou"], 1.0)
        self.assertGreater(diagnostics["bbox_center_displacement_image_fraction"], 0.0)
        self.assertAlmostEqual(diagnostics["body_canonical_coverage_delta"], -0.1)
        self.assertAlmostEqual(diagnostics["body_fit_confidence_delta"], -0.1)
        self.assertGreater(diagnostics["pose_parameter_rms_delta"], 0.0)
        self.assertEqual(diagnostics["decision_influence"], "NONE")

    def test_completed_body_run_is_resumable_and_componentwise_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = SyntheticBodyAdapter()
            output_root = root / "runs"

            first = run_body_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="body_idempotent",
            )
            first_call_count = len(adapter.calls)
            second = run_body_repeatability_shadow(
                image_paths=[source],
                output_root=output_root,
                adapter=adapter,
                run_id="body_idempotent",
            )

            self.assertEqual(first_call_count, 14)
            self.assertEqual(len(adapter.calls), first_call_count)
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(second["trial_state_counts"], {"COMPLETE": 13})
            self.assertTrue(
                second["axis_availability"]["body_core_shape"]["fully_observed"]
            )
            self.assertTrue(
                second["axis_availability"]["body_topology"]["fully_observed"]
            )
            body_axis = second["items"][0]["axes"]["body_core_shape"]
            self.assertFalse(body_axis["component_aggregation_allowed"])
            self.assertEqual(len(body_axis["components"]), 5)
            shoulder = body_axis["components"]["shoulder_width_to_torso"]
            self.assertEqual(set(shoulder), {
                "numerical_repeatability",
                "preprocessing_repeatability",
                "admissible_perturbation_stability",
            })
            self.assertIn(
                "native_absolute_residual_descriptor",
                shoulder["preprocessing_repeatability"],
            )
            self.assertIsNone(second["combined_repeatability_score"])
            self.assertIsNone(second["stable_unstable_classification"])
            self.assertFalse(
                second["cross_source_descriptors"]["component_aggregation_allowed"]
            )
            topology_axis = second["items"][0]["axes"]["body_topology"]
            self.assertTrue(topology_axis["baseline_available"])
            self.assertFalse(topology_axis["coordinate_axis_aggregation_allowed"])
            self.assertFalse(topology_axis["vertex_norm_aggregation_allowed"])
            topology_preprocessing = topology_axis["domains"]["preprocessing_repeatability"]
            self.assertGreater(topology_preprocessing["available_residual_count"], 0)
            self.assertIn("q95", topology_preprocessing["coordinate_axes"]["x"]["absolute_quantiles"])
            self.assertIsNone(topology_preprocessing["combined_topology_score"])
            self.assertGreater(
                topology_preprocessing["detector_chain_assessed_count"],
                0,
            )
            self.assertIn(
                "bbox_iou",
                topology_preprocessing["chain_diagnostic_descriptors"],
            )
            self.assertIn(
                "body_topology",
                second["cross_source_descriptors"]["axes"],
            )
            trial_files = [
                path
                for path in Path(first["run_dir"]).rglob("result.json")
                if "trials" in path.parts
            ]
            trial_payload = json.loads(trial_files[0].read_text(encoding="utf-8"))
            self.assertEqual(
                len(trial_payload["residuals"]["body_topology"]["residual_vector"]),
                20670,
            )
            self.assertNotIn("winner_bank", json.dumps(second))
            self.assertEqual(list(Path(first["run_dir"]).rglob("input.*")), [])

    def test_complete_execution_with_missing_topology_is_explicitly_partial_by_axis(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = CoreOnlySyntheticBodyAdapter()

            result = run_body_repeatability_shadow(
                image_paths=[source],
                output_root=root / "runs",
                adapter=adapter,
                run_id="body_core_only",
            )

            self.assertEqual(len(adapter.calls), 14)
            self.assertEqual(result["status"], "COMPLETE_WITH_UNAVAILABLE_MEASUREMENTS")
            self.assertTrue(
                result["axis_availability"]["body_core_shape"]["fully_observed"]
            )
            self.assertFalse(
                result["axis_availability"]["body_topology"]["fully_observed"]
            )
            self.assertEqual(
                result["axis_availability"]["body_topology"]["available_trial_count"],
                0,
            )

    def test_unavailable_body_baseline_does_not_launch_thirteen_heavy_trials(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            source = root / "source.png"
            _write_source(source)
            adapter = UnavailableBodyAdapter()

            result = run_body_repeatability_shadow(
                image_paths=[source],
                output_root=root / "runs",
                adapter=adapter,
                run_id="body_unavailable",
            )

            self.assertEqual(adapter.calls, 1)
            self.assertEqual(result["status"], "COMPLETE_WITH_UNAVAILABLE_MEASUREMENTS")
            self.assertEqual(result["baseline_state_counts"], {"MEASUREMENT_UNAVAILABLE": 1})
            self.assertEqual(result["planned_trial_count"], 13)
            self.assertEqual(result["executed_or_resumed_trial_count"], 0)


if __name__ == "__main__":
    unittest.main()
