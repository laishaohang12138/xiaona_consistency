from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.qa_body_evidence_shadow import (
    build_body_evidence_shadow,
    write_body_evidence_shadow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_contract(*, backend: str = "cuda", native_topology: bool = False) -> dict:
    contract = {
        "provider_name": "body_canonical_hmr2",
        "provider_version": "test_v1",
        "model_id": "hmr2.ckpt",
        "model_sha256": "a" * 64,
        "implementation_sha256": "b" * 64,
        "execution_backend": backend,
        "body_model_id": "SMPL_NEUTRAL.pkl",
        "body_model_sha256": "c" * 64,
        "preprocessing_contract_id": "hmr2_test_preprocess_v1",
        "measurement_schema_id": "hmr2_body25_3d_v2",
        "measurement_order": [
            "shoulder_width_to_torso",
            "hip_width_to_torso",
            "shoulder_to_hip_ratio",
            "leg_length_to_torso",
            "upper_to_lower_leg_ratio",
            "left_right_leg_balance",
            "foot_length_to_leg",
            "left_right_foot_balance",
        ],
        "shape_dimension": 3,
        "coordinate_convention": "hmr2_camera_relative_body25_3d",
        "source_field": "canonical_measurements",
    }
    if native_topology:
        contract.update(
            {
                "topology_schema_id": "smpl_neutral_zero_pose_vertices_v1",
                "topology_dimension": 20670,
                "topology_vertex_count": 6890,
                "topology_representation": "dense_smpl_vertices_exact_index_correspondence",
                "topology_coordinate_convention": "smpl_neutral_zero_pose_model_space",
                "canonicalization_contract_id": "smpl_identity_global_and_body_rotations_v1",
                "topology_alignment_contract_id": "centroid_translation_removal_only_v1",
                "topology_source_field": "canonical_smpl_vertices",
            }
        )
    else:
        contract.update(
            {
                "topology_schema_id": "hmr2_shape_beta_body25_signature_v1",
                "topology_dimension": 4,
                "topology_source_field": "body_topology_signature",
            }
        )
    return contract


def _artifact(
    *,
    backend: str = "cuda",
    scale: float = 1.0,
    native_topology: bool = False,
) -> dict:
    artifact = {
        "schema_version": "body_canonical_artifact_v3",
        "provider_name": "body_canonical_hmr2",
        "provider_version": "test_v1",
        "model_id": "hmr2_direct_bridge_v2",
        "shape_beta": [0.1, 0.2, 0.3],
        "body_topology_signature": [0.1, 0.2, 0.3, 0.4],
        "pose_vector": [0.0, 0.1, 0.2],
        "canonical_measurements": {
            "shoulder_width_to_torso": 0.50 * scale,
            "hip_width_to_torso": 0.40 * scale,
            "shoulder_to_hip_ratio": 1.25 * scale,
            "upper_to_lower_leg_ratio": 1.00 * scale,
            "foot_length_to_leg": 0.20 * scale,
        },
        "fit_confidence": 0.9,
        "coverage": 0.8,
        "body_canonical_contract": _complete_contract(
            backend=backend,
            native_topology=native_topology,
        ),
    }
    if native_topology:
        artifact["canonical_smpl_vertices"] = [
            [
                float(index % 37) * scale,
                float((index // 37) % 31) * scale,
                float(index // (37 * 31)) * scale,
            ]
            for index in range(6890)
        ]
    return artifact


class BodyEvidenceShadowTests(unittest.TestCase):
    def _runtime_and_item(
        self,
        root: Path,
        *,
        candidate_backend: str = "cuda",
        native_topology: bool = False,
    ) -> tuple[SimpleNamespace, dict]:
        output = root / "outputs"
        output.mkdir()
        body_truth = root / "Task-63987060-116-1.png"
        body_truth.write_bytes(b"body-truth")
        master_artifact = root / "body_master.json"
        candidate_artifact = root / "body_candidate.json"
        master_artifact.write_text(
            json.dumps(_artifact(native_topology=native_topology)),
            encoding="utf-8",
        )
        candidate_artifact.write_text(
            json.dumps(
                _artifact(
                    backend=candidate_backend,
                    scale=1.1,
                    native_topology=native_topology,
                )
            ),
            encoding="utf-8",
        )
        config = SimpleNamespace(
            paths=SimpleNamespace(base_dir=root, config_dir=root, dir_output=output),
            provider_policy={"anchor_source": "registry_only"},
            anchor_registry={
                "anchors": {
                    "body": {
                        "role": "FULL_BODY_MASTER",
                        "anchor_tier": "absolute",
                        "authority": "ABSOLUTE_FROZEN",
                        "mutable": False,
                        "may_modify_truth": False,
                        "path": body_truth.name,
                        "expected_sha256": _sha256(body_truth),
                    }
                },
                "rules": {"body_truth_anchor": "body"},
            },
        )
        item = {
            "image": "candidate.png",
            "selection_score": 0.99,
            "review_only_status_v2": "PASS",
            "review_only_breakdown_v2": {"observed_lane_family": "front"},
            "debug": {
                "source_path": str(root / "candidate.png"),
                "heavy_evidence": {
                    "provider_name": "segformer_body_truth_fusion",
                    "provider_version": "test_fusion",
                    "metrics": [
                        {"metric_name": "visible_body_ratio", "metric_value": 0.75},
                        {"metric_name": "garment_coverage_ratio", "metric_value": 0.60},
                    ],
                    "summary": {
                        "body_canonical_summary": {
                            "master_artifact_path": str(master_artifact),
                            "candidate_artifact_path": str(candidate_artifact),
                        }
                    },
                },
            },
        }
        return SimpleNamespace(config=config), item

    def test_shadow_body_measurement_is_vector_only_and_does_not_mutate_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            original = copy.deepcopy(item)
            payload = build_body_evidence_shadow(runtime, [item])

            self.assertEqual(item, original)
            self.assertEqual(payload["summary"]["validation_status"], "VALID")
            self.assertEqual(payload["governance"]["decision_influence"], "NONE")
            self.assertFalse(payload["truth_registry"]["pose_gait_creates_new_anchor"])
            body_axis = payload["items"][0]["axes"]["body_core_shape"]
            self.assertEqual(body_axis["observation"]["eligibility"], "PRIOR_DEPENDENT")
            self.assertTrue(body_axis["measurement"]["available"])
            self.assertIsNone(body_axis["measurement"]["residual"])
            self.assertEqual(len(body_axis["measurement"]["residual_vector"]), 5)
            self.assertEqual(
                body_axis["measurement"]["provider_contracts"]["comparison"]["comparison_state"],
                "MATCH",
            )
            self.assertEqual(
                payload["items"][0]["axes"]["body_topology"]["implementation_state"],
                "NATIVE_ZERO_POSE_VERTEX_RESIDUAL_SHADOW",
            )
            topology_readiness = payload["items"][0]["axes"]["body_topology"][
                "native_measurement_readiness"
            ]
            self.assertEqual(topology_readiness["readiness_state"], "BLOCKED")
            self.assertIn(
                "CANDIDATE_CANONICAL_SMPL_VERTICES_UNAVAILABLE",
                topology_readiness["blockers"],
            )
            self.assertEqual(
                payload["items"][0]["axes"]["body_topology"]["provider_contracts"]
                ["comparison"]["comparison_state"],
                "UNAVAILABLE",
            )

    def test_native_topology_is_vector_only_when_vertex_and_provider_contracts_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(
                Path(temp_dir),
                native_topology=True,
            )
            payload = build_body_evidence_shadow(runtime, [item])
            topology = payload["items"][0]["axes"]["body_topology"]
            measurement = topology["measurement"]

            self.assertEqual(
                topology["native_measurement_readiness"]["readiness_state"],
                "READY",
            )
            self.assertEqual(
                topology["provider_contracts"]["comparison"]["comparison_state"],
                "MATCH",
            )
            self.assertTrue(measurement["available"])
            self.assertIsNone(measurement["residual"])
            self.assertEqual(len(measurement["residual_vector"]), 20670)
            self.assertTrue(any(abs(value) > 0.01 for value in measurement["residual_vector"]))
            self.assertFalse(measurement["scalar_residual_authorized"])
            self.assertEqual(payload["summary"]["body_topology_native_measurements"], 1)
            self.assertEqual(payload["summary"]["validation_status"], "VALID")

    def test_backend_mismatch_withholds_body_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(
                Path(temp_dir),
                candidate_backend="cpu",
            )
            payload = build_body_evidence_shadow(runtime, [item])
            axis = payload["items"][0]["axes"]["body_core_shape"]

            self.assertEqual(
                axis["measurement"]["provider_contracts"]["comparison"]["comparison_state"],
                "MISMATCH",
            )
            self.assertFalse(axis["measurement"]["available"])
            self.assertIsNone(axis["measurement"]["residual_vector"])
            self.assertEqual(axis["observation"]["observation_chain_state"], "CHAIN_INVALID")
            self.assertEqual(
                payload["items"][0]["axes"]["body_topology"]["provider_contracts"]
                ["comparison"]["comparison_state"],
                "MISMATCH",
            )

    def test_shadow_writer_creates_standalone_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            result = write_body_evidence_shadow(runtime, [item])
            output_path = Path(result["path"])

            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.name, "body_evidence_shadow.json")
            self.assertEqual(result["sha256"], _sha256(output_path))
            self.assertEqual(result["decision_influence"], "NONE")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["validation_status"], "VALID")

    def test_decision_modules_do_not_import_body_shadow(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in [
            "core/qa_review_only_score.py",
            "core/qa_consistency_confidence_matrix.py",
            "core/qa_winner_bank.py",
        ]:
            source = (root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("qa_body_evidence", source)

        pipeline_source = (root / "core/qa_pipeline.py").read_text(encoding="utf-8")
        self.assertLess(
            pipeline_source.index("_write_report_outputs(runtime, report_payload)"),
            pipeline_source.index("write_body_evidence_shadow(runtime, report_items)"),
        )


if __name__ == "__main__":
    unittest.main()
