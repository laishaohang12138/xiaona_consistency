from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.qa_identity_evidence_shadow import (
    build_identity_evidence_shadow,
    write_identity_evidence_shadow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IdentityEvidenceShadowTests(unittest.TestCase):
    def _runtime_and_item(self, root: Path) -> tuple[SimpleNamespace, dict]:
        output = root / "outputs"
        output.mkdir()
        face_truth = root / "face.png"
        body_truth = root / "body.png"
        face_truth.write_bytes(b"face-truth")
        body_truth.write_bytes(b"body-truth")
        master_artifact = root / "face_master.json"
        candidate_artifact = root / "face_candidate.json"
        master_artifact.write_text(
            json.dumps(
                {
                    "canonical_identity_vector": [1.0, 0.0],
                    "canonical_landmarks": [
                        [-2.0, -1.0],
                        [-1.0, 1.0],
                        [0.0, -1.0],
                        [1.0, 2.0],
                        [2.0, 0.0],
                    ],
                    "landmark_visibility_weights": [1.0, 0.9, 0.8, 0.9, 1.0],
                    "landmark_schema_id": "test_5_point_v1",
                }
            ),
            encoding="utf-8",
        )
        candidate_artifact.write_text(
            json.dumps(
                {
                    "runtime_face_embedding_raw": [0.0, 1.0],
                    "canonical_landmarks": [
                        [0.0, -5.0],
                        [2.0, -1.0],
                        [4.0, -5.0],
                        [6.0, 1.0],
                        [8.0, -3.0],
                    ],
                    "landmark_visibility_weights": [0.9, 0.8, 0.7, 0.8, 0.9],
                    "landmark_schema_id": "test_5_point_v1",
                }
            ),
            encoding="utf-8",
        )
        config = SimpleNamespace(
            paths=SimpleNamespace(base_dir=root, config_dir=root, dir_output=output),
            provider_policy={"anchor_source": "registry_only"},
            anchor_registry={
                "anchors": {
                    "face": {
                        "role": "FACE_MASTER",
                        "anchor_tier": "absolute",
                        "authority": "ABSOLUTE_FROZEN",
                        "mutable": False,
                        "may_modify_truth": False,
                        "path": "face.png",
                        "expected_sha256": _sha256(face_truth),
                    },
                    "body": {
                        "role": "FULL_BODY_MASTER",
                        "anchor_tier": "absolute",
                        "authority": "ABSOLUTE_FROZEN",
                        "mutable": False,
                        "may_modify_truth": False,
                        "path": "body.png",
                        "expected_sha256": _sha256(body_truth),
                    },
                },
                "rules": {
                    "face_truth_anchor": "face",
                    "body_truth_anchor": "body",
                },
            },
        )
        item = {
            "image": "candidate.png",
            "selection_score": 0.99,
            "review_only_status_v2": "PASS",
            "review_only_breakdown_v2": {"observed_lane_family": "front"},
            "debug": {
                "source_path": str(root / "candidate.png"),
                "input_shape": [1024, 768],
                "candidate_face_bbox_area_ratio": 0.03,
                "candidate_face_lap_var": 80.0,
                "candidate_face_hf_energy": 1.4,
                "face_canonical_shadow": {
                    "provider_version": "test",
                    "master_artifact_path": str(master_artifact),
                    "candidate_artifact_path": str(candidate_artifact),
                    "visible_face_coverage": 0.9,
                },
            },
        }
        return SimpleNamespace(config=config), item

    def test_shadow_builder_does_not_mutate_decision_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            original = copy.deepcopy(item)
            payload = build_identity_evidence_shadow(runtime, [item])

            self.assertEqual(item, original)
            self.assertEqual(payload["governance"]["decision_influence"], "NONE")
            self.assertEqual(payload["lineage"]["status"], "VALID")
            self.assertEqual(payload["repeatability_protocol"]["validation_status"], "VALID")
            self.assertEqual(payload["repeatability_protocol"]["protocol_execution_state"], "NOT_EXECUTED")
            measurement = payload["items"][0]["axes"]["face_identity"]["measurement"]
            self.assertAlmostEqual(measurement["residual"], 1.5707963267948966)
            shape_measurement = payload["items"][0]["axes"]["face_shape"]["measurement"]
            self.assertTrue(shape_measurement["available"])
            self.assertAlmostEqual(shape_measurement["residual"], 0.0, places=12)
            self.assertEqual(payload["summary"]["available_face_shape_measurements"], 1)
            self.assertEqual(
                set(measurement["repeatability"]["domains"]),
                {
                    "numerical_repeatability",
                    "preprocessing_repeatability",
                    "admissible_perturbation_stability",
                },
            )
            self.assertIsNone(measurement["repeatability"]["combined_repeatability_score"])
            self.assertEqual(
                measurement["provider_contracts"]["comparison"]["comparison_state"],
                "PARTIAL_MATCH",
            )
            self.assertEqual(
                payload["summary"]["provider_contract_comparison_counts"]["face_identity"],
                {"PARTIAL_MATCH": 1},
            )
            self.assertNotIn("selection_score", payload["items"][0])
            self.assertNotIn("review_only_status_v2", payload["items"][0])

            shape_nodes = [
                node
                for node in payload["lineage"]["nodes"]
                if node["node_type"] == "NATIVE_MEASUREMENT"
                and node["attributes"].get("axis") == "face_shape"
            ]
            self.assertEqual(len(shape_nodes), 1)
            self.assertFalse(shape_nodes[0]["attributes"]["partition_diagnostics_independent_evidence"])

    def test_shadow_writer_creates_standalone_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            pointer = write_identity_evidence_shadow(runtime, [item])
            output_path = Path(pointer["path"])

            self.assertTrue(output_path.exists())
            self.assertEqual(pointer["decision_influence"], "NONE")
            self.assertEqual(pointer["sha256"], _sha256(output_path))

    def test_back_view_withholds_face_residual_even_if_embedding_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            item["review_only_breakdown_v2"]["observed_lane_family"] = "back"
            payload = build_identity_evidence_shadow(runtime, [item])
            axis = payload["items"][0]["axes"]["face_identity"]

            self.assertEqual(axis["observation"]["eligibility"], "UNOBSERVABLE")
            self.assertEqual(axis["observation"]["observation_chain_state"], "CHAIN_VALID")
            self.assertFalse(axis["measurement"]["available"])
            self.assertIsNone(axis["measurement"]["residual"])
            self.assertEqual(
                axis["measurement"]["errors"],
                ["MEASUREMENT_WITHHELD_UNOBSERVABLE_SCOPE"],
            )
            shape_axis = payload["items"][0]["axes"]["face_shape"]
            self.assertEqual(shape_axis["observation"]["eligibility"], "UNOBSERVABLE")
            self.assertEqual(shape_axis["observation"]["observation_chain_state"], "CHAIN_VALID")
            self.assertFalse(shape_axis["measurement"]["available"])
            self.assertIsNone(shape_axis["measurement"]["residual"])

    def test_landmark_schema_mismatch_withholds_shape_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime, item = self._runtime_and_item(root)
            candidate_path = root / "face_candidate.json"
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["landmark_schema_id"] = "different_5_point_order_v1"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            payload = build_identity_evidence_shadow(runtime, [item])
            axis = payload["items"][0]["axes"]["face_shape"]

            self.assertEqual(axis["observation"]["observation_chain_state"], "CHAIN_INVALID")
            self.assertFalse(axis["measurement"]["available"])
            self.assertIsNone(axis["measurement"]["residual"])
            self.assertEqual(axis["measurement"]["errors"], ["LANDMARK_SCHEMA_MISMATCH"])
            self.assertIn("LANDMARK_SCHEMA_MISMATCH", axis["measurement"]["contract_gaps"])

    def test_identity_provider_mismatch_withholds_identity_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime, item = self._runtime_and_item(root)
            for filename, model_id in [
                ("face_master.json", "identity_model_v1"),
                ("face_candidate.json", "identity_model_v2"),
            ]:
                artifact_path = root / filename
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                artifact["runtime_face_embedding_contract"] = {
                    "provider_name": "identity_provider",
                    "provider_version": "1",
                    "model_id": model_id,
                    "model_sha256": "a" * 64,
                    "detector_contract_id": "detector_v1",
                    "alignment_contract_id": "alignment_v1",
                    "preprocessing_contract_id": "preprocess_v1",
                    "source_field": "face.embedding",
                }
                artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            payload = build_identity_evidence_shadow(runtime, [item])
            axis = payload["items"][0]["axes"]["face_identity"]

            self.assertEqual(axis["observation"]["observation_chain_state"], "CHAIN_INVALID")
            self.assertFalse(axis["measurement"]["available"])
            self.assertIsNone(axis["measurement"]["residual"])
            self.assertIsNone(axis["measurement"]["embedding_contract"]["cosine"])
            self.assertEqual(axis["measurement"]["errors"], ["IDENTITY_PROVIDER_CONTRACT_MISMATCH"])

    def test_side_shape_is_available_but_prior_dependent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime, item = self._runtime_and_item(Path(temp_dir))
            item["review_only_breakdown_v2"]["observed_lane_family"] = "side"
            payload = build_identity_evidence_shadow(runtime, [item])
            axis = payload["items"][0]["axes"]["face_shape"]

            self.assertEqual(axis["observation"]["eligibility"], "PRIOR_DEPENDENT")
            self.assertTrue(axis["measurement"]["available"])
            self.assertEqual(axis["measurement"]["decision_influence"], "NONE")

    def test_present_but_degenerate_sources_invalidate_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime, item = self._runtime_and_item(root)
            (root / "face_candidate.json").write_text(
                json.dumps(
                    {
                        "runtime_face_embedding_raw": [0.0, 0.0],
                        "canonical_landmarks": [[1.0, 1.0]] * 5,
                    }
                ),
                encoding="utf-8",
            )
            payload = build_identity_evidence_shadow(runtime, [item])

            identity_axis = payload["items"][0]["axes"]["face_identity"]
            shape_axis = payload["items"][0]["axes"]["face_shape"]
            self.assertEqual(identity_axis["observation"]["observation_chain_state"], "CHAIN_INVALID")
            self.assertEqual(shape_axis["observation"]["observation_chain_state"], "CHAIN_INVALID")
            self.assertFalse(identity_axis["measurement"]["available"])
            self.assertFalse(shape_axis["measurement"]["available"])

    def test_decision_modules_do_not_import_shadow_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in [
            "core/qa_review_only_score.py",
            "core/qa_consistency_confidence_matrix.py",
            "core/qa_winner_bank.py",
        ]:
            source = (root / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("qa_identity_evidence", source)

        pipeline_source = (root / "core/qa_pipeline.py").read_text(encoding="utf-8")
        self.assertLess(
            pipeline_source.index("_write_report_outputs(runtime, report_payload)"),
            pipeline_source.index("write_identity_evidence_shadow(runtime, report_items)"),
        )


if __name__ == "__main__":
    unittest.main()
