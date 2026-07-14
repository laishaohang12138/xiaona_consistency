from __future__ import annotations

import copy
import unittest

from core.qa_provider_contract import (
    build_body_provider_contract,
    build_face_provider_contract,
    compare_provider_contracts,
    provider_contract_gap_codes,
)


def _complete_identity_artifact() -> dict:
    return {
        "source_path": "excluded-from-contract.png",
        "runtime_face_embedding_raw": [1.0, 0.0, 0.0],
        "runtime_face_embedding_contract": {
            "provider_name": "identity_provider",
            "provider_version": "1.2.3",
            "model_id": "identity_model_v1",
            "model_sha256": "a" * 64,
            "execution_backend": "CUDAExecutionProvider",
            "detector_contract_id": "detector_v1",
            "alignment_contract_id": "alignment_v1",
            "preprocessing_contract_id": "preprocess_v1",
            "source_field": "face.embedding",
        },
    }


class ProviderContractTests(unittest.TestCase):
    def test_face_builder_rejects_body_axis(self) -> None:
        with self.assertRaises(ValueError):
            build_face_provider_contract({}, axis="body_core_shape")

    def test_complete_contracts_match_independent_of_asset_path(self) -> None:
        reference_artifact = _complete_identity_artifact()
        candidate_artifact = copy.deepcopy(reference_artifact)
        candidate_artifact["source_path"] = "different-candidate.png"
        reference = build_face_provider_contract(reference_artifact, axis="face_identity")
        candidate = build_face_provider_contract(candidate_artifact, axis="face_identity")
        comparison = compare_provider_contracts(reference, candidate)

        self.assertEqual(reference["completeness_state"], "COMPLETE")
        self.assertIsNotNone(reference["comparable_contract_sha256"])
        self.assertEqual(reference["comparable_contract_sha256"], candidate["comparable_contract_sha256"])
        self.assertEqual(comparison["comparison_state"], "MATCH")
        self.assertTrue(comparison["comparable"])

    def test_partial_contract_never_claims_comparability(self) -> None:
        artifact = _complete_identity_artifact()
        artifact["runtime_face_embedding_contract"]["model_sha256"] = None
        contract = build_face_provider_contract(artifact, axis="face_identity")
        comparison = compare_provider_contracts(contract, copy.deepcopy(contract))

        self.assertEqual(contract["completeness_state"], "PARTIAL")
        self.assertIsNone(contract["comparable_contract_sha256"])
        self.assertEqual(comparison["comparison_state"], "PARTIAL_MATCH")
        self.assertFalse(comparison["comparable"])

    def test_known_model_conflict_is_explicit_mismatch(self) -> None:
        reference = build_face_provider_contract(_complete_identity_artifact(), axis="face_identity")
        candidate_artifact = _complete_identity_artifact()
        candidate_artifact["runtime_face_embedding_contract"]["model_id"] = "identity_model_v2"
        candidate = build_face_provider_contract(candidate_artifact, axis="face_identity")
        comparison = compare_provider_contracts(reference, candidate)
        gaps = provider_contract_gap_codes(reference, candidate, comparison)

        self.assertEqual(comparison["comparison_state"], "MISMATCH")
        self.assertEqual(comparison["conflicts"][0]["field"], "model_id")
        self.assertIn("FACE_IDENTITY_PROVIDER_CONTRACT_CONFLICT_MODEL_ID", gaps)

    def test_unresolved_coordinate_convention_remains_partial(self) -> None:
        artifact = {
            "provider_name": "shape_provider",
            "provider_version": "1",
            "model_id": "shape_model",
            "model_sha256": "b" * 64,
            "canonical_landmarks": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            "landmark_schema_id": "shape_3_v1",
            "landmark_coordinate_convention": "PROVIDER_NATIVE_UNRESOLVED",
            "canonical_preprocessing_contract_id": "shape_preprocess_v1",
            "landmark_source_field": "landmarks_3",
        }
        contract = build_face_provider_contract(artifact, axis="face_shape")

        self.assertEqual(contract["completeness_state"], "PARTIAL")
        self.assertIn("coordinate_convention", contract["missing_fields"])

    def test_body_contract_without_measurement_source_is_unavailable(self) -> None:
        artifact = {
            "shape_beta": [0.1, 0.2, 0.3],
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
                "measurement_order": [],
                "shape_dimension": 3,
                "coordinate_convention": "camera_relative",
                "source_field": None,
            },
        }
        contract = build_body_provider_contract(artifact, axis="body_core_shape")

        self.assertEqual(contract["completeness_state"], "UNAVAILABLE")
        self.assertIn("measurement_order", contract["missing_fields"])
        self.assertIn("source_field", contract["missing_fields"])

    def test_legacy_body_topology_signature_cannot_unlock_native_contract(self) -> None:
        artifact = {
            "body_topology_signature": [0.1, 0.2, 0.3],
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
                "topology_schema_id": "hmr2_shape_beta_body25_signature_v1",
                "topology_dimension": 3,
                "coordinate_convention": "camera_relative",
                "topology_source_field": "body_topology_signature",
            },
        }

        contract = build_body_provider_contract(artifact, axis="body_topology")

        self.assertEqual(contract["completeness_state"], "UNAVAILABLE")
        self.assertIsNone(contract["fields"]["topology_dimension"])
        self.assertEqual(contract["fields"]["source_field"], "body_topology_signature")
        self.assertIn(
            "source_field",
            [row["field"] for row in contract["incompatible_fields"]],
        )

    def test_native_body_topology_contract_requires_exact_zero_pose_semantics(self) -> None:
        vertices = [[float(index), 0.0, 0.0] for index in range(6890)]
        artifact = {
            "canonical_smpl_vertices": vertices,
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
                "topology_schema_id": "smpl_neutral_zero_pose_vertices_v1",
                "topology_coordinate_convention": "smpl_neutral_zero_pose_model_space",
                "canonicalization_contract_id": "smpl_identity_global_and_body_rotations_v1",
                "topology_alignment_contract_id": "centroid_translation_removal_only_v1",
                "topology_representation": "dense_smpl_vertices_exact_index_correspondence",
                "topology_source_field": "canonical_smpl_vertices",
            },
        }

        contract = build_body_provider_contract(artifact, axis="body_topology")

        self.assertEqual(contract["completeness_state"], "COMPLETE")
        self.assertEqual(contract["fields"]["topology_vertex_count"], 6890)
        self.assertEqual(contract["fields"]["topology_dimension"], 20670)
        self.assertFalse(contract["incompatible_fields"])

        incompatible = copy.deepcopy(artifact)
        incompatible["body_canonical_contract"][
            "topology_alignment_contract_id"
        ] = "procrustes_similarity_fit_v1"
        incompatible_contract = build_body_provider_contract(
            incompatible,
            axis="body_topology",
        )
        self.assertEqual(incompatible_contract["completeness_state"], "PARTIAL")
        self.assertEqual(
            incompatible_contract["incompatible_fields"][0]["field"],
            "topology_alignment_contract_id",
        )


if __name__ == "__main__":
    unittest.main()
