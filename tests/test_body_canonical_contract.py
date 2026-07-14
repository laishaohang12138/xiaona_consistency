from __future__ import annotations

import unittest
from pathlib import Path

from core.qa_heavy_body_canonical import _normalize_artifact


class BodyCanonicalContractTests(unittest.TestCase):
    def test_normalization_fills_derived_topology_dimension_and_preserves_provenance(self) -> None:
        raw = {
            "schema_version": "body_canonical_artifact_v2",
            "shape_beta": [0.1, 0.2, 0.3],
            "pose_vector": [0.0, 0.1],
            "canonical_measurements": {
                "shoulder_width_to_torso": 0.5,
                "hip_width_to_torso": 0.4,
                "shoulder_to_hip_ratio": 1.25,
                "leg_length_to_torso": 1.8,
                "upper_to_lower_leg_ratio": 1.0,
                "foot_length_to_leg": 0.2,
            },
            "conversion_meta": {"device": "cuda", "measurement_basis": "body25_v2"},
            "body_canonical_contract": {
                "model_id": "hmr2.ckpt",
                "model_sha256": "a" * 64,
                "implementation_sha256": "b" * 64,
                "body_model_id": "SMPL_NEUTRAL.pkl",
                "body_model_sha256": "c" * 64,
                "preprocessing_contract_id": "preprocess_v1",
                "topology_dimension": None,
            },
        }

        artifact = _normalize_artifact(
            raw,
            source_path=Path("candidate.png"),
            source_role="candidate",
        )

        self.assertEqual(artifact["conversion_meta"]["device"], "cuda")
        self.assertIsNotNone(artifact["body_topology_signature"])
        self.assertEqual(
            artifact["body_canonical_contract"]["topology_dimension"],
            len(artifact["body_topology_signature"]),
        )
        self.assertEqual(artifact["body_canonical_contract"]["model_id"], "hmr2.ckpt")

    def test_normalization_preserves_finite_native_vertices_and_rewrites_native_contract(self) -> None:
        vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        artifact = _normalize_artifact(
            {
                "shape_beta": [0.1, 0.2, 0.3],
                "canonical_smpl_vertices": vertices,
                "body_canonical_contract": {
                    "topology_schema_id": "legacy_signature",
                    "topology_source_field": "body_topology_signature",
                },
            },
            source_path=Path("candidate.png"),
            source_role="candidate",
        )

        contract = artifact["body_canonical_contract"]
        self.assertEqual(artifact["canonical_smpl_vertices"], vertices)
        self.assertEqual(contract["topology_vertex_count"], 4)
        self.assertEqual(contract["topology_dimension"], 12)
        self.assertEqual(contract["topology_source_field"], "canonical_smpl_vertices")
        self.assertEqual(
            contract["canonicalization_contract_id"],
            "smpl_identity_global_and_body_rotations_v1",
        )

    def test_normalization_withholds_invalid_native_vertices(self) -> None:
        artifact = _normalize_artifact(
            {
                "shape_beta": [0.1, 0.2, 0.3],
                "canonical_smpl_vertices": [[0.0, float("nan"), 0.0]],
            },
            source_path=Path("candidate.png"),
            source_role="candidate",
        )

        self.assertIsNone(artifact["canonical_smpl_vertices"])
        self.assertEqual(
            artifact["body_canonical_contract"]["topology_source_field"],
            "body_topology_signature",
        )


if __name__ == "__main__":
    unittest.main()
