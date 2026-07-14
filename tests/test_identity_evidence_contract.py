from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from core.qa_face_pose_canonical import _normalize_artifact
from core.qa_identity_evidence_contract import (
    angular_distance_radians,
    build_axis_observation,
    build_face_identity_measurement,
    build_face_projection_shape_measurement,
    validate_shadow_axis_record,
)


class IdentityEvidenceContractTests(unittest.TestCase):
    def test_angular_distance_native_geometry(self) -> None:
        same = angular_distance_radians([2.0, 0.0], [5.0, 0.0])
        orthogonal = angular_distance_radians([1.0, 0.0], [0.0, 3.0])
        opposite = angular_distance_radians([1.0, 0.0], [-4.0, 0.0])

        self.assertTrue(same["available"])
        self.assertAlmostEqual(same["residual"], 0.0)
        self.assertAlmostEqual(orthogonal["residual"], math.pi / 2.0)
        self.assertAlmostEqual(opposite["residual"], math.pi)

    def test_invalid_embeddings_do_not_fabricate_residuals(self) -> None:
        zero = angular_distance_radians([0.0, 0.0], [1.0, 0.0])
        mismatch = angular_distance_radians([1.0, 0.0], [1.0, 0.0, 0.0])

        self.assertFalse(zero["available"])
        self.assertIn("EMBEDDING_ZERO_NORM", zero["errors"])
        self.assertFalse(mismatch["available"])
        self.assertIn("EMBEDDING_DIMENSION_MISMATCH", mismatch["errors"])

    def test_face_measurement_is_shadow_uncalibrated(self) -> None:
        measurement = build_face_identity_measurement(
            [1.0, 0.0],
            [0.0, 1.0],
            provider_name="insightface_runtime_embedding",
            provider_version="test",
            model_id="test",
        )

        self.assertEqual(measurement["decision_influence"], "NONE")
        self.assertEqual(measurement["calibration_state"], "SHADOW_UNCALIBRATED")
        self.assertEqual(measurement["native_space"], "unit_hypersphere")
        self.assertEqual(measurement["unit"], "radian")

    def test_observation_enums_are_enforced(self) -> None:
        observation = build_axis_observation(
            axis="face_identity",
            eligibility="MEASURABLE",
            scope_state="FRONT_SUPPORTED",
            chain_state="CHAIN_VALID",
            raw_observations={},
        )
        self.assertEqual(observation["decision_influence"], "NONE")

        with self.assertRaises(ValueError):
            build_axis_observation(
                axis="face_identity",
                eligibility="STABLE",
                scope_state="FRONT_SUPPORTED",
                chain_state="CHAIN_VALID",
                raw_observations={},
            )

    def test_face_shape_measurement_is_one_uncalibrated_evidence_unit(self) -> None:
        reference = [[-2.0, -1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 2.0], [2.0, 0.0]]
        candidate = [[3.0 + 2.0 * x, -4.0 + 2.0 * y] for x, y in reference]
        measurement = build_face_projection_shape_measurement(
            reference,
            candidate,
            visibility_weights=[1.0] * len(reference),
            provider_name="test_provider",
            provider_version="test",
            model_id="test",
            visibility_weight_source="pairwise_min_visibility",
        )

        self.assertTrue(measurement["available"])
        self.assertAlmostEqual(measurement["residual"], 0.0, places=12)
        self.assertEqual(measurement["calibration_state"], "SHADOW_UNCALIBRATED")
        self.assertEqual(measurement["decision_influence"], "NONE")
        self.assertFalse(
            measurement["independence_contract"]["partition_diagnostics_are_independent_evidence"]
        )
        self.assertEqual(
            measurement["shape_contract"]["visibility_weight_source"],
            "pairwise_min_visibility",
        )

    def test_legacy_identity_field_is_preserved_as_runtime_embedding_alias(self) -> None:
        artifact = _normalize_artifact(
            {"canonical_identity_vector": [3.0, 4.0]},
            source_path=Path("candidate.png"),
            source_role="candidate",
        )

        np.testing.assert_allclose(artifact["canonical_identity_vector"], [3.0, 4.0])
        np.testing.assert_allclose(artifact["runtime_face_embedding_raw"], [3.0, 4.0])
        np.testing.assert_allclose(artifact["runtime_face_embedding_unit"], [0.6, 0.8])

    def test_landmark_visibility_alias_is_preserved(self) -> None:
        artifact = _normalize_artifact(
            {
                "canonical_landmarks": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "landmark_confidence": [0.9, 0.8, 0.7],
                "landmark_schema_id": "test_3_point_v1",
            },
            source_path=Path("candidate.png"),
            source_role="candidate",
        )

        np.testing.assert_allclose(artifact["landmark_visibility_weights"], [0.9, 0.8, 0.7])
        self.assertEqual(artifact["landmark_schema_id"], "test_3_point_v1")

    def test_axis_validator_rejects_fabricated_or_cross_axis_residuals(self) -> None:
        record = {
            "observation": build_axis_observation(
                axis="face_identity",
                eligibility="UNOBSERVABLE",
                scope_state="BACK_FACE_UNOBSERVABLE",
                chain_state="CHAIN_VALID",
                raw_observations={},
            ),
            "measurement": {
                "axis": "face_shape",
                "available": True,
                "residual": float("nan"),
                "calibration_state": "SHADOW_UNCALIBRATED",
                "decision_influence": "NONE",
            },
        }

        issues = validate_shadow_axis_record(record)
        self.assertIn("OBSERVATION_MEASUREMENT_AXIS_MISMATCH", issues)
        self.assertIn("AVAILABLE_MEASUREMENT_REQUIRES_FINITE_RESIDUAL", issues)
        self.assertIn("UNOBSERVABLE_AXIS_MUST_WITHHOLD_MEASUREMENT", issues)


if __name__ == "__main__":
    unittest.main()
