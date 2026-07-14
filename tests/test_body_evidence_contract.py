from __future__ import annotations

import math
import unittest

from core.qa_body_evidence_contract import (
    body_core_observation_scope,
    build_body_axis_observation,
    build_body_core_shape_measurement,
    build_body_topology_measurement,
    build_pose_gait_condition,
    build_surface_occlusion_condition,
    canonical_vertex_delta_vector,
    log_ratio_residual_vector,
    validate_body_shadow_axis_record,
)


class BodyEvidenceContractTests(unittest.TestCase):
    def test_log_ratio_residual_preserves_signed_components_without_scalar_score(self) -> None:
        reference = {
            "shoulder_width_to_torso": 0.50,
            "hip_width_to_torso": 0.40,
            "shoulder_to_hip_ratio": 1.25,
            "upper_to_lower_leg_ratio": 1.00,
            "foot_length_to_leg": 0.20,
        }
        candidate = {
            "shoulder_width_to_torso": 0.55,
            "hip_width_to_torso": 0.40,
            "shoulder_to_hip_ratio": 1.00,
            "upper_to_lower_leg_ratio": 1.10,
            "foot_length_to_leg": 0.20,
        }

        result = log_ratio_residual_vector(reference, candidate)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(
            result["component_residuals"]["shoulder_width_to_torso"],
            math.log(1.1),
        )
        self.assertAlmostEqual(
            result["component_residuals"]["shoulder_to_hip_ratio"],
            math.log(0.8),
        )

        measurement = build_body_core_shape_measurement(
            reference,
            candidate,
            provider_name="body_canonical_hmr2",
            provider_version="test",
            model_id="test",
        )
        self.assertIsNone(measurement["residual"])
        self.assertFalse(measurement["scalar_residual_authorized"])
        self.assertEqual(measurement["decision_influence"], "NONE")
        self.assertFalse(
            measurement["independence_contract"]["components_are_independent_votes"]
        )

    def test_invalid_or_missing_components_are_not_renormalized_into_availability(self) -> None:
        result = log_ratio_residual_vector(
            {"shoulder_width_to_torso": 0.5, "hip_width_to_torso": 0.0},
            {"shoulder_width_to_torso": 0.6, "hip_width_to_torso": 0.4},
        )
        self.assertFalse(result["available"])
        self.assertIsNone(result["residual_vector"])
        self.assertIn("hip_width_to_torso", result["invalid_components"])
        self.assertLess(result["coverage"], 1.0)

    def test_hmr2_measurement_is_prior_dependent_in_every_lane(self) -> None:
        for lane in ["front", "three_quarter", "side", "back"]:
            with self.subTest(lane=lane):
                eligibility, scope, reasons = body_core_observation_scope(
                    lane,
                    measurement_available=True,
                )
                self.assertEqual(eligibility, "PRIOR_DEPENDENT")
                self.assertIn("PRIOR_DEPENDENT", scope)
                self.assertTrue(reasons)

    def test_pose_and_occlusion_are_conditions_not_votes(self) -> None:
        pose = build_pose_gait_condition([0.0, 1.0], [0.2, 0.5])
        surface = build_surface_occlusion_condition(
            body_coverage=0.9,
            visible_body_ratio=0.7,
            garment_coverage_ratio=0.6,
        )
        self.assertTrue(pose["available"])
        self.assertEqual(pose["condition_role"], "NUISANCE_DESCRIPTOR_NOT_CONSISTENCY_VOTE")
        self.assertEqual(pose["decision_influence"], "NONE")
        self.assertTrue(surface["available"])
        self.assertIsNone(surface["threshold_classification"])
        self.assertEqual(surface["decision_influence"], "NONE")

    def test_body_axis_validator_rejects_scalar_residual(self) -> None:
        measurement = build_body_core_shape_measurement(
            {
                "shoulder_width_to_torso": 0.5,
                "hip_width_to_torso": 0.4,
                "shoulder_to_hip_ratio": 1.2,
            },
            {
                "shoulder_width_to_torso": 0.5,
                "hip_width_to_torso": 0.4,
                "shoulder_to_hip_ratio": 1.2,
            },
            provider_name="test",
            provider_version="test",
            model_id="test",
        )
        measurement["residual"] = 0.0
        observation = build_body_axis_observation(
            axis="body_core_shape",
            eligibility="PRIOR_DEPENDENT",
            scope_state="FRONT_HMR2_PRIOR_DEPENDENT",
            chain_state="CHAIN_VALID",
            raw_observations={},
        )
        issues = validate_body_shadow_axis_record(
            {"observation": observation, "measurement": measurement}
        )
        self.assertIn("SCALAR_BODY_CORE_RESIDUAL_NOT_ALLOWED", issues)

    def test_canonical_vertex_delta_removes_translation_only(self) -> None:
        reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        translated = [
            [vertex[0] + 10.0, vertex[1] - 3.0, vertex[2] + 2.0]
            for vertex in reference
        ]

        result = canonical_vertex_delta_vector(reference, translated)

        self.assertTrue(result["available"])
        self.assertEqual(result["coordinate_count"], 12)
        self.assertTrue(all(abs(value) < 0.0000001 for value in result["residual_vector"]))

    def test_canonical_vertex_delta_preserves_shape_and_scale_change(self) -> None:
        reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        scaled = [[component * 1.2 for component in vertex] for vertex in reference]

        measurement = build_body_topology_measurement(
            reference,
            scaled,
            provider_name="body_canonical_hmr2",
            provider_version="test",
            model_id="test",
        )

        self.assertTrue(measurement["available"])
        self.assertIsNone(measurement["residual"])
        self.assertEqual(len(measurement["residual_vector"]), 12)
        self.assertTrue(any(abs(value) > 0.01 for value in measurement["residual_vector"]))
        self.assertFalse(measurement["alignment_contract"]["rotation_fit_applied"])
        self.assertFalse(measurement["alignment_contract"]["scale_fit_applied"])
        self.assertFalse(measurement["alignment_contract"]["procrustes_fit_applied"])
        self.assertFalse(measurement["scalar_residual_authorized"])
        self.assertFalse(
            measurement["independence_contract"]["independent_from_body_core_shape"]
        )
        self.assertEqual(
            measurement["repeatability"]["protocol_execution_state"],
            "NOT_EXECUTED",
        )
        self.assertEqual(
            measurement["repeatability"]["measurement_axis"],
            "body_topology",
        )

    def test_canonical_vertex_delta_rejects_mismatch_and_nonfinite_values(self) -> None:
        mismatch = canonical_vertex_delta_vector(
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        nonfinite = canonical_vertex_delta_vector(
            [[0.0, 0.0, 0.0]],
            [[float("nan"), 0.0, 0.0]],
        )

        self.assertFalse(mismatch["available"])
        self.assertIn("CANONICAL_SMPL_VERTEX_COUNT_MISMATCH", mismatch["errors"])
        self.assertFalse(nonfinite["available"])
        self.assertIsNone(nonfinite["residual_vector"])


if __name__ == "__main__":
    unittest.main()
