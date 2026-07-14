from __future__ import annotations

import math
import unittest

import numpy as np

from core.qa_procrustes_shape import weighted_irls_procrustes


class WeightedIrlsProcrustesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.asarray(
            [
                [-2.0, -1.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [1.0, 2.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float64,
        )

    def test_translation_scale_and_rotation_are_removed(self) -> None:
        angle = 0.63
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        candidate = 3.7 * (self.reference @ rotation) + np.asarray([12.0, -5.0])
        result = weighted_irls_procrustes(self.reference, candidate)

        self.assertTrue(result["available"])
        self.assertLess(result["residual"], 0.000000000001)

    def test_reflection_is_not_removed(self) -> None:
        reflected = self.reference.copy()
        reflected[:, 0] *= -1.0
        result = weighted_irls_procrustes(self.reference, reflected)

        self.assertTrue(result["available"])
        self.assertGreater(result["residual"], 0.1)
        self.assertGreaterEqual(np.linalg.det(np.asarray(result["rotation_matrix"])), 0.0)

    def test_huber_irls_downweights_a_strong_outlier(self) -> None:
        candidate = self.reference.copy()
        candidate[-1] += np.asarray([30.0, -25.0])
        result = weighted_irls_procrustes(self.reference, candidate)

        self.assertTrue(result["available"])
        self.assertLess(result["residual"], result["raw_rms_residual"])
        self.assertLess(result["effective_weight_share"], 1.0)
        self.assertEqual(
            set(result["partition_diagnostics"]),
            {"low_y_band", "mid_y_band", "high_y_band", "lateral_band", "center_axis_band"},
        )

    def test_shape_mismatch_and_invalid_weights_withhold_residual(self) -> None:
        mismatch = weighted_irls_procrustes(self.reference, self.reference[:-1])
        invalid_weights = weighted_irls_procrustes(
            self.reference,
            self.reference,
            visibility_weights=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )

        self.assertFalse(mismatch["available"])
        self.assertEqual(mismatch["error"], "LANDMARK_SHAPE_MISMATCH")
        self.assertFalse(invalid_weights["available"])
        self.assertEqual(invalid_weights["error"], "LANDMARK_VISIBILITY_WEIGHTS_INVALID")


if __name__ == "__main__":
    unittest.main()
