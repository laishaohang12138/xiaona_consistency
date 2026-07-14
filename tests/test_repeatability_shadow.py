from __future__ import annotations

import unittest

from core.qa_repeatability_shadow import (
    REPEATABILITY_DOMAINS,
    empty_repeatability_contract,
    repeatability_protocol_snapshot,
    summarize_repeatability_cohort,
    summarize_repeatability_trials,
)


class RepeatabilityShadowTests(unittest.TestCase):
    def test_empty_contract_does_not_fabricate_stability(self) -> None:
        contract = empty_repeatability_contract()

        self.assertEqual(set(contract["domains"]), set(REPEATABILITY_DOMAINS))
        self.assertIsNone(contract["combined_repeatability_score"])
        self.assertFalse(contract["parameter_fitting_allowed"])
        self.assertEqual(contract["protocol_validation_status"], "VALID")
        self.assertEqual(len(contract["protocol_sha256"]), 64)
        for domain in contract["domains"].values():
            self.assertEqual(domain["measurement_state"], "NOT_MEASURED")
            self.assertEqual(domain["decision_influence"], "NONE")

    def test_trial_summary_preserves_native_descriptors_and_chain_jumps(self) -> None:
        summary = summarize_repeatability_trials(
            [
                {
                    "trial_id": "jpeg_1",
                    "perturbation_family": "jpeg_roundtrip",
                    "signed_strength": 1.0,
                    "native_residual": 0.1,
                    "baseline_chain_signature": "face_a",
                    "trial_chain_signature": "face_a",
                    "chain_diagnostics": {
                        "bbox_iou": 0.99,
                        "kps5_similarity_shape_residual": 0.01,
                    },
                },
                {
                    "trial_id": "jpeg_2",
                    "perturbation_family": "jpeg_roundtrip",
                    "signed_strength": 2.0,
                    "native_residual": 0.2,
                    "baseline_chain_signature": "face_a",
                    "trial_chain_signature": "face_b",
                },
                {
                    "trial_id": "crop_1",
                    "perturbation_family": "crop_translation",
                    "signed_strength": -1.0,
                    "native_residual": None,
                },
            ],
            domain="admissible_perturbation_stability",
            residual_unit="radian",
        )

        descriptor = summary["native_residual_descriptor"]
        self.assertEqual(summary["measurement_state"], "OBSERVED_UNCALIBRATED")
        self.assertEqual(summary["trial_count"], 3)
        self.assertEqual(summary["available_residual_count"], 2)
        self.assertAlmostEqual(descriptor["median"], 0.15)
        self.assertAlmostEqual(descriptor["spread"], 0.1)
        self.assertEqual(summary["detector_chain_transition_count"], 1)
        self.assertAlmostEqual(summary["chain_diagnostic_descriptors"]["bbox_iou"]["median"], 0.99)
        self.assertEqual(summary["trials"][0]["chain_diagnostics"]["bbox_iou"], 0.99)
        self.assertIsNone(summary["stable_unstable_classification"])
        self.assertEqual(summary["decision_influence"], "NONE")

    def test_domains_cannot_be_collapsed_into_generic_repeatability(self) -> None:
        with self.assertRaises(ValueError):
            summarize_repeatability_trials([], domain="repeatability", residual_unit="radian")

    def test_cross_source_descriptors_do_not_create_a_score_or_classification(self) -> None:
        def _domain(median: float, maximum: float, trial_value: float) -> dict:
            return {
                "trial_count": 1,
                "available_residual_count": 1,
                "native_residual_unit": "radian",
                "native_residual_descriptor": {
                    "min": median,
                    "median": median,
                    "max": maximum,
                    "spread": maximum - median,
                },
                "chain_diagnostic_descriptors": {
                    "bbox_iou": {"min": 0.9, "median": 0.95, "max": 1.0, "spread": 0.1},
                },
                "trials": [
                    {
                        "trial_id": "jpeg_95",
                        "perturbation_family": "jpeg_roundtrip",
                        "signed_strength": 95.0,
                        "native_residual": trial_value,
                    }
                ],
            }

        items = [
            {"axes": {"face_identity": {domain: _domain(0.1, 0.2, 0.12) for domain in REPEATABILITY_DOMAINS}}},
            {"axes": {"face_identity": {domain: _domain(0.3, 0.4, 0.32) for domain in REPEATABILITY_DOMAINS}}},
        ]

        summary = summarize_repeatability_cohort(items, axes=["face_identity"])
        domain = summary["axes"]["face_identity"]["preprocessing_repeatability"]

        self.assertEqual(summary["source_count"], 2)
        self.assertIsNone(summary["combined_repeatability_score"])
        self.assertIsNone(summary["stable_unstable_classification"])
        self.assertFalse(summary["parameter_fitting_allowed"])
        self.assertEqual(summary["decision_influence"], "NONE")
        self.assertEqual(domain["observed_source_count"], 2)
        self.assertEqual(domain["fully_observed_source_count"], 2)
        self.assertAlmostEqual(
            domain["source_native_residual_descriptors"]["source_medians"]["median"],
            0.2,
        )
        self.assertAlmostEqual(
            domain["trial_descriptors"]["jpeg_95"]["native_residual_descriptor"]["median"],
            0.22,
        )
        self.assertEqual(domain["calibration_state"], "SHADOW_UNCALIBRATED")

    def test_preregistered_protocol_snapshot_is_valid(self) -> None:
        snapshot = repeatability_protocol_snapshot()

        self.assertEqual(snapshot["validation_status"], "VALID")
        self.assertEqual(snapshot["validation_issues"], [])
        self.assertEqual(len(snapshot["protocol_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
