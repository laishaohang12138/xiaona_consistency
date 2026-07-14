from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.qa_admission import (
    build_batch_admission_advice,
    build_candidate_admission_advice,
)
from core.qa_governance import (
    RELEASE_GATE_SCHEMA_VERSION,
    fail_closed_release_gate,
    normalize_release_gate_config,
)
from core.qa_training_admission import seal_training_admission_entry


class AdmissionGovernanceTests(unittest.TestCase):
    def test_legacy_true_flags_are_migrated_fail_closed(self) -> None:
        payload = normalize_release_gate_config(
            {
                "schema_version": "qa_release_gates_v1",
                "release_gates": {
                    "BODY_GOLD.front_core": {
                        "release_state": "primary",
                        "machine_status_ceiling": "PASS",
                        "training_admission_allowed": True,
                        "manual_training_admission_required": True,
                        "optuna_fit_allowed": True,
                        "may_emit_final_admission": True,
                        "may_emit_final_image_set_membership": True,
                    }
                },
            },
            {
                "schema_version": RELEASE_GATE_SCHEMA_VERSION,
                "release_gates": {},
            },
        )

        gate = payload["release_gates"]["BODY_GOLD.front_core"]
        self.assertEqual(payload["schema_version"], RELEASE_GATE_SCHEMA_VERSION)
        self.assertEqual(gate["schema_state"], "LEGACY_MIGRATED_FAIL_CLOSED")
        self.assertEqual(gate["local_decision_authority"], "NONE")
        self.assertEqual(gate["external_review_route"], "PRIORITY_REVIEW")
        self.assertFalse(gate["training_admission_allowed"])
        self.assertFalse(gate["manual_training_admission_required"])
        self.assertFalse(gate["optuna_fit_allowed"])
        self.assertFalse(gate["parameter_fitting_allowed"])
        self.assertFalse(gate["may_emit_final_admission"])
        self.assertFalse(gate["may_emit_final_image_set_membership"])

    def test_missing_or_unknown_schema_holds_for_more_evidence(self) -> None:
        for source_schema in ("", "qa_release_gates_future"):
            with self.subTest(source_schema=source_schema):
                gate = fail_closed_release_gate(
                    {
                        "release_state": "primary",
                        "external_review_route": "PRIORITY_REVIEW",
                        "training_admission_allowed": True,
                    },
                    source_schema_version=source_schema,
                )
                self.assertEqual(gate["external_review_route"], "HOLD_FOR_MORE_EVIDENCE")
                self.assertEqual(gate["local_decision_authority"], "NONE")
                self.assertFalse(gate["training_admission_allowed"])

    def test_batch_and_candidate_advice_ignore_explicit_participation_true(self) -> None:
        batch = {
            "target_profile": "body_gold_fullbody",
            "lane_detail_counts": {"front": 4},
            "release_gate": {
                "schema_version": RELEASE_GATE_SCHEMA_VERSION,
                "release_state": "primary",
                "machine_status_ceiling": "PASS",
                "external_review_route": "PRIORITY_REVIEW",
                "training_admission_allowed": True,
                "required_lane_families": ["front"],
            },
            "training_admission_governance": {
                "participates_in_final_admission": True,
            },
            "batch_preflight": {"status": "PASS"},
            "evidence_completeness": {
                "status": "PASS",
                "replay_ready": True,
                "gpt_review_ready": True,
            },
            "identity_summary": {
                "batch_identity_cohesion": 0.95,
                "batch_hybrid_identity_cohesion": 0.95,
                "batch_clothfree_identity_cohesion": 0.95,
            },
            "geometry_summary": {
                "batch_world3d_cohesion": 0.98,
                "routing_consistency": 0.98,
            },
            "engine_status": {"fatal": False},
        }

        batch_advice = build_batch_admission_advice(batch)
        self.assertFalse(batch_advice["training_admission_participation"])
        self.assertFalse(batch_advice["eligible_for_training_seal"])
        self.assertEqual(batch_advice["local_decision_authority"], "NONE")
        self.assertEqual(batch_advice["external_review_route"], "PRIORITY_REVIEW")

        candidate_advice = build_candidate_admission_advice(
            {
                "status": "PASS",
                "lane": {"view_lane_detail": "front"},
                "master_consistency_card": {
                    "lane_family": "front",
                    "hybrid_master_alignment": 0.90,
                    "body_master_alignment": 0.90,
                },
            },
            {**batch_advice, "training_admission_participation": True},
        )
        self.assertFalse(candidate_advice["training_admission_participation"])
        self.assertFalse(candidate_advice["eligible_for_training_seal"])
        self.assertEqual(candidate_advice["local_decision_authority"], "NONE")
        self.assertEqual(candidate_advice["external_review_route"], "PRIORITY_REVIEW")

    def test_audit_requires_explicit_external_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "training_admission_manifest.json"
            candidate = {
                "target_profile": "body_gold_fullbody",
                "image": "candidate.png",
                "record_key": "candidate-1",
            }

            blocked = seal_training_admission_entry(
                candidate,
                manifest,
                manual_owner="external-reviewer",
            )
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["reason"], "external_decision_confirmation_required")
            self.assertFalse(manifest.exists())

            recorded = seal_training_admission_entry(
                candidate,
                manifest,
                release_gate={
                    "schema_version": "qa_release_gates_v1",
                    "training_admission_allowed": True,
                },
                admission_advice={
                    "eligible_for_training_seal": False,
                    "blockers": ["LOCAL_SCREENING_WARNING"],
                },
                batch_preflight={"status": "FAIL"},
                evidence_completeness={"status": "FAIL", "replay_ready": False},
                manual_owner="external-reviewer",
                external_decision_confirmed=True,
            )
            self.assertEqual(recorded["status"], "ok")
            self.assertFalse(recorded["sealed_for_training"])
            self.assertIn("BATCH_PREFLIGHT_FAILED_AT_AUDIT_TIME", recorded["local_evidence_warnings"])

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            entry = payload["entries"][0]
            self.assertEqual(payload["schema_version"], "external_admission_audit_manifest_v2")
            self.assertEqual(entry["local_decision_authority"], "NONE")
            self.assertFalse(entry["sealed_for_training"])
            self.assertTrue(entry["external_decision_recorded"])
            self.assertFalse(entry["release_gate"]["training_admission_allowed"])


if __name__ == "__main__":
    unittest.main()
