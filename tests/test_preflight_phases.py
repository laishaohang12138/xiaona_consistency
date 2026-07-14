from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.qa_preflight import (
    create_lightweight_preflight_config,
    run_preflight_batch,
    static_preflight_blockers,
)


class PreflightPhaseTests(unittest.TestCase):
    def _batch(self, root: Path, *, with_complete_manifest: bool, phase: str):
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "candidate.png").write_bytes(b"metadata-only fixture")
        if with_complete_manifest:
            (input_dir / "input_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "input_manifest_v1",
                        "items": [
                            {
                                "input_relative_path": "candidate.png",
                                "prompt_id": "face_lock_fl01",
                                "seed_unavailable_reason": "generator_did_not_expose_seed",
                                "anchor_source": "A-Core_01_0deg_MASTER.png",
                                "intended_view": "front",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return run_preflight_batch(
            None,
            config=create_lightweight_preflight_config(root),
            input_dir=input_dir,
            target_profile="body_gold_fullbody",
            preflight_phase=phase,
        )

    def test_metadata_phase_reports_only_metadata_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self._batch(
                Path(temp_dir),
                with_complete_manifest=False,
                phase="metadata_only",
            )

        batch = payload["batch_preflight"]
        self.assertEqual(payload["preflight_phase"], "metadata_only")
        self.assertFalse(payload["runtime_initialization_attempted"])
        self.assertEqual(payload["metadata_gate"]["blockers"], ["PROMPT_INTENT_METADATA_MISSING"])
        self.assertEqual(batch["reasons"], ["PROMPT_INTENT_METADATA_MISSING"])
        self.assertEqual(batch["visual_lane_assessment_state"], "DEFERRED")
        self.assertEqual(batch["lane_counts"], {})
        self.assertIsNone(batch["lane_purity_score"])
        self.assertNotIn("UNKNOWN_LANE_PRESENT", batch["reasons"])
        self.assertEqual(payload["items"][0]["issues"], [])

    def test_complete_metadata_allows_visual_preflight_without_claiming_lane_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self._batch(
                Path(temp_dir),
                with_complete_manifest=True,
                phase="metadata_only",
            )

        batch = payload["batch_preflight"]
        self.assertEqual(payload["metadata_gate"]["status"], "PASS")
        self.assertTrue(payload["metadata_gate"]["runtime_initialization_allowed"])
        self.assertEqual(static_preflight_blockers(payload), [])
        self.assertEqual(batch["status"], "PASS")
        self.assertEqual(batch["recommended_action"], "continue_to_visual_preflight")
        self.assertEqual(batch["visual_lane_assessment_state"], "DEFERRED")
        self.assertEqual(batch["dominant_lane_family"], "deferred")

    def test_visual_phase_without_runtime_is_a_runtime_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = self._batch(
                Path(temp_dir),
                with_complete_manifest=True,
                phase="visual",
            )

        batch = payload["batch_preflight"]
        self.assertTrue(payload["runtime_initialization_attempted"])
        self.assertFalse(payload["runtime_ready"])
        self.assertEqual(payload["observation_mode"], "visual_runtime_unavailable")
        self.assertEqual(payload["metadata_gate"]["status"], "PASS")
        self.assertEqual(batch["status"], "FAIL")
        self.assertEqual(batch["reasons"], ["VISUAL_PREFLIGHT_RUNTIME_UNAVAILABLE"])
        self.assertEqual(batch["visual_lane_assessment_state"], "RUNTIME_UNAVAILABLE")
        self.assertIn("PREFLIGHT_RUNTIME_UNAVAILABLE", payload["items"][0]["issues"][0])


if __name__ == "__main__":
    unittest.main()
