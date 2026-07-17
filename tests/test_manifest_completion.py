from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.qa_manifest_completion import build_manifest_completion_plan


class ManifestCompletionTests(unittest.TestCase):
    def test_explicit_input_target_reports_only_the_current_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "input_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "input_manifest_v1",
                        "items": [
                            {
                                "image": "missing.png",
                                "prompt_id": "",
                                "seed": None,
                                "seed_unavailable_reason": "",
                                "anchor_source": "",
                                "intended_view": "",
                            },
                            {
                                "image": "ready.png",
                                "prompt_id": "prompt-02",
                                "seed": None,
                                "seed_unavailable_reason": "nano_banana_seed_not_exposed",
                                "anchor_source": "confirmed-truth-inputs",
                                "intended_view": "front",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_file = root / "outputs" / "current_input_manifest_completion_plan.json"

            result = build_manifest_completion_plan(
                base_dir=root,
                output_file=output_file,
                targets=(("current_batch", input_dir),),
            )

            self.assertEqual(result["scope"], "explicit_input_targets")
            self.assertEqual(result["blocked_splits"], ["current_batch"])
            self.assertIsNone(result["ready_for_clean_replay"])
            self.assertFalse(result["ready_for_visual_preflight"])
            self.assertEqual(len(result["splits"]), 1)
            current = result["splits"][0]
            self.assertIsNone(current["ready_for_clean_replay"])
            self.assertEqual(current["item_count"], 2)
            self.assertEqual(current["missing_counts"]["prompt_id"], 1)
            self.assertEqual(current["missing_counts"]["seed"], 1)
            self.assertEqual(current["field_coverage"]["intended_view"], 0.5)
            self.assertEqual(
                {row["field"] for row in current["manual_required_fields"]},
                {
                    "prompt_id",
                    "seed_or_seed_unavailable_reason",
                    "anchor_source",
                    "intended_view",
                },
            )
            self.assertIn('--input-dir "input"', current["example_fill_command"])
            self.assertIn("Rerun preflight_batch", result["next_actions"][-1])
            self.assertTrue(output_file.exists())

    def test_default_scope_preserves_front_and_three_quarter_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for lane in ("front", "three_quarter"):
                input_dir = root / "input_split" / lane
                input_dir.mkdir(parents=True)
                (input_dir / "input_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "input_manifest_v1",
                            "items": [
                                {
                                    "image": f"{lane}.png",
                                    "prompt_id": f"prompt-{lane}",
                                    "seed": 7,
                                    "anchor_source": "confirmed-truth-inputs",
                                    "intended_view": lane,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            result = build_manifest_completion_plan(
                base_dir=root,
                output_file=root / "outputs" / "input_manifest_completion_plan.json",
            )

            self.assertEqual(result["scope"], "default_clean_replay_splits")
            self.assertEqual(result["overall_status"], "READY_FOR_CLEAN_REPLAY")
            self.assertEqual([row["split"] for row in result["splits"]], ["front", "three_quarter"])


if __name__ == "__main__":
    unittest.main()
