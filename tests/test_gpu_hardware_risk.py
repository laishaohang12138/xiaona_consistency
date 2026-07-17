from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_consistency

from core.qa_gpu_device_policy import (
    build_gpu_runtime_safety_preflight,
    configured_default_device,
    detect_windows_nvidia_whea_risk,
    evaluate_nvidia_whea_gate,
    nvidia_whea_risk_blockers,
)


def _watermark_policy() -> dict:
    return {
        "schema_version": "gpu_runtime_policy_v1",
        "execution_preference": "GPU_FIRST",
        "whea_gate": {
            "mode": "ACKNOWLEDGED_EVENT_WATERMARK",
            "acknowledged_boot_id": "boot-a",
            "acknowledged_event_count_since_boot": 74,
            "acknowledged_through": "2026-07-16T10:34:39.746701+00:00",
            "acknowledged_by": "operator",
            "acknowledged_at_utc": "2026-07-16T15:04:19+00:00",
            "reason_code": "RESOLVED_SOFTWARE_STACK_CONFLICT",
        },
    }


class NvidiaWheaRiskTests(unittest.TestCase):
    def test_observed_nvidia_whea_is_a_blocker(self) -> None:
        blockers = nvidia_whea_risk_blockers(
            {
                "probe_state": "OBSERVED",
                "nvidia_whea17_count_since_boot": 3,
            }
        )

        self.assertEqual(blockers, ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"])

    def test_clean_observation_has_no_blocker(self) -> None:
        self.assertEqual(
            nvidia_whea_risk_blockers(
                {
                    "probe_state": "OBSERVED",
                    "nvidia_whea17_count_since_boot": 0,
                }
            ),
            [],
        )

    def test_failed_probe_is_conservatively_blocked(self) -> None:
        self.assertEqual(
            nvidia_whea_risk_blockers({"probe_state": "FAILED"}),
            ["NVIDIA_WHEA_RISK_UNASSESSED"],
        )

    def test_non_windows_probe_is_not_applicable(self) -> None:
        self.assertEqual(
            nvidia_whea_risk_blockers({"probe_state": "NOT_APPLICABLE"}),
            [],
        )

    def test_windows_probe_uses_stable_boot_event_fallback(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "probe_state": "OBSERVED",
                    "risk_state": "NO_NVIDIA_WHEA_OBSERVED",
                    "boot_id": "2026-07-14T10:42:55.2066046Z",
                    "boot_time": "2026-07-14T18:42:55.2066046+08:00",
                    "boot_time_source": "kernel_general_event_12",
                    "event_query_state": "NO_MATCHING_EVENTS",
                    "whea17_count_since_boot": 0,
                    "nvidia_whea17_count_since_boot": 0,
                    "latest_nvidia_whea_time": None,
                }
            ),
        )
        with patch("core.qa_gpu_device_policy.os.name", "nt"), patch(
            "core.qa_gpu_device_policy.subprocess.run",
            return_value=completed,
        ) as run_probe:
            result = detect_windows_nvidia_whea_risk()

        script = run_probe.call_args.args[0][-1]
        self.assertIn("kernel_general_event_12", script)
        self.assertNotIn("TickCount64", script)
        self.assertEqual(result["boot_time_source"], "kernel_general_event_12")
        self.assertEqual(result["blockers"], [])

    def test_runtime_safety_preflight_is_narrow_and_fail_closed(self) -> None:
        failed = build_gpu_runtime_safety_preflight(
            {
                "probe_state": "OBSERVED",
                "risk_state": "NVIDIA_PCIE_WHEA_OBSERVED",
                "nvidia_whea17_count_since_boot": 4,
                "blockers": ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"],
            }
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(failed["gpu_runtime_allowed_by_whea_gate"])
        self.assertFalse(failed["runtime_or_provider_initialized_by_this_preflight"])
        self.assertIn("PCIe link stability under load", failed["does_not_certify"])

        clean = build_gpu_runtime_safety_preflight(
            {
                "probe_state": "OBSERVED",
                "risk_state": "NO_NVIDIA_WHEA_OBSERVED",
                "nvidia_whea17_count_since_boot": 0,
                "blockers": [],
            }
        )
        self.assertEqual(clean["status"], "PASS")
        self.assertTrue(clean["gpu_runtime_allowed_by_whea_gate"])

        unassessed = build_gpu_runtime_safety_preflight(
            {
                "probe_state": "FAILED",
                "risk_state": "UNASSESSED",
                "blockers": [],
            }
        )
        self.assertEqual(unassessed["status"], "FAIL")
        self.assertEqual(unassessed["blockers"], ["NVIDIA_WHEA_RISK_UNASSESSED"])

    def test_runtime_safety_cli_never_initializes_visual_runtime(self) -> None:
        risk = {
            "probe_state": "OBSERVED",
            "risk_state": "NVIDIA_PCIE_WHEA_OBSERVED",
            "nvidia_whea17_count_since_boot": 4,
            "blockers": ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"],
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "core.qa_gpu_device_policy.detect_windows_nvidia_whea_risk",
            return_value=risk,
        ), patch("check_consistency.create_runtime") as create_runtime, redirect_stdout(io.StringIO()):
            root = Path(temp_dir)
            result = check_consistency.cli(
                [
                    "--base-dir",
                    str(root),
                    "--workflow",
                    "gpu_runtime_safety_preflight",
                ]
            )

            create_runtime.assert_not_called()
            self.assertEqual(result, 2)
            payload = json.loads(
                (root / "outputs" / "gpu_runtime_safety_preflight.json").read_text(encoding="utf-8")
            )
            self.assertFalse(payload["runtime_or_provider_initialized_by_this_preflight"])

    def test_acknowledged_historical_events_pass_but_remain_visible(self) -> None:
        result = evaluate_nvidia_whea_gate(
            {
                "probe_state": "OBSERVED",
                "boot_id": "boot-a",
                "nvidia_whea17_count_since_boot": 74,
                "latest_nvidia_whea_time": "2026-07-16T18:34:39.746701+08:00",
            },
            _watermark_policy(),
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["disposition"], "ACKNOWLEDGED_HISTORICAL_EVENTS")
        self.assertTrue(result["acknowledgement_applied"])
        self.assertEqual(result["raw_nvidia_whea17_count_since_boot"], 74)
        self.assertEqual(result["blockers"], [])
        preflight = build_gpu_runtime_safety_preflight(
            {
                "probe_state": "OBSERVED",
                "risk_state": "NVIDIA_PCIE_WHEA_OBSERVED",
                "boot_id": "boot-a",
                "nvidia_whea17_count_since_boot": 74,
                "latest_nvidia_whea_time": "2026-07-16T18:34:39.746701+08:00",
                "gate_evaluation": result,
                "blockers": [],
            }
        )
        self.assertEqual(preflight["status"], "PASS")
        self.assertTrue(preflight["gpu_runtime_allowed_by_whea_gate"])

    def test_same_boot_count_advance_blocks_even_when_timestamp_matches(self) -> None:
        result = evaluate_nvidia_whea_gate(
            {
                "probe_state": "OBSERVED",
                "boot_id": "boot-a",
                "nvidia_whea17_count_since_boot": 75,
                "latest_nvidia_whea_time": "2026-07-16T18:34:39.746701+08:00",
            },
            _watermark_policy(),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["event_count_advanced"])
        self.assertEqual(
            result["blockers"],
            ["NVIDIA_PCIE_WHEA17_AFTER_ACKNOWLEDGED_BASELINE"],
        )

    def test_later_event_on_a_new_boot_blocks(self) -> None:
        result = evaluate_nvidia_whea_gate(
            {
                "probe_state": "OBSERVED",
                "boot_id": "boot-b",
                "nvidia_whea17_count_since_boot": 1,
                "latest_nvidia_whea_time": "2026-07-17T01:00:00+00:00",
            },
            _watermark_policy(),
        )

        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["same_acknowledged_boot"])
        self.assertTrue(result["event_timestamp_advanced"])

    def test_gpu_first_policy_resolves_to_cuda(self) -> None:
        self.assertEqual(configured_default_device(_watermark_policy()), "cuda")


if __name__ == "__main__":
    unittest.main()
