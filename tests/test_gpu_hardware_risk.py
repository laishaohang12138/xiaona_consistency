from __future__ import annotations

import unittest

from core.qa_gpu_device_policy import nvidia_whea_risk_blockers


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


if __name__ == "__main__":
    unittest.main()
