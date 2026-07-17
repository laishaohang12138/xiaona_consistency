from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.qa_gpu_execution_guard import (
    GpuExecutionGuardError,
    build_whea_delta,
    create_gpu_execution_guard,
    resolve_gpu_resource,
)
from core.qa_io import acquire_workflow_lock


def _policy(device: str = "cuda") -> dict:
    return {
        "device": device,
        "gpu_status": {
            "gpu_ready_for_torch_models": device != "cpu",
            "gpu_ready_for_insightface": device != "cpu",
        },
    }


def _whea(count: int, *, boot: str = "2026-07-14T00:00:00+08:00") -> dict:
    return {
        "probe_state": "OBSERVED",
        "risk_state": "NO_NVIDIA_WHEA_OBSERVED" if count == 0 else "NVIDIA_PCIE_WHEA_OBSERVED",
        "boot_time": boot,
        "nvidia_whea17_count_since_boot": count,
        "blockers": [] if count == 0 else ["NVIDIA_PCIE_WHEA17_SINCE_BOOT"],
    }


class GpuExecutionGuardTests(unittest.TestCase):
    def test_gpu_resource_is_device_scoped(self) -> None:
        self.assertIsNone(resolve_gpu_resource(_policy("cpu")))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            resource = resolve_gpu_resource(_policy("cuda"))
        self.assertEqual(resource["logical_device_id"], "cuda:0")
        self.assertEqual(resource["physical_selector"], "2")
        self.assertEqual(resource["resource_id"], "gpu-2")

    def test_whea_delta_requires_same_boot_and_detects_new_events(self) -> None:
        clean = build_whea_delta(_whea(0), _whea(0))
        self.assertEqual(clean["status"], "NO_NEW_EVENTS")
        self.assertEqual(clean["nvidia_whea17_delta"], 0)
        self.assertEqual(clean["blockers"], [])

        changed = build_whea_delta(_whea(0), _whea(2))
        self.assertEqual(changed["nvidia_whea17_delta"], 2)
        self.assertEqual(changed["blockers"], ["NVIDIA_PCIE_WHEA17_DURING_RUN"])

        rebooted = build_whea_delta(_whea(0), _whea(0, boot="later"))
        self.assertFalse(rebooted["same_boot"])
        self.assertEqual(rebooted["blockers"], ["NVIDIA_WHEA_BOOT_CHANGED_DURING_RUN"])

    def test_preexisting_whea_blocks_before_workload_and_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = create_gpu_execution_guard(
                base_dir=root,
                artifact_dir=root / "artifacts",
                policy=_policy(),
                workload_class="shot_review",
                lock_timeout_s=0.0,
            )
            self.assertIsNotNone(guard)
            observed_risk = _whea(4)
            observed_risk["blockers"] = []
            with patch(
                "core.qa_gpu_execution_guard.detect_windows_nvidia_whea_risk",
                return_value=observed_risk,
            ):
                with self.assertRaisesRegex(GpuExecutionGuardError, "before runtime"):
                    with guard:  # type: ignore[union-attr]
                        self.fail("workload must not start")

            self.assertFalse(guard.lock_path.exists())  # type: ignore[union-attr]
            payload = json.loads(guard.artifact_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self.assertEqual(payload["status"], "BLOCKED_BEFORE_RUNTIME")
            self.assertFalse(payload["qualification_evidence_eligible"])

    def test_clean_guard_records_no_event_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = create_gpu_execution_guard(
                base_dir=root,
                artifact_dir=root / "artifacts",
                policy=_policy(),
                workload_class="body_repeatability",
                lock_timeout_s=0.0,
            )
            with patch(
                "core.qa_gpu_execution_guard.detect_windows_nvidia_whea_risk",
                side_effect=[_whea(0), _whea(0)],
            ):
                with guard as active:  # type: ignore[union-attr]
                    self.assertEqual(active.whea_before["nvidia_whea17_count_since_boot"], 0)

            payload = json.loads(guard.artifact_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertEqual(payload["whea_delta"]["nvidia_whea17_delta"], 0)
            self.assertTrue(payload["qualification_evidence_eligible"])
            self.assertFalse(guard.lock_path.exists())  # type: ignore[union-attr]

    def test_acknowledged_historical_events_can_qualify_when_delta_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = create_gpu_execution_guard(
                base_dir=root,
                artifact_dir=root / "artifacts",
                policy=_policy(),
                workload_class="shot_review",
                lock_timeout_s=0.0,
            )
            acknowledged = _whea(74)
            acknowledged["blockers"] = []
            acknowledged["gate_evaluation"] = {
                "status": "PASS",
                "disposition": "ACKNOWLEDGED_HISTORICAL_EVENTS",
                "acknowledgement_applied": True,
                "blockers": [],
            }
            with patch(
                "core.qa_gpu_execution_guard.detect_windows_nvidia_whea_risk",
                side_effect=[acknowledged, acknowledged],
            ):
                with guard:  # type: ignore[union-attr]
                    pass

            payload = json.loads(guard.artifact_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self.assertEqual(payload["status"], "COMPLETED")
            self.assertEqual(payload["whea_delta"]["nvidia_whea17_delta"], 0)
            self.assertTrue(payload["qualification_evidence_eligible"])

    def test_new_whea_marks_completed_run_non_qualifying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = create_gpu_execution_guard(
                base_dir=root,
                artifact_dir=root / "artifacts",
                policy=_policy(),
                workload_class="identity_repeatability",
                lock_timeout_s=0.0,
            )
            with patch(
                "core.qa_gpu_execution_guard.detect_windows_nvidia_whea_risk",
                side_effect=[_whea(0), _whea(1)],
            ):
                with self.assertRaisesRegex(GpuExecutionGuardError, "qualification failed"):
                    with guard:  # type: ignore[union-attr]
                        pass

            payload = json.loads(guard.artifact_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self.assertEqual(payload["status"], "COMPLETED_WITH_HARDWARE_RISK")
            self.assertEqual(payload["post_run_blockers"], ["NVIDIA_PCIE_WHEA17_DURING_RUN"])
            self.assertFalse(payload["qualification_evidence_eligible"])

    def test_same_device_lock_blocks_a_second_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock_path = root / "outputs" / ".gpu_device_locks" / "gpu-0.lock"
            first = acquire_workflow_lock(lock_path, owner="first", timeout_s=0.0)
            try:
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                    guard = create_gpu_execution_guard(
                        base_dir=root,
                        artifact_dir=root / "artifacts",
                        policy=_policy(),
                        workload_class="second",
                        lock_timeout_s=0.0,
                    )
                with patch(
                    "core.qa_gpu_execution_guard.detect_windows_nvidia_whea_risk"
                ) as risk_probe:
                    with self.assertRaisesRegex(GpuExecutionGuardError, "guard setup failed"):
                        with guard:  # type: ignore[union-attr]
                            pass
                risk_probe.assert_not_called()
                payload = json.loads(guard.artifact_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
                self.assertEqual(payload["status"], "GUARD_SETUP_FAILED")
            finally:
                first.release()

    def test_dead_pid_lock_is_reclaimed_without_age_wait(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "gpu-0.lock"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": "workflow_file_lock_v2",
                        "lock_token": "dead-owner",
                        "pid": 999999,
                        "hostname": socket.gethostname(),
                        "owner": "old-run",
                        "created_at_epoch": 0,
                    }
                ),
                encoding="utf-8",
            )
            with patch("core.qa_io._pid_is_alive", return_value=False):
                lock = acquire_workflow_lock(
                    lock_path,
                    owner="replacement",
                    timeout_s=0.0,
                )
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["owner"], "replacement")
            finally:
                lock.release()
            self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
