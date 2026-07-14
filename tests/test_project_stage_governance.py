from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_consistency
from core.qa_invariance_status import build_review_invariance_status
from core.qa_optuna import run_optuna_search
from core.qa_pipeline import calibrate_quality_thresholds
from core.qa_project_stage import (
    ProjectStageError,
    ProjectStagePermissionError,
    load_project_stage,
    require_project_permission,
)


def _write_project_stage(root: Path, **permissions: bool) -> Path:
    path = root / "configs" / "project_stage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "project_stage_v1",
                "stage": "TEST_STAGE",
                "permissions": permissions,
            }
        ),
        encoding="utf-8",
    )
    return path


class ProjectStageGovernanceTests(unittest.TestCase):
    def test_repository_stage_denies_current_sensitive_actions(self) -> None:
        stage = load_project_stage()

        self.assertEqual(stage["stage"], "MEASUREMENT_QUALIFICATION")
        for permission in (
            "calibrate_quality_thresholds",
            "optuna_parameter_fitting",
            "winner_bank_freeze",
            "allow_preflight_fail_override",
            "allow_gpu_hardware_risk_override",
        ):
            with self.subTest(permission=permission):
                self.assertFalse(stage["permissions"][permission])

    def test_missing_or_malformed_stage_config_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "configs" / "project_stage.json"
            with patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", path):
                with self.assertRaisesRegex(ProjectStageError, "config is missing"):
                    load_project_stage()

            path = _write_project_stage(root, optuna_parameter_fitting=1)  # type: ignore[arg-type]
            with patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", path):
                with self.assertRaisesRegex(ProjectStageError, "must be boolean"):
                    load_project_stage()
            self.assertTrue(path.exists())

    def test_calibration_is_blocked_before_image_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_path = _write_project_stage(root, calibrate_quality_thresholds=False)
            runtime = SimpleNamespace(
                config=SimpleNamespace(
                    paths=SimpleNamespace(base_dir=root, dir_calib=root / "calib_pass")
                )
            )

            with (
                patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", stage_path),
                patch("core.qa_pipeline.list_images_in_dir") as list_images,
            ):
                with self.assertRaises(ProjectStagePermissionError):
                    calibrate_quality_thresholds(runtime)
                list_images.assert_not_called()

    def test_optuna_is_blocked_before_runtime_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_path = _write_project_stage(root, optuna_parameter_fitting=False)

            with (
                patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", stage_path),
                patch("core.qa_optuna._build_runtime") as build_runtime,
            ):
                with self.assertRaises(ProjectStagePermissionError):
                    run_optuna_search(
                        base_dir=root,
                        report_path=root / "report.json",
                        labels_path=root / "labels.json",
                        search_space_path=root / "search.json",
                    )
                build_runtime.assert_not_called()

    def test_static_preflight_blocks_before_device_or_visual_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir(parents=True)
            (input_dir / "candidate.png").write_bytes(b"metadata-only preflight")

            with (
                patch("check_consistency.create_runtime") as create_runtime,
                patch("check_consistency._apply_cli_device_policy") as apply_device_policy,
            ):
                exit_code = check_consistency.cli(
                    [
                        "--base-dir",
                        str(root),
                        "--workflow",
                        "shot_review",
                        "--profile",
                        "body_gold_fullbody",
                    ]
                )

            self.assertEqual(exit_code, 2)
            create_runtime.assert_not_called()
            apply_device_policy.assert_not_called()

    def test_hardware_risk_override_is_stage_gated_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_path = _write_project_stage(root, allow_gpu_hardware_risk_override=False)

            with (
                patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", stage_path),
                patch(
                    "core.qa_gpu_device_policy.detect_windows_nvidia_whea_risk"
                ) as risk_probe,
            ):
                with self.assertRaises(ProjectStagePermissionError):
                    check_consistency._prepare_cuda_hardware_risk(
                        device="cuda",
                        allow_override=True,
                        context="test",
                    )
                risk_probe.assert_not_called()

    def test_ordinary_qa_auto_cuda_path_checks_hardware_risk(self) -> None:
        args = SimpleNamespace(
            heavy_provider="segformer_body_fusion",
            benchmark_compare_heavy_providers=[],
            device_policy=None,
            surface_occlusion_auto=None,
            require_gpu=False,
            allow_gpu_hardware_risk=False,
        )
        gpu_status = {
            "gpu_ready_for_torch_models": True,
            "gpu_ready_for_insightface": True,
            "torch": {},
            "onnxruntime": {},
        }
        policy = {
            "device": "auto",
            "require_gpu": False,
            "surface_auto_export": "",
            "gpu_status": gpu_status,
            "requirement_blockers": [],
        }

        with (
            patch(
                "check_consistency._resolve_cli_heavy_provider_for_device_policy",
                return_value="segformer_body_fusion",
            ),
            patch("core.qa_gpu_device_policy.apply_truth_fusion_gpu_env", return_value=policy),
            patch(
                "check_consistency._prepare_cuda_hardware_risk",
                return_value={"probe_state": "OBSERVED", "blockers": []},
            ) as risk_check,
            patch("check_consistency._print_gpu_policy_summary"),
        ):
            result = check_consistency._apply_cli_device_policy(
                parser=argparse.ArgumentParser(add_help=False),
                args=args,
                base_dir=Path(__file__).resolve().parent.parent,
                selected_profile="body_gold_fullbody",
                effective_mode="qa",
            )

        self.assertIsNotNone(result)
        risk_check.assert_called_once()
        self.assertEqual(risk_check.call_args.kwargs["device"], "auto")
        self.assertIs(risk_check.call_args.kwargs["gpu_status"], gpu_status)

    def test_winner_bank_freeze_output_obeys_project_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_path = _write_project_stage(root, winner_bank_freeze=False)
            output_file = root / "outputs" / "review_invariance_status.json"

            with patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", stage_path):
                payload = build_review_invariance_status(base_dir=root, output_file=output_file)

            self.assertFalse(payload["winner_bank_freeze_allowed"])
            self.assertFalse(payload["winner_bank_freeze_prerequisites_satisfied"])
            self.assertTrue(payload["winner_bank_mutable_memory_allowed"])
            self.assertEqual(payload["project_stage"], "TEST_STAGE")

    def test_explicit_permission_requires_the_stage_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_path = _write_project_stage(root, optuna_parameter_fitting=True)

            with patch("core.qa_project_stage.PROJECT_STAGE_CONFIG_PATH", stage_path):
                stage = require_project_permission(
                    "optuna_parameter_fitting",
                    action="test permission",
                )

            self.assertEqual(stage["stage"], "TEST_STAGE")


if __name__ == "__main__":
    unittest.main()
