# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from pathlib import Path

from core.qa_pipeline import (
    calibrate_quality_thresholds,
    create_runtime,
    load_anchor_set,
    load_thresholds_from_file,
    main,
    print_runtime_config,
    run_pipeline,
)
from core.qa_runtime import (
    AnchorSet,
    EngineState,
    FaceFeat,
    PoseFeat,
    ProjectPaths,
    QualityThresholds,
    RuntimeConfig,
    RuntimeContext,
    save_thresholds_to_file,
)


BASE_DIR = Path(__file__).resolve().parent

__all__ = [
    "AnchorSet",
    "EngineState",
    "FaceFeat",
    "PoseFeat",
    "ProjectPaths",
    "QualityThresholds",
    "RuntimeConfig",
    "RuntimeContext",
    "calibrate_quality_thresholds",
    "create_runtime",
    "load_anchor_set",
    "load_thresholds_from_file",
    "main",
    "print_runtime_config",
    "run_pipeline",
    "save_thresholds_to_file",
]


if __name__ == "__main__":
    main(base_dir=BASE_DIR)
