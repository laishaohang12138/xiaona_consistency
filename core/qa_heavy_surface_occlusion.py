import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

from .providers import HeavyEvidenceProvider
from .qa_utils import dedupe_keep_order

_PROVIDER_NAME = "clothing_surface_occlusion_bridge"
_PROVIDER_FAMILY = "clothing_invariant_surface"
_PROVIDER_VERSION = "surface_occlusion_sidecar_bridge_v1"
_MODEL_ID = "densepose_or_sam2_sidecar_bridge"
_ARTIFACT_SCHEMA = "clothing_surface_occlusion_artifact_v1"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    numeric = _safe_float(value)
    return None if numeric is None else round(float(numeric), digits)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _pick(raw: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw.get(key)
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    for key in keys:
        if key in metrics:
            return metrics.get(key)
    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    for key in keys:
        if key in summary:
            return summary.get(key)
    return None


def _sidecar_candidates(image_path: Path) -> List[Path]:
    return [
        image_path.with_suffix(image_path.suffix + ".surface_occlusion.json"),
        image_path.with_name(f"{image_path.stem}.surface_occlusion.json"),
        image_path.with_suffix(image_path.suffix + ".densepose.json"),
        image_path.with_name(f"{image_path.stem}.densepose.json"),
        image_path.with_suffix(image_path.suffix + ".sam2.json"),
        image_path.with_name(f"{image_path.stem}.sam2.json"),
    ]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _auto_export_order() -> List[str]:
    raw = str(os.getenv("XIAONA_SURFACE_OCCLUSION_AUTO_EXPORT", "") or "").strip().lower()
    if not raw or raw in {"off", "none", "false", "0"}:
        return []
    aliases = {
        "densepose_then_sam2": ["densepose", "sam2"],
        "sam2_then_densepose": ["sam2", "densepose"],
        "all": ["densepose", "sam2"],
    }
    if raw in aliases:
        return aliases[raw]
    order: List[str] = []
    for item in raw.replace(";", ",").split(","):
        name = item.strip().lower()
        if name in {"densepose", "sam2"} and name not in order:
            order.append(name)
    return order


def _run_export_command(command: List[str], *, timeout_s: int) -> Dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:
        return {"ok": False, "reason": f"surface_export_exception:{exc}", "command": command}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-800:],
        "stderr": (completed.stderr or "")[-800:],
        "command": command,
    }


def _try_auto_export_sidecar(image_path: Path) -> List[Dict[str, Any]]:
    order = _auto_export_order()
    if not order:
        return []
    repo_root = Path(__file__).resolve().parents[1]
    attempts: List[Dict[str, Any]] = []
    device = str(
        os.getenv(
            "XIAONA_SURFACE_OCCLUSION_DEVICE",
            os.getenv("XIAONA_SAM2_DEVICE", os.getenv("XIAONA_DENSEPOSE_DEVICE", "cuda")),
        )
        or "cuda"
    ).strip()
    for exporter in order:
        if exporter == "densepose":
            output = image_path.with_suffix(image_path.suffix + ".densepose.json")
            command = [
                sys.executable,
                str(repo_root / "export_surface_occlusion_densepose.py"),
                "--image",
                str(image_path),
                "--output",
                str(output),
                "--device",
                device,
            ]
            result = _run_export_command(command, timeout_s=1800)
        else:
            output = image_path.with_suffix(image_path.suffix + ".surface_occlusion.json")
            command = [
                sys.executable,
                str(repo_root / "export_surface_occlusion_sam2.py"),
                "--image",
                str(image_path),
                "--output",
                str(output),
                "--device",
                device,
            ]
            result = _run_export_command(command, timeout_s=1200)
        result.update({"exporter": exporter, "output": str(output), "device": device})
        attempts.append(result)
        if output.exists() and _load_json(output) is not None:
            break
    return attempts


def _metric_spec(metric_name: str, value: Any, *, confidence: Optional[float], coverage: Optional[float]) -> Dict[str, Any]:
    return {
        "metric_name": str(metric_name),
        "metric_value": _round_or_none(_safe_float(value)),
        "confidence": _round_or_none(confidence),
        "coverage": _round_or_none(coverage),
        "lane_scope": "report_item",
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "failure_reason": None,
        "signature_ref": {
            "kind": "scalar",
            "name": str(metric_name),
            "available": _safe_float(value) is not None,
        },
    }


def _available_metric_count(values: Sequence[Optional[float]]) -> int:
    return sum(1 for value in values if _safe_float(value) is not None)


def _normalize_artifact(raw: Dict[str, Any], *, image_path: Path, sidecar_file: Path) -> Dict[str, Any]:
    visible_body_ratio = _safe_float(_pick(raw, "visible_body_ratio", "densepose_visible_body_ratio", "human_surface_visible_ratio"))
    visible_face_ratio = _safe_float(_pick(raw, "visible_face_ratio", "face_visible_ratio"))
    visible_arm_ratio = _safe_float(_pick(raw, "visible_arm_ratio", "arm_visible_ratio"))
    visible_leg_ratio = _safe_float(_pick(raw, "visible_leg_ratio", "leg_visible_ratio"))
    visible_body_surface_alignment = _safe_float(
        _pick(raw, "visible_body_surface_alignment", "densepose_surface_alignment", "body_surface_alignment")
    )
    garment_occlusion_index = _safe_float(_pick(raw, "garment_occlusion_index", "occlusion_index", "clothing_occlusion_index"))
    if garment_occlusion_index is None:
        clothing_mask_coverage = _safe_float(_pick(raw, "clothing_mask_coverage", "garment_coverage_ratio"))
        if clothing_mask_coverage is not None:
            garment_occlusion_index = _clamp(clothing_mask_coverage)
    garment_boundary_risk = _safe_float(_pick(raw, "garment_boundary_risk", "silhouette_boundary_risk"))
    clothing_surface_confidence = _safe_float(
        _pick(raw, "clothing_surface_confidence", "surface_confidence", "occlusion_confidence", "confidence")
    )
    if clothing_surface_confidence is None:
        clothing_surface_confidence = _safe_float(raw.get("confidence"), 0.0)

    metric_values = [
        visible_body_ratio,
        visible_face_ratio,
        visible_arm_ratio,
        visible_leg_ratio,
        visible_body_surface_alignment,
        garment_occlusion_index,
        garment_boundary_risk,
        clothing_surface_confidence,
    ]
    coverage = float(_available_metric_count(metric_values) / max(1, len(metric_values)))
    confidence = _weighted_confidence(clothing_surface_confidence, coverage, visible_body_surface_alignment)
    specs = [
        _metric_spec("visible_body_surface_alignment", visible_body_surface_alignment, confidence=confidence, coverage=coverage),
        _metric_spec("garment_occlusion_index", garment_occlusion_index, confidence=confidence, coverage=coverage),
        _metric_spec("garment_boundary_risk", garment_boundary_risk, confidence=confidence, coverage=coverage),
        _metric_spec("visible_body_ratio", visible_body_ratio, confidence=confidence, coverage=coverage),
        _metric_spec("visible_face_ratio", visible_face_ratio, confidence=confidence, coverage=coverage),
        _metric_spec("visible_arm_ratio", visible_arm_ratio, confidence=confidence, coverage=coverage),
        _metric_spec("visible_leg_ratio", visible_leg_ratio, confidence=confidence, coverage=coverage),
        _metric_spec("clothing_surface_confidence", clothing_surface_confidence, confidence=confidence, coverage=coverage),
    ]
    return {
        "ok": coverage > 0.0,
        "provider_name": _PROVIDER_NAME,
        "provider_family": _PROVIDER_FAMILY,
        "provider_version": _PROVIDER_VERSION,
        "model_id": str(raw.get("model_id") or _MODEL_ID),
        "device": raw.get("device"),
        "source_path": str(image_path.resolve()),
        "sidecar_file": str(sidecar_file),
        "cache_state": "sidecar",
        "confidence": _round_or_none(confidence),
        "coverage": _round_or_none(coverage),
        "metric_specs": specs,
        "summary": {
            "schema_version": str(raw.get("schema_version") or _ARTIFACT_SCHEMA),
            "sidecar_file": str(sidecar_file),
            "source_provider": raw.get("provider_name"),
            "candidate_count": 1,
            "guidance": [
                "Surface occlusion sidecar is available; use it to separate clothing coverage from body identity evidence.",
            ],
        },
        "reasons": dedupe_keep_order(str(reason) for reason in raw.get("reasons", []) if str(reason).strip())[:12],
    }


def _weighted_confidence(
    clothing_surface_confidence: Optional[float],
    coverage: Optional[float],
    visible_body_surface_alignment: Optional[float],
) -> Optional[float]:
    values = [
        (clothing_surface_confidence, 0.48),
        (coverage, 0.34),
        (visible_body_surface_alignment, 0.18),
    ]
    numerator = 0.0
    denominator = 0.0
    for value, weight in values:
        numeric = _safe_float(value)
        if numeric is None:
            continue
        numerator += _clamp(numeric) * float(weight)
        denominator += float(weight)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


class ClothingSurfaceOcclusionBridgeProvider(HeavyEvidenceProvider):
    provider_name = _PROVIDER_NAME
    provider_family = _PROVIDER_FAMILY
    provider_version = _PROVIDER_VERSION

    def get_provider_status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "external_sidecar",
            "integration_state": "sidecar_bridge_ready",
            "requires_candidate_artifact": True,
            "auto_export_order": _auto_export_order(),
            "candidate_artifact_suffixes": [
                ".surface_occlusion.json",
                ".densepose.json",
                ".sam2.json",
            ],
            "evidence_schema_version": "heavy_evidence_v1",
        }

    def get_heavy_evidence_metrics(self, runtime: Any, image_path: Path) -> Dict[str, Any]:
        del runtime
        resolved = Path(image_path).resolve()
        for sidecar in _sidecar_candidates(resolved):
            if not sidecar.exists():
                continue
            payload = _load_json(sidecar)
            if payload is None:
                continue
            return _normalize_artifact(payload, image_path=resolved, sidecar_file=sidecar)
        auto_export_attempts = _try_auto_export_sidecar(resolved)
        for sidecar in _sidecar_candidates(resolved):
            if not sidecar.exists():
                continue
            payload = _load_json(sidecar)
            if payload is None:
                continue
            artifact = _normalize_artifact(payload, image_path=resolved, sidecar_file=sidecar)
            summary = dict(artifact.get("summary") or {})
            summary["auto_export_attempts"] = auto_export_attempts
            artifact["summary"] = summary
            return artifact
        return {
            "ok": False,
            "provider_name": self.provider_name,
            "provider_family": self.provider_family,
            "provider_version": self.provider_version,
            "model_id": _MODEL_ID,
            "device": "external_sidecar",
            "source_path": str(resolved),
            "cache_state": "missing",
            "confidence": 0.0,
            "coverage": 0.0,
            "metric_specs": [],
            "summary": {
                "candidate_count": 1,
                "auto_export_order": _auto_export_order(),
                "auto_export_attempts": auto_export_attempts,
                "guidance": [
                    "DensePose/SAM2 surface occlusion sidecar is missing; clothing-invariant review falls back to parser and body topology.",
                ],
            },
            "reasons": ["SURFACE_OCCLUSION_SIDECAR_MISSING"],
        }
