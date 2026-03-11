from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .providers import build_provider_bundle
from .qa_consistency import (
    apply_consistency_soft_gate,
    extract_body_constitution_metrics,
    extract_depth_3d_lite_metrics,
    extract_skin_consistency_metrics,
)
from .qa_features import extract_face_feat, extract_pose_feat, init_engines
from .qa_runtime import (
    AnchorSet,
    FaceFeat,
    PoseFeat,
    RuntimeContext,
    anchor_registry_summary,
    create_runtime_config,
    load_thresholds_from_file as load_runtime_thresholds_from_file,
    resolve_anchor_paths,
    save_thresholds_to_file,
)
from .qa_scoring import (
    build_quality_reference_stats,
    classify_module,
    filter_face_anchors_by_view,
    fuse_overall,
    get_identity_anchor_pool,
    get_profile_policy,
    get_quality_anchor_pool,
    get_stats_for_bucket,
    make_recommendations,
    score_face_against_anchor_set,
    score_full_against_anchor_set,
    score_upper_against_anchor_set,
)
from .qa_utils import (
    SKIMAGE_SSIM_AVAILABLE,
    dedupe_keep_order,
    estimate_view_bucket_and_side,
    get_quality_tolerances_by_face_size,
    image_read_bgr,
    list_images_in_dir,
    robust_percentile,
    valid_face_feats,
)


def create_runtime(base_dir: Optional[Path] = None) -> RuntimeContext:
    config = create_runtime_config(base_dir)
    providers = build_provider_bundle(config.provider_policy)
    engines = init_engines()
    return RuntimeContext(config=config, providers=providers, engines=engines)


def print_runtime_config(runtime: RuntimeContext) -> None:
    config = runtime.config
    print(f"[CONFIG] RUN_MODE={config.run_mode}")
    print(f"[CONFIG] ACTIVE_PROFILE={config.review.active_profile}")
    print(f"[CONFIG] CONFIG_DIR={config.paths.config_dir}")
    print(f"[CONFIG] EXTERNAL_CONFIG_STATUS={config.external_config_status}")
    print(f"[CONFIG] PROVIDER_POLICY={config.provider_policy}")
    print(f"[CONFIG] PROVIDERS={runtime.providers.describe()}")
    print(f"[CONFIG] ANCHOR_REGISTRY_SUMMARY={anchor_registry_summary(config)}")
    print("[CONFIG] FACE_CONF_MAP = linear_map_to_01(bbox_ratio, 0.006, 0.035)")
    print(f"[CONFIG] FACE_NO_RELIABLE_SIGNAL_TH = {config.review.face_no_signal_conf_th}")
    print(f"[CONFIG] MIN_CONF_FOR_STRICT_FAIL = {config.review.min_conf_for_strict_fail}")
    print(f"[CONFIG] CONSISTENCY_MODE={config.consistency.mode}")


def load_anchor_set(runtime: RuntimeContext) -> AnchorSet:
    anchors = AnchorSet()
    resolved_anchor_paths = resolve_anchor_paths(runtime.config)

    anchors.meta["anchor_source_mode"] = runtime.config.provider_policy.get(
        "anchor_source", "registry_then_directory_fallback"
    )
    anchors.meta["face_paths"] = list(resolved_anchor_paths.get("face_paths", []))
    anchors.meta["upper_paths"] = list(resolved_anchor_paths.get("upper_paths", []))
    anchors.meta["full_paths"] = list(resolved_anchor_paths.get("full_paths", []))

    print("\n[初始化] 加载 Anchor Set...")
    print(f"  Anchor Source Mode: {anchors.meta['anchor_source_mode']}")
    print(f"  Face Anchors : {len(anchors.meta['face_paths'])}")
    print(f"  Upper Anchors: {len(anchors.meta['upper_paths'])}")
    print(f"  Full Anchors : {len(anchors.meta['full_paths'])}")

    for path_str in anchors.meta["face_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        face_feat = extract_face_feat(runtime, img, path) if img is not None else FaceFeat(
            ok=False,
            reasons=["IMAGE_READ_ERROR"],
            source_path=str(path),
        )
        anchors.face_feats.append(face_feat)

    for path_str in anchors.meta["upper_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            anchors.upper_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.upper_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(path)))
            continue
        anchors.upper_pose_feats.append(extract_pose_feat(runtime, img))
        anchors.upper_face_feats.append(extract_face_feat(runtime, img, path))

    for path_str in anchors.meta["full_paths"]:
        path = Path(path_str)
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            anchors.full_pose_feats.append(PoseFeat(ok=False, reasons=["IMAGE_READ_ERROR"]))
            anchors.full_face_feats.append(FaceFeat(ok=False, reasons=["IMAGE_READ_ERROR"], source_path=str(path)))
            continue
        anchors.full_pose_feats.append(extract_pose_feat(runtime, img))
        anchors.full_face_feats.append(extract_face_feat(runtime, img, path))

    print(f"  Upper Face-like Quality Refs: {len(valid_face_feats(anchors.upper_face_feats))}")
    print(f"  Full  Face-like Quality Refs: {len(valid_face_feats(anchors.full_face_feats))}")

    face_front = sum(1 for feat in anchors.face_feats if feat.ok and "/front/" in str(feat.source_path).replace("\\", "/").lower())
    face_3q = sum(1 for feat in anchors.face_feats if feat.ok and "/three_quarter/" in str(feat.source_path).replace("\\", "/").lower())
    face_profile = sum(
        1
        for feat in anchors.face_feats
        if feat.ok and (
            "/profile_like/" in str(feat.source_path).replace("\\", "/").lower()
            or "/profile/" in str(feat.source_path).replace("\\", "/").lower()
        )
    )
    print(f"  Face Buckets => front={face_front} | three_quarter={face_3q} | profile_like={face_profile}")
    return anchors


def calibrate_quality_thresholds(
    runtime: RuntimeContext,
    calib_dir: Optional[Path] = None,
) -> Dict[str, float]:
    target_dir = calib_dir or runtime.config.paths.dir_calib
    images = list_images_in_dir(target_dir)
    if len(images) == 0:
        raise RuntimeError(f"校准目录为空: {target_dir}")

    lumas: List[float] = []
    lap_vars: List[float] = []
    hf_energies: List[float] = []
    used = 0

    for path in images:
        img = image_read_bgr(path, runtime.config.standardization)
        if img is None:
            continue
        face_feat = extract_face_feat(runtime, img, path)
        if not face_feat.ok or face_feat.lab_mean is None:
            continue

        lumas.append(float(face_feat.lab_mean[0]))
        if face_feat.lap_var > 0:
            lap_vars.append(float(face_feat.lap_var))
        if face_feat.hf_energy > 0:
            hf_energies.append(float(face_feat.hf_energy))
        used += 1

    if used < 8:
        raise RuntimeError(f"有效校准样本太少，仅 {used} 张，建议至少 8 张，最好 20–40 张")

    return {
        "FACE_LUMA_DARK_WARN_L": robust_percentile(lumas, 10),
        "FACE_LAPVAR_SOFT_WARN": robust_percentile(lap_vars, 15),
        "FACE_HFENERGY_SOFT_WARN": robust_percentile(hf_energies, 15),
        "num_used": used,
        "luma_mean": float(np.mean(lumas)) if lumas else 0.0,
        "lap_var_mean": float(np.mean(lap_vars)) if lap_vars else 0.0,
        "hf_energy_mean": float(np.mean(hf_energies)) if hf_energies else 0.0,
    }


def load_thresholds_from_file(
    runtime: RuntimeContext,
    path: Optional[Path] = None,
) -> None:
    load_runtime_thresholds_from_file(runtime.config, path or runtime.config.paths.thresh_file)


def run_pipeline(
    runtime: RuntimeContext,
    profile_name: Optional[str] = None,
) -> None:
    config = runtime.config
    target_profile = profile_name or config.review.active_profile
    if target_profile not in config.task_profiles:
        raise ValueError(f"未知任务模板: {target_profile}. 可选: {list(config.task_profiles.keys())}")

    profile = config.task_profiles[target_profile]
    weights = profile["weights"]
    reqs = profile["require"]
    th = profile["thresholds"]
    policy = get_profile_policy(runtime, target_profile)

    if not config.paths.dir_input.exists():
        print(f"[致命错误] 输入目录不存在: {config.paths.dir_input}")
        return

    images = list_images_in_dir(config.paths.dir_input)
    if len(images) == 0:
        print(f"[提示] 输入目录为空: {config.paths.dir_input}")
        return

    anchors = load_anchor_set(runtime)
    face_identity_anchors = get_identity_anchor_pool(runtime, target_profile, anchors)
    face_quality_anchors = get_quality_anchor_pool(runtime, target_profile, anchors)
    quality_ref_stats = build_quality_reference_stats(face_quality_anchors)

    if len(face_identity_anchors) == 0:
        print("[警告] 没有可用面部身份锚点，将导致 face 模块不可用")
    if len(anchors.upper_pose_feats) == 0:
        print("[警告] 没有半身锚点，将导致 upper 模块不可用")
    if len(anchors.full_pose_feats) == 0:
        print("[警告] 没有全身锚点，将导致 full 模块不可用")

    report: List[Dict[str, Any]] = []
    print(f"\n[运行中] 任务模板: {target_profile}")
    print(f"[运行中] 身份锚池(face): {len(face_identity_anchors)}")
    print(f"[运行中] 质量锚池(face-like): {len(face_quality_anchors)}")
    print(
        f"[运行中] STANDARDIZE_INPUT={config.standardization.enabled} "
        f"long_side={config.standardization.long_side}"
    )
    print(f"[运行中] CONSISTENCY_MODE={config.consistency.mode}")
    print("[运行中] 开始批处理质检...\n")

    for img_path in images:
        print(f"-> 检测: {img_path.name}")
        try:
            img = image_read_bgr(img_path, config.standardization)
            if img is None:
                raise RuntimeError("IMAGE_READ_ERROR")

            cand_face = extract_face_feat(runtime, img, img_path)
            cand_pose = extract_pose_feat(runtime, img)
            view_bucket, view_side, yaw_proxy = estimate_view_bucket_and_side(cand_face)

            constitution_metrics = extract_body_constitution_metrics(
                runtime,
                img,
                cand_face,
                cand_pose,
                view_bucket=view_bucket,
            )
            skin_metrics = extract_skin_consistency_metrics(runtime, img, cand_face, cand_pose)
            depth_3d_metrics = extract_depth_3d_lite_metrics(
                cand_face,
                cand_pose,
                view_bucket=view_bucket,
                yaw_proxy=yaw_proxy,
            )

            constitution_score = constitution_metrics.get("body_constitution_score", None)
            skin_score = skin_metrics.get("skin_uniformity_score", None)
            depth_3d_score = depth_3d_metrics.get("depth_3d_score", None)

            face_identity_anchors_view = filter_face_anchors_by_view(face_identity_anchors, view_bucket)
            face_score_o, face_conf_o, face_reasons_o, face_debug_o = score_face_against_anchor_set(
                runtime,
                cand_face,
                face_identity_anchors_view,
                view_bucket=view_bucket,
            )

            face_score = face_score_o
            face_conf = face_conf_o
            face_reasons = face_reasons_o
            face_debug = {
                "view_bucket": view_bucket,
                "view_side": view_side,
                "yaw_proxy": yaw_proxy,
                "flip_canonicalized": False,
                "identity_anchor_count_view": len(face_identity_anchors_view),
                "original": face_debug_o,
            }

            if view_bucket != "front":
                img_flipped = cv2.flip(img, 1)
                cand_face_flip = extract_face_feat(runtime, img_flipped, None)
                face_score_f, face_conf_f, face_reasons_f, face_debug_f = score_face_against_anchor_set(
                    runtime,
                    cand_face_flip,
                    face_identity_anchors_view,
                    view_bucket=view_bucket,
                )
                face_debug["flipped"] = face_debug_f
                if face_score_f > face_score_o:
                    face_score = face_score_f
                    face_conf = face_conf_f
                    face_reasons = ["FACE_FLIP_CANONICALIZED"] + face_reasons_f
                    face_debug["flip_canonicalized"] = True

            upper_score, upper_conf, upper_reasons, upper_debug = score_upper_against_anchor_set(
                cand_pose,
                anchors.upper_pose_feats,
                view_bucket=view_bucket,
            )
            full_score, full_conf, full_reasons, full_debug = score_full_against_anchor_set(
                cand_pose,
                anchors.full_pose_feats,
            )

            face_state, face_state_reasons = classify_module(
                runtime, face_score, face_conf, th["face_pass"], th["face_warn"], "face"
            )
            upper_state, upper_state_reasons = classify_module(
                runtime, upper_score, upper_conf, th["upper_pass"], th["upper_warn"], "upper"
            )
            full_state, full_state_reasons = classify_module(
                runtime, full_score, full_conf, th["full_pass"], th["full_warn"], "full"
            )

            scores = {"face": face_score, "upper": upper_score, "full": full_score}
            confs = {"face": face_conf, "upper": upper_conf, "full": full_conf}
            overall_score = fuse_overall(scores, confs, weights)

            if overall_score >= th["overall_pass"]:
                overall_state = "PASS"
            elif overall_score >= th["overall_warn"]:
                overall_state = "WARN"
            else:
                overall_state = "FAIL"

            hard_fail = False
            hard_warn = False
            strict_fail_min_conf = config.review.min_conf_for_strict_fail

            if reqs.get("face", False):
                if face_state == "FAIL" and face_conf >= strict_fail_min_conf:
                    hard_fail = True
                elif face_state != "PASS":
                    hard_warn = True

            if reqs.get("upper", False):
                if upper_state == "FAIL" and upper_conf >= strict_fail_min_conf:
                    hard_fail = True
                elif upper_state != "PASS":
                    hard_warn = True

            if reqs.get("full", False):
                if full_state == "FAIL" and full_conf >= strict_fail_min_conf:
                    hard_fail = True
                elif full_state != "PASS":
                    hard_warn = True

            if hard_fail or overall_state == "FAIL":
                final_status = "FAIL"
            elif hard_warn or overall_state == "WARN":
                final_status = "WARN"
            else:
                final_status = "PASS"

            reasons_all = (
                face_state_reasons
                + upper_state_reasons
                + full_state_reasons
                + face_reasons
                + upper_reasons
                + full_reasons
            )

            extra_flags: List[str] = []
            quality_debug: Dict[str, Any] = {}
            quality_thresholds = config.quality_thresholds

            if not cand_face.ok or face_conf < config.review.face_no_signal_conf_th:
                extra_flags.append("FACE_NO_RELIABLE_SIGNAL")
            else:
                qtol = get_quality_tolerances_by_face_size(cand_face.bbox_area_ratio, quality_thresholds)
                bucket = qtol["bucket"]
                bucket_stats = get_stats_for_bucket(quality_ref_stats, bucket)

                quality_debug["face_size_bucket"] = bucket
                quality_debug["bucket_quality_ref_count"] = bucket_stats.get("count", 0)
                quality_debug["bucket_quality_tolerances"] = qtol

                if cand_face.lab_mean is not None:
                    cand_L = float(cand_face.lab_mean[0])
                    quality_debug["candidate_face_L"] = cand_L
                    if cand_L < qtol["abs_luma_warn"]:
                        extra_flags.append("FACE_UNDEREXPOSED_DARK")

                    if len(bucket_stats.get("L", [])) > 0:
                        anchor_L_mean = float(np.mean(bucket_stats["L"]))
                        quality_debug["anchor_face_L_mean_bucket"] = anchor_L_mean
                        if cand_L < (anchor_L_mean - qtol["dark_delta_L"]):
                            extra_flags.append("FACE_DARKER_THAN_ANCHOR")

                if cand_face.lap_var > 0:
                    quality_debug["candidate_face_lap_var"] = cand_face.lap_var
                    if cand_face.lap_var < qtol["abs_lap_warn"]:
                        extra_flags.append("FACE_TOO_SOFT_POSSIBLE_SMOOTHING")

                    if len(bucket_stats.get("lap", [])) > 0:
                        anchor_lap_mean = float(np.mean(bucket_stats["lap"]))
                        quality_debug["anchor_face_lap_mean_bucket"] = anchor_lap_mean
                        if cand_face.lap_var < (anchor_lap_mean * qtol["sharp_ratio_floor"]):
                            extra_flags.append("FACE_SOFTER_THAN_ANCHOR")

                if cand_face.hf_energy > 0:
                    quality_debug["candidate_face_hf_energy"] = cand_face.hf_energy
                    if cand_face.hf_energy < qtol["abs_hf_warn"]:
                        extra_flags.append("FACE_LOW_MICROTEXTURE")

                    if len(bucket_stats.get("hf", [])) > 0:
                        anchor_hf_mean = float(np.mean(bucket_stats["hf"]))
                        quality_debug["anchor_face_hf_mean_bucket"] = anchor_hf_mean
                        if cand_face.hf_energy < (anchor_hf_mean * qtol["texture_ratio_floor"]):
                            extra_flags.append("FACE_LOWER_TEXTURE_THAN_ANCHOR")

            reasons_all.extend(extra_flags)
            reasons_all = dedupe_keep_order(reasons_all)

            reasons_all, final_status, overall_state, consistency_gate_debug = apply_consistency_soft_gate(
                runtime=runtime,
                reasons_all=reasons_all,
                final_status=final_status,
                overall_state=overall_state,
                constitution_metrics=constitution_metrics,
                skin_metrics=skin_metrics,
                depth_3d_metrics=depth_3d_metrics,
                view_bucket=view_bucket,
            )

            hard_quality_flags = set(policy.get("hard_quality_flags", set()))
            soft_quality_flags = quality_thresholds.degrade_flags - hard_quality_flags
            hard_hits = sum(1 for reason in reasons_all if reason in hard_quality_flags)
            soft_hits = sum(1 for reason in reasons_all if reason in soft_quality_flags)
            soft_hit_limit = int(policy.get("soft_quality_hits_to_warn", 2))

            if final_status == "PASS":
                if hard_hits >= 1 or soft_hits >= soft_hit_limit:
                    final_status = "WARN"
                    overall_state = "WARN"

            if "HIP_POP_DETECTED_POSSIBLE_MODEL_POSE" in reasons_all and final_status == "PASS":
                final_status = "WARN"
                overall_state = "WARN"

            if target_profile == "body_gold_fullbody":
                if view_bucket == "profile_like" and final_status == "PASS":
                    final_status = "WARN"
                    overall_state = "WARN"
                    reasons_all.append("PROFILE_LIKE_NO_SIDE_ANCHOR_PASS_CAPPED")
                elif view_bucket == "three_quarter":
                    reasons_all.append("THREE_QUARTER_SOFT_REVIEW")

            reasons_all = dedupe_keep_order(reasons_all)

            result_node = {
                "image": img_path.name,
                "task_profile": target_profile,
                "status": final_status,
                "scores": {
                    "face": round(face_score, 4),
                    "upper": round(upper_score, 4),
                    "full": round(full_score, 4),
                    "overall": round(overall_score, 4),
                    "constitution": round(float(constitution_score), 4) if constitution_score is not None else None,
                    "skin": round(float(skin_score), 4) if skin_score is not None else None,
                    "depth_3d": round(float(depth_3d_score), 4) if depth_3d_score is not None else None,
                },
                "confidence": {
                    "face": round(face_conf, 4),
                    "upper": round(upper_conf, 4),
                    "full": round(full_conf, 4),
                    "constitution": round(float(constitution_metrics.get("confidence", 0.0)), 4),
                    "skin": round(float(skin_metrics.get("confidence", 0.0)), 4),
                    "depth_3d": round(float(depth_3d_metrics.get("confidence", 0.0)), 4),
                },
                "module_state": {
                    "face": face_state,
                    "upper": upper_state,
                    "full": full_state,
                    "overall": overall_state,
                },
                "reasons": reasons_all,
                "reasons_face": face_reasons,
                "reasons_upper": upper_reasons,
                "reasons_full": full_reasons,
                "recommendations": [],
                "engine": {
                    "face": runtime.engines.face_mode,
                    "pose": runtime.engines.pose_mode,
                    "ssim_backend": "skimage" if SKIMAGE_SSIM_AVAILABLE else "ncc_fallback",
                },
                "debug": {
                    "face": face_debug,
                    "upper": upper_debug,
                    "full": full_debug,
                    "constitution_metrics": constitution_metrics,
                    "skin_metrics": skin_metrics,
                    "depth_3d_metrics": depth_3d_metrics,
                    "consistency_gate": consistency_gate_debug,
                    "candidate_pose_framing": cand_pose.framing,
                    "candidate_upper_geom": cand_pose.upper_geom,
                    "candidate_full_geom": cand_pose.full_geom,
                    "candidate_face_bbox_area_ratio": cand_face.bbox_area_ratio if cand_face.ok else 0.0,
                    "candidate_face_lab_mean": cand_face.lab_mean.tolist() if (cand_face.ok and cand_face.lab_mean is not None) else None,
                    "candidate_face_lap_var": cand_face.lap_var if cand_face.ok else 0.0,
                    "candidate_face_hf_energy": cand_face.hf_energy if cand_face.ok else 0.0,
                    "quality_gate_flags": extra_flags,
                    "quality_gate_soft_hits": soft_hits,
                    "quality_gate_hard_hits": hard_hits,
                    "quality_anchor_pool_mode": policy.get("quality_anchor_pool"),
                    "quality_ref_stats": quality_debug,
                    "view_bucket": view_bucket,
                    "view_side": view_side,
                    "yaw_proxy": yaw_proxy,
                    "identity_anchor_count_view": len(face_identity_anchors_view),
                    "input_shape": list(img.shape[:2]),
                },
            }

            result_node["recommendations"] = make_recommendations(runtime, result_node, target_profile)
            report.append(result_node)

            constitution_show = "NA" if constitution_score is None else f"{constitution_score:.3f}"
            skin_show = "NA" if skin_score is None else f"{skin_score:.3f}"
            depth_show = "NA" if depth_3d_score is None else f"{depth_3d_score:.3f}"

            if final_status == "PASS":
                shutil.copy2(img_path, config.paths.dir_out_pass / img_path.name)
                print(
                    f"   PASS | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            elif final_status == "WARN":
                shutil.copy2(img_path, config.paths.dir_out_warn / img_path.name)
                print(
                    f"   WARN | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )
            else:
                shutil.copy2(img_path, config.paths.dir_out_fail / img_path.name)
                print(
                    f"   FAIL | overall={overall_score:.3f} | face={face_score:.3f} upper={upper_score:.3f} full={full_score:.3f} "
                    f"| constitution={constitution_show} skin={skin_show} depth3d={depth_show} "
                    f"| view={view_bucket}/{view_side} | faceAnchors={len(face_identity_anchors_view)}"
                )

        except Exception as exc:
            print(f"   FAIL (异常): {exc}")
            traceback.print_exc()

            report.append(
                {
                    "image": img_path.name,
                    "task_profile": target_profile,
                    "status": "FAIL",
                    "scores": {
                        "face": 0.0,
                        "upper": 0.0,
                        "full": 0.0,
                        "overall": 0.0,
                        "constitution": None,
                        "skin": None,
                        "depth_3d": None,
                    },
                    "confidence": {
                        "face": 0.0,
                        "upper": 0.0,
                        "full": 0.0,
                        "constitution": 0.0,
                        "skin": 0.0,
                        "depth_3d": 0.0,
                    },
                    "module_state": {"face": "FAIL", "upper": "FAIL", "full": "FAIL", "overall": "FAIL"},
                    "reasons": ["RUNTIME_EXCEPTION", str(exc)],
                    "reasons_face": [],
                    "reasons_upper": [],
                    "reasons_full": [],
                    "recommendations": ["检查依赖、图片可读性、模型环境与日志"],
                    "engine": {"face": runtime.engines.face_mode, "pose": runtime.engines.pose_mode},
                    "debug": {
                        "constitution_metrics": None,
                        "skin_metrics": None,
                        "depth_3d_metrics": None,
                        "consistency_gate": None,
                    },
                }
            )

            try:
                shutil.copy2(img_path, config.paths.dir_out_fail / img_path.name)
            except Exception:
                pass

    config.paths.dir_output.mkdir(parents=True, exist_ok=True)
    with open(config.paths.report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n[完工] 质检完成 ✅")
    print(f"[报告] {config.paths.report_file}")
    print(
        f"[输出目录] PASS={config.paths.dir_out_pass} | WARN={config.paths.dir_out_warn} | FAIL={config.paths.dir_out_fail}"
    )


def main(base_dir: Optional[Path] = None) -> None:
    runtime = create_runtime(base_dir)
    print_runtime_config(runtime)

    if runtime.config.run_mode not in {"qa", "calibrate"}:
        raise ValueError("RUN_MODE 只能是 'qa' 或 'calibrate'")

    if runtime.config.run_mode == "calibrate":
        print(f"[校准模式] 从目录读取样本: {runtime.config.paths.dir_calib}")
        thresholds = calibrate_quality_thresholds(runtime, runtime.config.paths.dir_calib)
        save_thresholds_to_file(thresholds, runtime.config.paths.thresh_file)
        print("\n[自动校准完成 ✅]")
        print(json.dumps(thresholds, indent=2, ensure_ascii=False))
        print(f"[阈值文件已保存] {runtime.config.paths.thresh_file}")
        return

    if runtime.config.auto_load_thresholds:
        load_thresholds_from_file(runtime, runtime.config.paths.thresh_file)
    run_pipeline(runtime, profile_name=runtime.config.review.active_profile)
