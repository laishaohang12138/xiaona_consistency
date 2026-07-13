from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .qa_runtime import EngineState, FaceFeat, PoseFeat, ReviewPolicy, RuntimeContext
from .qa_utils import (
    bbox_area_ratio_xyxy,
    bbox_xywh_to_xyxy,
    clamp,
    compute_hog_vec,
    compute_lbp_hist,
    compute_phash64,
    crop_safe,
    high_freq_energy,
    laplacian_var,
    linear_map_to_01,
    mean_lab,
    resize_gray,
)


def _try_init_insightface(allow_classic_fallback: bool) -> Tuple[str, object, Optional[str]]:
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception as exc:
                print(f"[警告] ONNXRuntime preload_dlls 失败，继续按默认方式加载 CUDA DLL。原因: {exc}")

        providers = ort.get_available_providers()
        print(f"[系统] ONNXRuntime providers: {providers}")
        device_preference = str(os.getenv("XIAONA_INSIGHTFACE_DEVICE", "auto") or "auto").strip().lower()
        require_gpu = str(os.getenv("XIAONA_REQUIRE_GPU", "")).strip().lower() in {"1", "true", "yes", "on"}

        if device_preference == "cpu":
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[系统] InsightFace 已按 XIAONA_INSIGHTFACE_DEVICE=cpu 强制使用 CPU")
        elif "CUDAExecutionProvider" in providers:
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            try:
                app.prepare(ctx_id=0, det_size=(640, 640))
                print("[系统] InsightFace 已启用：GPU (CUDAExecutionProvider)")
            except Exception as exc:
                if require_gpu:
                    reason = f"INSIGHTFACE_CUDA_INIT_FAILED:{exc}"
                    print(f"[致命] InsightFace CUDA 初始化失败，且当前要求 GPU：{exc}")
                    return "disabled", None, reason
                print(f"[警告] InsightFace GPU 初始化失败，回退 CPU。原因: {exc}")
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                print("[系统] InsightFace 已启用：CPU")
        else:
            if require_gpu:
                reason = "INSIGHTFACE_CUDA_PROVIDER_MISSING"
                print("[致命] InsightFace 缺少 CUDAExecutionProvider，且当前要求 GPU。")
                return "disabled", None, reason
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[系统] InsightFace 已启用：CPU（未检测到 CUDAExecutionProvider）")

        return "insightface", app, None

    except ModuleNotFoundError as exc:
        reason = f"INSIGHTFACE_MODULE_MISSING:{exc.name}"
        if allow_classic_fallback:
            print(f"[警告] InsightFace 不可用（缺依赖: {exc.name}），回退 OpenCV。")
            return "opencv", None, reason
        print(f"[致命] InsightFace 不可用（缺依赖: {exc.name}），禁止回退 OpenCV。")
        return "disabled", None, reason
    except Exception as exc:
        reason = f"INSIGHTFACE_INIT_FAILED:{exc}"
        if allow_classic_fallback:
            print(f"[警告] InsightFace 初始化失败（{exc}），回退 OpenCV。")
            return "opencv", None, reason
        print(f"[致命] InsightFace 初始化失败（{exc}），禁止回退 OpenCV。")
        return "disabled", None, reason


def _try_init_mediapipe_pose(allow_classic_fallback: bool) -> Tuple[str, object, object, Optional[str]]:
    try:
        import mediapipe as mp

        print("[系统] 启动 MediaPipe Pose (骨骼引擎)...")
        print(f"[系统] mediapipe module: {getattr(mp, '__file__', None)}")
        print(f"[系统] mediapipe has solutions: {hasattr(mp, 'solutions')}")

        mp_pose = None
        tried: List[str] = []

        if hasattr(mp, "solutions"):
            try:
                mp_pose = mp.solutions.pose
                print("[系统] MediaPipe 使用经典入口: mp.solutions.pose")
            except Exception as exc:
                tried.append(f"mp.solutions.pose -> {exc}")

        if mp_pose is None:
            for mod_name in ["mediapipe.python.solutions.pose", "mediapipe.modules.pose_landmark"]:
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, "Pose"):
                        mp_pose = mod
                        print(f"[系统] MediaPipe 使用兼容入口: {mod_name}")
                        break
                    tried.append(f"{mod_name} imported but no Pose class")
                except Exception as exc:
                    tried.append(f"{mod_name} -> {exc}")

        if mp_pose is None:
            raise RuntimeError(" ; ".join(tried) if tried else "No valid MediaPipe Pose entry found")

        pose_engine = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=1,
        )
        return "mediapipe", pose_engine, mp_pose, None

    except ModuleNotFoundError as exc:
        reason = f"MEDIAPIPE_MODULE_MISSING:{exc.name}"
        if allow_classic_fallback:
            print(f"[警告] MediaPipe 不可用（缺依赖: {exc.name}），回退 OpenCV HOG。")
            return "opencv", None, None, reason
        print(f"[致命] MediaPipe 不可用（缺依赖: {exc.name}），禁止回退 OpenCV HOG。")
        return "disabled", None, None, reason
    except Exception as exc:
        reason = f"MEDIAPIPE_INIT_FAILED:{exc}"
        if allow_classic_fallback:
            print(f"[警告] MediaPipe 初始化失败（{exc}），回退 OpenCV HOG。")
            return "opencv", None, None, reason
        print(f"[致命] MediaPipe 初始化失败（{exc}），禁止回退 OpenCV HOG。")
        return "disabled", None, None, reason


def _engine_fatal_reasons(
    review: ReviewPolicy,
    *,
    face_mode: str,
    pose_mode: str,
    face_reason: Optional[str],
    pose_reason: Optional[str],
) -> List[str]:
    if not review.fatal_on_engine_unavailable:
        return []
    reasons: List[str] = []
    if face_mode != "insightface" and face_reason:
        reasons.append(face_reason)
    if pose_mode != "mediapipe" and pose_reason:
        reasons.append(pose_reason)
    return reasons


def init_engines(review: Optional[ReviewPolicy] = None) -> EngineState:
    review = review or ReviewPolicy()
    face_mode, face_app, face_reason = _try_init_insightface(review.allow_classic_cv_fallback)
    pose_mode, pose_engine, mp_pose, pose_reason = _try_init_mediapipe_pose(review.allow_classic_cv_fallback)

    hog_people = None
    if pose_mode == "opencv":
        hog_people = cv2.HOGDescriptor()
        detector = cv2.HOGDescriptor_getDefaultPeopleDetector()  # type: ignore[attr-defined]
        hog_people.setSVMDetector(detector)
        print("[系统] OpenCV HOG 人体检测器已启用（兜底）")

    fatal_reasons = _engine_fatal_reasons(
        review,
        face_mode=face_mode,
        pose_mode=pose_mode,
        face_reason=face_reason,
        pose_reason=pose_reason,
    )
    classic_cv_fallback_active = face_mode == "opencv" or pose_mode == "opencv"
    if fatal_reasons:
        print("[致命] 核心视觉引擎不可用，本次运行将标记为 engine_fatal。")

    return EngineState(
        face_mode=face_mode,
        pose_mode=pose_mode,
        face_app=face_app,
        pose_engine=pose_engine,
        mp_pose=mp_pose,
        hog_people=hog_people,
        face_reason=face_reason,
        pose_reason=pose_reason,
        classic_cv_fallback_active=classic_cv_fallback_active,
        fatal=len(fatal_reasons) > 0,
        fatal_reasons=fatal_reasons,
        policy_allow_classic_cv_fallback=review.allow_classic_cv_fallback,
        policy_fatal_on_engine_unavailable=review.fatal_on_engine_unavailable,
    )


def opencv_largest_face_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = str(Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    return bbox_xywh_to_xyxy(int(x), int(y), int(w), int(h))


def detect_face_insightface(runtime: RuntimeContext, img_bgr: np.ndarray) -> Optional[Any]:
    if runtime.engines.face_mode != "insightface" or runtime.engines.face_app is None:
        return None
    faces = runtime.engines.face_app.get(img_bgr)
    if not faces:
        return None

    def area(face: Any) -> float:
        x1, y1, x2, y2 = face.bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    return max(faces, key=area)


def compute_face_geom_from_kps5(
    kps5: np.ndarray,
    face_xyxy: Tuple[int, int, int, int],
) -> Dict[str, float]:
    x1, y1, x2, y2 = face_xyxy
    fw = max(1.0, x2 - x1)
    fh = max(1.0, y2 - y1)

    le, re, nose, ml, mr = [kps5[i] for i in range(5)]
    eye_dist = float(np.linalg.norm(le - re))
    eye_center = (le + re) / 2.0
    mouth_center = (ml + mr) / 2.0
    mouth_w = float(np.linalg.norm(ml - mr))
    eye_tilt = math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))

    return {
        "eye_dist_norm": eye_dist / fw,
        "eye_y_norm": (float(eye_center[1]) - y1) / fh,
        "nose_y_norm": (float(nose[1]) - y1) / fh,
        "mouth_y_norm": (float(mouth_center[1]) - y1) / fh,
        "mouth_w_norm": mouth_w / fw,
        "face_ar": fw / fh,
        "eye_tilt_deg": eye_tilt,
    }


def _joint_angle_deg(
    xy: np.ndarray,
    vis: np.ndarray,
    idx_a: int,
    idx_b: int,
    idx_c: int,
    vis_th: float = 0.35,
) -> Optional[float]:
    if any(vis[idx] <= vis_th for idx in [idx_a, idx_b, idx_c]):
        return None

    v1 = xy[idx_a] - xy[idx_b]
    v2 = xy[idx_c] - xy[idx_b]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cos_theta = float(np.dot(v1, v2) / max(1e-6, n1 * n2))
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def _balance_ratio(left_value: Optional[float], right_value: Optional[float]) -> Optional[float]:
    if left_value is None or right_value is None:
        return None
    left_abs = abs(float(left_value))
    right_abs = abs(float(right_value))
    hi = max(left_abs, right_abs)
    if hi < 1e-6:
        return None
    return float(min(left_abs, right_abs) / hi)


def extract_face_feat(
    runtime: RuntimeContext,
    img_bgr: np.ndarray,
    source_path: Optional[Path] = None,
) -> FaceFeat:
    feat = FaceFeat(ok=False, source_path=str(source_path) if source_path else None)
    if img_bgr is None:
        feat.reasons.append("IMAGE_READ_ERROR")
        return feat

    if runtime.engines.face_mode == "disabled":
        feat.reasons.append("FACE_ENGINE_UNAVAILABLE_FATAL")
        if runtime.engines.face_reason:
            feat.reasons.append(str(runtime.engines.face_reason))
        return feat

    face_xyxy = None
    embedding = None
    kps5 = None

    if runtime.engines.face_mode == "insightface":
        face = detect_face_insightface(runtime, img_bgr)
        if face is not None:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            face_xyxy = (x1, y1, x2, y2)
            emb = getattr(face, "embedding", None)
            if emb is not None:
                embedding = np.array(emb, dtype=np.float32)
            kps = getattr(face, "kps", None)
            if kps is not None:
                kps5 = np.array(kps, dtype=np.float32)

    if face_xyxy is None:
        face_xyxy = opencv_largest_face_bbox(img_bgr)

    if face_xyxy is None:
        feat.reasons.append("FACE_NOT_FOUND")
        return feat

    crop = crop_safe(img_bgr, face_xyxy)
    if crop is None:
        feat.reasons.append("FACE_CROP_FAILED")
        return feat

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray128 = resize_gray(gray, 128)

    feat.ok = True
    feat.bbox_xyxy = face_xyxy
    feat.bbox_area_ratio = bbox_area_ratio_xyxy(face_xyxy, img_bgr.shape)
    feat.embedding = embedding
    feat.kps5 = kps5
    feat.crop_gray_128 = gray128
    feat.hog_vec = compute_hog_vec(gray128)
    feat.lbp_hist = compute_lbp_hist(gray128)
    feat.phash64 = compute_phash64(gray128)
    feat.geom = compute_face_geom_from_kps5(kps5, face_xyxy) if kps5 is not None else {}
    feat.confidence = linear_map_to_01(feat.bbox_area_ratio, 0.006, 0.035)
    feat.lab_mean = mean_lab(crop)
    feat.lap_var = laplacian_var(gray128)
    feat.hf_energy = high_freq_energy(gray128)

    if embedding is not None:
        feat.reasons.append("FACE_EMBEDDING_READY")
    else:
        feat.reasons.append("FACE_EMBEDDING_MISSING_USING_TEXTURE_GEOM")

    if kps5 is not None:
        feat.reasons.append("FACE_LANDMARKS_READY")
    else:
        feat.reasons.append("FACE_LANDMARKS_MISSING")

    if feat.bbox_area_ratio < 0.01:
        feat.reasons.append("FACE_TOO_SMALL")

    return feat


def extract_pose_feat(runtime: RuntimeContext, img_bgr: np.ndarray) -> PoseFeat:
    feat = PoseFeat(ok=False, mode=runtime.engines.pose_mode)
    if img_bgr is None:
        feat.reasons.append("IMAGE_READ_ERROR")
        return feat

    if runtime.engines.pose_mode == "disabled":
        feat.reasons.append("POSE_ENGINE_UNAVAILABLE_FATAL")
        if runtime.engines.pose_reason:
            feat.reasons.append(str(runtime.engines.pose_reason))
        return feat

    if runtime.engines.pose_mode == "mediapipe":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = runtime.engines.pose_engine.process(img_rgb)
        if not results.pose_landmarks:
            feat.reasons.append("POSE_NOT_DETECTED")
            return feat

        lms = results.pose_landmarks.landmark
        xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
        vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)
        world_xyz = None
        if getattr(results, "pose_world_landmarks", None):
            world_xyz = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark],
                dtype=np.float32,
            )

        feat.ok = True
        feat.lm_xy = xy
        feat.lm_vis = vis
        feat.lm_world = world_xyz

        reasons: List[str] = []
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_HEEL, R_HEEL = 29, 30
        L_FOOT, R_FOOT = 31, 32

        ankles = [xy[L_ANKLE], xy[R_ANKLE]]
        ankles_vis = [vis[L_ANKLE], vis[R_ANKLE]]
        visible_ankles = [ankle for ankle, visibility in zip(ankles, ankles_vis) if visibility > 0.35]
        if len(visible_ankles) > 0:
            max_ankle_y = float(max(ankle[1] for ankle in visible_ankles))
            feet_in_frame = 1.0 if 0.80 <= max_ankle_y <= 0.995 else 0.0
            reasons.append("FEET_IN_FRAME" if feet_in_frame > 0 else "FEET_CROPPED_OR_TOO_HIGH")
        else:
            max_ankle_y = 1.0
            feet_in_frame = 0.0
            reasons.append("ANKLES_NOT_VISIBLE")

        nose_y = float(xy[NOSE][1]) if vis[NOSE] > 0.2 else 0.08
        top_y_est = max(0.0, nose_y - 0.09)
        subject_height = max(0.0, max_ankle_y - top_y_est)
        headroom = top_y_est

        feat.framing = {
            "nose_y": nose_y,
            "top_y_est": top_y_est,
            "max_ankle_y": max_ankle_y,
            "subject_height_ratio": subject_height,
            "headroom_ratio": headroom,
            "feet_in_frame": feet_in_frame,
        }

        upper_geom: Dict[str, float] = {}
        if vis[L_SHOULDER] > 0.35 and vis[R_SHOULDER] > 0.35:
            shoulder_w = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
            upper_geom["shoulder_width_norm"] = shoulder_w
            upper_geom["shoulder_tilt_deg"] = math.degrees(
                math.atan2(
                    float(xy[R_SHOULDER][1] - xy[L_SHOULDER][1]),
                    float(xy[R_SHOULDER][0] - xy[L_SHOULDER][0]),
                )
            )

        if vis[L_HIP] > 0.35 and vis[R_HIP] > 0.35:
            upper_geom["hip_width_norm"] = float(np.linalg.norm(xy[L_HIP] - xy[R_HIP]))

        if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]):
            shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
            hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
            upper_geom["torso_len_norm"] = float(np.linalg.norm(shoulder_mid - hip_mid))

            spine_dx = float(shoulder_mid[0] - hip_mid[0])
            spine_dy = float(hip_mid[1] - shoulder_mid[1])
            if abs(spine_dy) > 1e-6:
                spine_angle_deg = float(math.degrees(math.atan2(abs(spine_dx), abs(spine_dy))))
            else:
                spine_angle_deg = 90.0
            upper_geom["spine_angle_deg"] = spine_angle_deg
            upper_geom["shoulder_hip_center_offset_norm"] = abs(spine_dx) / max(1e-6, upper_geom["torso_len_norm"])
            if "shoulder_width_norm" in upper_geom and "hip_width_norm" in upper_geom:
                upper_geom["hip_shoulder_ratio"] = upper_geom["hip_width_norm"] / max(
                    1e-6, upper_geom["shoulder_width_norm"]
                )
                upper_geom["torso_compactness"] = (
                    upper_geom["shoulder_width_norm"] + upper_geom["hip_width_norm"]
                ) / max(1e-6, 2.0 * upper_geom["torso_len_norm"])
            if spine_angle_deg > 12.0:
                reasons.append("HIP_POP_DETECTED_POSSIBLE_MODEL_POSE")

        feat.upper_geom = upper_geom

        full_geom: Dict[str, float] = {}
        if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, NOSE]):
            shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
            head_proxy = float(np.linalg.norm(xy[NOSE] - shoulder_mid)) * 1.6
            if head_proxy > 1e-5 and subject_height > 1e-5:
                full_geom["head_body_ratio"] = subject_height / head_proxy
        if vis[L_SHOULDER] > 0.35 and vis[R_SHOULDER] > 0.35:
            shoulder_w = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
            if shoulder_w > 1e-5:
                full_geom["shoulder_level_delta_norm"] = abs(float(xy[L_SHOULDER][1] - xy[R_SHOULDER][1])) / shoulder_w
        if vis[L_HIP] > 0.35 and vis[R_HIP] > 0.35:
            hip_w = float(np.linalg.norm(xy[L_HIP] - xy[R_HIP]))
            if hip_w > 1e-5:
                full_geom["hip_level_delta_norm"] = abs(float(xy[L_HIP][1] - xy[R_HIP][1])) / hip_w

        if all(vis[idx] > 0.35 for idx in [L_HIP, R_HIP, L_ANKLE, R_ANKLE]):
            hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
            ankles_mid = (xy[L_ANKLE] + xy[R_ANKLE]) / 2
            leg_len = float(np.linalg.norm(hip_mid - ankles_mid))
            if subject_height > 1e-5:
                full_geom["leg_ratio"] = leg_len / subject_height
                full_geom["ankle_gap_norm"] = float(np.linalg.norm(xy[L_ANKLE] - xy[R_ANKLE])) / subject_height

        knee_angles = [
            _joint_angle_deg(xy, vis, L_HIP, L_KNEE, L_ANKLE),
            _joint_angle_deg(xy, vis, R_HIP, R_KNEE, R_ANKLE),
        ]
        knee_angles = [float(angle) for angle in knee_angles if angle is not None]
        if len(knee_angles) > 0:
            full_geom["leg_straightness_min_deg"] = float(min(knee_angles))
            full_geom["leg_straightness_mean_deg"] = float(np.mean(np.array(knee_angles, dtype=np.float32)))

        left_thigh_len = None
        right_thigh_len = None
        left_calf_len = None
        right_calf_len = None
        if all(vis[idx] > 0.35 for idx in [L_HIP, L_KNEE]):
            left_thigh_len = float(np.linalg.norm(xy[L_HIP] - xy[L_KNEE]))
        if all(vis[idx] > 0.35 for idx in [R_HIP, R_KNEE]):
            right_thigh_len = float(np.linalg.norm(xy[R_HIP] - xy[R_KNEE]))
        if all(vis[idx] > 0.35 for idx in [L_KNEE, L_ANKLE]):
            left_calf_len = float(np.linalg.norm(xy[L_KNEE] - xy[L_ANKLE]))
        if all(vis[idx] > 0.35 for idx in [R_KNEE, R_ANKLE]):
            right_calf_len = float(np.linalg.norm(xy[R_KNEE] - xy[R_ANKLE]))

        thigh_balance = _balance_ratio(left_thigh_len, right_thigh_len)
        calf_balance = _balance_ratio(left_calf_len, right_calf_len)
        if thigh_balance is not None:
            full_geom["thigh_length_balance"] = thigh_balance
        if calf_balance is not None:
            full_geom["calf_length_balance"] = calf_balance
        if thigh_balance is not None or calf_balance is not None:
            valid_balances = [value for value in [thigh_balance, calf_balance] if value is not None]
            full_geom["lower_limb_balance"] = float(np.mean(np.array(valid_balances, dtype=np.float32)))

        foot_lengths = []
        foot_lengths_lr: Dict[str, float] = {}
        for side_name, heel_idx, foot_idx in [("left", L_HEEL, L_FOOT), ("right", R_HEEL, R_FOOT)]:
            if vis[heel_idx] > 0.35 and vis[foot_idx] > 0.35 and subject_height > 1e-5:
                foot_len = float(np.linalg.norm(xy[heel_idx] - xy[foot_idx])) / subject_height
                foot_lengths.append(foot_len)
                foot_lengths_lr[side_name] = foot_len
        if len(foot_lengths) > 0:
            full_geom["foot_length_proxy_norm"] = float(np.mean(np.array(foot_lengths, dtype=np.float32)))
        foot_balance = _balance_ratio(foot_lengths_lr.get("left"), foot_lengths_lr.get("right"))
        if foot_balance is not None:
            full_geom["foot_length_balance"] = foot_balance

        feat.full_geom = full_geom

        upper_ids = [0, 11, 12, 13, 14, 23, 24]
        full_ids = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        upper_vis = float(np.mean([1.0 if vis[i] > 0.35 else 0.0 for i in upper_ids]))
        full_vis = float(np.mean([1.0 if vis[i] > 0.35 else 0.0 for i in full_ids]))

        feat.confidence_upper = clamp(upper_vis, 0.0, 1.0)
        feat.confidence_full = clamp(full_vis, 0.0, 1.0)

        if feat.confidence_upper < 0.5:
            reasons.append("UPPER_KEYPOINTS_LOW_CONFIDENCE")
        if feat.confidence_full < 0.5:
            reasons.append("FULL_KEYPOINTS_LOW_CONFIDENCE")

        feat.reasons.extend(reasons)
        return feat

    h0, w0 = img_bgr.shape[:2]
    max_side = max(h0, w0)
    scale = 1.0
    img_small = img_bgr
    if max_side > 1000:
        scale = 1000.0 / max_side
        img_small = cv2.resize(
            img_bgr,
            (int(w0 * scale), int(h0 * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if runtime.engines.hog_people is None:
        feat.reasons.append("POSE_ENGINE_UNAVAILABLE_FATAL")
        if runtime.engines.pose_reason:
            feat.reasons.append(str(runtime.engines.pose_reason))
        return feat

    rects, weights = runtime.engines.hog_people.detectMultiScale(
        img_small,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    if len(rects) == 0:
        feat.reasons.append("PERSON_NOT_DETECTED_OPENCV_HOG")
        return feat

    best_i = int(np.argmax(weights)) if len(weights) > 0 else 0
    x, y, bw, bh = rects[best_i]
    x, y, bw, bh = int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)

    feat.ok = True
    feat.person_bbox_xywh = (x, y, bw, bh)
    feat.person_bbox_area_ratio = float((bw * bh) / max(1, h0 * w0))
    feat.confidence_upper = 0.35
    feat.confidence_full = 0.35

    top = y
    bottom = y + bh
    subject_height = bh / float(h0)
    feet_in_frame = 1.0 if bottom < int(h0 * 0.995) else 0.0
    feat.framing = {
        "subject_height_ratio": subject_height,
        "headroom_ratio": top / float(h0),
        "feet_in_frame": feet_in_frame,
    }
    feat.reasons.extend(["POSE_ENGINE_FALLBACK_OPENCV_HOG", "FRAMING_APPROXIMATE"])
    return feat
