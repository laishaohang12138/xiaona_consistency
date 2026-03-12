from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .qa_runtime import EngineState, FaceFeat, PoseFeat, RuntimeContext
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


def _try_init_insightface() -> Tuple[str, object]:
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

        if "CUDAExecutionProvider" in providers:
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            try:
                app.prepare(ctx_id=0, det_size=(640, 640))
                print("[系统] InsightFace 已启用：GPU (CUDAExecutionProvider)")
            except Exception as exc:
                print(f"[警告] InsightFace GPU 初始化失败，回退 CPU。原因: {exc}")
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                print("[系统] InsightFace 已启用：CPU")
        else:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[系统] InsightFace 已启用：CPU（未检测到 CUDAExecutionProvider）")

        return "insightface", app

    except ModuleNotFoundError as exc:
        print(f"[警告] InsightFace 不可用（缺依赖: {exc.name}），回退 OpenCV。")
        return "opencv", None
    except Exception as exc:
        print(f"[警告] InsightFace 初始化失败（{exc}），回退 OpenCV。")
        return "opencv", None


def _try_init_mediapipe_pose() -> Tuple[str, object, object]:
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
        return "mediapipe", pose_engine, mp_pose

    except ModuleNotFoundError as exc:
        print(f"[警告] MediaPipe 不可用（缺依赖: {exc.name}），回退 OpenCV HOG。")
        return "opencv", None, None
    except Exception as exc:
        print(f"[警告] MediaPipe 初始化失败（{exc}），回退 OpenCV HOG。")
        return "opencv", None, None


def init_engines() -> EngineState:
    face_mode, face_app = _try_init_insightface()
    pose_mode, pose_engine, mp_pose = _try_init_mediapipe_pose()

    hog_people = None
    if pose_mode == "opencv":
        hog_people = cv2.HOGDescriptor()
        detector = cv2.HOGDescriptor_getDefaultPeopleDetector()  # type: ignore[attr-defined]
        hog_people.setSVMDetector(detector)
        print("[系统] OpenCV HOG 人体检测器已启用（兜底）")

    return EngineState(
        face_mode=face_mode,
        pose_mode=pose_mode,
        face_app=face_app,
        pose_engine=pose_engine,
        mp_pose=mp_pose,
        hog_people=hog_people,
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


def extract_face_feat(
    runtime: RuntimeContext,
    img_bgr: np.ndarray,
    source_path: Optional[Path] = None,
) -> FaceFeat:
    feat = FaceFeat(ok=False, source_path=str(source_path) if source_path else None)
    if img_bgr is None:
        feat.reasons.append("IMAGE_READ_ERROR")
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

    if runtime.engines.pose_mode == "mediapipe":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = runtime.engines.pose_engine.process(img_rgb)
        if not results.pose_landmarks:
            feat.reasons.append("POSE_NOT_DETECTED")
            return feat

        lms = results.pose_landmarks.landmark
        xy = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
        vis = np.array([getattr(lm, "visibility", 1.0) for lm in lms], dtype=np.float32)

        feat.ok = True
        feat.lm_xy = xy
        feat.lm_vis = vis

        reasons: List[str] = []
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_ANKLE, R_ANKLE = 27, 28

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
            if spine_angle_deg > 12.0:
                reasons.append("HIP_POP_DETECTED_POSSIBLE_MODEL_POSE")

        feat.upper_geom = upper_geom

        full_geom: Dict[str, float] = {}
        if all(vis[idx] > 0.35 for idx in [L_SHOULDER, R_SHOULDER, NOSE]):
            shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
            head_proxy = float(np.linalg.norm(xy[NOSE] - shoulder_mid)) * 1.6
            if head_proxy > 1e-5 and subject_height > 1e-5:
                full_geom["head_body_ratio"] = subject_height / head_proxy

        if all(vis[idx] > 0.35 for idx in [L_HIP, R_HIP, L_ANKLE, R_ANKLE]):
            hip_mid = (xy[L_HIP] + xy[R_HIP]) / 2
            ankles_mid = (xy[L_ANKLE] + xy[R_ANKLE]) / 2
            leg_len = float(np.linalg.norm(hip_mid - ankles_mid))
            if subject_height > 1e-5:
                full_geom["leg_ratio"] = leg_len / subject_height

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
