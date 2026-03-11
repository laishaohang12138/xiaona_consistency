from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np

from .human_parsing_engine import HumanParsingEngine


def _warn_once(warned_keys: set[str], key: str, message: str) -> None:
    if key in warned_keys:
        return
    warned_keys.add(key)
    print(message)


def _clip_rect(x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> Optional[tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask_u8)
    out[labels == best] = 255
    return out


def _normalize_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        return None
    return np.where(mask > 0, 255, 0).astype(np.uint8)


class SubjectMaskProvider(ABC):
    provider_name = "subject_mask_base"

    @abstractmethod
    def get_subject_mask(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError


class SkinRegionProvider(ABC):
    provider_name = "skin_region_base"

    @abstractmethod
    def get_skin_region(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError


class LegacyForegroundSubjectMaskProvider(SubjectMaskProvider):
    provider_name = "legacy_foreground"

    def get_subject_mask(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        bd = max(4, int(round(min(h, w) * 0.04)))
        border_pixels = np.concatenate(
            [
                lab[:bd, :, :].reshape(-1, 3),
                lab[-bd:, :, :].reshape(-1, 3),
                lab[:, :bd, :].reshape(-1, 3),
                lab[:, -bd:, :].reshape(-1, 3),
            ],
            axis=0,
        )

        bg_med = np.median(border_pixels, axis=0).astype(np.float32)
        dist = np.sqrt(np.sum((lab - bg_med[None, None, :]) ** 2, axis=2))
        mask = (dist > 10.5).astype(np.uint8) * 255

        if getattr(pose_feat, "ok", False) and getattr(pose_feat, "lm_xy", None) is not None and getattr(pose_feat, "lm_vis", None) is not None:
            vis_ids = [i for i in range(len(pose_feat.lm_vis)) if float(pose_feat.lm_vis[i]) > 0.20]
            if len(vis_ids) > 0:
                xs = [float(pose_feat.lm_xy[i][0]) for i in vis_ids]
                ys = [float(pose_feat.lm_xy[i][1]) for i in vis_ids]
                x1 = int(max(0, (min(xs) - 0.08) * w))
                y1 = int(max(0, (min(ys) - 0.08) * h))
                x2 = int(min(w, (max(xs) + 0.08) * w))
                y2 = int(min(h, (max(ys) + 0.08) * h))
                roi = np.zeros((h, w), dtype=np.uint8)
                roi[y1:y2, x1:x2] = 255
                mask = cv2.bitwise_and(mask, roi)

        if getattr(face_feat, "ok", False) and getattr(face_feat, "bbox_xyxy", None) is not None:
            x1, y1, x2, y2 = face_feat.bbox_xyxy
            ex = int((x2 - x1) * 0.18)
            ey = int((y2 - y1) * 0.18)
            face_roi = np.zeros((h, w), dtype=np.uint8)
            rr = _clip_rect(x1 - ex, y1 - ey, x2 + ex, y2 + ey, h, w)
            if rr is not None:
                fx1, fy1, fx2, fy2 = rr
                face_roi[fy1:fy2, fx1:fx2] = 255
                mask = cv2.bitwise_or(mask, face_roi)

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
        return _largest_component(mask)


class LegacyYCrCbSkinRegionProvider(SkinRegionProvider):
    provider_name = "legacy_ycrcb"

    def get_skin_region(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        if img_bgr is None or img_bgr.size == 0:
            return None
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]
        mask = (
            (y_channel > 35)
            & (cr_channel >= 132)
            & (cr_channel <= 180)
            & (cb_channel >= 75)
            & (cb_channel <= 135)
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask


class HumanParsingProvider(SubjectMaskProvider, SkinRegionProvider):
    provider_name = "human_parsing"
    SUBJECT_LABELS = (
        "hair",
        "upper_clothes",
        "pants",
        "skirt",
        "dress",
        "face",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "left_shoe",
        "right_shoe",
        "skin",
    )
    SKIN_LABELS = ("face", "left_arm", "right_arm", "left_leg", "right_leg", "skin")

    def __init__(self, engine: Optional[HumanParsingEngine] = None) -> None:
        self.engine = engine or HumanParsingEngine()
        self._cache_key: Optional[tuple[int, tuple[int, ...]]] = None
        self._cache_output = None

    def _get_output(self, img_bgr: np.ndarray):
        cache_key = (id(img_bgr), tuple(img_bgr.shape))
        if self._cache_key == cache_key and self._cache_output is not None:
            return self._cache_output
        output = self.engine.parse(img_bgr)
        self._cache_key = cache_key
        self._cache_output = output
        return output

    def _compose_mask(self, img_bgr: np.ndarray, label_names: Iterable[str]) -> Optional[np.ndarray]:
        output = self._get_output(img_bgr)
        mask = np.zeros(output.label_map.shape, dtype=np.uint8)
        matched = 0
        for label_name in label_names:
            cur_mask = output.masks.get(label_name)
            if cur_mask is None:
                continue
            mask = cv2.bitwise_or(mask, cur_mask)
            matched += 1
        if matched == 0:
            return None
        return mask

    def get_subject_mask(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        mask = self._compose_mask(img_bgr, self.SUBJECT_LABELS)
        if mask is None:
            return None
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        return _normalize_mask(mask)

    def get_skin_region(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        mask = self._compose_mask(img_bgr, self.SKIN_LABELS)
        if mask is None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return _normalize_mask(mask)


@dataclass
class ProviderBundle:
    requested_policy: Dict[str, str]
    subject_mask_provider: SubjectMaskProvider
    skin_region_provider: SkinRegionProvider
    subject_mask_fallback: SubjectMaskProvider
    skin_region_fallback: SkinRegionProvider
    warned_keys: set[str] = field(default_factory=set)

    def get_subject_mask(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        provider_name = getattr(self.subject_mask_provider, "provider_name", "unknown")
        provider_failed = False
        try:
            mask = self.subject_mask_provider.get_subject_mask(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
        except Exception as exc:
            mask = None
            provider_failed = True
            _warn_once(
                self.warned_keys,
                f"subject_mask_provider_error::{provider_name}",
                f"[警告] subject_mask provider={provider_name} 调用失败，已回退 legacy。原因: {exc}",
            )
        if mask is not None:
            return _normalize_mask(mask)
        if provider_name != self.subject_mask_fallback.provider_name and not provider_failed:
            _warn_once(
                self.warned_keys,
                f"subject_mask_provider_fallback::{provider_name}",
                f"[警告] subject_mask provider={provider_name} 未产出有效 mask，已回退 legacy。",
            )
        return _normalize_mask(self.subject_mask_fallback.get_subject_mask(img_bgr, face_feat=face_feat, pose_feat=pose_feat))

    def get_skin_region(
        self,
        img_bgr: np.ndarray,
        face_feat: Optional[Any] = None,
        pose_feat: Optional[Any] = None,
    ) -> np.ndarray:
        provider_name = getattr(self.skin_region_provider, "provider_name", "unknown")
        provider_failed = False
        try:
            mask = self.skin_region_provider.get_skin_region(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
        except Exception as exc:
            mask = None
            provider_failed = True
            _warn_once(
                self.warned_keys,
                f"skin_region_provider_error::{provider_name}",
                f"[警告] skin_region provider={provider_name} 调用失败，已回退 legacy。原因: {exc}",
            )
        if mask is not None:
            normalized_mask = _normalize_mask(mask)
            if normalized_mask is not None:
                return normalized_mask
            return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        if provider_name != self.skin_region_fallback.provider_name and not provider_failed:
            _warn_once(
                self.warned_keys,
                f"skin_region_provider_fallback::{provider_name}",
                f"[警告] skin_region provider={provider_name} 未产出有效 mask，已回退 legacy。",
            )
        fallback_mask = self.skin_region_fallback.get_skin_region(img_bgr, face_feat=face_feat, pose_feat=pose_feat)
        normalized_fallback = _normalize_mask(fallback_mask)
        if normalized_fallback is not None:
            return normalized_fallback
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    def describe(self) -> Dict[str, str]:
        return {
            "requested_subject_mask": str(self.requested_policy.get("subject_mask", "")),
            "requested_skin_region": str(self.requested_policy.get("skin_region", "")),
            "active_subject_mask": getattr(self.subject_mask_provider, "provider_name", "unknown"),
            "active_skin_region": getattr(self.skin_region_provider, "provider_name", "unknown"),
            "subject_fallback": getattr(self.subject_mask_fallback, "provider_name", "unknown"),
            "skin_fallback": getattr(self.skin_region_fallback, "provider_name", "unknown"),
        }


def build_provider_bundle(provider_policy: Dict[str, str]) -> ProviderBundle:
    requested_policy = {
        "subject_mask": str(provider_policy.get("subject_mask", "human_parsing")),
        "skin_region": str(provider_policy.get("skin_region", "human_parsing")),
    }

    legacy_subject = LegacyForegroundSubjectMaskProvider()
    legacy_skin = LegacyYCrCbSkinRegionProvider()
    human_provider = HumanParsingProvider()

    subject_provider_map: Dict[str, SubjectMaskProvider] = {
        "human_parsing": human_provider,
        "legacy_foreground": legacy_subject,
    }
    skin_provider_map: Dict[str, SkinRegionProvider] = {
        "human_parsing": human_provider,
        "legacy_ycrcb": legacy_skin,
    }

    subject_name = requested_policy["subject_mask"]
    skin_name = requested_policy["skin_region"]
    subject_provider = subject_provider_map.get(subject_name, legacy_subject)
    skin_provider = skin_provider_map.get(skin_name, legacy_skin)

    if subject_name not in subject_provider_map:
        print(f"[警告] 未知 subject_mask provider={subject_name}，已改用 legacy_foreground。")
    if skin_name not in skin_provider_map:
        print(f"[警告] 未知 skin_region provider={skin_name}，已改用 legacy_ycrcb。")

    return ProviderBundle(
        requested_policy=requested_policy,
        subject_mask_provider=subject_provider,
        skin_region_provider=skin_provider,
        subject_mask_fallback=legacy_subject,
        skin_region_fallback=legacy_skin,
    )
