"""Self-Blended Images 생성 (mapooon/SelfBlendedImages 로직 포팅).

실제 얼굴 이미지 + 68-pt 랜드마크 -> self-blended 가짜(블렌딩 경계+통계 불일치) 생성.
핵심: fake 데이터셋 없이 '모든 face-swap 공통 아티팩트'만 학습하도록 pseudo-fake를 만든다.
"""
import random

import cv2
import numpy as np
import albumentations as alb


class RandomDownScale(alb.ImageOnlyTransform):
    """해상도 불일치 생성: 다운스케일 후 원복(블러/저해상 아티팩트)."""
    def apply(self, img, **params):
        h, w = img.shape[:2]
        r = random.choice([2, 3, 4])
        small = cv2.resize(img, (w // r, h // r), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def source_transforms():
    """source/target 통계차 생성용 증강."""
    return alb.Compose([
        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                               val_shift_limit=(-0.3, 0.3), p=1),
        alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                     contrast_limit=(-0.1, 0.1), p=1),
        alb.OneOf([RandomDownScale(p=1),
                   alb.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=1)], p=1),
    ], p=1)


_SRC = source_transforms()


def random_hull_mask(landmark, img):
    """landmark 점들의 convex hull -> 채운 얼굴 마스크 (tool-agnostic).
    mediapipe full mesh면 이마까지 자연 포함."""
    pts = landmark.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    return mask


def get_blend_mask(mask):
    """마스크 랜덤 리사이즈 + 이중 GaussianBlur로 소프트 블렌딩 경계."""
    H, W = mask.shape
    sh, sw = np.random.randint(192, 257), np.random.randint(192, 257)
    m = cv2.resize(mask, (sw, sh))
    k1 = random.randrange(5, 26, 2)
    k2 = random.randrange(5, 26, 2)
    m = cv2.GaussianBlur(m, (k1, k1), 0)
    mx = m.max()
    if mx <= 0:
        return np.zeros((H, W, 1), np.float32)
    m = m / mx                          # max -> 정확히 1.0 (원본 SBI)
    m[m < 1] = 0                        # 내부 plateau만 유지
    m = cv2.GaussianBlur(m, (k2, k2), np.random.randint(5, 46))
    m = m / (m.max() if m.max() > 0 else 1.0)
    m = cv2.resize(m, (W, H))
    return m[:, :, None]


def dynamic_blend(source, target, mask):
    mb = get_blend_mask(mask)
    ratio = [0.25, 0.5, 0.75, 1, 1, 1][np.random.randint(6)]
    mb = mb * ratio
    blended = mb * source + (1 - mb) * target
    return blended, mb


def randaffine(img, mask):
    """source에 소폭 affine(정합 오차 모사)."""
    H, W = img.shape[:2]
    tx, ty = np.random.uniform(-0.03, 0.03, 2) * [W, H]
    sc = np.random.uniform(0.95, 1.05)
    ang = np.random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    M[:, 2] += [tx, ty]
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (W, H))
    return img, mask


def self_blend(img, landmark):
    """실제 img + landmark -> (real_img, fake_blended). 실패 시 None."""
    img = img.copy()
    mask = random_hull_mask(landmark, img)
    if mask.sum() < 100:
        return None
    source = img.copy()
    if np.random.rand() < 0.5:
        source = _SRC(image=source)["image"]
    else:
        img = _SRC(image=img)["image"]
    source, mask = randaffine(source, mask)
    blended, _ = dynamic_blend(source.astype(np.float32), img.astype(np.float32), mask)
    return img.astype(np.uint8), np.clip(blended, 0, 255).astype(np.uint8)
