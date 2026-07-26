"""SBIDataset: real 크롭 -> 온더플라이 self-blend(균형) + 야생 열화 증강.

핵심: 분류-시 증강은 real/fake에 동일 분포로 적용(모델이 증강이 아니라 블렌드를 학습하도록).
video_id(부모 폴더) 단위 train/val 분할.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as alb

from sbi_gen import self_blend

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def wild_aug(size):
    """야생 저화질 시뮬 + 일반 증강 (real/fake 공통)."""
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        alb.OneOf([
            alb.ImageCompression(quality_range=(30, 90), p=1),
            alb.Downscale(scale_range=(0.4, 0.9), p=1),
            alb.GaussNoise(p=1),
            alb.GaussianBlur(blur_limit=(3, 7), p=1),
            alb.MotionBlur(blur_limit=7, p=1),
        ], p=0.6),
        alb.Resize(size, size),
    ])


def build_manifest(root):
    """랜드마크 캐시가 있는 크롭만 -> [(png, lmk, video_id)]."""
    items = []
    for lmk in Path(root).rglob("*.lmk.npy"):
        png = lmk.with_suffix("").with_suffix(".png")
        if png.exists():
            items.append((str(png), str(lmk), png.parent.name))
    return items


class SBIDataset(Dataset):
    def __init__(self, items, size=256, train=True):
        self.items = items
        self.size = size
        self.train = train
        self.aug = wild_aug(size)

    def __len__(self):
        return len(self.items)

    def _finalize(self, img):
        img = self.aug(image=img)["image"]
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1))

    def __getitem__(self, i):
        png, lmkp, _ = self.items[i]
        bgr = cv2.imread(png)
        if bgr is None:
            return self.__getitem__((i + 1) % len(self))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lmk = np.load(lmkp)
        make_fake = self.train and (random.random() < 0.5)
        if make_fake:
            out = self_blend(img, lmk)
            if out is None:
                real_img, label = img, 0
            else:
                _, real_img = out          # blended = fake
                label = 1
        else:
            real_img, label = img, 0
        return self._finalize(real_img), torch.tensor(float(label))
