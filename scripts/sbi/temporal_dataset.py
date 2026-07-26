"""TemporalDataset: real 클립 npz -> 온더플라이 temporal pseudo-fake + 시간-일관 증강.

핵심: 야생 증강은 클립 내 프레임에 '동일 파라미터'로 적용(프레임마다 다르면 real에 인공
시간 불일치가 생겨 오염). ReplayCompose로 보장.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as alb

from temporal_gen import make_fake

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def clip_aug(size):
    return alb.ReplayCompose([
        alb.HorizontalFlip(p=0.5),
        alb.OneOf([
            alb.ImageCompression(quality_range=(35, 90), p=1),
            alb.Downscale(scale_range=(0.5, 0.9), p=1),
            alb.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.5),
        alb.RandomBrightnessContrast(0.1, 0.1, p=0.3),
        alb.Resize(size, size),
    ])


def build_clip_manifest(clips_dir):
    items = []
    for p in Path(clips_dir).glob("*.npz"):
        vid = p.stem.rsplit("_", 1)[0]
        items.append((str(p), vid))
    return items


class TemporalDataset(Dataset):
    def __init__(self, items, size=160, train=True):
        self.items = items
        self.size = size
        self.train = train
        self.aug = clip_aug(size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            d = np.load(self.items[i][0])
            frames, lms = d["frames"], d["lms"]
        except Exception:
            return self.__getitem__((i + 1) % len(self))
        label = 0
        if self.train and random.random() < 0.5:
            frames = make_fake(frames.copy(), lms)
            label = 1
        # 시간-일관 증강: 프레임0로 replay 확정 후 전 프레임 동일 적용
        first = self.aug(image=frames[0])
        rep = first["replay"]
        out = [first["image"]]
        for t in range(1, len(frames)):
            out.append(alb.ReplayCompose.replay(rep, image=frames[t])["image"])
        clip = np.stack(out).astype(np.float32) / 255.0
        clip = (clip - MEAN) / STD                      # [T,H,W,3]
        clip = torch.from_numpy(clip.transpose(0, 3, 1, 2))  # [T,3,H,W]
        return clip, torch.tensor(float(label))
