#!/usr/bin/env python
"""판단 근거 시각화 (Grad-CAM): SBI + Temporal 브랜치가 얼굴 어디를 보고 fake로 판정하는지.

- SBI(EfficientNet-b4): 프레임별 얼굴 공간 Grad-CAM -> 얼굴 중앙(블렌딩 경계) 근거.
- Temporal(EffB0+BiGRU): 클립의 프레임별 Grad-CAM -> 입/턱(시간 불일치) 근거.
빨강 = fake 근거 영역. Temporal 초록테 = 판정 주도 프레임.

입력(사전 추출): 크롭 <crops>/<vid>/*.png, 클립 <clips>/<vid>_*.npz
  (local_video_crops.py / local_video_clips.py 로 생성)

  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/gradcam_viz.py \
      --crops gradcam2/crops --clips gradcam2/clips --vids v_759298 v_817590 --out gradcam2/out.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MEAN as SM, STD as SS               # noqa: E402
from temporal_dataset import MEAN as TM, STD as TS      # noqa: E402
from temporal_train import TemporalNet                  # noqa: E402

CELL = 240


def overlay(bgr, cam, label=None, hi=False):
    """이미지에 Grad-CAM 히트맵(JET) 오버레이."""
    base = cv2.resize(bgr, (CELL, CELL))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    o = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    if hi:
        cv2.rectangle(o, (2, 2), (CELL - 3, CELL - 3), (0, 255, 0), 3)
    if label is not None:
        cv2.rectangle(o, (0, 0), (CELL, 22), (0, 0, 0), -1)
        cv2.putText(o, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return o


def _cam_from(feats, grad, t):
    """Grad-CAM: 채널별 grad 평균 가중 x 활성 -> ReLU -> 정규화."""
    w = grad[t].mean((1, 2))
    cam = torch.relu((w[:, None, None] * feats[t]).sum(0)).detach().cpu().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cv2.resize(cam, (CELL, CELL))


# ---------------- SBI (프레임 공간 Grad-CAM) ----------------

class SBIGradCAM:
    def __init__(self, dev):
        import timm
        ck = torch.load("outputs/sbi/best.pt", map_location=dev, weights_only=False)
        self.size = ck["cfg"].get("size", 256)
        self.m = timm.create_model(ck["cfg"]["model"], pretrained=False, num_classes=1).to(dev)
        self.m.load_state_dict(ck["model"]); self.m.eval()
        self.dev = dev

    def score_cam(self, bgr):
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (self.size, self.size)).astype(np.float32) / 255.
        x = torch.from_numpy(((x - SM) / SS).transpose(2, 0, 1)).unsqueeze(0).to(self.dev)
        feats = self.m.forward_features(x); feats.retain_grad()
        logit = self.m.forward_head(feats).squeeze()
        self.m.zero_grad(); logit.backward()
        return float(torch.sigmoid(logit).item()), _cam_from(feats, feats.grad, 0)


# ---------------- Temporal (클립 프레임별 Grad-CAM) ----------------

class TemporalGradCAM:
    def __init__(self, dev):
        torch.backends.cudnn.enabled = False      # cuDNN GRU는 eval backward 불가 -> 우회
        ck = torch.load("outputs/temporal/best.pt", map_location=dev, weights_only=False)
        self.size = ck["cfg"].get("size", 160)
        self.m = TemporalNet().to(dev); self.m.load_state_dict(ck["model"]); self.m.eval()
        self.dev = dev

    def score_cams(self, npz):
        fr = np.load(npz)["frames"]
        x = np.stack([cv2.resize(f, (self.size, self.size)) for f in fr]).astype(np.float32) / 255.
        x = torch.from_numpy(((x - TM) / TS).transpose(0, 3, 1, 2)).to(self.dev)
        feats = self.m.backbone.forward_features(x); feats.retain_grad()
        pooled = feats.mean((2, 3)).unsqueeze(0)          # [1,T,D] (backbone global_pool과 동치)
        out, _ = self.m.gru(pooled)
        logit = self.m.head(out.mean(1)).squeeze()
        self.m.zero_grad(); logit.backward()
        cams, contrib = [], []
        for t in range(len(fr)):
            cam = _cam_from(feats, feats.grad, t)
            cams.append(cam); contrib.append(float(cam.sum()))
        return float(torch.sigmoid(logit).item()), np.array(contrib), cams, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--vids", nargs="+", required=True)
    ap.add_argument("--out", default="gradcam_viz.png")
    ap.add_argument("--n-top", type=int, default=4, help="SBI 상위 몇 프레임 표시")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sbi, temp = SBIGradCAM(dev), TemporalGradCAM(dev)

    print(f"{'video':12s} {'SBI(swap)':>10s} {'Temporal':>10s}  최종")
    sbi_rows, temp_rows = [], []
    for vid in args.vids:
        # SBI: 프레임별 점수+CAM, 상위 n개
        imgs = [cv2.imread(str(p)) for p in sorted((Path(args.crops) / vid).glob("*.png"))]
        sres = [sbi.score_cam(im) for im in imgs]
        sscs = np.array([r[0] for r in sres])
        s5 = float(np.sort(sscs)[::-1][:5].mean())
        so = np.argsort(sscs)[::-1][:args.n_top]
        tag = np.full((CELL, 120, 3), 40, np.uint8)
        cv2.putText(tag, vid, (4, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(tag, f"S {s5:.2f}", (4, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        sbi_rows.append(np.concatenate([tag] + [overlay(imgs[i], sres[i][1], f"{sres[i][0]:.2f}") for i in so], axis=1))

        # Temporal: 최고 점수 클립의 프레임별 CAM
        clips = sorted(Path(args.clips).glob(f"{vid}_*.npz"))
        tres = [temp.score_cams(str(c)) for c in clips]
        tsc, contrib, cams, tfr = max(tres, key=lambda z: z[0])
        top = set(np.argsort(contrib)[::-1][:3])
        tag = np.full((CELL, 120, 3), 40, np.uint8)
        cv2.putText(tag, vid, (4, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(tag, f"T {tsc:.2f}", (4, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cells = [overlay(cv2.cvtColor(tfr[i], cv2.COLOR_RGB2BGR), cams[i], hi=i in top) for i in range(0, len(tfr), 2)]
        temp_rows.append(np.concatenate([tag] + cells, axis=1))

        verdict = "FAKE(" + "+".join([b for b, ok in [("swap", s5 >= .5), ("reenact", tsc >= .5)] if ok]) + ")" if (s5 >= .5 or tsc >= .5) else "REAL"
        print(f"{vid:12s} {s5:10.2f} {tsc:10.2f}  {verdict}")

    def stack(rows):
        w = max(r.shape[1] for r in rows)
        return np.concatenate([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for r in rows], 0)

    banner = lambda txt, w: cv2.putText(np.full((30, w, 3), 30, np.uint8), txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    S, T = stack(sbi_rows), stack(temp_rows)
    w = max(S.shape[1], T.shape[1])
    S = cv2.copyMakeBorder(S, 0, 0, 0, w - S.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    T = cv2.copyMakeBorder(T, 0, 0, 0, w - T.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
    grid = np.concatenate([banner("SBI (swap) - face-center blending", w), S,
                           banner("Temporal (reenact) - mouth inconsistency", w), T], 0)
    cv2.imwrite(args.out, grid)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
