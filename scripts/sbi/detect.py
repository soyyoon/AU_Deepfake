#!/usr/bin/env python
"""최종 2-브랜치 딥페이크 탐지기: SBI(공간 블렌딩) + Temporal(시간 불일치).

SBI는 face-swap, Temporal은 reenactment/lip-sync를 잡는 상보 쌍. 각 브랜치는 서로 다른 조작
유형의 전문가 -> 탐지 융합은 OR(max): "swap이든 reenact든 하나라도 잡히면 fake". (mean은 AUC
랭킹엔 소폭 유리하나 한 브랜치가 다른 브랜치를 희석해 threshold 판정서 reenactment를 놓침.)

입력(사전 추출): 얼굴 크롭 <crops>/<vid>/frames/*.png, dense 클립 <clips>/<vid>_*.npz
  (크롭: dfdc_extract.py / local_video_crops.py, 클립: stream_clips.py 로 생성)

  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/detect.py \
      --crops usertest/crops --clips data_sbi/eval_kakao --vids kakao_1 kakao_2 kakao_3
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MEAN, STD                    # noqa: E402
from temporal_train import TemporalNet           # noqa: E402


def load_models(dev):
    import timm
    cs = torch.load("outputs/sbi/best.pt", map_location=dev, weights_only=False)
    sbi = timm.create_model(cs["cfg"]["model"], pretrained=False, num_classes=1).to(dev)
    sbi.load_state_dict(cs["model"]); sbi.eval()
    ct = torch.load("outputs/temporal/best.pt", map_location=dev, weights_only=False)
    tmp = TemporalNet().to(dev); tmp.load_state_dict(ct["model"]); tmp.eval()
    return sbi, cs["cfg"].get("size", 256), tmp, ct["cfg"].get("size", 160)


def sbi_score(sbi, size, crops_dir, vid, dev, topk=5):
    d = Path(crops_dir) / vid / "frames"
    if not d.exists():
        d = Path(crops_dir) / vid
    pngs = sorted(d.glob("*.png"))[:32]
    if not pngs:
        return None
    b = [((cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (size, size))
           .astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1) for p in pngs]
    x = torch.from_numpy(np.stack(b)).to(dev)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        pf = torch.sigmoid(sbi(x).squeeze(1)).float().cpu().numpy()
    return float(np.sort(pf)[::-1][:min(topk, len(pf))].mean())


def temporal_score(tmp, size, clips_dir, vid, dev):
    clips = sorted(Path(clips_dir).glob(f"{vid}_*.npz"))
    if not clips:
        return None
    scores = []
    for p in clips:
        fr = np.load(p)["frames"]
        fr = np.stack([cv2.resize(f, (size, size)) for f in fr]).astype(np.float32) / 255.
        fr = ((fr - MEAN) / STD).transpose(0, 3, 1, 2)
        x = torch.from_numpy(fr).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            scores.append(float(torch.sigmoid(tmp(x)).item()))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--vids", nargs="+", required=True)
    ap.add_argument("--thr-sbi", type=float, default=0.6, help="SBI(swap) fake 임계값")
    ap.add_argument("--thr-temp", type=float, default=0.5, help="Temporal(reenact) fake 임계값")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sbi, ss, tmp, ts = load_models(dev)

    print(f"{'video':12s} {'SBI(swap)':>10s} {'Temporal':>10s} {'FUSED(OR)':>10s}  판정(근거)")
    print("-" * 62)
    for vid in args.vids:
        s = sbi_score(sbi, ss, args.crops, vid, dev)
        t = temporal_score(tmp, ts, args.clips, vid, dev)
        s = -1 if s is None else s
        t = -1 if t is None else t
        fused = max(s, t)                                  # OR 융합
        hits = []
        if s >= args.thr_sbi:
            hits.append("swap")
        if t >= args.thr_temp:
            hits.append("reenact")
        verdict = f"FAKE ({'+'.join(hits)})" if hits else "REAL"
        print(f"{vid:12s} {s:10.3f} {t:10.3f} {fused:10.3f}  {verdict}")


if __name__ == "__main__":
    main()
