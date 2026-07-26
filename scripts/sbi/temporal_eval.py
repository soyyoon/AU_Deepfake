#!/usr/bin/env python
"""Temporal 브랜치 평가: 비디오별 dense 클립 -> 클립 점수 평균 -> per-domain AUC.

클립 dir: <clips>/<video_id>_<ci>.npz. 타깃: video_id,label[,method].
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/temporal_eval.py \
      --ckpt outputs/temporal/best.pt --clips data_sbi/eval_ffpp --targets convnext_stage/ffpp_targets.csv --tag FFpp
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from temporal_dataset import MEAN, STD           # noqa: E402
from temporal_train import TemporalNet           # noqa: E402
import cv2


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    size = ck["cfg"].get("size", 160)
    model = TemporalNet().to(dev); model.load_state_dict(ck["model"]); model.eval()

    # video_id -> [clip paths]
    byvid = defaultdict(list)
    for p in Path(args.clips).glob("*.npz"):
        byvid[p.stem.rsplit("_", 1)[0]].append(p)

    def clip_score(p):
        d = np.load(p); fr = d["frames"]
        fr = np.stack([cv2.resize(f, (size, size)) for f in fr]).astype(np.float32) / 255.0
        fr = ((fr - MEAN) / STD).transpose(0, 3, 1, 2)
        x = torch.from_numpy(fr).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return float(torch.sigmoid(model(x)).item())

    y, s, meth = [], [], []
    miss = 0
    for r in csv.DictReader(open(args.targets)):
        vid = r["video_id"]
        if vid not in byvid:
            miss += 1; continue
        sc = np.mean([clip_score(p) for p in byvid[vid]])
        y.append(int(r["label"])); s.append(sc); meth.append(r.get("method", ""))
    y, s = np.array(y), np.array(s)
    print(f"\n[{args.tag}] n={len(y)} (real={int((y==0).sum())}, fake={int((y==1).sum())}) 누락={miss}")
    print(f"  Temporal AUC = {roc_auc(y, s):.4f}")
    ms = np.array(meth)
    if (ms != "").any():
        reals = s[y == 0]
        for m in sorted(set(ms[y == 1])):
            fk = (y == 1) & (ms == m)
            yy = np.r_[np.zeros(len(reals)), np.ones(fk.sum())]; ss = np.r_[reals, s[fk]]
            print(f"    [{m}] AUC={roc_auc(yy, ss):.4f} (fake={int(fk.sum())})")
    if args.out:
        np.savez(args.out, y=y, s=s, meth=ms)


if __name__ == "__main__":
    main()
