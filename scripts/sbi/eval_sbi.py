#!/usr/bin/env python
"""SBI 모델 비디오-레벨 평가: 프레임 점수 평균 -> video score -> per-domain AUC.

SBI는 face 크롭 학습 -> 평가도 face 크롭. frames layout: <base>/<vid>/frames/*.png 또는
<base>/<vid>/*.png 둘 다 지원.
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/eval_sbi.py \
      --ckpt outputs/sbi/best.pt --frames-base convnext_stage/ffpp_frames \
      --targets convnext_stage/ffpp_cn_targets.csv --tag FFpp
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MEAN, STD             # noqa: E402


def roc_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def frames_of(base, vid, maxf):
    d = Path(base) / vid / "frames"
    if not d.exists():
        d = Path(base) / vid
    return sorted(d.glob("*.png"))[:maxf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames-base", required=True)
    ap.add_argument("--targets", required=True, help="video_id,label[,method]")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--size", type=int, default=0, help="0=ckpt cfg")
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--pad", type=float, default=0.0,
                    help="reflect-pad 비율(tight 크롭에 마진 추가해 학습 스케일 매칭)")
    ap.add_argument("--agg", default="topk", choices=["mean", "max", "topk"],
                    help="프레임->비디오 집계(기본 topk: 간헐적 fake에 강건)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import timm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck.get("cfg", {})
    size = args.size or cfg.get("size", 256)
    model = timm.create_model(cfg.get("model", "tf_efficientnet_b4"),
                              pretrained=False, num_classes=1).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[eval] {args.ckpt} (ep{ck.get('epoch')}) size={size} -> {args.tag}", flush=True)

    rows = list(csv.DictReader(open(args.targets)))
    y, s, meth, vids = [], [], [], []
    miss = 0
    for r in rows:
        vid = r["video_id"]
        pngs = frames_of(args.frames_base, vid, args.max_frames)
        if not pngs:
            miss += 1
            continue
        batch = []
        for p in pngs:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if args.pad > 0:
                ph, pw = int(im.shape[0] * args.pad), int(im.shape[1] * args.pad)
                im = cv2.copyMakeBorder(im, ph, ph, pw, pw, cv2.BORDER_REFLECT)
            im = cv2.resize(im, (size, size)).astype(np.float32) / 255.0
            im = (im - MEAN) / STD
            batch.append(im.transpose(2, 0, 1))
        if not batch:
            miss += 1
            continue
        x = torch.from_numpy(np.stack(batch)).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            p = torch.sigmoid(model(x).squeeze(1)).float().cpu().numpy()
        if args.agg == "max":
            score = float(p.max())
        elif args.agg == "topk":
            score = float(np.sort(p)[::-1][:min(args.topk, len(p))].mean())
        else:
            score = float(p.mean())
        y.append(int(r["label"])); s.append(score)
        meth.append(r.get("method", "")); vids.append(vid)

    y, s = np.array(y), np.array(s)
    print(f"\n[{args.tag}] n={len(y)} (real={int((y==0).sum())}, fake={int((y==1).sum())}) 누락={miss}")
    print(f"  SBI AUC = {roc_auc(y, s):.4f}")
    ms = np.array(meth)
    if (ms != "").any():
        reals = s[y == 0]
        for m in sorted(set(ms[y == 1])):
            fk = (y == 1) & (ms == m)
            yy = np.r_[np.zeros(len(reals)), np.ones(fk.sum())]
            ss = np.r_[reals, s[fk]]
            print(f"    [{m}] AUC={roc_auc(yy, ss):.4f} (fake={int(fk.sum())})")
    if args.out:
        np.savez(args.out, y=y, s=s, meth=ms, vids=np.array(vids))
        print("saved", args.out)


if __name__ == "__main__":
    main()
