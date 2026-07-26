#!/usr/bin/env python
"""real 크롭마다 68-pt 랜드마크(pyfeat) -> 캐시(<png>.lmk.npy). SBI 마스크용.

pyfeat env에서 실행(이미 작동 확인). 배치 detect_image. 샤딩/resumable.
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/cache_landmarks.py \
      --root data_sbi/real [--shard i --nshard N] [--batch 32]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_sbi/real")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from feat import Detector
    det = Detector(device=args.device)
    XC = [f"x_{i}" for i in range(68)]
    YC = [f"y_{i}" for i in range(68)]

    pngs = sorted(Path(args.root).rglob("*.png"))
    pngs = [p for i, p in enumerate(pngs) if i % args.nshard == args.shard]
    todo = [p for p in pngs if not p.with_suffix(".lmk.npy").exists()]
    print(f"[lmk] shard {args.shard}: {len(todo)}/{len(pngs)} todo", flush=True)

    n_ok = n_none = 0
    for b in range(0, len(todo), args.batch):
        batch = todo[b:b + args.batch]
        try:
            fex = det.detect_image([str(p) for p in batch])
        except Exception as e:
            print(f"  [batch err] {str(e)[:80]}", flush=True)
            continue
        fex = fex.reset_index(drop=True)
        # 'input' 컬럼으로 파일 매칭(가장 큰 얼굴 1개)
        for p in batch:
            sub = fex[fex["input"] == str(p)] if "input" in fex.columns else fex
            if len(sub) == 0:
                n_none += 1
                continue
            if {"FaceRectWidth", "FaceRectHeight"}.issubset(sub.columns):
                sub = sub.assign(_a=sub["FaceRectWidth"].fillna(0) * sub["FaceRectHeight"].fillna(0))
                row = sub.sort_values("_a").iloc[-1]
            else:
                row = sub.iloc[0]
            try:
                lm = np.stack([row[XC].to_numpy(float), row[YC].to_numpy(float)], axis=1)
            except Exception:
                n_none += 1
                continue
            if np.isnan(lm).any():
                n_none += 1
                continue
            np.save(p.with_suffix(".lmk.npy"), lm.astype(np.float32))
            n_ok += 1
        if (b // args.batch) % 20 == 0:
            print(f"[lmk] {b+len(batch)}/{len(todo)} ok={n_ok} none={n_none}", flush=True)
    print(f"완료 shard{args.shard}: ok={n_ok} none={n_none}")


if __name__ == "__main__":
    main()
