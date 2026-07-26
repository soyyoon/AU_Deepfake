#!/usr/bin/env python
"""로컬 영상 -> dense 연속 클립 + 랜드마크 (Temporal 브랜치 추론용).

local_video_crops.py 가 SBI용 얼굴 크롭을 만든다면, 이 스크립트는 Temporal용 연속 클립을 만든다.
targets: video_id,path  -> <out>/<video_id>_<ci>.npz (frames[16,256,256,3], lms[16,68,2])
  PY=~/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/local_video_clips.py \
      --targets videos.csv --out clips
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_clips import crop_lm, CLIP_LEN, N_CLIPS      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="video_id,path csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=N_CLIPS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    from feat import Detector
    det = Detector(device=args.device)
    rows = list(csv.DictReader(open(args.targets)))
    print(f"[local-clips] {len(rows)} videos", flush=True)

    for r in rows:
        vid, path = r["video_id"], r["path"]
        if list(Path(args.out).glob(f"{vid}_*.npz")):
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < CLIP_LEN + 2:
            cap.release()
            print(f"  [skip] {vid}: too short"); continue
        starts = np.linspace(2, max(3, total - CLIP_LEN - 2), args.n_clips).astype(int)
        n = 0
        for ci, st in enumerate(starts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(st))
            frames = []
            for _ in range(CLIP_LEN):
                ok, im = cap.read()
                if not ok:
                    break
                frames.append(im)
            if len(frames) < CLIP_LEN:
                continue
            out = crop_lm(det, frames)
            if out is None:
                continue
            cr, lm = out
            np.savez_compressed(Path(args.out) / f"{vid}_{ci}.npz", frames=cr, lms=lm)
            n += 1
        cap.release()
        print(f"  {vid}: {n} clips", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
