#!/usr/bin/env python
"""실제 영상 -> 연속(dense) 프레임 클립 + 랜드마크 (temporal 브랜치 학습용).

per video: 스트리밍 다운로드 -> 여러 구간에서 CLIP_LEN 연속 프레임 -> 프레임별 얼굴크롭+68랜드마크
-> npz 저장 -> 삭제. 샤딩/resumable. pyfeat env.
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/stream_clips.py \
      --targets convnext_stage/sbi_real_targets.csv --out data_sbi/clips [--shard i --nshard N]
"""
import argparse
import base64
import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_extract import download_one, load_creds       # noqa: E402
from stream_ffpp_convnext_frames import square_crop        # noqa: E402

CLIP_LEN = 16
N_CLIPS = 2
XC = [f"x_{i}" for i in range(68)]
YC = [f"y_{i}" for i in range(68)]


def crop_lm(det, frames):
    """연속 프레임 리스트 -> (crops[N,256,256,3], lms[N,68,2]) 또는 None."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    paths = []
    for i, im in enumerate(frames):
        p = f"{tmpdir}/{i:03d}.png"
        cv2.imwrite(p, im); paths.append(p)
    fex = det.detect_image(paths).reset_index(drop=True)
    crops, lms = [], []
    for i, p in enumerate(paths):
        sub = fex[fex["input"] == p] if "input" in fex.columns else fex.iloc[[i]]
        if len(sub) == 0:
            return None
        row = sub.iloc[0]
        try:
            x, y, w, h = (float(row["FaceRectX"]), float(row["FaceRectY"]),
                          float(row["FaceRectWidth"]), float(row["FaceRectHeight"]))
        except Exception:
            return None
        c = square_crop(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), x, y, w, h)
        if c is None:
            return None
        lm = np.stack([row[XC].to_numpy(float), row[YC].to_numpy(float)], 1)
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * (1 + 2 * 0.35)
        x0, y0 = cx - side / 2, cy - side / 2
        lm = (lm - [x0, y0]) / side * 256
        crops.append(cv2.resize(c, (256, 256))); lms.append(lm)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return np.stack(crops), np.stack(lms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default="data_sbi/clips")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default="./.clip_work")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    rows = [r for i, r in enumerate(rows) if i % args.nshard == args.shard]
    user, key = load_creds()
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    from feat import Detector
    det = Detector(device=args.device)
    print(f"[clips] shard {args.shard}: {len(rows)} videos", flush=True)

    n_ok = n_err = 0
    t0 = time.time()
    for j, r in enumerate(rows):
        vid = r["video_id"]
        if list(Path(args.out).glob(f"{vid}_*.npz")):
            continue
        tmp = Path(args.workdir) / f"{args.shard}_{vid}.mp4"
        try:
            download_one(r["slug"], r["name"], tmp, auth)
            cap = cv2.VideoCapture(str(tmp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < CLIP_LEN + 4:
                cap.release(); tmp.unlink(); n_err += 1; continue
            starts = np.linspace(2, total - CLIP_LEN - 2, N_CLIPS).astype(int)
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
                crops, lms = out
                np.savez_compressed(Path(args.out) / f"{vid}_{ci}.npz", frames=crops, lms=lms)
            cap.release()
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [err] {vid}: {str(e)[:70]}", flush=True)
        finally:
            if tmp.exists():
                tmp.unlink()
        if (j + 1) % 25 == 0 or j + 1 == len(rows):
            print(f"[clips] {j+1}/{len(rows)} ok={n_ok} err={n_err} "
                  f"({(j+1)/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    print(f"완료 shard{args.shard}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
