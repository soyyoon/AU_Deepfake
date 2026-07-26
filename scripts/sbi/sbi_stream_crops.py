#!/usr/bin/env python
"""SBI 학습용 real 얼굴 크롭 스트리밍 추출 (CelebDF/FF++ train real).

각 real 영상: 스트리밍 다운로드 -> 균일 샘플 N프레임 -> 얼굴검출(pyfeat) 크롭 -> 저장 -> 삭제.
per-row slug 지원(CelebDF/FF++ 혼재). 디스크 안전, 샤딩/resumable.
출력: <out>/<video_id>/*.png  (SBI는 이미지 단위라 시퀀스 구조 불필요, 평면 저장)

  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/sbi_stream_crops.py \
      --targets convnext_stage/sbi_real_targets.csv --out data_sbi/real \
      --per-video 24 [--shard i --nshard N]
"""
import argparse
import base64
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
from stream_extract import download_one, load_creds            # noqa: E402
from stream_ffpp_convnext_frames import square_crop, boxes_per_frame  # noqa: E402


def process(detector, tmp, out_dir, vid, per_video):
    cap = cv2.VideoCapture(str(tmp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return 0
    k = max(1, total // max(per_video, 1))
    fex = detector.detect_video(str(tmp), skip_frames=k, batch_size=32)
    boxes = boxes_per_frame(fex)
    if not boxes:
        cap.release(); return 0
    idxs = list(range(0, total, k))
    fdir = Path(out_dir) / vid
    fdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in range(min(len(idxs), len(boxes), per_video)):
        if boxes[i] is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[i])
        ok, img = cap.read()
        if not ok or img is None:
            continue
        crop = square_crop(img, *boxes[i])
        if crop is None:
            continue
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fdir / f"{saved:03d}.png"), crop)
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default="data_sbi/real")
    ap.add_argument("--per-video", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default="./.sbi_work")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    if args.nshard > 1:
        rows = [r for i, r in enumerate(rows) if i % args.nshard == args.shard]
    if args.limit:
        rows = rows[: args.limit]
    user, key = load_creds()
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    from feat import Detector
    detector = Detector(device=args.device)
    print(f"[sbi-crops] {len(rows)} real videos | Detector on {args.device}", flush=True)

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid = r["video_id"]
        fdir = Path(args.out) / vid
        if fdir.exists() and any(fdir.glob("*.png")):
            n_skip += 1; continue
        tmp = Path(args.workdir) / f"{args.shard}_{vid}.mp4"
        try:
            download_one(r["slug"], r["name"], tmp, auth)
            saved = process(detector, tmp, args.out, vid, args.per_video)
            n_ok += (saved > 0); n_err += (saved == 0)
        except Exception as e:
            n_err += 1
            print(f"  [err] {vid}: {str(e)[:80]}", flush=True)
        finally:
            if tmp.exists():
                tmp.unlink()
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[sbi-crops] {i+1}/{len(rows)} ok={n_ok} skip={n_skip} err={n_err} "
                  f"({(i+1)/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    print(f"\n완료 shard{args.shard}: ok={n_ok} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
