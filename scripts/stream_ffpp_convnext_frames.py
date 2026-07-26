#!/usr/bin/env python
"""FF++ c23 영상을 스트리밍 다운로드 -> 얼굴 크롭 프레임 PNG 저장(ConvNeXt 입력용).

ConvNeXt가 DFD에 쓴 크롭 전략을 복제:
  largest_face_square_bbox_with_margin_no_alignment, margin_ratio=0.35
  (얼굴검출 -> 가장 큰 얼굴 -> 정사각 bbox + 35% 마진 -> 크롭 -> 저장)

디스크 안전: 영상 1개 받아 프레임 뽑고 즉시 삭제(피크 ~1영상). Resumable.
pyfeat env에서 실행(얼굴검출=Py-Feat Detector, GPU). 저장:
  <out>/<video_id>/frames/NNNN.png

사용:
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/stream_ffpp_convnext_frames.py \
      --targets convnext_stage/ffpp_targets.csv --slug xdxd003/ff-c23 \
      --out convnext_stage/ffpp_frames [--limit 3]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_extract import download_one, load_creds     # noqa: E402

MARGIN = 0.35
MAX_FRAMES = 64


def square_crop(img, x, y, w, h, margin=MARGIN):
    """가장 큰 얼굴 box -> 정사각 + 마진 크롭(경계 클램프). 실패시 None."""
    H, W = img.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)
    x0 = int(round(cx - side / 2.0)); y0 = int(round(cy - side / 2.0))
    x1 = int(round(cx + side / 2.0)); y1 = int(round(cy + side / 2.0))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c - x0c < 8 or y1c - y0c < 8:
        return None
    return img[y0c:y1c, x0c:x1c]


def boxes_per_frame(fex):
    """Fex -> {처리순서 index: (x,y,w,h)} 가장 큰 얼굴. 처리 순서 기준으로 정렬."""
    import pandas as pd
    df = pd.DataFrame(fex.copy())
    if any(n == "frame" for n in (list(df.index.names) if df.index.names else [])):
        df = df.reset_index(drop=("frame" in df.columns))
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))
    need = {"FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"}
    if not need.issubset(df.columns):
        return []
    df["_area"] = df["FaceRectWidth"].fillna(0) * df["FaceRectHeight"].fillna(0)
    df = df.sort_values("_area").groupby("frame", as_index=False).last()
    df = df.sort_values("frame")
    out = []
    for _, r in df.iterrows():
        if r["_area"] <= 0 or np.isnan(r["FaceRectX"]):
            out.append(None)
        else:
            out.append((float(r.FaceRectX), float(r.FaceRectY),
                        float(r.FaceRectWidth), float(r.FaceRectHeight)))
    return out


def process_video(detector, tmp_path, out_dir, video_id):
    """영상 -> 얼굴 크롭 프레임 저장. 저장한 프레임 수 반환."""
    cap = cv2.VideoCapture(str(tmp_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    k = max(1, total // MAX_FRAMES)
    processed_idx = list(range(0, total, k))[:MAX_FRAMES]

    fex = detector.detect_video(str(tmp_path), skip_frames=k, batch_size=64)
    boxes = boxes_per_frame(fex)              # 처리 순서대로 box (largest face)
    if not boxes:
        cap.release()
        return 0

    fdir = Path(out_dir) / video_id / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    n = min(len(processed_idx), len(boxes))
    for i in range(n):
        box = boxes[i]
        if box is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, processed_idx[i])
        ok, img = cap.read()
        if not ok or img is None:
            continue
        crop = square_crop(img, *box)
        if crop is None:
            continue
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(fdir / f"{saved:04d}.png"), crop)
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="video_id,name,label,method csv")
    ap.add_argument("--slug", default="xdxd003/ff-c23")
    ap.add_argument("--out", default="convnext_stage/ffpp_frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workdir", default="./.ffpp_cn_work")
    ap.add_argument("--shard", type=int, default=0, help="이 워커의 인덱스")
    ap.add_argument("--nshard", type=int, default=1, help="총 워커 수(round-robin 분할)")
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
    print(f"[ffpp-cn] {len(rows)} videos | Detector on {args.device}", flush=True)

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, name = r["video_id"], r["name"]
        fdir = Path(args.out) / vid / "frames"
        if fdir.exists() and any(fdir.glob("*.png")):
            n_skip += 1
            continue
        tmp = Path(args.workdir) / f"{vid}.mp4"
        try:
            download_one(args.slug, name, tmp, auth)
            saved = process_video(detector, tmp, args.out, vid)
            if saved == 0:
                n_err += 1
            else:
                n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [err] {vid}: {str(e)[:100]}", flush=True)
        finally:
            if tmp.exists():
                tmp.unlink()
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[ffpp-cn] {i+1}/{len(rows)} ok={n_ok} skip={n_skip} "
                  f"err={n_err} ({rate:.2f} vid/s)", flush=True)

    print(f"\n완료: ok={n_ok} skip={n_skip} err={n_err}  -> {args.out}")


if __name__ == "__main__":
    main()
