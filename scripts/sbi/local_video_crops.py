#!/usr/bin/env python
"""로컬 영상에서 얼굴 크롭 추출 (SBI/ConvNeXt 평가용). 스트리밍 대신 로컬 파일.

targets: video_id,path[,label]  -> <out>/<video_id>/*.png (평면). pyfeat 얼굴검출+크롭.
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/local_video_crops.py \
      --targets dfdc_stage/targets.csv --out dfdc_stage/crops --per-video 24 [--shard i --nshard N]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_ffpp_convnext_frames import square_crop, boxes_per_frame  # noqa: E402


def process(detector, path, out_dir, vid, per_video):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return 0
    k = max(1, total // max(per_video, 1))
    fex = detector.detect_video(str(path), skip_frames=k, batch_size=32)
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
        cv2.imwrite(str(fdir / f"{saved:03d}.png"),
                    cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA))
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    rows = [r for i, r in enumerate(rows) if i % args.nshard == args.shard]
    from feat import Detector
    detector = Detector(device=args.device)
    print(f"[local-crops] {len(rows)} videos", flush=True)
    n_ok = n_err = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid = r["video_id"]
        if (Path(args.out) / vid).exists() and any((Path(args.out) / vid).glob("*.png")):
            continue
        try:
            n_ok += process(detector, r["path"], args.out, vid, args.per_video) > 0
        except Exception as e:
            n_err += 1
            print(f"  [err] {vid}: {str(e)[:80]}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"[local-crops] {i+1}/{len(rows)} ok={n_ok} err={n_err} "
                  f"({(i+1)/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    print(f"완료 shard{args.shard}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
