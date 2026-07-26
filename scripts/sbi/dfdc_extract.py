#!/usr/bin/env python
"""DFDC 로컬 영상 -> 얼굴 크롭(SBI/ConvNeXt) + 30ch AU feature(AU) 동시 추출 (pyfeat 1회 통과).

크롭: <crops_out>/<vid>/frames/NNNN.png (24장), AU: <au_out>/<vid>/features.npy [64,30].
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/sbi/dfdc_extract.py \
      --targets dfdc_stage/targets.csv --crops-out dfdc_stage/crops --au-out dfdc_stage/au_data \
      [--shard i --nshard N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_ffpp_convnext_frames import square_crop, boxes_per_frame  # noqa: E402
from extract_features import per_frame_matrix, resample_time          # noqa: E402

GROUPS = ["aus", "poses", "emotions"]
CROP_N = 24
SEQ = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--crops-out", required=True)
    ap.add_argument("--au-out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.targets)))
    rows = [r for i, r in enumerate(rows) if i % args.nshard == args.shard]
    from feat import Detector
    det = Detector(device=args.device)
    print(f"[dfdc] shard {args.shard}: {len(rows)} videos", flush=True)

    n_ok = n_err = 0
    t0 = time.time()
    for j, r in enumerate(rows):
        vid = r["video_id"]
        crop_dir = Path(args.crops_out) / vid / "frames"
        au_npy = Path(args.au_out) / vid / "features.npy"
        if au_npy.exists() and crop_dir.exists() and any(crop_dir.glob("*.png")):
            continue
        try:
            cap = cv2.VideoCapture(r["path"])
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release(); n_err += 1; continue
            k = max(1, total // SEQ)
            fex = det.detect_video(r["path"], skip_frames=k, batch_size=32)
            # AU features
            mat, au_mat, cols = per_frame_matrix(fex, GROUPS)
            seq = resample_time(mat, SEQ)
            if seq is not None:
                au_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(au_npy, seq.astype(np.float32))
                np.save(au_npy.parent / "au_sequence.npy", resample_time(au_mat, SEQ).astype(np.float32))
                (au_npy.parent / "meta.json").write_text(json.dumps(
                    {"video_id": vid, "label": int(r["label"]), "source": "DFDC"}))
            # crops
            boxes = boxes_per_frame(fex)
            idxs = list(range(0, total, k))
            crop_dir.mkdir(parents=True, exist_ok=True)
            step = max(1, len(boxes) // CROP_N)
            saved = 0
            for i in range(0, min(len(idxs), len(boxes)), step):
                if boxes[i] is None or saved >= CROP_N:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[i])
                ok, img = cap.read()
                if not ok or img is None:
                    continue
                c = square_crop(img, *boxes[i])
                if c is None:
                    continue
                cv2.imwrite(str(crop_dir / f"{saved:03d}.png"),
                            cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA))
                saved += 1
            cap.release()
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [err] {vid}: {str(e)[:80]}", flush=True)
        if (j + 1) % 25 == 0 or j + 1 == len(rows):
            print(f"[dfdc] {j+1}/{len(rows)} ok={n_ok} err={n_err} "
                  f"({(j+1)/max(time.time()-t0,1e-6):.2f}/s)", flush=True)
    print(f"완료 shard{args.shard}: ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
