#!/usr/bin/env python
"""Disk-safe chunked feature extraction straight out of a Kaggle .zip.

For datasets that ship as ONE big zip (FaceForensics++ c23, Celeb-DF v2) we never
unpack the whole thing. We stream the archive member-by-member: read one video
from the zip -> Py-Feat detect -> write tiny .npy features -> delete the temp
video -> next. Peak extra disk = one video (a few MB), regardless of total size.

Labels come from the path inside the zip (these datasets have no metadata.json):
  fake (1): manipulated / synthesis / a known forgery-method folder
  real (0): original / *-real / youtube / actors
Each video also gets a `method` tag (Deepfakes, Face2Face, synthesis, real, ...)
so evaluation can be decomposed by forgery method later.

Output mirrors scripts/extract_features.py (so downstream is identical):
  data/<video_id>/features.npy      [seq_len, 30] float32  (AUs+pose+emotions)
  data/<video_id>/au_sequence.npy   [seq_len, 20] float32  (AU-only)
  data/<video_id>/meta.json
  data/dataset_metadata_v2.csv      (appended; + dataset/method columns)
  data/errors_v2.jsonl

Resumable: a video whose features.npy already exists is skipped *before* it is
unzipped, so an interrupted/re-run job costs almost nothing.

Usage:
  PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
  export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0
  # 1) sanity-check labels WITHOUT processing:
  $PY scripts/chunked_extract_zip.py --zip dl/ff-c23.zip --dataset FFpp --dry-run
  # 2) real run:
  $PY scripts/chunked_extract_zip.py --zip dl/ff-c23.zip --dataset FFpp
  $PY scripts/chunked_extract_zip.py --zip dl/celeb-df-v2.zip --dataset CelebDF
  # smoke: add --limit 5
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

# reuse the (now-fixed) per-frame logic from the standalone extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import per_frame_matrix, resample_time  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# path-keyword -> forgery method (checked first => fake). Lowercased substring match.
FAKE_METHODS = [
    ("neuraltextures", "NeuralTextures"), ("face2face", "Face2Face"),
    ("faceshifter", "FaceShifter"), ("faceswap", "FaceSwap"),
    ("deepfakedetection", "DeepFakeDetection"), ("deepfakes", "Deepfakes"),
    ("synthesis", "synthesis"), ("manipulated", "manipulated"),
]
REAL_KEYS = ["original", "celeb-real", "youtube-real", "actors", "real", "youtube"]


def classify(path):
    """(label:int, method:str) or (None, None) if the path isn't a labelable video."""
    p = path.lower()
    for key, method in FAKE_METHODS:
        if key in p:
            return 1, method
    for key in REAL_KEYS:
        if key in p:
            return 0, "real"
    return None, None


def safe_id(dataset, method, stem):
    s = f"{dataset.lower()}_{method}_{stem}"
    return re.sub(r"[^0-9A-Za-z_.-]", "_", s)


def list_videos(zf):
    """-> list of (member_name, video_id, label, method) for every labelable video."""
    out = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            continue
        label, method = classify(name)
        if label is None:
            continue
        out.append((name, None, label, method))  # video_id filled by caller (needs dataset)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", required=True)
    ap.add_argument("--dataset", required=True, help="tag, e.g. FFpp or CelebDF")
    ap.add_argument("--out", dest="out_dir", default="data")
    ap.add_argument("--work", default=os.environ.get("WORKDIR", "/tmp/cev_work"),
                    help="scratch dir for the single in-flight video")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--features", default="aus,poses,emotions")
    ap.add_argument("--skip-frames", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print inferred label/method counts from the zip and exit")
    args = ap.parse_args()

    groups = [g.strip() for g in args.features.split(",") if g.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    meta_csv = out_root / "dataset_metadata_v2.csv"
    err_log = out_root / "errors_v2.jsonl"

    zf = zipfile.ZipFile(args.zip_path)
    items = list_videos(zf)
    # fill video_id now that we know the dataset tag
    items = [(name, safe_id(args.dataset, method, Path(name).stem), label, method)
             for (name, _, label, method) in items]
    if args.limit:
        items = items[: args.limit]

    # report inferred labels
    from collections import Counter
    by_lm = Counter((lab, m) for _, _, lab, m in items)
    print(f"[chunk] {args.zip_path}: {len(items)} labelable videos")
    for (lab, m), n in sorted(by_lm.items()):
        print(f"        label={lab} method={m:<18} n={n}")
    # warn about skipped (unlabelable) video members
    n_skipped_unlabel = sum(
        1 for n in zf.namelist()
        if (not n.endswith("/")) and Path(n).suffix.lower() in VIDEO_EXTS
        and classify(n)[0] is None
    )
    if n_skipped_unlabel:
        print(f"        [warn] {n_skipped_unlabel} video(s) had no label keyword -> skipped")
    if args.dry_run:
        return 0
    if not items:
        print("[chunk] nothing to do", file=sys.stderr)
        return 1

    from feat import Detector
    detector = Detector(device=args.device)
    print(f"[chunk] Detector ready on {args.device}")

    fields = ["video_id", "label", "source", "dataset", "method",
              "n_detected_frames", "target_frames", "feat_shape", "video_path"]
    new_rows, n_ok, n_skip, n_err = [], 0, 0, 0
    t0 = time.time()
    for i, (member, vid_id, label, method) in enumerate(items):
        vdir = out_root / vid_id
        feat_npy = vdir / "features.npy"
        if feat_npy.exists():          # resume: skip before unzipping
            n_skip += 1
            continue

        tmp = work / (vid_id + Path(member).suffix.lower())
        try:
            tmp.write_bytes(zf.read(member))             # extract ONE video
            fex = detector.detect_video(
                str(tmp), skip_frames=args.skip_frames, batch_size=args.batch_size
            )
            mat, au_mat, cols = per_frame_matrix(fex, groups)
            seq = resample_time(mat, args.seq_len)
            au_seq = resample_time(au_mat, args.seq_len)
            if seq is None or au_seq is None:
                raise RuntimeError("no detected faces in any frame")

            vdir.mkdir(parents=True, exist_ok=True)
            np.save(feat_npy, seq.astype(np.float32))
            np.save(vdir / "au_sequence.npy", au_seq.astype(np.float32))
            (vdir / "meta.json").write_text(json.dumps({
                "video_id": vid_id, "label": int(label), "source": args.dataset,
                "dataset": args.dataset, "method": method,
                "n_detected_frames": int(mat.shape[0]), "target_frames": args.seq_len,
                "feat_shape": list(seq.shape), "feat_columns": cols,
                "zip_member": member, "zip_path": str(args.zip_path),
            }, indent=2))
            new_rows.append({
                "video_id": vid_id, "label": int(label), "source": args.dataset,
                "dataset": args.dataset, "method": method,
                "n_detected_frames": int(mat.shape[0]), "target_frames": args.seq_len,
                "feat_shape": str(list(seq.shape)), "video_path": member,
            })
            n_ok += 1
        except Exception as e:
            n_err += 1
            with open(err_log, "a") as f:
                f.write(json.dumps({"video_id": vid_id, "member": member,
                                    "error": str(e),
                                    "trace": traceback.format_exc()[-500:]}) + "\n")
        finally:
            if tmp.exists():
                tmp.unlink()                              # free disk immediately

        if (i + 1) % 20 == 0 or i + 1 == len(items):
            # flush csv rows periodically so progress survives a crash
            if new_rows:
                exists = meta_csv.exists()
                with open(meta_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    if not exists:
                        w.writeheader()
                    w.writerows(new_rows)
                new_rows = []
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[chunk] {i+1}/{len(items)} ok={n_ok} skip={n_skip} "
                  f"err={n_err} ({rate:.2f} vid/s)", flush=True)

    print(f"[chunk] DONE {args.dataset} ok={n_ok} skip={n_skip} err={n_err} -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
