#!/usr/bin/env python
"""Re-preprocessing: raw video -> richer per-frame facial feature sequence.

Replaces the old OpenFace `[64, 17]` AU-intensity-only representation (the data
bottleneck) with a richer Py-Feat representation per video:

    AUs (20)  +  head-pose (3: pitch/roll/yaw)  +  emotions (7)  = 30 channels

Output per video (mirrors the existing layout under data/<video_id>/):
    data/<video_id>/features.npy     [seq_len, C] float32   (the rich features)
    data/<video_id>/au_sequence.npy  [seq_len, 20] float32  (AU-only, back-compat)
    data/<video_id>/meta.json        label / source / columns / shapes

Designed for the storage-constrained chunked workflow: process a chunk of raw
videos, write tiny .npy features, then delete the raw chunk. Re-runnable: a video
whose features.npy already exists is skipped, so an interrupted run resumes.

Labels: if the input dir (or --labels) has a DFDC-style metadata.json
({"<file>.mp4": {"label": "FAKE"|"REAL", "split": ..., "original": ...}}), labels
are read from it (FAKE->1, REAL->0). Otherwise pass --label to force one, or labels
stay -1 (unknown, e.g. for an inference-only eval set).

Usage:
    PY=/home/soyoon/anaconda3/envs/au/bin/python
    $PY scripts/extract_features.py --in /tmp/dfdc/part_00 --source DFDC
    $PY scripts/extract_features.py --in /tmp/dfdc/train_sample --source DFDC --limit 5  # smoke
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Feature groups -> Py-Feat Fex accessor. Order here defines channel order.
FEATURE_GROUPS = ["aus", "poses", "emotions"]


def find_videos(root):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def load_dfdc_labels(in_dir, labels_arg):
    """Return {filename(str): (label:int, original:str|None)} or {}."""
    path = None
    if labels_arg:
        path = Path(labels_arg)
    else:
        cand = Path(in_dir) / "metadata.json"
        if cand.exists():
            path = cand
    if not path or not path.exists():
        return {}
    with open(path) as f:
        meta = json.load(f)
    out = {}
    for fname, info in meta.items():
        lab = info.get("label", "")
        y = 1 if str(lab).upper() == "FAKE" else 0 if str(lab).upper() == "REAL" else -1
        out[fname] = (y, info.get("original"))
    return out


def per_frame_matrix(fex, groups):
    """Fex (one+ rows per frame) -> (mat [n_frames, C], au_mat [n_frames, 20], colnames).

    Keeps the largest detected face per frame; orders by frame index; fills NaN.
    """
    import pandas as pd

    df = fex.copy()
    # py-feat 0.6.2 returns 'frame' as BOTH an index level and a column, which makes
    # groupby('frame') ambiguous -> de-conflict by flattening the index first.
    if any(n == "frame" for n in (list(df.index.names) if df.index.names else [])):
        df = df.reset_index(drop=("frame" in df.columns))
    df = pd.DataFrame(df)                 # drop Fex subclass quirks for plain ops
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))

    # largest face per frame (by facebox area) when multiple faces present
    if {"FaceRectWidth", "FaceRectHeight"}.issubset(df.columns):
        df["_area"] = df["FaceRectWidth"].fillna(0) * df["FaceRectHeight"].fillna(0)
        df = df.sort_values("_area").groupby("frame", as_index=False).last()
    else:
        df = df.groupby("frame", as_index=False).last()
    df = df.sort_values("frame")

    # Resolve columns per group via py-feat 0.6.x Fex sub-frame accessors.
    # (Older `fex.au_columns`/`emotion_columns`/`facepose_columns` no longer exist;
    # 0.6.2 exposes `.aus` / `.poses` / `.emotions` DataFrame properties instead.)
    sub = {
        "aus": list(fex.aus.columns),
        "poses": list(fex.poses.columns),
        "emotions": list(fex.emotions.columns),
    }
    cols = []
    for g in groups:
        cols.extend([c for c in sub[g] if c in df.columns])
    au_cols = [c for c in sub["aus"] if c in df.columns]

    mat = df[cols].to_numpy(dtype=np.float32)
    au_mat = df[au_cols].to_numpy(dtype=np.float32)
    # forward/backward fill then zero remaining NaN (undetected frames)
    mat = _fill_nan(mat)
    au_mat = _fill_nan(au_mat)
    return mat, au_mat, cols


def _fill_nan(a):
    a = a.copy()
    n, c = a.shape
    for j in range(c):
        col = a[:, j]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
        elif mask.any():
            idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], idx, col[idx])
        a[:, j] = col
    return a


def resample_time(mat, T):
    """Uniformly sample/pad rows to exactly T (handles up- and down-sampling)."""
    n = mat.shape[0]
    if n == 0:
        return None
    if n == T:
        return mat
    idx = np.clip(np.linspace(0, n - 1, T).round().astype(int), 0, n - 1)
    return mat[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of raw videos")
    ap.add_argument("--out", dest="out_dir", default="data", help="output data dir")
    ap.add_argument("--source", default="DFDC", help="source tag stored in meta")
    ap.add_argument("--labels", default=None, help="DFDC metadata.json (auto-detected if in --in)")
    ap.add_argument("--label", type=int, default=None, help="force a fixed label for all videos")
    ap.add_argument("--id-prefix", default="", help="prefix for video_id")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--features", default=",".join(FEATURE_GROUPS),
                    help="comma list from: aus,poses,emotions")
    ap.add_argument("--skip-frames", type=int, default=2,
                    help="run detector every k-th frame (speed); resampled to seq-len after")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (0=all)")
    args = ap.parse_args()

    groups = [g.strip() for g in args.features.split(",") if g.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_csv = out_root / "dataset_metadata_v2.csv"
    err_log = out_root / "errors_v2.jsonl"

    videos = find_videos(args.in_dir)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"[extract] no videos under {args.in_dir}", file=sys.stderr)
        return 1

    labels = load_dfdc_labels(args.in_dir, args.labels)
    print(f"[extract] {len(videos)} videos | labels for {len(labels)} | groups={groups}")

    # Heavy init once (downloads model weights on first ever run)
    from feat import Detector
    detector = Detector(device=args.device)
    print(f"[extract] Py-Feat Detector ready on {args.device}")

    new_rows, n_ok, n_skip, n_err = [], 0, 0, 0
    t0 = time.time()
    for i, vid in enumerate(videos):
        video_id = f"{args.id_prefix}{vid.stem}"
        vdir = out_root / video_id
        feat_npy = vdir / "features.npy"
        if feat_npy.exists():
            n_skip += 1
            continue
        if args.label is not None:
            label, original = args.label, None
        else:
            label, original = labels.get(vid.name, (-1, None))
        try:
            fex = detector.detect_video(
                str(vid), skip_frames=args.skip_frames, batch_size=args.batch_size
            )
            mat, au_mat, cols = per_frame_matrix(fex, groups)
            seq = resample_time(mat, args.seq_len)
            au_seq = resample_time(au_mat, args.seq_len)
            if seq is None or au_seq is None:
                raise RuntimeError("no detected faces in any frame")

            vdir.mkdir(parents=True, exist_ok=True)
            np.save(feat_npy, seq.astype(np.float32))
            np.save(vdir / "au_sequence.npy", au_seq.astype(np.float32))
            meta = {
                "video_id": video_id, "label": int(label), "source": args.source,
                "original": original, "n_detected_frames": int(mat.shape[0]),
                "target_frames": args.seq_len, "feat_shape": list(seq.shape),
                "feat_columns": cols, "video_path": str(vid),
            }
            (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
            new_rows.append({
                "video_id": video_id, "label": int(label), "source": args.source,
                "n_detected_frames": int(mat.shape[0]), "target_frames": args.seq_len,
                "feat_shape": str(list(seq.shape)), "video_path": str(vid),
                "original": original or "",
            })
            n_ok += 1
        except Exception as e:
            n_err += 1
            with open(err_log, "a") as f:
                f.write(json.dumps({"video_id": video_id, "error": str(e),
                                    "trace": traceback.format_exc()[-500:]}) + "\n")
        if (i + 1) % 20 == 0 or i + 1 == len(videos):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[extract] {i+1}/{len(videos)} ok={n_ok} skip={n_skip} "
                  f"err={n_err} ({rate:.2f} vid/s)", flush=True)

    # append rows to the v2 metadata csv
    if new_rows:
        fields = ["video_id", "label", "source", "n_detected_frames",
                  "target_frames", "feat_shape", "video_path", "original"]
        exists = meta_csv.exists()
        with open(meta_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    print(f"[extract] DONE ok={n_ok} skip={n_skip} err={n_err} -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
