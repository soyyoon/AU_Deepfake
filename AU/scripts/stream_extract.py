#!/usr/bin/env python
"""Phase 2: stream a Kaggle dataset's videos one at a time through Py-Feat.

For each row in the phase-1 filelist: download ONE video (stdlib urllib, fully
URL-encoded path so Kaggle's per-file endpoint accepts the `+`/`/` in the name)
-> Py-Feat detect -> save tiny .npy (30ch AUs+pose+emotions, +20ch AU-only)
-> delete the video. Peak extra disk = one video (~1 MB). No big zip is ever
materialised, which is the whole point for the disk-constrained setup.

Runs with PYTHONNOUSERSITE=1 (Py-Feat) and needs NO kaggle lib — only the
username/key in ~/.kaggle/kaggle.json. Resumable: a video whose features.npy
already exists is skipped before download.

    PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
    PYTHONNOUSERSITE=1 $PY scripts/stream_extract.py \
        --slug xdxd003/ff-c23 --tag FFpp --filelist data/_filelist_FFpp.csv
"""
import argparse
import base64
import csv
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import per_frame_matrix, resample_time  # noqa: E402

KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download"


def load_creds():
    p = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
    d = json.loads(p.read_text())
    return d["username"], d["key"]


def download_one(slug, name, dest, auth, retries=3):
    """Download a single dataset file to `dest`. Handles the rare zip-wrapped case."""
    enc = urllib.parse.quote(name, safe="")          # encode EVERYTHING incl. '/' and '+'
    url = f"{KAGGLE_API}/{slug}/{enc}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", "Basic " + auth)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if data[:4] == b"PK\x03\x04":            # zip-wrapped single file
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    inner = zf.namelist()[0]
                    data = zf.read(inner)
            if data[:1] == b"<" or len(data) < 1024:  # HTML error / truncated
                raise RuntimeError(f"bad response ({len(data)}B)")
            Path(dest).write_bytes(data)
            return
        except Exception as e:                        # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} tries: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--work", default=os.environ.get("WORKDIR", "/tmp/cev_work"))
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--features", default="aus,poses,emotions")
    ap.add_argument("--skip-frames", type=int, default=0,
                    help="0 = auto: stride chosen per-video to yield ~seq_len frames "
                         "(Py-Feat AU stage is CPU-bound, so fewer frames = big speedup)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker handles filelist rows where idx %% num_shards == shard")
    args = ap.parse_args()

    groups = [g.strip() for g in args.features.split(",") if g.strip()]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) / f"shard{args.shard}"   # per-shard scratch (no temp-name clashes)
    work.mkdir(parents=True, exist_ok=True)
    # per-shard error log -> merged by finalize_metadata.py (avoids concurrent-append corruption)
    err_log = out_root / f".err_shard_{args.shard}.jsonl"

    with open(args.filelist) as f:
        items = list(csv.DictReader(f))
    if args.limit:
        items = items[: args.limit]
    # shard: this worker takes every num_shards-th row
    if args.num_shards > 1:
        items = [it for idx, it in enumerate(items) if idx % args.num_shards == args.shard]
    if not items:
        print("[stream] empty filelist/shard", file=sys.stderr); return 0

    import cv2  # frame-count probe for auto skip_frames

    def auto_skip(path):
        if args.skip_frames > 0:
            return args.skip_frames
        try:
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            return max(1, round(n / args.seq_len)) if n > 0 else 1
        except Exception:       # noqa: BLE001
            return 1

    u, k = load_creds()
    auth = base64.b64encode(f"{u}:{k}".encode()).decode()

    import torch
    torch.set_num_threads(1)            # 1 CPU thread/worker -> no oversubscription
    from feat import Detector
    detector = Detector(device=args.device)
    tag = f"{args.tag}#{args.shard}/{args.num_shards}"
    print(f"[stream {tag}] {len(items)} videos | Detector on {args.device}", flush=True)

    n_ok, n_skip, n_err = 0, 0, 0
    t0 = time.time()
    for i, row in enumerate(items):
        vid_id, name = row["video_id"], row["name"]
        label, method = int(row["label"]), row["method"]
        vdir = out_root / vid_id
        if (vdir / "features.npy").exists():          # resume
            n_skip += 1
            continue
        tmp = work / (vid_id + Path(name).suffix.lower())
        try:
            download_one(args.slug, name, tmp, auth)
            sk = auto_skip(str(tmp))
            fex = detector.detect_video(str(tmp), skip_frames=sk,
                                        batch_size=args.batch_size)
            mat, au_mat, cols = per_frame_matrix(fex, groups)
            seq = resample_time(mat, args.seq_len)
            au_seq = resample_time(au_mat, args.seq_len)
            if seq is None or au_seq is None:
                raise RuntimeError("no detected faces in any frame")
            vdir.mkdir(parents=True, exist_ok=True)
            np.save(vdir / "features.npy", seq.astype(np.float32))
            np.save(vdir / "au_sequence.npy", au_seq.astype(np.float32))
            # meta.json is the per-video source of truth; finalize_metadata.py
            # rebuilds dataset_metadata_v2.csv from these (no concurrent CSV writes).
            (vdir / "meta.json").write_text(json.dumps({
                "video_id": vid_id, "label": label, "source": args.tag,
                "dataset": args.tag, "method": method, "skip_frames": sk,
                "n_detected_frames": int(mat.shape[0]), "target_frames": args.seq_len,
                "feat_shape": list(seq.shape), "feat_columns": cols,
                "kaggle_slug": args.slug, "kaggle_member": name,
            }, indent=2))
            n_ok += 1
        except Exception as e:                        # noqa: BLE001
            n_err += 1
            with open(err_log, "a") as f:
                f.write(json.dumps({"video_id": vid_id, "member": name,
                                    "error": str(e),
                                    "trace": traceback.format_exc()[-500:]}) + "\n")
        finally:
            if tmp.exists():
                tmp.unlink()
            try:
                torch.cuda.empty_cache()   # avoid slow GPU-memory creep over a long run
            except Exception:              # noqa: BLE001
                pass

        if (i + 1) % 20 == 0 or i + 1 == len(items):
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"[stream {tag}] {i+1}/{len(items)} ok={n_ok} skip={n_skip} "
                  f"err={n_err} ({rate:.2f} vid/s)", flush=True)

    print(f"[stream {tag}] DONE ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
