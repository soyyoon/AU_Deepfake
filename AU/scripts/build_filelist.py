#!/usr/bin/env python
"""Phase 1 of the per-file streaming pipeline: list a Kaggle dataset's videos and
label them by path, writing a small CSV that phase 2 (stream_extract.py) consumes.

Run this WITHOUT PYTHONNOUSERSITE (it needs the user-site `kaggle` lib):
    python scripts/build_filelist.py --slug xdxd003/ff-c23 --tag FFpp

Why split listing from extraction: the kaggle lib lives in user-site, but Py-Feat
must run with PYTHONNOUSERSITE=1. So listing (kaggle lib) and extraction (urllib,
stdlib-only download) are separate processes with opposite env needs.

Output: data/_filelist_<tag>.csv  columns: name,video_id,label,method
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunked_extract_zip import classify, safe_id, VIDEO_EXTS  # noqa: E402


def list_all_files(slug):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    names, tok = [], None
    while True:
        res = api.dataset_list_files(slug, page_token=tok, page_size=200)
        files = getattr(res, "files", None) or []
        names.extend(getattr(f, "name", "") for f in files)
        tok = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not tok or not files:
            break
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="e.g. xdxd003/ff-c23")
    ap.add_argument("--tag", required=True, help="dataset tag, e.g. FFpp / CelebDF")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    names = list_all_files(args.slug)
    rows, lm = [], Counter()
    n_unlabel = 0
    for name in names:
        if name.endswith("/") or Path(name).suffix.lower() not in VIDEO_EXTS:
            continue
        label, method = classify(name)
        if label is None:
            n_unlabel += 1
            continue
        vid = safe_id(args.tag, method, Path(name).stem)
        rows.append({"name": name, "video_id": vid, "label": label, "method": method})
        lm[(label, method)] += 1

    out = Path(args.out) / f"_filelist_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "video_id", "label", "method"])
        w.writeheader(); w.writerows(rows)

    print(f"[filelist] {args.slug}: {len(rows)} labelable videos -> {out}")
    for (lab, m), n in sorted(lm.items()):
        print(f"           label={lab} method={m:<18} n={n}")
    if n_unlabel:
        print(f"           [warn] {n_unlabel} video(s) had no label keyword -> skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
