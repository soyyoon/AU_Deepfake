#!/usr/bin/env python
"""Rebuild data/dataset_metadata_v2.csv from every data/<id>/meta.json, and merge
the per-shard .err_shard_*.jsonl into errors_v2.jsonl.

meta.json (written atomically once per video) is the source of truth, so this is
safe to run anytime — during, after, or between parallel streaming runs.

    python scripts/finalize_metadata.py
"""
import csv
import json
import sys
from pathlib import Path

FIELDS = ["video_id", "label", "source", "dataset", "method", "skip_frames",
          "n_detected_frames", "target_frames", "feat_shape", "video_path"]


def main():
    out = Path("data")
    rows = []
    for mj in sorted(out.glob("*/meta.json")):
        try:
            m = json.loads(mj.read_text())
        except Exception:                       # noqa: BLE001
            continue
        if not (mj.parent / "features.npy").exists():
            continue
        rows.append({
            "video_id": m.get("video_id", mj.parent.name),
            "label": m.get("label", -1), "source": m.get("source", ""),
            "dataset": m.get("dataset", ""), "method": m.get("method", ""),
            "skip_frames": m.get("skip_frames", ""),
            "n_detected_frames": m.get("n_detected_frames", ""),
            "target_frames": m.get("target_frames", ""),
            "feat_shape": str(m.get("feat_shape", "")),
            "video_path": m.get("kaggle_member", m.get("video_path", "")),
        })
    csv_path = out / "dataset_metadata_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    # merge per-shard error logs
    err_out, n_err = out / "errors_v2.jsonl", 0
    shard_errs = sorted(out.glob(".err_shard_*.jsonl"))
    if shard_errs:
        with open(err_out, "w") as fo:
            for se in shard_errs:
                for line in se.read_text().splitlines():
                    if line.strip():
                        fo.write(line + "\n"); n_err += 1

    import collections
    by = collections.Counter((r["dataset"], r["label"]) for r in rows)
    print(f"[finalize] {len(rows)} videos -> {csv_path}  ({n_err} errors -> {err_out})")
    for k in sorted(by):
        print(f"           dataset={k[0]:<8} label={k[1]} n={by[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
