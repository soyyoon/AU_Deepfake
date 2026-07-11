#!/usr/bin/env python
"""Write a class/method-balanced subset filelist from a full one.

Policy per dataset: keep ALL real videos; give fakes a budget == #reals, split
evenly across forgery methods and uniformly stride-sampled (deterministic). This
yields a ~balanced real/fake set covering every method, small enough to validate
the richer-feature hypothesis in a few hours before committing to the full run.

    python scripts/make_subset.py --tag FFpp      # data/_filelist_FFpp.csv -> _sub.csv
    python scripts/make_subset.py --tag CelebDF

Reads  data/_filelist_<tag>.csv   (full, from build_filelist.py)
Writes data/_filelist_<tag>_sub.csv
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def stride_sample(rows, k):
    """Pick k items spread uniformly across rows (deterministic, order-preserving)."""
    n = len(rows)
    if k >= n:
        return rows
    step = n / k
    idx = sorted({min(n - 1, int(i * step)) for i in range(k)})
    return [rows[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--fake-ratio", type=float, default=1.0,
                    help="fake budget = ratio * (#reals); 1.0 = balanced")
    args = ap.parse_args()

    src = Path(args.out) / f"_filelist_{args.tag}.csv"
    with open(src) as f:
        rows = list(csv.DictReader(f))
    reals = [r for r in rows if r["label"] == "0"]
    fakes_by_m = defaultdict(list)
    for r in rows:
        if r["label"] == "1":
            fakes_by_m[r["method"]].append(r)

    fake_budget = int(round(len(reals) * args.fake_ratio))
    methods = sorted(fakes_by_m)
    per_method = max(1, math.ceil(fake_budget / max(1, len(methods))))
    picked_fakes = []
    for m in methods:
        picked_fakes.extend(stride_sample(fakes_by_m[m], per_method))

    out_rows = reals + picked_fakes
    dst = Path(args.out) / f"_filelist_{args.tag}_sub.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "video_id", "label", "method"])
        w.writeheader(); w.writerows(out_rows)

    print(f"[subset] {args.tag}: reals={len(reals)} fakes={len(picked_fakes)} "
          f"(~{per_method}/method x {len(methods)}) total={len(out_rows)} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
