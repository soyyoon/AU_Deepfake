#!/usr/bin/env python
"""Quality-check the extracted v2 features before training.

Loads every data/<id>/features.npy listed in the metadata csv and reports:
  - integrity: shape mismatches, NaN/Inf, constant (zero-variance) channels
  - per-channel value ranges (grouped AUs / pose / emotions)
  - real-vs-fake signal: univariate ROC-AUC of each channel's temporal mean
    (and temporal std) — a quick read on which channels carry discriminative
    signal, and whether the 10 non-AU channels (pose/emotion) add anything.
  - per dataset/method counts
Writes a bar plot of per-channel |AUC-0.5| to outputs/qc/channel_auc.png.

    PY=/home/soyoon/anaconda3/envs/pyfeat/bin/python
    PYTHONNOUSERSITE=1 $PY scripts/qc_features.py            # all of dataset_metadata_v2.csv
    PYTHONNOUSERSITE=1 $PY scripts/qc_features.py --balanced # only the _sub.csv subset
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from au_detect.metrics import roc_auc  # noqa: E402


def load_rows(meta_csv, balanced):
    rows = list(csv.DictReader(open(meta_csv)))
    if balanced:
        keep = set()
        for tag in ("FFpp", "CelebDF"):
            fl = Path("data") / f"_filelist_{tag}_sub.csv"
            if fl.exists():
                keep |= {r["video_id"] for r in csv.DictReader(open(fl))}
        rows = [r for r in rows if r["video_id"] in keep]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/dataset_metadata_v2.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--balanced", action="store_true", help="restrict to the _sub.csv subset")
    ap.add_argument("--feature-file", default="features.npy")
    ap.add_argument("--out", default="outputs/qc")
    args = ap.parse_args()

    rows = load_rows(args.meta, args.balanced)
    # channel names from any meta.json
    cols = None
    for r in rows:
        mj = Path(args.data_dir) / r["video_id"] / "meta.json"
        if mj.exists():
            cols = json.loads(mj.read_text()).get("feat_columns")
            if cols:
                break
    print(f"[qc] {len(rows)} videos | feature={args.feature_file} | "
          f"{'BALANCED subset' if args.balanced else 'all'}")
    print(f"[qc] datasets: {dict(Counter(r['dataset'] for r in rows))}")
    print(f"[qc] methods : {dict(Counter(r['method'] for r in rows))}")

    # ---- load all, collect integrity + per-video summaries ----
    X_mean, X_std, labels, bad = [], [], [], []
    n_nan = n_inf = n_badshape = 0
    C = None
    for r in rows:
        p = Path(args.data_dir) / r["video_id"] / args.feature_file
        if not p.exists():
            bad.append((r["video_id"], "missing")); continue
        a = np.load(p)
        if C is None:
            C = a.shape[1]
        if a.shape[1] != C:
            n_badshape += 1; bad.append((r["video_id"], f"shape {a.shape}")); continue
        if np.isnan(a).any(): n_nan += 1
        if np.isinf(a).any(): n_inf += 1
        X_mean.append(a.mean(axis=0))
        X_std.append(a.std(axis=0))
        labels.append(int(r["label"]))
    X_mean = np.array(X_mean); X_std = np.array(X_std); y = np.array(labels)
    if cols is None or len(cols) != C:
        cols = [f"ch{j}" for j in range(C)]

    print(f"\n[integrity] channels={C}  videos_loaded={len(y)}  "
          f"NaN_videos={n_nan} Inf_videos={n_inf} badshape={n_badshape} missing={len(bad)}")
    # constant channels (zero variance across all videos' frame-means)
    const = [cols[j] for j in range(C) if X_mean[:, j].std() < 1e-8]
    print(f"[integrity] constant(zero-var) channels: {const or 'none'}")
    print(f"[integrity] label balance: real(0)={int((y==0).sum())} fake(1)={int((y==1).sum())}")

    # ---- per-channel value ranges ----
    print("\n[ranges] per-channel  (mean-of-frame-means / global min / global max)")
    for j in range(C):
        print(f"   {cols[j]:<10} mean={X_mean[:,j].mean():8.3f}  "
              f"min={X_mean[:,j].min():8.3f}  max={X_mean[:,j].max():8.3f}")

    # ---- univariate real-vs-fake AUC per channel (temporal mean & std) ----
    print("\n[signal] univariate ROC-AUC (real vs fake) per channel — |AUC-0.5| ranks usefulness")
    aucs = []
    for j in range(C):
        a_mean = roc_auc(y, X_mean[:, j])
        a_std = roc_auc(y, X_std[:, j])
        best = max(a_mean, a_std, key=lambda v: abs(v - 0.5))
        aucs.append((cols[j], a_mean, a_std, abs(best - 0.5)))
    for name, am, asd, gap in sorted(aucs, key=lambda t: -t[3]):
        flag = "  <== AU" if name.startswith("AU") else ("  <== pose/emo")
        print(f"   {name:<10} AUC(mean)={am:.3f}  AUC(std)={asd:.3f}  |Δ|={gap:.3f}{flag}")

    au_gap = np.mean([g for n, _, _, g in aucs if n.startswith("AU")])
    extra_gap = np.mean([g for n, _, _, g in aucs if not n.startswith("AU")]) if C > 20 else float("nan")
    print(f"\n[signal] mean |AUC-0.5|: AU channels={au_gap:.3f}"
          + (f"  |  pose+emotion channels={extra_gap:.3f}" if C > 20 else ""))
    print("[signal] (higher = more real/fake separation; compares whether the extra "
          "10 non-AU channels carry signal beyond AUs)")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        order = sorted(range(C), key=lambda j: -aucs[j][3])
        names = [aucs[j][0] for j in order]
        gaps = [aucs[j][3] for j in order]
        colors = ["#4c72b0" if names[i].startswith("AU") else "#dd8452" for i in range(C)]
        plt.figure(figsize=(11, 4))
        plt.bar(range(C), gaps, color=colors)
        plt.xticks(range(C), names, rotation=90, fontsize=7)
        plt.ylabel("|AUC - 0.5|"); plt.title("Per-channel real-vs-fake signal "
                                              "(blue=AU, orange=pose/emotion)")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "channel_auc.png"), dpi=120)
        print(f"\n[qc] wrote {os.path.join(args.out, 'channel_auc.png')}")
    except Exception as e:                       # noqa: BLE001
        print(f"[qc] plot skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
