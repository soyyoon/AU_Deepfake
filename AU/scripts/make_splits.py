#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from au_detect.data.splits import make_splits
from au_detect.utils import load_config, set_seed, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dir(cfg["paths"]["output_dir"])
    out_csv = cfg["paths"]["splits_csv"]
    make_splits(cfg["paths"]["metadata_csv"], out_csv, seed=cfg["seed"])
    print(f"[make_splits] wrote {out_csv}")


if __name__ == "__main__":
    main()
