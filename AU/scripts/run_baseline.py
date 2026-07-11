#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from au_detect.baseline import run_baseline
from au_detect.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_baseline(cfg)


if __name__ == "__main__":
    main()
