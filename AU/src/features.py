"""Fixed-length statistical features per AU channel for the ML baseline.

Per channel: mean, std, min, max, range, plus 1st-difference (velocity) mean & std.
-> 7 stats * 17 channels = 119-dim vector.
"""
import numpy as np


def stat_features(seq):
    """seq: [T, C] -> [7*C] feature vector."""
    seq = np.asarray(seq, dtype=np.float64)
    diff = np.diff(seq, axis=0)
    feats = [
        seq.mean(0), seq.std(0), seq.min(0), seq.max(0),
        seq.max(0) - seq.min(0),
        diff.mean(0), diff.std(0),
    ]
    return np.concatenate(feats).astype(np.float32)


def build_feature_matrix(rows, data_dir, split, n_channels=17,
                         feature_file="au_sequence.npy"):
    import os
    X, y, src = [], [], []
    for r in rows:
        if r["split"] != split:
            continue
        a = np.load(os.path.join(data_dir, r["video_id"], feature_file))
        X.append(stat_features(a))
        y.append(int(r["label"]))
        src.append(r["source"])
    return np.stack(X), np.array(y), np.array(src)
