"""Classic ML baseline on statistical AU features.

Tries XGBoost first (plan preference). If XGBoost/sklearn are unavailable — in
this environment scipy is ABI-broken, which both import — it falls back to a
self-contained, class-weighted logistic regression trained in torch. Same split,
same metrics as the DL model for a fair comparison.
"""
import json

import numpy as np

from .data.dataset import read_splits
from .features import build_feature_matrix
from .metrics import full_report, select_threshold
from .utils import ensure_dir


def _standardize(Xtr, *others):
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-6] = 1e-6
    return [(X - mean) / std for X in (Xtr, *others)]


def _try_xgboost(Xtr, ytr, Xva, Xte):
    try:
        import xgboost as xgb  # noqa
    except Exception as e:
        return None, f"xgboost unavailable ({type(e).__name__})"
    n_pos = (ytr == 1).sum(); n_neg = (ytr == 0).sum()
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, eval_metric="auc",
        scale_pos_weight=n_neg / max(n_pos, 1), n_jobs=4,
    )
    clf.fit(Xtr, ytr)
    return (clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]), "xgboost"


def _torch_logreg(Xtr, ytr, Xva, Xte, epochs=300, lr=0.05, wd=1e-4):
    import torch
    Xtr, Xva, Xte = _standardize(Xtr, Xva, Xte)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    w = torch.zeros(Xtr.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    n_pos = (ytr == 1).sum(); n_neg = (ytr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=dev)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        logit = xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, yt, pos_weight=pos_weight)
        loss.backward(); opt.step()
    with torch.no_grad():
        pva = torch.sigmoid(torch.tensor(Xva, dtype=torch.float32, device=dev) @ w + b).cpu().numpy()
        pte = torch.sigmoid(torch.tensor(Xte, dtype=torch.float32, device=dev) @ w + b).cpu().numpy()
    return (pva, pte), "torch_logreg"


def run_baseline(cfg):
    rows = read_splits(cfg["paths"]["splits_csv"])
    dd, nc = cfg["paths"]["data_dir"], cfg["data"]["n_channels"]
    ff = cfg["data"].get("feature_file", "au_sequence.npy")
    Xtr, ytr, _ = build_feature_matrix(rows, dd, "train", nc, ff)
    Xva, yva, _ = build_feature_matrix(rows, dd, "val", nc, ff)
    Xte, yte, ste = build_feature_matrix(rows, dd, "test", nc, ff)
    print(f"[baseline] features train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    out, kind = _try_xgboost(Xtr, ytr, Xva, Xte)
    if out is None:
        print(f"[baseline] {kind}; falling back to torch logistic regression")
        out, kind = _torch_logreg(Xtr, ytr, Xva, Xte)
    pva, pte = out
    print(f"[baseline] model = {kind}")

    thr, _ = select_threshold(yva, pva, cfg["train"]["threshold_metric"])
    report = {"model": kind, "overall": full_report(yte, pte, thr), "by_source": {}}
    for s in sorted(set(ste.tolist())):
        m = ste == s
        report["by_source"][s] = full_report(yte[m], pte[m], thr)

    ensure_dir(cfg["paths"]["output_dir"])
    path = cfg["paths"]["metrics_json"].replace(".json", "_baseline.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    o = report["overall"]
    print(f"[baseline] TEST roc_auc={o['roc_auc']:.4f} pr_auc={o['pr_auc']:.4f} "
          f"f1={o['f1']:.4f} bal_acc={o['balanced_acc']:.4f}")
    for s, r in report["by_source"].items():
        print(f"           {s:8s} roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f} n={r['n']}")
    print(f"[baseline] wrote {path}")
    return report
