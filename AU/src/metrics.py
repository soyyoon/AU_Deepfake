"""Pure-numpy classification metrics (no sklearn/scipy dependency).

ROC-AUC, average precision (PR-AUC), F1, balanced accuracy, EER, confusion
matrix, and val-driven threshold selection.
"""
import numpy as np


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-based (Mann-Whitney U) with tie handling via average ranks
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true, y_score):
    """Area under precision-recall curve (AP), step interpolation."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if (y_true == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt == 1)
    fp = np.cumsum(yt == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / (yt == 1).sum()
    # AP = sum over thresholds of (R_n - R_{n-1}) * P_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def f1_score(y_true, y_pred):
    c = confusion(y_true, y_pred)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    rec = c["tp"] / max(c["tp"] + c["fn"], 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def balanced_accuracy(y_true, y_pred):
    c = confusion(y_true, y_pred)
    tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
    tnr = c["tn"] / max(c["tn"] + c["fp"], 1)
    return 0.5 * (tpr + tnr)


def eer(y_true, y_score):
    """Equal error rate via threshold sweep over unique scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan")
    thr = np.unique(y_score)
    best_gap = float("inf")
    best_eer = 0.5
    for t in thr:
        pred = (y_score >= t).astype(int)
        c = confusion(y_true, pred)
        far = c["fp"] / max(c["fp"] + c["tn"], 1)   # positive class = fake (label 1)
        frr = c["fn"] / max(c["fn"] + c["tp"], 1)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (far + frr) / 2.0
    return float(best_eer)


def select_threshold(y_true, y_score, metric="f1"):
    """Pick threshold maximizing the chosen metric on (val) data."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    cand = np.unique(y_score)
    if len(cand) > 512:
        cand = np.quantile(cand, np.linspace(0, 1, 512))
    fn = f1_score if metric == "f1" else balanced_accuracy
    best_t, best_v = 0.5, -1.0
    for t in cand:
        pred = (y_score >= t).astype(int)
        v = fn(y_true, pred)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t, best_v


def full_report(y_true, y_score, threshold):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": average_precision(y_true, y_score),
        "f1": f1_score(y_true, pred),
        "balanced_acc": balanced_accuracy(y_true, pred),
        "eer": eer(y_true, y_score),
        "threshold": float(threshold),
        "confusion": confusion(y_true, pred),
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }
