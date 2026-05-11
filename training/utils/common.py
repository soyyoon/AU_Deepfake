"""
Shared training helpers: metrics, checkpoint, config loading.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalMetrics:
    auc: float
    ap: float
    f1: float
    acc: float

    def to_dict(self) -> dict:
        return self.__dict__

    def get(self, name: str) -> float:
        return getattr(self, name)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> EvalMetrics:
    preds = (probs >= threshold).astype(int)
    # AUC/AP need both classes present
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(labels, probs)
    except ValueError:
        ap = float("nan")
    return EvalMetrics(
        auc=auc,
        ap=ap,
        f1=f1_score(labels, preds, zero_division=0),
        acc=accuracy_score(labels, preds),
    )


class CheckpointSaver:
    """Saves best model by configurable metric."""

    def __init__(self, ckpt_dir: Path | str, best_metric: str = "auc", mode: str = "max"):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = best_metric
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best
        return value < self.best

    def save_if_best(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: EvalMetrics,
        extra: dict | None = None,
    ) -> bool:
        v = metrics.get(self.metric_name)
        if not self.is_better(v):
            return False
        self.best = v
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics.to_dict(),
            **(extra or {}),
        }
        torch.save(ckpt, self.dir / "best.pt")
        with open(self.dir / "best_metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info("New best: %s=%.4f at epoch %d", self.metric_name, v, epoch)
        return True


def compute_pos_weight(labels: np.ndarray) -> float:
    """For BCEWithLogitsLoss when classes are imbalanced."""
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg / pos)
