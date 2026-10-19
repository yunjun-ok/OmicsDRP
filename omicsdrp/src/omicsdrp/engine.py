"""Training / evaluation loops, early stopping and optimiser construction.

``evaluate`` always returns the *full* metric dict and can additionally return
per-sample predictions, so the recorder can persist everything.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import regression_metrics


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def build_adamw_optimizer(model, lr, weight_decay):
    no_decay_kw = ("bias", "bn", "batchnorm", "layernorm", "ln", "norm")
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in n.lower() for k in no_decay_kw) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def _move_batch(gene_features, drug_idx, labels, device):
    # gene_features is a single [B, n_gene, n_omics] tensor (non_blocking for pinned)
    return (gene_features.to(device, non_blocking=True),
            drug_idx.to(device, non_blocking=True),
            labels.to(device, non_blocking=True))


def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, Dict[str, float]]:
    model.train()
    running = 0.0
    ys, ps = [], []
    for sample_features, drug_idx, labels, _ in loader:
        sample_features, drug_idx, labels = _move_batch(sample_features, drug_idx, labels, device)
        optimizer.zero_grad()
        out = model(sample_features, drug_idx)
        loss = criterion(out.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
        ys.append(labels.detach().cpu().numpy())
        ps.append(out.squeeze(-1).detach().cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return running / len(loader), regression_metrics(y, p)


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect: bool = False):
    model.eval()
    running = 0.0
    ys, ps, sidx, didx = [], [], [], []
    for sample_features, drug_idx, labels, (sample_idx, d_idx) in loader:
        sample_features, drug_idx, labels = _move_batch(sample_features, drug_idx, labels, device)
        out = model(sample_features, drug_idx)
        loss = criterion(out.squeeze(-1), labels)
        running += loss.item()
        ys.append(labels.cpu().numpy())
        ps.append(out.squeeze(-1).cpu().numpy())
        if collect:
            sidx.append(sample_idx.cpu().numpy())
            didx.append(d_idx.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    metrics = regression_metrics(y, p)
    loss = running / len(loader)
    if collect:
        preds = {
            "sample_idx": np.concatenate(sidx),
            "drug_idx": np.concatenate(didx),
            "true": y, "pred": p,
        }
        return loss, metrics, preds
    return loss, metrics


class EarlyStopping:
    """Tracks best inner-val RMSE and keeps the best model state in memory."""

    def __init__(self, patience: int, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.best_epoch = -1
        self.count = 0
        self.early_stop = False
        self.best_state: Optional[dict] = None

    def step(self, val_rmse: float, model, epoch: int) -> None:
        if self.best is None or val_rmse < self.best - self.delta:
            self.best = val_rmse
            self.best_epoch = epoch
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
