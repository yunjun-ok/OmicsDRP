"""Deployable inference models via plain 5-fold CV + ensemble.

This is a SEPARATE track from ``nested_cv`` and does not touch its invariants.

Why a second track
------------------
``nested_cv`` exists to produce an *honest performance estimate* (outer-test is
untouched until a single final eval). It deliberately throws its per-fold models
away -- they are only a means to a number.

To later run *inference on brand-new test data* and compare conditions, we need
the actual trained models. The standard pattern is:

  * plain K-fold CV (no inner split): for fold ``k`` train on the other K-1
    folds and use the held-out fold ``k`` **as the early-stopping validation
    set**; keep the model.
  * this yields K models per condition -> average their predictions (ensemble)
    on new data.

The held-out fold is used for early stopping here, so these models' fold scores
are optimistic and must NOT be reported as performance -- that is what the
nested-CV numbers are for. These models exist only to predict on genuinely
external data (never in any fold's train/val), where no such leak exists.

Implementation reuses ``splits.build_folds``: for each outer fold we merge its
``train_pair_idx`` + ``val_pair_idx`` into one training pool and use
``test_pair_idx`` as the early-stopping set. The gene scaler is still fit on the
training pool's cells only (never the early-stopping fold), and it is SAVED
alongside each model -- inference on new cells is impossible without it.

Checkpoint layout (per condition ``tag``)::

    <out_root>/<tag>/
        config.json          # ExperimentConfig used
        fold_k.pt            # {model_state, scaler_mean, scaler_scale,
                             #  omics_indices, genes, best_epoch, val_metrics}
        summary.json         # per-fold best_epoch + early-stopping val metrics
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import (RawData, OmicsDrugDataset, select_omics, stack_gene_data,
                   apply_noise_omics)
from .splits import build_folds
from .models import DRPModel, initialize_weights
from .engine import (set_seed, build_adamw_optimizer, train_epoch, evaluate,
                     EarlyStopping)
from .metrics import regression_metrics


# --------------------------------------------------------------------------- #
# scaling: fit on train cells only, but ALSO return the stats so we can persist
# them for inference (data.scale_gene_data discards them).
# --------------------------------------------------------------------------- #
def _scale_with_stats(gene_data: Dict[str, torch.Tensor],
                      train_sample_indices, genes: List[str]
                      ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """StandardScaler per gene, fit on train cells only (identical maths to
    ``data.scale_gene_data``), returning the scaled dict plus stacked
    ``mean``/``scale`` tensors ``[n_gene, n_omics]`` in ``genes`` order."""
    train_idx = np.asarray(sorted(set(int(i) for i in train_sample_indices)))
    scaled: Dict[str, torch.Tensor] = {}
    means, scales = [], []
    for g in genes:
        mat = gene_data[g]
        mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)
        sc = StandardScaler().fit(mat_np[train_idx])
        scaled[g] = torch.tensor(sc.transform(mat_np), dtype=torch.float32)
        means.append(torch.tensor(sc.mean_, dtype=torch.float32))
        scales.append(torch.tensor(sc.scale_, dtype=torch.float32))
    return scaled, torch.stack(means), torch.stack(scales)


def _apply_saved_scaler(gene_data: Dict[str, torch.Tensor], genes: List[str],
                        omics_indices: List[int], mean: torch.Tensor,
                        scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Apply a persisted (mean, scale) to NEW gene data, selecting the same omics
    columns the model was trained on. ``mean``/``scale`` are ``[n_gene, n_omics]``
    in ``genes`` order."""
    # scale on CPU (one-off transform); the DataLoader moves tensors to GPU later.
    mean = mean.cpu(); scale = scale.cpu()
    idx = torch.as_tensor(omics_indices, dtype=torch.long)
    out: Dict[str, torch.Tensor] = {}
    for gi, g in enumerate(genes):
        mat = gene_data[g]
        mat = mat if isinstance(mat, torch.Tensor) else torch.as_tensor(mat)
        mat = mat.detach().cpu().index_select(1, idx).to(torch.float32)
        out[g] = (mat - mean[gi]) / scale[gi]
    return out


def _make_loader(gene_tensor, ic50, pair_idx, pairs, batch_size, shuffle,
                 drop_last=False, num_workers=4):
    ds = OmicsDrugDataset(gene_tensor, ic50, pairs[pair_idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _fold_pools(raw: RawData, config: ExperimentConfig):
    cell_of = raw.pairs[:, 0]
    for f in build_folds(raw, config):
        train_idx = np.concatenate([f.train_pair_idx, f.val_pair_idx])
        esval_idx = f.test_pair_idx
        train_sample_indices = np.unique(cell_of[train_idx])
        yield f.fold, train_idx, esval_idx, train_sample_indices


def train_one_fold(raw: RawData, config: ExperimentConfig, fold: int,
                   train_idx, esval_idx, train_sample_indices, device: str):
    set_seed(config.seed + fold)
    omics_gene = select_omics(raw.gene_data, config.omics_indices())
    scaled, mean_t, scale_t = _scale_with_stats(omics_gene, train_sample_indices, raw.genes)
    gene_tensor = stack_gene_data(scaled, raw.genes)
    gene_tensor = apply_noise_omics(gene_tensor, config.omics_indices(),
                                    config.noise_indices(), config.seed + fold)

    nw = config.num_workers
    train_loader = _make_loader(gene_tensor, raw.ic50, train_idx, raw.pairs,
                                config.batch_size, shuffle=True, drop_last=True,
                                num_workers=nw)
    val_loader = _make_loader(gene_tensor, raw.ic50, esval_idx, raw.pairs,
                              config.batch_size, shuffle=False, num_workers=nw)

    model = DRPModel(raw.genes, raw.drug_meta, config).to(device)
    model.apply(initialize_weights)
    criterion = nn.MSELoss()
    optimizer = build_adamw_optimizer(model, config.lr, config.weight_decay)
    stopper = EarlyStopping(patience=config.patience)

    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_m = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_m = evaluate(model, val_loader, criterion, device)
        print(f"[infer fold {fold}] epoch {epoch:3d} "
              f"train_rmse={tr_m['rmse']:.4f} esval_rmse={va_m['rmse']:.4f}")
        stopper.step(va_m["rmse"], model, epoch)
        if stopper.early_stop:
            print(f"[infer fold {fold}] early stop @ epoch {epoch} "
                  f"(best epoch {stopper.best_epoch})")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    best_epoch = stopper.best_epoch
    # early-stopping-fold metrics (record only; NOT a performance claim)
    _, val_metrics = evaluate(model, val_loader, criterion, device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    del model, train_loader, val_loader, stopper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"model_state": best_state, "scaler_mean": mean_t, "scaler_scale": scale_t,
            "noise_seed": config.seed + fold,
            "best_epoch": best_epoch, "val_metrics": val_metrics}


def _completed_folds(cond_dir: str, k: int) -> bool:
    return all(os.path.exists(os.path.join(cond_dir, f"fold_{i}.pt"))
               for i in range(1, k + 1))


def train_inference_models(raw: RawData, config: ExperimentConfig,
                           out_root: str, device: str) -> str:
    """Train K ensemble models for one condition; save to <out_root>/<tag>/.

    Idempotent: a condition whose K fold checkpoints already exist is skipped."""
    tag = config.tag()
    cond_dir = os.path.join(out_root, tag)
    os.makedirs(cond_dir, exist_ok=True)
    with open(os.path.join(cond_dir, "config.json"), "w") as fh:
        json.dump(config.to_dict(), fh, indent=2)

    if _completed_folds(cond_dir, config.outer_folds):
        print(f"[skip] {tag}: all {config.outer_folds} fold models present")
        return cond_dir

    for fold, tr, es, tr_cells in _fold_pools(raw, config):
        fpath = os.path.join(cond_dir, f"fold_{fold}.pt")
        if os.path.exists(fpath):
            print(f"[skip] {tag} fold {fold}")
            continue
        res = train_one_fold(raw, config, fold, tr, es, tr_cells, device)
        torch.save({
            "model_state": res["model_state"],
            "scaler_mean": res["scaler_mean"],
            "scaler_scale": res["scaler_scale"],
            "omics_indices": config.omics_indices(),
            "noise_indices": config.noise_indices(),
            "noise_seed": res["noise_seed"],
            "genes": list(raw.genes),
            "best_epoch": res["best_epoch"],
            "val_metrics": res["val_metrics"],
        }, fpath)
        # small sidecar so summary.json can be rebuilt without loading weights
        _write_fold_meta(cond_dir, fold, res["best_epoch"], res["val_metrics"])
        print(f"[saved] {fpath}")

    write_summary(cond_dir, tag)
    return cond_dir


def _write_fold_meta(cond_dir: str, fold: int, best_epoch, val_metrics) -> None:
    with open(os.path.join(cond_dir, f"fold_{fold}_meta.json"), "w") as fh:
        json.dump({"fold": fold, "best_epoch": best_epoch,
                   "val_metrics": val_metrics}, fh, indent=2)


def write_summary(cond_dir: str, tag: Optional[str] = None) -> dict:
    """Rebuild summary.json from the per-fold sidecars.
    Reads sidecars (not the .pt weights) so a RESUMED condition still summarises
    every fold, including ones trained in an earlier run."""
    import glob
    folds = []
    for p in sorted(glob.glob(os.path.join(cond_dir, "fold_*_meta.json"))):
        with open(p) as fh:
            folds.append(json.load(fh))
    folds.sort(key=lambda d: d["fold"])
    summary = {"tag": tag or os.path.basename(cond_dir.rstrip("/")), "folds": folds}
    with open(os.path.join(cond_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# ensemble inference
# --------------------------------------------------------------------------- #
class InferenceEnsemble:
    """Load a condition's K fold models and average their predictions.
    Use for prediction on NEW data in the SAME feature space (same 909 genes,
    same drug set / drug_meta the models were trained with). Each fold applies
    its own saved scaler before predicting."""

    def __init__(self, config: ExperimentConfig, folds: List[dict], device: str):
        self.config = config
        self.folds = folds
        self.device = device

    @classmethod
    def load(cls, cond_dir: str, dataset_path: Optional[str] = None,
             device: Optional[str] = None) -> "InferenceEnsemble":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(cond_dir, "config.json")) as fh:
            cfg = ExperimentConfig(**json.load(fh))
        if dataset_path:
            cfg = _with_dataset_path(cfg, dataset_path)
        folds = []
        i = 1
        while os.path.exists(os.path.join(cond_dir, f"fold_{i}.pt")):
            folds.append(torch.load(os.path.join(cond_dir, f"fold_{i}.pt"),
                                    map_location=device))
            i += 1
        if not folds:
            raise FileNotFoundError(f"no fold_*.pt in {cond_dir}")
        return cls(cfg, folds, device)

    @torch.no_grad()
    def predict(self, gene_data: Dict[str, torch.Tensor], drug_meta,
                pairs: np.ndarray, ic50: Optional[torch.Tensor] = None,
                batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Ensemble-predict LN_IC50 for ``pairs`` [(cell_idx, drug_idx), ...].

        Returns {"pred": mean over folds, "per_fold": [K, M], and if ``ic50``
        given, "true" + "metrics"}. ``gene_data`` must use the same gene keys as
        training; ``drug_meta`` must index the same drug set."""
        pairs = np.asarray(pairs)
        genes = self.folds[0]["genes"]
        bs = batch_size or self.config.batch_size
        dummy_ic50 = ic50 if ic50 is not None else torch.zeros(
            (max(int(pairs[:, 0].max()) + 1, 1), max(int(pairs[:, 1].max()) + 1, 1)))

        per_fold = []
        for ck in self.folds:
            scaled = _apply_saved_scaler(gene_data, genes, ck["omics_indices"],
                                         ck["scaler_mean"], ck["scaler_scale"])
            gene_tensor = stack_gene_data(scaled, genes)
            # same noise ablation the model was trained with
            gene_tensor = apply_noise_omics(gene_tensor, ck["omics_indices"],
                                            ck.get("noise_indices", []),
                                            ck.get("noise_seed", 0))
            model = DRPModel(genes, drug_meta, self.config).to(self.device)
            model.load_state_dict(ck["model_state"], strict=False)
            model.eval()

            ds = OmicsDrugDataset(gene_tensor, dummy_ic50, pairs)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False,
                                num_workers=0)
            preds = []
            for feats, didx, _, _ in loader:
                feats = feats.to(self.device)
                didx = didx.to(self.device)
                out = model(feats, didx).squeeze(-1)
                preds.append(out.cpu().numpy())
            per_fold.append(np.concatenate(preds))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_fold = np.vstack(per_fold)          # [K, M]
        mean_pred = per_fold.mean(axis=0)
        out = {"pred": mean_pred, "per_fold": per_fold}
        if ic50 is not None:
            true = np.array([float(ic50[int(s), int(d)]) for s, d in pairs])
            out["true"] = true
            out["metrics"] = regression_metrics(true, mean_pred)
        return out

    @torch.no_grad()
    def evaluate_raw(self, raw_new: RawData) -> Dict[str, object]:
        """Convenience: ensemble-evaluate on a RawData built over new data that
        shares the training feature space. Uses raw_new.pairs / raw_new.ic50."""
        return self.predict(raw_new.gene_data, raw_new.drug_meta,
                            raw_new.pairs, ic50=raw_new.ic50)


def _with_dataset_path(cfg: ExperimentConfig, dataset_path: str) -> ExperimentConfig:
    from dataclasses import replace
    return replace(cfg, dataset_path=dataset_path)
