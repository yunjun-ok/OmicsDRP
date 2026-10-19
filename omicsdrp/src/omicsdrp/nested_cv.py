"""Nested cross-validation engine (outer-test / inner-val).

Fixes the Stage-1 metric-bias flaw: the original code selected the early-stopping
epoch on a validation set and then *reported that same set* as the score, which
is optimistically biased. Here, for every outer fold:

    outer-train  ->  inner-train + inner-val   (inner-val drives early stopping)
    outer-test                                 (untouched until final evaluation)

The reported metric is the model -- restored to its best inner-val epoch --
evaluated **once** on the held-out outer test. Feature scaling is fit on
inner-train cells only, so no information crosses the boundary.

Every epoch's train and val loss/metrics are streamed to the recorder, so the
entire optimisation trajectory (not just the best epoch) is available later.
"""
from __future__ import annotations

import gc
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

from .config import ExperimentConfig
from .data import (RawData, select_omics, scale_gene_data, stack_gene_data,
                   apply_noise_omics,
                   OmicsDrugDataset)
from .splits import FoldSpec, build_folds
from .models import DRPModel, initialize_weights, count_parameters
from .engine import (set_seed, build_adamw_optimizer, train_epoch, evaluate,
                     EarlyStopping)
from .recorder import ExperimentRecorder


def _make_loader(gene_tensor, ic50, pair_idx, pairs, batch_size, shuffle,
                 drop_last=False, num_workers=4):
    ds = OmicsDrugDataset(gene_tensor, ic50, pairs[pair_idx])
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0)


def run_fold(raw: RawData, config: ExperimentConfig, fold: FoldSpec,
             recorder: ExperimentRecorder, device: str) -> dict:
    # per-fold seed so a fold's result is identical whether it runs fresh or as a
    # resume (independent of whether earlier folds ran in this session).
    set_seed(config.seed + fold.fold)
    omics_gene = select_omics(raw.gene_data, config.omics_indices())
    gene_scaled = scale_gene_data(omics_gene, fold.train_sample_indices)
    # stack the per-gene dict into one [N_cell, n_gene, n_omics] tensor ONCE, so the
    # DataLoader returns a cheap slice (not a 909-key dict) per sample.
    gene_tensor = stack_gene_data(gene_scaled, raw.genes)
    # noise ablation (no-op unless config.noise_omics is set)
    gene_tensor = apply_noise_omics(gene_tensor, config.omics_indices(),
                                    config.noise_indices(), config.seed + fold.fold)

    nw = config.num_workers
    # drop_last on TRAIN only: the drug-embedding & response-head BatchNorm1d still
    # crash on a trailing batch of size 1; eval loaders use BN eval mode so are safe.
    train_loader = _make_loader(gene_tensor, raw.ic50, fold.train_pair_idx,
                                raw.pairs, config.batch_size, shuffle=True,
                                drop_last=True, num_workers=nw)
    val_loader = _make_loader(gene_tensor, raw.ic50, fold.val_pair_idx,
                              raw.pairs, config.batch_size, shuffle=False, num_workers=nw)
    test_loader = _make_loader(gene_tensor, raw.ic50, fold.test_pair_idx,
                               raw.pairs, config.batch_size, shuffle=False, num_workers=nw)

    model = DRPModel(raw.genes, raw.drug_meta, config).to(device)
    model.apply(initialize_weights)
    criterion = nn.MSELoss()
    optimizer = build_adamw_optimizer(model, config.lr, config.weight_decay)
    stopper = EarlyStopping(patience=config.patience)

    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_m = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_m = evaluate(model, val_loader, criterion, device)
        recorder.log_epoch(fold.fold, epoch, "train", tr_loss, tr_m)
        recorder.log_epoch(fold.fold, epoch, "val", va_loss, va_m)
        print(f"[fold {fold.fold}] epoch {epoch:3d} "
              f"train_rmse={tr_m['rmse']:.4f} val_rmse={va_m['rmse']:.4f}")
        stopper.step(va_m["rmse"], model, epoch)
        if stopper.early_stop:
            print(f"[fold {fold.fold}] early stop @ epoch {epoch} "
                  f"(best epoch {stopper.best_epoch})")
            break

    # restore best-inner-val weights, then evaluate ONCE on the held-out test
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    test_loss, test_m, preds = evaluate(model, test_loader, criterion, device, collect=True)
    test_m = {**test_m, "loss": test_loss}

    recorder.save_predictions(fold.fold, df=pd.DataFrame({
        "sample_idx": preds["sample_idx"], "drug_idx": preds["drug_idx"],
        "true": preds["true"], "pred": preds["pred"]}))
    # log test LAST so the fold-completion marker (test_metrics.json) only appears
    # once predictions are safely on disk -> a crash never leaves a fold "done"
    # without its predictions.
    recorder.log_fold_test(fold.fold, test_m, best_epoch=stopper.best_epoch)

    # free GPU memory before the next fold/experiment
    del model, train_loader, val_loader, test_loader, stopper
    _free_gpu()
    return test_m


def run_nested_cv(raw: RawData, config: ExperimentConfig,
                  recorder: ExperimentRecorder, device: str) -> dict:
    set_seed(config.seed)
    folds = build_folds(raw, config)
    recorder.event("folds_built", n_folds=len(folds),
                   fold_sizes=[{"fold": f.fold,
                                "n_train": int(len(f.train_pair_idx)),
                                "n_val": int(len(f.val_pair_idx)),
                                "n_test": int(len(f.test_pair_idx)),
                                **f.meta} for f in folds])
    # record model size once (Stage-2 complexity comparison)
    probe = DRPModel(raw.genes, raw.drug_meta, config)
    recorder.event("model_info", n_params=count_parameters(probe))
    del probe

    # for unseen splits, save the clustering diagnostics for the k that was
    # actually used (skip if already saved on a previous run).
    import os
    if config.split_mode in ("unseen_cell", "unseen_drug"):
        diag_dir = os.path.join(recorder.dir, "cluster_diag")
        if not os.path.isdir(diag_dir):
            try:
                from .diagnostics import (embedding_plot, ood_distance_report,
                                          cluster_fold_table)
                target = "cell" if config.split_mode == "unseen_cell" else "drug"
                k = config.n_cluster_cell if target == "cell" else config.n_cluster_drug
                embedding_plot(raw, config, target, k, diag_dir, do_tsne=False)
                ood = ood_distance_report(raw, config, target, k, diag_dir)
                cluster_fold_table(raw, config, target, k, diag_dir)
                recorder.event("cluster_diagnostics_saved", target=target, k=int(k),
                               ood_summary=ood.to_dict(orient="records"))
            except Exception as e:  # diagnostics must never block training
                recorder.event("cluster_diagnostics_failed", error=str(e))

    # fold-level resume: skip folds already completed on disk (crash/power-off safe)
    done = recorder.completed_folds()
    if done:
        print(f"  resuming: folds {sorted(done)} already done, skipping")
        recorder.event("resume", completed_folds=sorted(done))
    for fold in folds:
        if fold.fold in done:
            continue
        run_fold(raw, config, fold, recorder, device)
        _free_gpu()
    return recorder.finalize()
