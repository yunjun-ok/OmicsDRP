"""Shared training loop for every competitor adapter.

An adapter only has to describe its *native* data — how to turn a set of
(cell, drug) pair indices into batches, and how to build its model. Everything
that must be held identical across models (fold indices, seeding, the
early-stopping rule, the leakage boundary, what gets written to disk) lives
here, so no adapter can quietly diverge from the protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


class BaseAdapter:
    """Subclasses implement the four hooks below; the loop is fixed."""

    name = "unnamed"
    default_lr = 1e-4
    default_batch_size = 64
    default_epochs = 100
    default_patience = 10

    # ---- hooks ------------------------------------------------------------ #
    def load_native(self, export: common.FoldExport) -> None:
        """Load this model's own input tables once, before any fold runs."""
        raise NotImplementedError

    def fit_scaler(self, export, fit_idx):
        """Fit feature scaling on training cells only. Return (mean, scale) or None."""
        return None

    def make_loader(self, export, pair_idx, scaler, *, batch_size, shuffle):
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, model, batch):
        """Return ``(prediction, target)`` both as 1-D float tensors."""
        raise NotImplementedError

    # ---- fixed machinery -------------------------------------------------- #
    def usable_pairs(self, export, pair_idx: np.ndarray) -> np.ndarray:
        """Drop pairs this model has no native features for.

        Every model covers a slightly different slice of GDSC, so a few pairs
        fall out. The dropped counts are recorded per fold in config.json — they
        must stay small, otherwise the comparison is not like-for-like.
        """
        return pair_idx

    def to_native_label(self, values):
        """Map predictions back to ln(IC50).

        Models that rescale the target internally (PaccMann min-maxes it)
        override this so every model is scored on the same axis. Identity for
        models already trained on ln(IC50).
        """
        return np.asarray(values)


def _epoch(adapter, model, loader, optimizer, device):
    model.train()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        pred, y = adapter.forward(model, batch)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.numel()
        n += y.numel()
    return total / max(n, 1)


@torch.no_grad()
def _predict(adapter, model, loader):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        pred, y = adapter.forward(model, batch)
        preds.append(pred.detach().cpu().numpy().ravel())
        trues.append(y.detach().cpu().numpy().ravel())
    return np.concatenate(trues), np.concatenate(preds)


def run(adapter: BaseAdapter, argv=None) -> None:
    ap = argparse.ArgumentParser(description=f"{adapter.name} benchmark adapter")
    ap.add_argument("--split", required=True, choices=list(common.SPLIT_MODES))
    ap.add_argument("--regime", required=True, choices=list(common.REGIMES))
    ap.add_argument("--folds", default="all", help="'all' or comma list e.g. 1,3")
    ap.add_argument("--epochs", type=int, default=adapter.default_epochs)
    ap.add_argument("--patience", type=int, default=adapter.default_patience)
    ap.add_argument("--lr", type=float, default=adapter.default_lr)
    ap.add_argument("--batch-size", type=int, default=adapter.default_batch_size)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out-root", default=str(common.BENCH_DIR / "BenchmarkResults"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, first fold only, subsampled pairs")
    args = ap.parse_args(argv)

    if args.smoke:
        # An explicit --epochs wins, so smoke can also be used for a quick
        # "does this model actually learn" check on a subsample.
        if args.epochs == adapter.default_epochs:
            args.epochs = 2
        args.folds = "1"

    device = torch.device(args.device)
    out_root = Path(args.out_root)
    export = common.FoldExport(args.split)

    print(f"[{adapter.name}] loading native inputs …", flush=True)
    adapter.device = device
    adapter.load_native(export)

    fold_ids = (list(range(1, export.n_folds + 1)) if args.folds == "all"
                else [int(x) for x in args.folds.split(",")])

    for k in fold_ids:
        if not args.overwrite and common.fold_outputs_exist(
                out_root, adapter.name, args.split, args.regime, k):
            print(f"[{adapter.name}] fold {k} already done — skipping", flush=True)
            continue

        # Per-fold seed, matching nested_cv.py, so a single fold reruns exactly.
        common.set_seed(args.seed + k)

        idx = export.regime_indices(k, args.regime)
        fit_idx = adapter.usable_pairs(export, idx["fit"])
        es_idx = adapter.usable_pairs(export, idx["early_stop"])
        eval_idx = adapter.usable_pairs(export, idx["eval"])

        if args.smoke:
            rng = np.random.default_rng(0)
            fit_idx = rng.choice(fit_idx, size=min(2000, len(fit_idx)), replace=False)
            es_idx = rng.choice(es_idx, size=min(500, len(es_idx)), replace=False)
            eval_idx = rng.choice(eval_idx, size=min(500, len(eval_idx)), replace=False)

        dropped = {
            "fit": int(len(idx["fit"]) - len(fit_idx)),
            "early_stop": int(len(idx["early_stop"]) - len(es_idx)),
            "eval": int(len(idx["eval"]) - len(eval_idx)),
        }

        # Leakage boundary: scaler sees training cells only.
        scaler = adapter.fit_scaler(export, fit_idx)

        loaders = {
            "fit": adapter.make_loader(export, fit_idx, scaler,
                                       batch_size=args.batch_size, shuffle=True),
            "early_stop": adapter.make_loader(export, es_idx, scaler,
                                              batch_size=args.batch_size, shuffle=False),
            "eval": adapter.make_loader(export, eval_idx, scaler,
                                        batch_size=args.batch_size, shuffle=False),
        }

        model = adapter.build_model().to(device)
        params = common.count_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_rmse, best_state, stale = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            tr_loss = _epoch(adapter, model, loaders["fit"], optimizer, device)
            y_t, y_p = _predict(adapter, model, loaders["early_stop"])
            rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
            print(f"  fold {k} epoch {epoch:3d} | train {tr_loss:.4f} | es-RMSE {rmse:.4f}",
                  flush=True)
            if rmse < best_rmse - 1e-6:
                best_rmse, stale = rmse, 0
                best_state = {kk: v.detach().cpu().clone()
                              for kk, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"  fold {k} early stop @ epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Single, final evaluation of the held-out fold, scored in ln(IC50).
        y_true, y_pred = _predict(adapter, model, loaders["eval"])
        y_true = adapter.to_native_label(y_true)
        y_pred = adapter.to_native_label(y_pred)

        mean, scale = (scaler if scaler is not None else (None, None))
        metrics = common.save_fold(
            out_root, adapter.name, args.split, args.regime, k,
            state_dict=best_state or model.state_dict(),
            scaler_mean=mean, scaler_scale=scale,
            config={
                "epochs_max": args.epochs, "patience": args.patience,
                "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
                "seed_used": args.seed + k, "device": str(device),
                "smoke": args.smoke,
            },
            pair_idx_eval=eval_idx, y_true=y_true, y_pred=y_pred,
            param_counts=params, export=export,
            extra_meta={
                "best_early_stop_rmse": best_rmse,
                "dropped_pairs_no_native_features": dropped,
                "train_seconds": round(time.time() - t0, 1),
            },
        )
        print(f"[{adapter.name}] fold {k} {args.regime}/{args.split} -> "
              f"{json.dumps({m: round(v, 4) for m, v in metrics.items()})}", flush=True)
