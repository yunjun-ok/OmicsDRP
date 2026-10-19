"""Env-neutral helpers shared by every competitor adapter.

Each adapter runs in its own conda env (``benchmark_deeptta`` /
``benchmark_graphdrp`` / ``benchmark_paccmann``), so this module deliberately
imports nothing heavier than numpy/pandas at module scope. ``torch`` is only
touched inside the functions that need it.

The single source of truth for *what data each model sees* is the frozen fold
export produced by ``export_folds.py``: every adapter reads the same
``exports/<split_mode>.npz`` and never calls ``build_folds`` itself. That is
what makes the split byte-identical across three different environments.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "omicsdrp" / "benchmark"
EXPORT_DIR = BENCH_DIR / "exports"
VENDOR_DIR = REPO_ROOT / "BenchSources"          # upstream clones + their native data
OMICSDRP_SRC = REPO_ROOT / "omicsdrp" / "src"
DATA_DIR = REPO_ROOT / "data"

SPLIT_MODES = ("mixed", "unseen_cell", "unseen_drug")
REGIMES = ("nested", "ensemble")


# --------------------------------------------------------------------------- #
# metrics — loaded straight from the omicsdrp source file
# --------------------------------------------------------------------------- #
def _load_omicsdrp_metrics():
    """Import ``omicsdrp/metrics.py`` by path.

    Loading the file directly (rather than ``import omicsdrp.metrics``) avoids
    dragging in the package ``__init__``, which pulls torch-side modules that
    some adapter envs do not have installed. metrics.py itself is pure
    numpy+scipy, so it imports cleanly everywhere.
    """
    path = OMICSDRP_SRC / "omicsdrp" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_omicsdrp_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regression_metrics = _load_omicsdrp_metrics().regression_metrics


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Replicate ``omicsdrp.engine.set_seed`` exactly, including cudnn flags.

    Adapters call this at the top of every fold with ``seed + fold`` so a rerun
    of a single fold reproduces bit-for-bit, matching how nested_cv.py seeds.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# frozen fold export
# --------------------------------------------------------------------------- #
class FoldExport:
    """Read-only view over ``exports/<split_mode>.npz``."""

    def __init__(self, split_mode: str):
        path = EXPORT_DIR / f"{split_mode}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python export_folds.py` in the omicsdrp env first."
            )
        self.split_mode = split_mode
        self._z = np.load(path, allow_pickle=False)
        self.meta = json.loads((EXPORT_DIR / f"{split_mode}.meta.json").read_text())

        self.pairs = self._z["pairs"]                # [M, 2] (cell_idx, drug_idx)
        self.y = self._z["y"]                        # [M] ln(IC50)
        self.cell_model_ids = self._z["cell_model_ids"]
        self.cell_cosmic_ids = self._z["cell_cosmic_ids"]   # -1 where unmapped
        self.drug_ids = self._z["drug_ids"]
        self.drug_names = self._z["drug_names"]
        self.drug_smiles = self._z["drug_smiles"]
        self.n_folds = int(self.meta["outer_folds"])

    def fold(self, k: int) -> Dict[str, np.ndarray]:
        """Indices into ``pairs`` for outer fold ``k`` (1-based)."""
        return {
            "train": self._z[f"fold{k}_train"],
            "val": self._z[f"fold{k}_val"],
            "test": self._z[f"fold{k}_test"],
        }

    def regime_indices(self, k: int, regime: str) -> Dict[str, np.ndarray]:
        """Map a fold onto the two training regimes.

        ``nested``   — train on inner-train, early-stop on inner-val, evaluate
                       outer-test exactly once. This is the honest metric.
        ``ensemble`` — train on the whole outer-train pool (train+val) and
                       early-stop on the held-out fold, mirroring
                       ``inference_models.py``. Its fold scores are
                       early-stopping-optimistic and are NOT performance numbers;
                       these weights exist to predict on external data.
        """
        f = self.fold(k)
        if regime == "nested":
            return {"fit": f["train"], "early_stop": f["val"], "eval": f["test"]}
        if regime == "ensemble":
            pool = np.concatenate([f["train"], f["val"]])
            pool.sort()
            return {"fit": pool, "early_stop": f["test"], "eval": f["test"]}
        raise ValueError(f"unknown regime {regime!r}")

    def train_cells(self, pair_idx: np.ndarray) -> np.ndarray:
        """Unique cell rows in a pair subset — the only cells a scaler may see."""
        return np.unique(self.pairs[pair_idx, 0])


def fold_signature(idx: np.ndarray) -> str:
    """Stable short hash of a fold index array, recorded in every meta.json."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(idx)).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# output layout
# --------------------------------------------------------------------------- #
def fold_dir(out_root: Path, model: str, split_mode: str, regime: str, k: int) -> Path:
    return Path(out_root) / model / f"{regime}__{split_mode}" / f"fold_{k}"


def fold_outputs_exist(out_root: Path, model: str, split_mode: str,
                       regime: str, k: int) -> bool:
    """Fold-level resume check — a fold is done only once metrics landed."""
    d = fold_dir(out_root, model, split_mode, regime, k)
    return (d / "metrics.json").exists() and (d / "model.pt").exists()


def count_params(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_params_total": int(total), "n_params_trainable": int(trainable)}


def save_fold(out_root: Path, model_name: str, split_mode: str, regime: str, k: int,
              *, state_dict, scaler_mean: Optional[np.ndarray],
              scaler_scale: Optional[np.ndarray], config: dict,
              pair_idx_eval: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
              param_counts: Dict[str, int], export: FoldExport,
              extra_meta: Optional[dict] = None) -> Dict[str, float]:
    """Persist everything needed to reproduce and to re-score this fold.

    Returns the metric dict so the caller can log it.
    """
    import pandas as pd
    import torch

    d = fold_dir(out_root, model_name, split_mode, regime, k)
    d.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, d / "model.pt")

    if scaler_mean is not None:
        np.savez(d / "scaler.npz", mean=scaler_mean, scale=scaler_scale)

    preds = pd.DataFrame({
        "sample_idx": export.pairs[pair_idx_eval, 0],
        "drug_idx": export.pairs[pair_idx_eval, 1],
        "true": y_true,
        "pred": y_pred,
    })
    preds.to_parquet(d / "predictions.parquet", index=False)

    metrics = regression_metrics(y_true, y_pred)
    metrics.update(param_counts)
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))

    meta = {
        "model": model_name,
        "split_mode": split_mode,
        "regime": regime,
        "fold": k,
        "n_eval_pairs": int(len(pair_idx_eval)),
        "fold_signature": fold_signature(pair_idx_eval),
        "export_meta": export.meta,
        "config": config,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "?"),
        "scores_are_performance": regime == "nested",
    }
    if extra_meta:
        meta.update(extra_meta)
    (d / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    return metrics


# --------------------------------------------------------------------------- #
# scaling — the leakage boundary
# --------------------------------------------------------------------------- #
def fit_scaler(matrix: np.ndarray):
    """StandardScaler fit on training rows only. Returns (mean, scale)."""
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def apply_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / scale).astype(np.float32)
