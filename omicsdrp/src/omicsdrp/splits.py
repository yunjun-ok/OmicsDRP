"""Data-splitting regimes for nested cross-validation.

Three modes, all returning the same ``FoldSpec`` contract so the training engine
never needs to know which regime it is running:

  * ``mixed``       -- random split on (cell, drug) pairs. Same cell & drug may
                       appear in both train and test (the original behaviour).
  * ``unseen_cell`` -- split by cell line. Test cells never appear in train.
  * ``unseen_drug`` -- split by drug. Test drugs never appear in train.

Conservative (stratified) unseen splits
----------------------------------------
A naive random group split can, by chance, place an entire similarity-cluster of
cells/drugs into the test fold -> the test set becomes *fully* out-of-distribution
and the score is dominated by luck. To avoid this we first cluster the groups
(cells by omics profile, drugs by Morgan fingerprint) and then use
``StratifiedKFold`` over the cluster labels, so every fold's test set contains a
representative mix of clusters and each held-out group has similar neighbours in
train. This is the "보다 보수적" split the review plan calls for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold

from .data import RawData


@dataclass
class FoldSpec:
    fold: int
    train_pair_idx: np.ndarray   # indices into RawData.pairs
    val_pair_idx: np.ndarray     # inner validation (early stopping)
    test_pair_idx: np.ndarray    # outer test (unbiased evaluation)
    train_sample_indices: np.ndarray  # unique cell rows used to fit scaling
    meta: Dict[str, object]


# --------------------------------------------------------------------------- #
# clustering helpers
# --------------------------------------------------------------------------- #
def _cell_feature_matrix(raw: RawData, omics_indices: List[int]) -> np.ndarray:
    """[N_cell, n_gene * n_omics] matrix built from the selected omics."""
    cols = [raw.gene_data[g][:, omics_indices] for g in raw.genes]
    return torch.cat(cols, dim=1).cpu().numpy()


def _drug_feature_matrix(raw: RawData) -> np.ndarray:
    """[N_drug, 512] Morgan fingerprint matrix aligned to IC50 column order.

    IC50 columns are DRUG_IDs (as strings); align the drug_meta rows to that
    order so drug indices are consistent everywhere.
    (drug_meta row order == IC50 column order == drug index order; verified).
    """
    fps = [[float(b) for b in str(fp).split(",")]
           for fp in raw.drug_meta["Morgan_Fingerprint"]]
    return np.asarray(fps, dtype=np.float32)


def group_features(raw: RawData, config, target: str):
    """Return (feature_matrix, natural_metric) for the clustering target.

    ``target`` is "cell" (omics profile, euclidean geometry) or "drug"
    (binary Morgan fingerprint, Jaccard/Tanimoto geometry). Diagnostics and the
    split builder both call this so they cluster identical inputs.
    """
    if target == "cell":
        return _cell_feature_matrix(raw, config.omics_indices()), "euclidean"
    if target == "drug":
        return _drug_feature_matrix(raw), "jaccard"
    raise ValueError(f"target must be 'cell' or 'drug', got {target!r}")


def cluster_labels(features: np.ndarray, n_clusters: int, seed: int,
                   pca_dim: Optional[int] = 50) -> np.ndarray:
    n = features.shape[0]
    k = int(max(2, min(n_clusters, n)))
    x = StandardScaler().fit_transform(features)
    if pca_dim and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)


# --------------------------------------------------------------------------- #
# split builders
# --------------------------------------------------------------------------- #
def _inner_val_from_pairs(train_pool: np.ndarray, groups: Optional[np.ndarray],
                          strat: Optional[np.ndarray], val_frac: float,
                          seed: int) -> (np.ndarray, np.ndarray):
    """Carve an inner-validation set out of the outer-train pool.

    * ``mixed``  -> random split on pairs.
    * unseen     -> split on *groups* (cells/drugs) so val stays unseen wrt train,
                    stratified by the group's cluster label.
    """
    rng = np.random.default_rng(seed)
    if groups is None:  # mixed
        perm = rng.permutation(train_pool)
        n_val = max(1, int(len(perm) * val_frac))
        return perm[n_val:], perm[:n_val]

    # unseen: pick a fraction of groups for validation, stratified by cluster
    uniq_groups = np.unique(groups[train_pool])
    g_strat = np.array([strat[g] for g in uniq_groups])
    n_splits = max(2, int(round(1.0 / val_frac)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_g_local = next(iter(skf.split(uniq_groups, g_strat)))
    val_groups = set(uniq_groups[val_g_local].tolist())
    is_val = np.array([groups[p] in val_groups for p in train_pool])
    return train_pool[~is_val], train_pool[is_val]


def build_folds(raw: RawData, config) -> List[FoldSpec]:
    omics_indices = config.omics_indices()
    pairs = raw.pairs
    cell_of = pairs[:, 0]
    drug_of = pairs[:, 1]
    n_pairs = len(pairs)
    seed = config.seed
    K = config.outer_folds

    folds: List[FoldSpec] = []

    if config.split_mode == "mixed":
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        for i, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(n_pairs)), start=1):
            tr, val = _inner_val_from_pairs(trainval_idx, None, None,
                                            config.inner_val_frac, seed + i)
            folds.append(FoldSpec(
                fold=i, train_pair_idx=tr, val_pair_idx=val, test_pair_idx=test_idx,
                train_sample_indices=np.unique(cell_of[tr]),
                meta={"mode": "mixed"}))
        return folds

    # ---- group-based unseen splits ----
    if config.split_mode == "unseen_cell":
        groups = cell_of
        feats = _cell_feature_matrix(raw, omics_indices)
        n_clusters = config.n_cluster_cell
        n_groups = raw.n_cell
    elif config.split_mode == "unseen_drug":
        # Duplicate molecules are already merged at the data level (see
        # data.merge_duplicate_drugs), so every drug is a distinct molecule and a
        # plain per-drug split is leakage-free.
        groups = drug_of
        feats = _drug_feature_matrix(raw)
        n_clusters = config.n_cluster_drug
        n_groups = raw.n_drug
    else:
        raise ValueError(f"Unknown split_mode {config.split_mode}")

    strat = cluster_labels(feats, n_clusters, seed)  # per-group cluster label
    # Guardrail: clusters smaller than the fold count cannot be spread across
    # folds, so such groups fall into a single test fold (fully OOD). Warn so the
    # user can lower k (see scripts/inspect_clusters.py).
    sizes = np.bincount(strat)
    n_small = int((sizes < K).sum())
    if n_small:
        import warnings
        warnings.warn(
            f"[{config.split_mode}] {n_small}/{len(sizes)} clusters have < "
            f"{K} members; those groups cannot be stratified across {K} folds "
            f"and will land in a single test fold. Consider lowering "
            f"n_cluster_{'cell' if config.split_mode=='unseen_cell' else 'drug'} "
            f"(inspect via scripts/inspect_clusters.py).")
    group_ids = np.arange(n_groups)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for i, (trainval_g, test_g) in enumerate(skf.split(group_ids, strat), start=1):
        test_groups = set(group_ids[test_g].tolist())
        is_test = np.array([g in test_groups for g in groups])
        test_idx = np.where(is_test)[0]
        trainval_idx = np.where(~is_test)[0]
        tr, val = _inner_val_from_pairs(trainval_idx, groups, strat,
                                        config.inner_val_frac, seed + i)
        folds.append(FoldSpec(
            fold=i, train_pair_idx=tr, val_pair_idx=val, test_pair_idx=test_idx,
            train_sample_indices=np.unique(cell_of[tr]),
            meta={"mode": config.split_mode,
                  "n_clusters": int(len(np.unique(strat))),
                  "n_test_groups": len(test_groups)}))
    return folds
