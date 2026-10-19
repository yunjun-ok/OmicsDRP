"""Data loading with omics-subset support.

Loads the three raw artefacts (per-gene omics tensors, drug metadata, IC50
matrix) and exposes helpers to:

  * select an omics subset (a column slice of every per-gene [N, 4] tensor),
  * fit feature scaling **only on training samples** (no leakage across the
    outer-test / inner-val boundary),
  * build an encoder-agnostic ``Dataset`` that yields *indices*; the drug
    representation itself is owned by the drug encoder (so a GNN can hold graph
    objects while Morgan holds a fingerprint table, without changing the loop).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .config import OMICS_ORDER


@dataclass
class RawData:
    gene_data: Dict[str, torch.Tensor]   # gene -> [N, 4]  (SNP, MET, CNV, RNA)
    genes: List[str]
    drug_meta: pd.DataFrame              # columns incl DRUG_ID, SMILE, Morgan_Fingerprint
    ic50: torch.Tensor                   # [N_cell, N_drug]
    pairs: np.ndarray                    # [M, 2] -> (sample_idx, drug_idx), IC50 non-NaN
    n_cell: int
    n_drug: int


def _canonical_smiles(smiles: List[str]) -> List[str]:
    """RDKit-canonicalise SMILES (so same molecule/different string collapses);
    fall back to the raw string if rdkit is unavailable or parsing fails."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        out = []
        for s in smiles:
            m = Chem.MolFromSmiles(str(s))
            out.append(Chem.MolToSmiles(m) if m is not None else str(s))
        return out
    except Exception:
        return [str(s) for s in smiles]


def merge_duplicate_drugs(ic50_df: pd.DataFrame, drug_meta: pd.DataFrame):
    """Collapse drugs that are the SAME molecule (identical canonical SMILES) into
    one, at the DATA level.

    GDSC2 registers 10 molecules twice under different DRUG_IDs, with slightly
    different IC50s per screening batch. Structure-based encoders can't tell them
    apart, so we merge them: IC50 becomes the (NaN-aware) mean across the duplicate
    columns, the merged label is ``name1 (id1)/name2 (id2)``, and a ``_source_row``
    column records the representative original row (used to remap pretrained
    embedding tables, whose rows are in the original 241-drug order).

    Returns (merged_ic50_df [n_cell x n_unique], merged_drug_meta [n_unique]).
    """
    keys = _canonical_smiles(drug_meta["SMILE"].astype(str).tolist())
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)

    ic50_vals = ic50_df.values  # [n_cell, n_drug], column order == drug_meta rows
    merged_rows, merged_cols = [], []
    import warnings
    with warnings.catch_warnings():
        # a cell that is NaN in BOTH duplicate columns -> nanmean of all-NaN ->
        # NaN (correct; that pair is simply absent). Silence the noisy warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for idxs in groups.values():
            rep = idxs[0]
            row = drug_meta.iloc[rep].copy()
            if len(idxs) > 1:
                ids = [str(drug_meta.iloc[j]["DRUG_ID"]) for j in idxs]
                names = [f"{drug_meta.iloc[j]['DRUG_NAME']} ({drug_meta.iloc[j]['DRUG_ID']})"
                         for j in idxs]
                row["DRUG_ID"] = "/".join(ids)
                row["DRUG_NAME"] = "/".join(names)
                merged_cols.append(np.nanmean(ic50_vals[:, idxs], axis=1))
            else:
                merged_cols.append(ic50_vals[:, rep])
            row["_source_row"] = rep
            merged_rows.append(row)

    merged_drug_meta = pd.DataFrame(merged_rows).reset_index(drop=True)
    merged_ic50 = np.column_stack(merged_cols).astype(np.float32)
    merged_ic50_df = pd.DataFrame(merged_ic50, index=ic50_df.index,
                                  columns=list(merged_drug_meta["DRUG_ID"]))
    return merged_ic50_df, merged_drug_meta


# --------------------------------------------------------------------------- #
# gene-set control: 909 random genes instead of the 909 curated pharmacogenes
# --------------------------------------------------------------------------- #
ALLGENE_FILES = {"SNP": "SNP_Gene_Level_count.csv", "MET": "MET_Gene_Level_mean.csv",
                 "CNV": "CNV.csv", "RNA": "RNA.csv"}   # column order = OMICS_ORDER


def build_gene_dict(dataset_path: str, seed: int, n_genes: int = 909
                    ) -> Dict[str, torch.Tensor]:
    """Size-matched RANDOM gene set, in the same format as ``PGKB_Gene_data_dict.pth``.

    Sampling pool = genes present in all four ``raw_data_allgene`` matrices with
    **no missing values** and non-zero variance, EXCLUDING the 909 PGKB genes
    (15,072 usable genes, of which 14,193 are non-PGKB). Restricting to complete
    genes matters: sampling from the union would hand the control set constant
    imputed columns, so it would lose for being missing rather than for being
    uninformative.

    The ``raw_data_allgene`` matrices are already on the exact scale used by the
    PGKB dict (873 GDSC2 cell lines, same row order, RNA already log-scaled), so
    no transform is applied -- only the column pick.

    Cached at ``<dataset_path>/gene_dicts/random_<seed>.pth`` (+ ``.genes.txt``)
    because building it parses ~900 MB of CSV.
    """
    cache_dir = os.path.join(dataset_path, "gene_dicts")
    cache = os.path.join(cache_dir, f"random_{seed}.pth")
    if os.path.isfile(cache):
        return torch.load(cache)

    src = os.path.join(dataset_path, "raw_data_allgene")
    heads = {}
    for k, f in ALLGENE_FILES.items():
        with open(os.path.join(src, f)) as fh:
            cols = fh.readline().rstrip("\n").split(",")
        heads[k] = [c for c in cols if c]          # RNA has a leading index col
    pool = set.intersection(*(set(v) for v in heads.values()))
    pgkb = set(pd.read_csv(os.path.join(dataset_path, "gene_list.txt"),
                           header=None)[0])
    cand = sorted(pool - pgkb)

    mats = {}
    for k, f in ALLGENE_FILES.items():
        d = pd.read_csv(os.path.join(src, f), usecols=cand, low_memory=False)
        mats[k] = d.loc[:, ~d.columns.duplicated()].reindex(columns=cand).astype("float32")
    ok = np.ones(len(cand), dtype=bool)
    for k in ALLGENE_FILES:
        v = mats[k].values
        ok &= ~np.isnan(v).any(axis=0)             # complete
        ok &= np.nanstd(v, axis=0) > 1e-6          # non-degenerate
    usable = [g for g, f in zip(cand, ok) if f]
    if len(usable) < n_genes:
        raise ValueError(f"only {len(usable)} usable genes in pool")

    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(len(usable), n_genes, replace=False))
    picked = [usable[i] for i in picked]
    cols = {k: mats[k][picked].values for k in ALLGENE_FILES}
    gene_data = {g: torch.from_numpy(
        np.column_stack([cols[k][:, i] for k in OMICS_ORDER])).float()
        for i, g in enumerate(picked)}

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(gene_data, cache)
    with open(cache + ".genes.txt", "w") as fh:
        fh.write("\n".join(picked) + "\n")
    print(f"[gene_set] random:{seed} -> {len(picked)} genes "
          f"(pool {len(usable)}), cached at {cache}")
    return gene_data


def load_gene_data(dataset_path: str, gene_set: str = "pgkb"
                   ) -> Dict[str, torch.Tensor]:
    """Dispatch ``ExperimentConfig.gene_set`` to a gene -> [873, 4] dict."""
    if gene_set == "pgkb":
        return torch.load(f"{dataset_path}/PGKB_Gene_data_dict.pth")
    if gene_set.startswith("random:"):
        return build_gene_dict(dataset_path, int(gene_set.split(":", 1)[1]))
    raise ValueError(f"unknown gene_set {gene_set!r}")


def load_raw(dataset_path: str, merge_duplicates: bool = True,
             gene_set: str = "pgkb") -> RawData:
    gene_data = load_gene_data(dataset_path, gene_set)
    genes = list(gene_data.keys())

    ic50_df = pd.read_csv(f"{dataset_path}/IC50_GDSC2.csv", index_col=0)
    drug_meta = pd.read_csv(
        f"{dataset_path}/TargetDrugs_with_MorganFingerprint_GDSC2_512.txt", sep="\t")

    if merge_duplicates:
        ic50_df, drug_meta = merge_duplicate_drugs(ic50_df, drug_meta)

    ic50 = torch.from_numpy(ic50_df.values).type(torch.FloatTensor)
    pairs = np.argwhere(~np.isnan(ic50_df.values))  # [M, 2]

    return RawData(
        gene_data=gene_data,
        genes=genes,
        drug_meta=drug_meta,
        ic50=ic50,
        pairs=pairs,
        n_cell=ic50.shape[0],
        n_drug=ic50.shape[1],
    )


def select_omics(gene_data: Dict[str, torch.Tensor],
                 omics_indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    """Column-slice every per-gene tensor down to the chosen modalities."""
    idx = torch.as_tensor(list(omics_indices), dtype=torch.long)
    return {g: t.index_select(1, idx).contiguous() for g, t in gene_data.items()}


def scale_gene_data(gene_data: Dict[str, torch.Tensor],
                    train_sample_indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    """StandardScaler fit on *train* cell rows only, applied to all rows.

    Fitting on the union of train samples (never val/test) is what keeps the
    nested-CV estimate honest.
    """
    train_idx = np.asarray(sorted(set(int(i) for i in train_sample_indices)))
    scaled: Dict[str, torch.Tensor] = {}
    for gene, mat in gene_data.items():
        mat_np = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)
        scaler = StandardScaler()
        scaler.fit(mat_np[train_idx])
        scaled[gene] = torch.tensor(scaler.transform(mat_np), dtype=torch.float32)
    return scaled


def stack_gene_data(gene_data: Dict[str, torch.Tensor],
                    genes: Sequence[str]) -> torch.Tensor:
    """Stack the per-gene dict into a single ``[N_cell, n_gene, n_omics]`` tensor,
    in ``genes`` order (must match the model's per-gene weight order).

    Done ONCE per fold so the Dataset can return a cheap slice instead of building
    a 909-key dict per sample (the old data-loading bottleneck)."""
    return torch.stack([gene_data[g] for g in genes], dim=1).contiguous()


def apply_noise_omics(gene_tensor: torch.Tensor,
                      omics_indices: Sequence[int],
                      noise_indices: Sequence[int],
                      seed: int) -> torch.Tensor:
    """Replace the chosen modality columns of a stacked ``[N_cell, n_gene, n_omics]``
    tensor with i.i.d. N(0, 1) noise.

    Called AFTER ``scale_gene_data`` so the noise matches the scale of the real
    (standardised) features -- the only thing removed is the information, not the
    input dimensionality or the parameter count. Deterministic given ``seed``
    (pass ``config.seed + fold`` so a fold is reproducible on resume).
    """
    noise_indices = list(noise_indices)
    if not noise_indices:
        return gene_tensor
    pos = [list(omics_indices).index(i) for i in noise_indices]
    g = torch.Generator().manual_seed(int(seed))
    out = gene_tensor.clone()
    for p in pos:
        out[:, :, p] = torch.randn(out.shape[0], out.shape[1], generator=g)
    return out.contiguous()


class OmicsDrugDataset(Dataset):
    """Yields (gene_features[n_gene, n_omics], drug_idx, ic50, (sample_idx, drug_idx)).

    Gene features are a slice of the pre-stacked cell tensor; drug features are
    gathered by the model's drug encoder from ``drug_idx``.
    """

    def __init__(self, gene_tensor: torch.Tensor, ic50: torch.Tensor,
                 pairs: Sequence[Sequence[int]]):
        self.gene_tensor = gene_tensor          # [N_cell, n_gene, n_omics]
        self.ic50 = ic50
        self.pairs = [(int(s), int(d)) for s, d in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sample_idx, drug_idx = self.pairs[idx]
        gene_features = self.gene_tensor[sample_idx]        # [n_gene, n_omics]
        ic50_value = self.ic50[sample_idx, drug_idx]
        return gene_features, drug_idx, ic50_value, (sample_idx, drug_idx)
