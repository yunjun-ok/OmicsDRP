"""Builds a frozen CCLE/PRISM export (MTS or HTS) for external validation.

Single source of truth for what data the CCLE inference scripts see: reads
the already-prepared ``data/ccle_processed/`` tables once, resolves
DeepTTA/PaccMann's trained gene lists against ``RNA_combat.csv`` via
``gene_alias``, and writes one export every model's inference script reads --
so no two inference scripts can silently diverge on cell/drug ordering or
gene resolution.

Genuinely-missing genes are left as NaN here. The imputation strategy
(training-mean vs CCLE-cohort-mean) is applied at inference time by each
model's own script, not baked in here, so the two variants can be compared.

The 909-gene PGKB panel omicsdrp uses has an exact 909/909 match already
(verified separately) -- this export does not duplicate it; the omicsdrp
inference script reads ``CCLE_PGKB_Gene_data_dict.pth`` directly, using the
``cell_ids`` order recorded here to confirm alignment.

Usage:
    python ccle_preprocess.py --split mts   # 172 drugs, 488 cells (default)
    python ccle_preprocess.py --split hts   # 1346 drugs, 338 cells
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_alias  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CCLE_DIR = REPO_ROOT / "data" / "ccle_processed"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
GDSC_RMA = REPO_ROOT / "Benchmark" / "our_data" / "Cell_line_RMA_proc_basalExp.txt"
PACCMANN_PANEL = REPO_ROOT / "Benchmark" / "PaccMann" / "datasets" / "2128_genes.pkl"


def _canonical_cell_order() -> list:
    """The 666-cell CCLE order shared by CCLE_PGKB_Gene_data_dict.pth and
    RNA_combat_PGKB.csv (verified identical elsewhere)."""
    idx = pd.read_csv(CCLE_DIR / "RNA_combat_PGKB.csv", index_col=0, usecols=[0]).index
    return list(idx)


SPLIT_CONFIG = {
    "mts": {"col_idx_field": "mts_col_idx", "label_file": "IC50_PRIMS_MTS.csv"},
    "hts": {"col_idx_field": "hts_col_idx", "label_file": "IC50_PRIMS_HTS.csv"},
}


def _load_drug_table(split: str) -> pd.DataFrame:
    """MTS/HTS-filtered, col_idx-sorted subset of PRISM_drug_smiles.csv.

    Keeps ``_source_row`` = each drug's row position in the FULL,
    unfiltered 1389-row PRISM_drug_smiles.csv -- that's the row order the
    CCLE pretrained drug embeddings (data/ccle_processed/drug_embeddings/
    *.npy) were computed in, per their own meta.json ("row i == prism_name
    row i in PRISM_drug_smiles.csv"), which is a different order/subset
    than this split's drug table. omicsdrp's own PretrainedEmbeddingDrugEncoder
    already supports remapping via a drug_meta["_source_row"] column
    (drug_encoders.py:155-159), the same mechanism used to align GDSC's
    241-row tables down to the 231-drug merged order.
    """
    col_idx_field = SPLIT_CONFIG[split]["col_idx_field"]
    df = pd.read_csv(CCLE_DIR / "PRISM_drug_smiles.csv")
    sub = df[df[col_idx_field].notna()].copy()
    sub[col_idx_field] = sub[col_idx_field].astype(int)
    sub = sub.sort_values(col_idx_field)
    sub["_source_row"] = sub.index
    return sub.reset_index(drop=True)


def _gdsc_rma_index() -> pd.Index:
    """Gene-symbol index, deduplicated on symbol (keep first) -- what
    PaccMann's adapter uses (paccmann_adapter.py dedups the pre-transpose
    GENE_SYMBOLS index)."""
    gdsc = pd.read_csv(GDSC_RMA, sep="\t")
    gdsc = gdsc.drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS")
    return gdsc[~gdsc.index.duplicated(keep="first")].index


def _deeptta_raw_gene_columns() -> list:
    """DeepTTA's adapter does NOT deduplicate gene symbols (only cell IDs,
    post-transpose) -- its trained MLP input is all 17,737 raw columns,
    including 318 blank/NaN-symbol probesets and zero real-symbol repeats
    (verified: 17,419 unique real symbols + 318 NaN == 17,737). Reproducing
    that exact, non-deduplicated column list/order here so the CCLE input
    matches the trained model's dimensionality."""
    gdsc = pd.read_csv(GDSC_RMA, sep="\t")
    gdsc = gdsc.drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS")
    cols = [c for c in gdsc.columns if c.startswith("DATA.")]
    expr = gdsc[cols].T
    expr = expr[~expr.index.duplicated(keep="first")]  # dedup cell IDs only
    # Normalise the ~318 blank/NaN gene-symbol probesets to a clean sentinel
    # string (valid JSON, unambiguous) -- these can never match a CCLE column
    # by name and always fall into "missing", same outcome as raw NaN would.
    return ["<no_symbol>" if pd.isna(c) else c for c in expr.columns]


def _deeptta_trained_gene_list() -> list:
    return _deeptta_raw_gene_columns()


def _paccmann_trained_gene_list() -> list:
    with open(PACCMANN_PANEL, "rb") as fh:
        panel = list(pickle.load(fh))
    gdsc_index = set(_gdsc_rma_index())
    return [g for g in panel if g in gdsc_index]


def _resolved_gene_matrix(trained_genes, cell_order, rna):
    """[n_cell, n_trained_gene] matrix from RNA_combat.csv, columns aligned to
    ``trained_genes`` order via gene_alias resolution. Missing genes -> NaN."""
    resolved, missing, ambiguous = gene_alias.resolve_genes(trained_genes, rna.columns)
    mat = np.full((len(cell_order), len(trained_genes)), np.nan, dtype=np.float32)
    for gi, g in enumerate(trained_genes):
        col = resolved.get(g)
        if col is not None:
            mat[:, gi] = rna[col].values.astype(np.float32)
    return mat, resolved, missing, ambiguous


def build(split: str = "mts") -> None:
    label_file = SPLIT_CONFIG[split]["label_file"]

    cell_order = _canonical_cell_order()
    drug_table = _load_drug_table(split)
    n_cell, n_drug = len(cell_order), len(drug_table)
    print(f"[{split}] cells={n_cell} drugs={n_drug}")

    ic50_df = pd.read_csv(CCLE_DIR / label_file, index_col=0)
    ic50_df = ic50_df.reindex(index=cell_order, columns=drug_table["prism_name"])
    ic50 = ic50_df.values.astype(np.float32)
    # A handful of labels (23 in HTS, 0 in MTS) are exactly float32's max
    # magnitude (+/-3.4028235e38) -- a sentinel for "couldn't fit a
    # dose-response curve", not a real ln(IC50). Not NaN, so they'd silently
    # corrupt every metric (one such value dominates a mean-squared-error by
    # itself). Treat as missing, same as NaN.
    sentinel = np.abs(ic50) > 1e30
    if sentinel.any():
        print(f"[{split}] dropping {int(sentinel.sum())} sentinel (unfit-curve) labels")
    ic50[sentinel] = np.nan
    pairs = np.argwhere(~np.isnan(ic50))
    print(f"[{split}] pairs (non-NaN IC50): {len(pairs)}")

    morgan = np.stack([
        np.array([int(b) for b in s.split(",")], dtype=np.float32)
        for s in drug_table["morgan_fingerprint_512"]
    ])

    rna = pd.read_csv(CCLE_DIR / "RNA_combat.csv", index_col=0).reindex(cell_order)

    deeptta_genes = _deeptta_trained_gene_list()
    paccmann_genes = _paccmann_trained_gene_list()
    deeptta_mat, deeptta_resolved, deeptta_missing, deeptta_amb = _resolved_gene_matrix(
        deeptta_genes, cell_order, rna)
    paccmann_mat, paccmann_resolved, paccmann_missing, paccmann_amb = _resolved_gene_matrix(
        paccmann_genes, cell_order, rna)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        EXPORT_DIR / f"ccle_{split}.npz",
        pairs=pairs, ic50=ic50, morgan_fingerprint=morgan,
        deeptta_expr=deeptta_mat, paccmann_expr=paccmann_mat,
    )
    drug_table.to_csv(EXPORT_DIR / f"ccle_{split}_drug_meta.csv", index=False)

    meta = {
        "split": split,
        "n_cell": n_cell, "n_drug": n_drug, "n_pairs": int(len(pairs)),
        "cell_ids": cell_order,
        "drug_names": drug_table["prism_name"].tolist(),
        "deeptta": {
            "n_trained_genes": len(deeptta_genes),
            "n_missing": len(deeptta_missing), "n_ambiguous": len(deeptta_amb),
            "gene_order": deeptta_genes, "missing_genes": deeptta_missing,
            "ambiguous": deeptta_amb,
        },
        "paccmann": {
            "n_trained_genes": len(paccmann_genes),
            "n_missing": len(paccmann_missing), "n_ambiguous": len(paccmann_amb),
            "gene_order": paccmann_genes, "missing_genes": paccmann_missing,
            "ambiguous": paccmann_amb,
        },
        "omicsdrp_note": (
            "use data/ccle_processed/CCLE_PGKB_Gene_data_dict.pth directly -- "
            "same cell_ids order as this export, exact 909/909 gene match, "
            "no imputation needed."
        ),
    }
    (EXPORT_DIR / f"ccle_{split}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[{split}] deeptta gene coverage: "
          f"{len(deeptta_genes) - len(deeptta_missing) - len(deeptta_amb)}/{len(deeptta_genes)}"
          f" ({len(deeptta_missing)} missing, {len(deeptta_amb)} ambiguous)")
    print(f"[{split}] paccmann gene coverage: "
          f"{len(paccmann_genes) - len(paccmann_missing) - len(paccmann_amb)}/{len(paccmann_genes)}"
          f" ({len(paccmann_missing)} missing, {len(paccmann_amb)} ambiguous)")
    print("saved:", EXPORT_DIR / f"ccle_{split}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_CONFIG), default="mts")
    args = ap.parse_args()
    build(args.split)
