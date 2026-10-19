"""Standalone: run DeepTTA's deployed ensemble__mixed fold models on CCLE/PRISM.

Reads the frozen export from ``ccle_preprocess.py`` (pairs, IC50 matrix,
gene-resolved CCLE expression matrix in DeepTTA's trained gene order, drug
SMILES table). DeepTTA feeds raw (unscaled) expression directly -- 639 of its
17,737 trained gene columns have no CCLE measurement at all (617 genuinely
missing + 22 ambiguous HGNC renames, left unresolved on purpose), so those
columns are imputed at the GDSC training-set mean (computed once here from
the same RMA table DeepTTA was trained on) before scoring.

Run inside the ``benchmark_deeptta`` env:
    conda run -n benchmark_deeptta python ccle_infer_deeptta.py --split mts
    conda run -n benchmark_deeptta python ccle_infer_deeptta.py --split hts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CCLE_DIR = REPO_ROOT / "data" / "ccle_processed"
EXPORT_DIR = HERE / "exports"
GDSC_RMA = REPO_ROOT / "Benchmark" / "our_data" / "Cell_line_RMA_proc_basalExp.txt"
UPSTREAM = REPO_ROOT / "Benchmark" / "DeepTTA"
ENSEMBLE_DIR = HERE / "BenchmarkResults" / "deeptta" / "ensemble__mixed"

sys.path.insert(0, str(UPSTREAM))
import Step3_model  # noqa: E402
from Step2_DataEncoding import DataEncoding  # noqa: E402
from Step3_model import MLP, Classifier, transformer  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))
from omicsdrp.metrics import regression_metrics  # noqa: E402


def gdsc_training_mean(n_gene: int) -> np.ndarray:
    """Per-column-position mean over all GDSC training cells.

    Positional, not symbol-based: DeepTTA's adapter never deduplicates gene
    symbols (only cell IDs, post-transpose), and ~318 columns have no gene
    symbol at all (see ccle_preprocess.py's _deeptta_raw_gene_columns), so a
    symbol lookup can't address every column. Reproduces the exact same
    (cell-deduped, transposed) frame construction to guarantee identical
    column order/count to what the model was trained on.
    """
    gdsc = pd.read_csv(GDSC_RMA, sep="\t")
    gdsc = gdsc.drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS")
    cols = [c for c in gdsc.columns if c.startswith("DATA.")]
    expr = gdsc[cols].T
    expr = expr[~expr.index.duplicated(keep="first")]
    assert expr.shape[1] == n_gene, f"expected {n_gene} gene columns, got {expr.shape[1]}"
    return expr.mean(axis=0).values.astype(np.float32)


def encode_drugs(smiles_list) -> tuple:
    enc = DataEncoding(str(UPSTREAM))
    tokens, masks = [], []
    for s in smiles_list:
        i, m = enc._drug2emb_encoder(str(s))
        tokens.append(np.asarray(i, dtype=np.int64))
        masks.append(np.asarray(m, dtype=np.float32))
    return np.stack(tokens), np.stack(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mts", "hts"], default="mts")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Step3_model.device = device

    meta = json.loads((EXPORT_DIR / f"ccle_{args.split}.meta.json").read_text())
    z = np.load(EXPORT_DIR / f"ccle_{args.split}.npz")
    drug_table = pd.read_csv(EXPORT_DIR / f"ccle_{args.split}_drug_meta.csv")

    gene_order = meta["deeptta"]["gene_order"]
    expr = z["deeptta_expr"].copy()  # [n_cell, n_gene], NaN for unresolved genes
    nan_cols = np.isnan(expr).all(axis=0)
    print(f"[deeptta] imputing {nan_cols.sum()} gene columns at GDSC training mean "
          f"({meta['deeptta']['n_missing']} missing + {meta['deeptta']['n_ambiguous']} ambiguous)")
    fill = gdsc_training_mean(len(gene_order))
    expr[:, nan_cols] = fill[nan_cols]
    assert not np.isnan(expr).any(), "unexpected residual NaN after imputation"

    print(f"[deeptta] tokenising {len(drug_table)} CCLE/PRISM-{args.split.upper()} drug SMILES ...")
    tokens, masks = encode_drugs(drug_table["smiles"])

    pairs = z["pairs"]
    true = z["ic50"][pairs[:, 0], pairs[:, 1]]

    fold_dirs = sorted(ENSEMBLE_DIR.glob("fold_*"))
    print(f"[deeptta] found {len(fold_dirs)} ensemble folds")
    per_fold_preds = []
    for fd in fold_dirs:
        model = Classifier(transformer(), MLP()).to(device)
        state = torch.load(fd / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        preds = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                sl = pairs[i:i + bs]
                t = torch.from_numpy(tokens[sl[:, 1]]).long().to(device)
                m = torch.from_numpy(masks[sl[:, 1]]).float().to(device)
                e = torch.from_numpy(expr[sl[:, 0]]).float().to(device)
                out = model((t, m), e).squeeze(-1)
                preds.append(out.cpu().numpy())
        per_fold_preds.append(np.concatenate(preds))
        print(f"  {fd.name} done")

    per_fold_arr = np.vstack(per_fold_preds)
    pred = per_fold_arr.mean(axis=0)
    metrics = regression_metrics(true, pred)

    out_dir = HERE / "BenchmarkResults" / "ccle_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_ids = meta["cell_ids"]
    tag = f"deeptta__{args.split}"
    out_df = pd.DataFrame({
        "cell_idx": pairs[:, 0], "drug_idx": pairs[:, 1],
        "cell_id": [cell_ids[i] for i in pairs[:, 0]],
        "drug_name": [drug_table["prism_name"].iloc[i] for i in pairs[:, 1]],
        "true": true, "pred": pred,
    })
    for fi, fold_pred in enumerate(per_fold_arr, start=1):
        out_df[f"pred_fold{fi}"] = fold_pred
    out_df.to_parquet(out_dir / f"{tag}_predictions.parquet", index=False)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("[deeptta] metrics:", json.dumps(metrics, indent=2))
    print("saved:", out_dir / f"{tag}_predictions.parquet")


if __name__ == "__main__":
    main()
