"""Standalone: run the deployed omicsdrp InferenceEnsemble on CCLE/PRISM-MTS.

Reads the frozen export from ``ccle_preprocess.py`` (pairs, IC50 matrix, drug
Morgan-fingerprint table, cell order) plus
``data/ccle_processed/CCLE_PGKB_Gene_data_dict.pth`` directly -- the PGKB
909-gene panel has an exact 909/909 match against training (verified
separately), so no gene imputation is needed for this model.

Usage:
    conda run -n omicsdrp python ccle_infer_omicsdrp.py \\
        --condition SNP+MET+CNV+RNA__attention__morgan__mixed__c94ea3
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
INFERENCE_MODELS_DIR = REPO_ROOT / "omicsdrp" / "scripts" / "InferenceModels"

sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))
from omicsdrp.inference_models import InferenceEnsemble  # noqa: E402


def load_export(split: str):
    meta = json.loads((EXPORT_DIR / f"ccle_{split}.meta.json").read_text())
    z = np.load(EXPORT_DIR / f"ccle_{split}.npz")
    drug_table = pd.read_csv(EXPORT_DIR / f"ccle_{split}_drug_meta.csv")
    return meta, z, drug_table


def build_drug_meta(drug_table: pd.DataFrame) -> pd.DataFrame:
    """MorganFPEncoder only reads drug_meta['Morgan_Fingerprint']; row position
    is the drug index, matching the export's pairs[:, 1].

    ``_source_row`` is read by PretrainedEmbeddingDrugEncoder (chemberta/
    molformer/graphormer/unimol conditions) to remap the CCLE embedding
    tables -- computed over the full 1389-row PRISM_drug_smiles.csv -- down
    to this split's drug order (drug_encoders.py:155-159)."""
    return pd.DataFrame({
        "DRUG_ID": drug_table["prism_name"],
        "SMILE": drug_table["smiles"],
        "Morgan_Fingerprint": drug_table["morgan_fingerprint_512"],
        "_source_row": drug_table["_source_row"],
    })


OMICS_NAMES = ["SNP", "MET", "CNV", "RNA"]


def impute_missing_channels(gene_data: dict, ensemble) -> tuple:
    """Fill any (gene, omics-channel) that is NaN for every CCLE cell.
    """
    used_channels = set()
    for ck in ensemble.folds:
        used_channels.update(ck["omics_indices"])

    imputed = []
    for gene, tensor in gene_data.items():
        nan_cols = torch.isnan(tensor).any(dim=0)
        if not nan_cols.any():
            continue
        for local_ch in nan_cols.nonzero(as_tuple=True)[0].tolist():
            if local_ch not in used_channels:
                continue
            fold_means = []
            for ck in ensemble.folds:
                omics_indices = ck["omics_indices"]
                if local_ch not in omics_indices:
                    continue
                gi = ck["genes"].index(gene)
                oi = omics_indices.index(local_ch)
                fold_means.append(float(ck["scaler_mean"][gi, oi]))
            if not fold_means:
                raise RuntimeError(
                    f"{gene}/{OMICS_NAMES[local_ch]} is all-NaN in CCLE data but "
                    f"no fold's saved scaler covers omics channel {local_ch} -- "
                    f"cannot impute.")
            fill_value = sum(fold_means) / len(fold_means)
            nan_rows = torch.isnan(tensor[:, local_ch])
            tensor[nan_rows, local_ch] = fill_value
            imputed.append((gene, OMICS_NAMES[local_ch]))
    return gene_data, imputed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mts", "hts"], default="mts")
    ap.add_argument("--condition", default="SNP+MET+CNV+RNA__attention__morgan__mixed__c94ea3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(HERE / "BenchmarkResults" / "ccle_external"))
    args = ap.parse_args()

    meta, z, drug_table = load_export(args.split)
    cell_ids = meta["cell_ids"]

    gene_data = torch.load(CCLE_DIR / "CCLE_PGKB_Gene_data_dict.pth", weights_only=False)
    # sanity: this export's cell order must match the .pth's implicit row order
    pgkb_order = list(pd.read_csv(CCLE_DIR / "CNV_combat_PGKB.csv", index_col=0,
                                  usecols=[0]).index)
    if pgkb_order != cell_ids:
        raise RuntimeError("cell order mismatch between export and CCLE_PGKB_Gene_data_dict.pth")

    drug_meta = build_drug_meta(drug_table)
    pairs = z["pairs"]
    ic50 = torch.from_numpy(np.nan_to_num(z["ic50"], nan=0.0))

    cond_dir = INFERENCE_MODELS_DIR / args.condition
    ensemble = InferenceEnsemble.load(str(cond_dir), dataset_path=str(CCLE_DIR),
                                      device=args.device)
    print(f"[omicsdrp] loaded {len(ensemble.folds)}-fold ensemble from {cond_dir}")

    gene_data, imputed = impute_missing_channels(gene_data, ensemble)
    if imputed:
        print(f"[omicsdrp] imputed {len(imputed)} gene/omics-channel gaps "
              f"(entirely NaN for all {meta['n_cell']} CCLE cells) at the "
              f"GDSC training-mean, averaged across folds:")
        for gene, ch in imputed:
            print(f"    {gene} / {ch}")
    print(f"[omicsdrp] predicting {len(pairs)} CCLE/PRISM-{args.split.upper()} pairs "
          f"({meta['n_cell']} cells x {meta['n_drug']} drugs) ...")

    result = ensemble.predict(gene_data, drug_meta, pairs, ic50=ic50)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"omicsdrp__{args.condition}__{args.split}"
    out_df = pd.DataFrame({
        "cell_idx": pairs[:, 0], "drug_idx": pairs[:, 1],
        "cell_id": [cell_ids[i] for i in pairs[:, 0]],
        "drug_name": [drug_table["prism_name"].iloc[i] for i in pairs[:, 1]],
        "true": result["true"], "pred": result["pred"],
    })
    for fi, fold_pred in enumerate(result["per_fold"], start=1):
        out_df[f"pred_fold{fi}"] = fold_pred
    out_df.to_parquet(out_dir / f"{tag}_predictions.parquet", index=False)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(result["metrics"], indent=2))

    print("[omicsdrp] metrics:", json.dumps(result["metrics"], indent=2))
    print("saved:", out_dir / f"{tag}_predictions.parquet")


if __name__ == "__main__":
    main()
