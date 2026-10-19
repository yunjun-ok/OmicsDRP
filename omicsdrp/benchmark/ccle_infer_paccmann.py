"""Standalone: run PaccMann's deployed ensemble__mixed fold models on CCLE/PRISM.

Reads the frozen export from ``ccle_preprocess.py`` (pairs, IC50 matrix,
gene-resolved CCLE expression matrix in PaccMann's trained gene order, drug
SMILES table). 13 of PaccMann's 2,089 trained gene columns have no CCLE
measurement (11 genuinely missing + 2 ambiguous HGNC renames, left
unresolved on purpose) -- imputed per-fold at that fold's own saved training
mean (``scaler.npz``, post scaler-leak fix), which becomes exactly 0 once
standardized.

PaccMann min-maxes the label internally; the min/max used to invert
predictions back to ln(IC50) isn't persisted per fold, so it's recomputed
here deterministically from the frozen GDSC fold export -- identical to what
pytoda computed at training time (min/max of ln(IC50) over that fold's
outer-train+val pool, the same set the adapter used as ``fit_idx``).

Run inside the ``benchmark_paccmann`` env:
    conda run -n benchmark_paccmann python ccle_infer_paccmann.py --split mts
    conda run -n benchmark_paccmann python ccle_infer_paccmann.py --split hts
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
UPSTREAM = REPO_ROOT / "Benchmark" / "PaccMann"
ENSEMBLE_DIR = HERE / "BenchmarkResults" / "paccmann" / "ensemble__mixed"
PARAMS_FILE = UPSTREAM / "examples" / "IC50" / "paccmann_v2_params.json"

sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path.insert(0, str(UPSTREAM))
from paccmann_predictor.models import MODEL_FACTORY  # noqa: E402
from pytoda.smiles.smiles_language import SMILESTokenizer  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))
from omicsdrp.metrics import regression_metrics  # noqa: E402


def build_test_smiles_language(params: dict) -> SMILESTokenizer:
    """Deterministic (augment=False, canonical=True) tokenizer, matching
    paccmann_adapter.py's ``test_smiles_language`` exactly."""
    lang = SMILESTokenizer.from_pretrained(str(UPSTREAM / "datasets" / "smiles_language"))
    lang.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=params.get("smiles_padding_length", None),
    )
    lang.set_smiles_transforms(
        augment=False,
        canonical=params.get("test_smiles_canonical", True),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    return lang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mts", "hts"], default="mts")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = json.loads(PARAMS_FILE.read_text())

    meta = json.loads((EXPORT_DIR / f"ccle_{args.split}.meta.json").read_text())
    z = np.load(EXPORT_DIR / f"ccle_{args.split}.npz")
    drug_table = pd.read_csv(EXPORT_DIR / f"ccle_{args.split}_drug_meta.csv")

    n_gene = meta["paccmann"]["n_trained_genes"]
    raw_expr = z["paccmann_expr"]  # [n_cell, n_gene], NaN for unresolved genes
    nan_cols = np.isnan(raw_expr).all(axis=0)
    print(f"[paccmann] {nan_cols.sum()} gene columns have no CCLE data "
          f"({meta['paccmann']['n_missing']} missing + {meta['paccmann']['n_ambiguous']} ambiguous) "
          f"-- imputed per-fold at that fold's own training mean")

    lang = build_test_smiles_language(params)
    print(f"[paccmann] tokenising {len(drug_table)} CCLE/PRISM-{args.split.upper()} drug SMILES ...")
    smiles_tokens = torch.stack([
        lang.smiles_to_token_indexes(str(s)) for s in drug_table["smiles"]
    ])
    vocab_size = lang.number_of_tokens

    pairs = z["pairs"]
    true = z["ic50"][pairs[:, 0], pairs[:, 1]]

    export = common.FoldExport("mixed")
    fold_dirs = sorted(ENSEMBLE_DIR.glob("fold_*"))
    print(f"[paccmann] found {len(fold_dirs)} ensemble folds")

    per_fold_preds = []
    for fd in fold_dirs:
        k = int(fd.name.split("_")[1])
        scaler = np.load(fd / "scaler.npz")
        gmean, gscale = scaler["mean"], scaler["scale"]

        expr = raw_expr.copy()
        expr[:, nan_cols] = gmean[nan_cols]  # neutral: -> 0 once standardized
        expr = (expr - gmean) / gscale

        fit_idx = export.regime_indices(k, "ensemble")["fit"]
        y_min, y_max = float(export.y[fit_idx].min()), float(export.y[fit_idx].max())

        model_params = dict(params)
        model_params["number_of_genes"] = n_gene
        model_params["smiles_vocabulary_size"] = vocab_size
        model = MODEL_FACTORY["mca"](model_params).to(device)
        state = torch.load(fd / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        preds = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                sl = pairs[i:i + bs]
                smiles_b = smiles_tokens[sl[:, 1]].to(device)
                gep_b = torch.from_numpy(expr[sl[:, 0]]).float().to(device)
                out, _ = model(smiles_b, gep_b)
                preds.append(out.view(-1).cpu().numpy())
        pred_raw = np.concatenate(preds)
        pred_ln = pred_raw * (y_max - y_min) + y_min
        per_fold_preds.append(pred_ln)
        print(f"  {fd.name} done (fit-fold ln(IC50) range [{y_min:.3f}, {y_max:.3f}])")

    per_fold_arr = np.vstack(per_fold_preds)
    pred = per_fold_arr.mean(axis=0)
    metrics = regression_metrics(true, pred)

    out_dir = HERE / "BenchmarkResults" / "ccle_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_ids = meta["cell_ids"]
    tag = f"paccmann__{args.split}"
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

    print("[paccmann] metrics:", json.dumps(metrics, indent=2))
    print("saved:", out_dir / f"{tag}_predictions.parquet")


if __name__ == "__main__":
    main()
