import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap



HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
INFERENCE_MODELS_DIR = REPO_ROOT / "omicsdrp" / "scripts" / "InferenceModels"

sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))
from omicsdrp.data import load_raw, stack_gene_data
from omicsdrp.models import DRPModel
from omicsdrp.inference_models import InferenceEnsemble, _apply_saved_scaler


class FlatWrapper(nn.Module):
    def __init__(self, base_model, n_gene, n_omics=4):
        super().__init__()
        self.base = base_model
        self.n_gene = n_gene
        self.n_omics = n_omics

    def forward(self, x):
        drug_fp = x[:, :512]
        genes_3d = x[:, 512:].view(x.size(0), self.n_gene, self.n_omics)

        cell_emb = self.base.cell_encoder(genes_3d)

        de = self.base.drug_encoder
        d_out = de.dropout(F.relu(de.batchnorm(de.fc1(drug_fp))))
        drug_emb = de.dropout(de.layernorm(de.fc2(d_out)))

        y = self.base.response(cell_emb, drug_emb)
        return y.unsqueeze(1) if y.dim() == 1 else y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="SNP+MET+CNV+RNA__attention__morgan__mixed__c94ea3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_path", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out_dir", default=None, help="defaults to <condition>/ under this script's dir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.dataset_path, merge_duplicates=True)
    cond_dir = INFERENCE_MODELS_DIR / args.condition
    ensemble = InferenceEnsemble.load(cond_dir, device=args.device)

    genes = ensemble.folds[0]["genes"]
    n_gene = len(genes)
    n_omics = len(ensemble.folds[0]["omics_indices"])

    valid_drugs = {}
    for idx, row in raw.drug_meta.iterrows():
        drug_name = str(row['DRUG_NAME']).replace("/", "_")
        cell_idxs = raw.pairs[raw.pairs[:, 1] == idx, 0]
        
        if len(cell_idxs) > 0:
            target_fp = np.array(str(row['Morgan_Fingerprint']).split(","), dtype=np.float32)
            valid_drugs[drug_name] = {'fp': target_fp, 'cell_idxs': cell_idxs}

    print(f'Run Gradient SHAP : {args.condition}, {len(valid_drugs)} drugs, {n_gene} genes, {n_omics} omics types, {len(ensemble.folds)} folds, {len(raw.pairs)} pairs')
    n_folds = len(ensemble.folds)
    for i, ck in enumerate(ensemble.folds, start=1):
        print(f"[Fold {i}/{len(ensemble.folds)}] Loading model and scaling entire gene data...")
        
        scaled_genes = _apply_saved_scaler(raw.gene_data, genes, ck["omics_indices"], ck["scaler_mean"], ck["scaler_scale"])
        gene_tensor = stack_gene_data(scaled_genes, genes)
        
        model = DRPModel(genes, raw.drug_meta, ensemble.config).to(args.device)
        model.load_state_dict(ck["model_state"], strict=False)
        surrogate = FlatWrapper(model.eval(), n_gene, n_omics).to(args.device).eval()

        fold_dir = out_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for d_i, (drug_name, d_info) in enumerate(valid_drugs.items(), start=1):
            print(f"  -> ({d_i}/{len(valid_drugs)}) Extracting SHAP for {drug_name}...")

            cell_idxs = d_info['cell_idxs']
            N_samples = len(cell_idxs)
            drug_fp_rep = np.tile(d_info['fp'], (N_samples, 1))
            target_gene_flat = gene_tensor[cell_idxs].numpy().reshape(N_samples, -1)

            X_np = np.concatenate([drug_fp_rep, target_gene_flat], axis=1).astype(np.float32)
            X_t = torch.from_numpy(X_np).to(args.device).float()
            bg_t = X_t.clone()

            explainer = shap.GradientExplainer(surrogate, bg_t)
            shap_vals = explainer.shap_values(X_t)
            current_shap = (shap_vals[0] if isinstance(shap_vals, list) else shap_vals).squeeze(-1).astype(np.float32)

            # write per-drug results straight to their own file (accumulating in a
            # dict for 231 drugs x 5 folds is what OOM-killed the previous run)
            np.savez_compressed(fold_dir / f"{drug_name}.npz", shap=current_shap, X=X_np)

            del X_t, bg_t, current_shap, X_np

        print(f"\n[Fold {i}] Saved per-drug results to {fold_dir}\n")

        del model, surrogate, scaled_genes, gene_tensor
        torch.cuda.empty_cache()

    print("Aggregating final ensemble results and saving...")
    final_shaps = {}
    final_X = {}
    for drug_name in valid_drugs:
        shap_sum, X_sum = None, None
        for i in range(1, n_folds + 1):
            npz = np.load(out_dir / f"fold_{i}" / f"{drug_name}.npz")
            shap_sum = npz["shap"] if shap_sum is None else shap_sum + npz["shap"]
            X_sum = npz["X"] if X_sum is None else X_sum + npz["X"]
        final_shaps[drug_name] = (shap_sum / n_folds).astype(np.float32)
        final_X[drug_name] = (X_sum / n_folds).astype(np.float32)

    np.savez_compressed(out_dir / "all_drugs_ensemble_shap.npz", **final_shaps)
    np.savez_compressed(out_dir / "all_drugs_ensemble_X.npz", **final_X)



if __name__ == "__main__":
    main()