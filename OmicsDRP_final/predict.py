"""OmicsDRP final framework: predict LN_IC50 on new cell lines x drugs.

Final model = SNP+MET+CNV+RNA, attention cell encoder, MoLFormer drug encoder,
5-fold ensemble trained on GDSC2 (weights in ./weights).

Input directory layout (see example/depmap_prism_mts):
    omics.pth                    {"cell_ids": [N], "gene_data": {gene: float32 [N, 4]}}
                                 columns = [SNP, MET, CNV, RNA], same 909 PGKB genes
    drugs.csv                    drug_name, smiles  (row order == embedding rows)
    drug_embeddings/molformer.npy  float32 [n_drug, 768]  (ibm/MoLFormer-XL-both-10pct,
                                 mean-pooled over tokens)
    ic50.csv  (optional)         cell_id, drug_name, ln_ic50 -> pairs to score + metrics
                                 (if absent, every cell x drug pair is predicted)

Pipeline: raw omics -> per-fold z-score with the scaler saved in each fold
checkpoint (GDSC training statistics) -> fold model -> mean over 5 folds.

Usage:
    python predict.py --input example/depmap_prism_mts --out depmap_predictions.parquet
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "omicsdrp", "src"))
from omicsdrp.inference_models import InferenceEnsemble  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "example", "depmap_prism_mts"))
    ap.add_argument("--weights", default=os.path.join(HERE, "weights"))
    ap.add_argument("--out", default="predictions.parquet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    omics = torch.load(os.path.join(args.input, "omics.pth"), weights_only=False)
    cell_ids, gene_data = omics["cell_ids"], omics["gene_data"]
    drugs = pd.read_csv(os.path.join(args.input, "drugs.csv"))

    # dataset_path -> input dir, so the drug encoder reads <input>/drug_embeddings/molformer.npy
    ens = InferenceEnsemble.load(args.weights, dataset_path=args.input, device=args.device)
    missing = set(ens.folds[0]["genes"]) - set(gene_data)
    if missing:
        raise ValueError(f"{len(missing)} model genes missing from omics.pth, e.g. {sorted(missing)[:5]}")

    cell_pos = {c: i for i, c in enumerate(cell_ids)}
    drug_pos = {d: i for i, d in enumerate(drugs["drug_name"])}
    ic50_path = os.path.join(args.input, "ic50.csv")
    if os.path.exists(ic50_path):
        obs = pd.read_csv(ic50_path)
        pairs = np.stack([obs["cell_id"].map(cell_pos), obs["drug_name"].map(drug_pos)], 1).astype(int)
        ic50 = torch.full((len(cell_ids), len(drugs)), float("nan"))
        ic50[pairs[:, 0], pairs[:, 1]] = torch.tensor(obs["ln_ic50"].values, dtype=torch.float32)
    else:
        pairs = np.array([(c, d) for c in range(len(cell_ids)) for d in range(len(drugs))])
        ic50 = None

    drug_meta = pd.DataFrame({"DRUG_ID": drugs["drug_name"], "SMILE": drugs["smiles"]})
    res = ens.predict(gene_data, drug_meta, pairs, ic50=ic50)

    out = pd.DataFrame({"cell_id": [cell_ids[i] for i in pairs[:, 0]],
                        "drug_name": drugs["drug_name"].values[pairs[:, 1]],
                        "pred_ln_ic50": res["pred"]})
    if ic50 is not None:
        out["true_ln_ic50"] = res["true"]
        print("metrics:", json.dumps(res["metrics"], indent=2))
    out.to_parquet(args.out, index=False)
    print(f"saved {len(out)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
