"""Standalone: run GraphDRP's deployed ensemble__mixed fold models (all 4
graph encoders) on CCLE/PRISM.

Reads the same frozen, model-agnostic export every other CCLE inference
script reads (``ccle_preprocess.py``'s ``pairs``/``ic50``/drug SMILES table)
plus GraphDRP's own native cell-feature matrix built separately by
``graphdrp_ccle_preprocess.py`` (735-dim binary genomic vector -- see
``data/ccle_processed/graphdrp_prep/DECISIONS.md`` for how that matrix was
built and its coverage/limitations).

Leakage guards, matching the other ccle_infer_*.py scripts exactly:
  - Only ``ensemble__mixed`` fold weights are used -- these were trained on
    the outer-train+val pool with outer-test as the early-stopping set only
    (never the eval set), so no GDSC test fold or CCLE data touched training.
  - GraphDRP's cell features are fed raw/unscaled (no fitted scaler exists to
    leak in the first place -- binary indicators, same as the GDSC adapter).
  - CCLE features with no resolvable evidence are NaN in the stored matrix
    and filled with 0 only at inference time, mirroring GDSC's own
    ``pivot_table(...).fillna(0)`` convention (graphdrp_adapter.py:63-67) --
    not an average/statistic computed from CCLE itself.
  - Same 5-fold averaging, same predictions.parquet/metrics.json schema, same
    ``BenchmarkResults/ccle_external/`` output location as every other model.

Run inside the ``benchmark_graphdrp`` env:
    conda run -n benchmark_graphdrp python ccle_infer_graphdrp.py --split mts
    conda run -n benchmark_graphdrp python ccle_infer_graphdrp.py --split mts --variant gcn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CCLE_DIR = REPO_ROOT / "data" / "ccle_processed"
EXPORT_DIR = HERE / "exports"
GRAPHDRP_PREP_DIR = CCLE_DIR / "graphdrp_prep"
UPSTREAM = REPO_ROOT / "BenchSources" / "GraphDRP"

sys.path.insert(0, str(UPSTREAM))
from models.ginconv import GINConvNet  # noqa: E402
from models.gcn import GCNNet  # noqa: E402
from models.gat import GATNet  # noqa: E402
from models.gat_gcn import GAT_GCN  # noqa: E402
from preprocess import smile_to_graph  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))
from omicsdrp.metrics import regression_metrics  # noqa: E402

VARIANTS = {"gin": GINConvNet, "gcn": GCNNet, "gat": GATNet, "gat_gcn": GAT_GCN}


def _normalise(ln_ic50):
    """GraphDRP's target transform, preprocess.py:243 -- identical to
    graphdrp_adapter.py, duplicated here since this script never imports the
    adapter (same self-contained convention as ccle_infer_deeptta.py /
    ccle_infer_paccmann.py)."""
    return 1.0 / (1.0 + np.exp(ln_ic50) ** -0.1)


def _denormalise(y):
    y = np.clip(np.asarray(y, dtype=np.float64), 1e-6, 1 - 1e-6)
    return 10.0 * np.log(y / (1.0 - y))


def build_drug_graphs(smiles_list):
    graphs, ok = [], []
    for s in smiles_list:
        try:
            c_size, features, edge_index = smile_to_graph(str(s))
            if c_size == 0 or len(edge_index) == 0:
                raise ValueError("empty graph")
            graphs.append((
                torch.tensor(np.asarray(features), dtype=torch.float),
                torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            ))
            ok.append(True)
        except Exception:
            graphs.append(None)
            ok.append(False)
    return graphs, np.asarray(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["mts", "hts"], default="mts")
    ap.add_argument("--variant", choices=list(VARIANTS), default="gin")
    args = ap.parse_args()

    model_name = "graphdrp" if args.variant == "gin" else f"graphdrp_{args.variant}"
    ensemble_dir = HERE / "BenchmarkResults" / model_name / "ensemble__mixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.loads((EXPORT_DIR / f"ccle_{args.split}.meta.json").read_text())
    z = np.load(EXPORT_DIR / f"ccle_{args.split}.npz")
    drug_table = pd.read_csv(EXPORT_DIR / f"ccle_{args.split}_drug_meta.csv")
    cell_ids = meta["cell_ids"]

    # Native cell features: same 666-cell order as every other CCLE export
    # (both built from data.py's _canonical_cell_order() over RNA_combat.csv).
    feat_df = pd.read_csv(GRAPHDRP_PREP_DIR / "CCLE_PANCANCER_Genetic_feature_binary.csv",
                           index_col=0)
    if list(feat_df.index) != cell_ids:
        raise RuntimeError("cell order mismatch between graphdrp CCLE feature matrix "
                            "and the shared ccle_preprocess.py export")
    n_nan = int(feat_df.isna().sum().sum())
    print(f"[graphdrp/{args.variant}] cell feature matrix {feat_df.shape}, "
          f"filling {n_nan} unresolved entries at 0 (GDSC's own pivot_table "
          f"convention -- see DECISIONS.md)")
    cell_feat = feat_df.fillna(0.0).values.astype(np.float32)

    print(f"[graphdrp/{args.variant}] building molecular graphs for "
          f"{len(drug_table)} CCLE/PRISM-{args.split.upper()} drug SMILES ...")
    graphs, drug_ok = build_drug_graphs(drug_table["smiles"])
    if not drug_ok.all():
        print(f"[graphdrp/{args.variant}] WARNING: {(~drug_ok).sum()} drug(s) had "
              f"no valid molecular graph -- pairs involving them are dropped")

    pairs = z["pairs"]
    usable = drug_ok[pairs[:, 1]]
    dropped = int((~usable).sum())
    if dropped:
        print(f"[graphdrp/{args.variant}] dropping {dropped}/{len(pairs)} pairs "
              f"with no valid drug graph")
    pairs = pairs[usable]
    true = z["ic50"][pairs[:, 0], pairs[:, 1]]

    items = []
    for c, d in pairs:
        x, edge_index = graphs[d]
        items.append(Data(
            x=x, edge_index=edge_index,
            target=torch.tensor(cell_feat[c]).unsqueeze(0),
        ))
    loader = DataLoader(items, batch_size=512, shuffle=False)

    fold_dirs = sorted(ensemble_dir.glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"no ensemble__mixed folds found at {ensemble_dir} -- "
                          f"train with: python run_benchmark.py --models {model_name} "
                          f"--splits mixed --regimes ensemble")
    print(f"[graphdrp/{args.variant}] found {len(fold_dirs)} ensemble folds")

    per_fold_preds = []
    for fd in fold_dirs:
        model = VARIANTS[args.variant]().to(device)
        state = torch.load(fd / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out, _ = model(batch)
                preds.append(out.squeeze(-1).cpu().numpy())
        pred_norm = np.concatenate(preds)
        per_fold_preds.append(_denormalise(pred_norm))
        print(f"  {fd.name} done")

    per_fold_arr = np.vstack(per_fold_preds)
    pred = per_fold_arr.mean(axis=0)
    metrics = regression_metrics(true, pred)

    out_dir = HERE / "BenchmarkResults" / "ccle_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}__{args.split}"
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

    print(f"[graphdrp/{args.variant}] metrics:", json.dumps(metrics, indent=2))
    print("saved:", out_dir / f"{tag}_predictions.parquet")


if __name__ == "__main__":
    main()
