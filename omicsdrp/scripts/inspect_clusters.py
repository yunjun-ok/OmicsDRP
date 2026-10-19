#!/usr/bin/env python
"""Visualise the clustering behind the unseen-cell / unseen-drug splits so YOU
pick the number of clusters (the split "threshold"), instead of a hidden default.

Workflow
--------
1. Run the k-sweep and look at ``<target>_k_sweep.png`` (elbow + silhouette).
2. Look at ``<target>_ood_distance_k{K}.png`` for a couple of candidate k to see
   how out-of-distribution the resulting test folds are.
3. Set ``--n_cluster_cell`` / ``--n_cluster_drug`` in your ExperimentConfig to the
   value you chose.

Examples
--------
    python inspect_clusters.py --target drug --k_min 2 --k_max 30 --chosen_k 12
    python inspect_clusters.py --target cell --k_min 2 --k_max 40 --chosen_k 20
    python inspect_clusters.py --target both                      # sweeps both
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.config import ExperimentConfig            # noqa: E402
from omicsdrp.data import load_raw                      # noqa: E402
from omicsdrp.diagnostics import run_full_diagnostics   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./ClusterDiagnostics")
    p.add_argument("--target", choices=["cell", "drug", "both"], default="both")
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=30)
    p.add_argument("--k_step", type=int, default=1)
    p.add_argument("--chosen_k_cell", type=int, default=20)
    p.add_argument("--chosen_k_drug", type=int, default=15)
    p.add_argument("--omics", nargs="*", default=["SNP", "MET", "CNV", "RNA"],
                   help="omics used to build the cell profile for clustering")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig(omics=args.omics, dataset_path=args.dataset_path)
    raw = load_raw(args.dataset_path)
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))

    targets = ["cell", "drug"] if args.target == "both" else [args.target]
    for t in targets:
        chosen = args.chosen_k_cell if t == "cell" else args.chosen_k_drug
        out_dir = os.path.join(args.out_root, t)
        run_full_diagnostics(raw, cfg, t, k_values, chosen, out_dir)

    print(f"\nOpen the PNGs under {args.out_root}/<target>/ and set "
          f"n_cluster_cell / n_cluster_drug accordingly.")


if __name__ == "__main__":
    main()
