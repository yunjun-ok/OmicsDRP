#!/usr/bin/env python
"""Sequentially execute the Stage-1 ablation grid.

Examples
--------
    # everything (stubbed drug encoders are auto-skipped and recorded):
    python run_ablations.py --dataset_path ../../data --out_root ./Results

    # only the split-regime ablation, quick smoke run:
    python run_ablations.py --groups split --smoke

Each experiment writes a self-contained folder under ``out_root`` and a row is
appended to ``out_root/ablation_summary.csv`` as it finishes, so partial runs are
never lost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd

# allow running from the scripts/ dir without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.ablations import build_grid            # noqa: E402
from omicsdrp.data import load_raw                   # noqa: E402
from omicsdrp.experiment import run_experiment       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./Results")
    p.add_argument("--groups", nargs="*", default=["omics", "encoder", "drug", "split"],
                   choices=["omics", "encoder", "drug", "split"])
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (few epochs) to validate the pipeline end-to-end")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root)
    if args.smoke:
        overrides.update(num_epochs=2, outer_folds=2, patience=2,
                         n_cluster_cell=6, n_cluster_drug=5)

    grid = build_grid(groups=args.groups, **overrides)
    print(f"Ablation grid: {len(grid)} experiments  (groups={args.groups}, smoke={args.smoke})")

    raw = load_raw(args.dataset_path)
    os.makedirs(args.out_root, exist_ok=True)
    summary_path = os.path.join(args.out_root, "ablation_summary.csv")
    rows = []
    for i, cfg in enumerate(grid, 1):
        print(f"\n=== [{i}/{len(grid)}] {cfg.tag()} ===")
        try:
            summary = run_experiment(cfg, raw=raw)
        except Exception as e:  # never let one config kill the whole sweep
            traceback.print_exc()
            summary = {"tag": cfg.tag(), "status": f"error: {e}"}
        row = {"tag": cfg.tag(), "name": cfg.name, "omics": "+".join(cfg.omics),
               "cell_encoder": cfg.cell_encoder, "drug_encoder": cfg.drug_encoder,
               "split_mode": cfg.split_mode, **summary}
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)  # checkpoint after each

    print(f"\nDone. Summary -> {summary_path}")


if __name__ == "__main__":
    main()
