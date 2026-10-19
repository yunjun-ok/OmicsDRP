#!/usr/bin/env python
"""Run the full Stage-1 ladder under BOTH unseen regimes, sequentially.

`run_stage1.py` runs one split_mode; this thin wrapper runs the whole ladder for
``unseen_cell`` then ``unseen_drug`` in one process, so a single detached
`run_sweep.sh RUNNER=run_unseen.py` invocation covers both — with the same
fold-level resume, auto-restart and email behaviour.

Cluster counts use the config defaults (n_cluster_cell=6, n_cluster_drug=8);
override only after inspecting scripts/inspect_clusters.py.

Examples
--------
    python run_unseen.py --dataset_path ../../data --out_root ./Results \
        --email-to jmj3078@gmail.com
    python run_unseen.py --smoke               # tiny end-to-end check, both modes
    python run_unseen.py --only unseen_drug    # just one of the two
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.data import load_raw                                  # noqa: E402
from omicsdrp.pipeline import build_stage1_stages, run_pipeline     # noqa: E402


UNSEEN_MODES = ["unseen_cell", "unseen_drug"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./Results")
    p.add_argument("--only", default=None, choices=UNSEEN_MODES,
                   help="run just one of the two unseen regimes (default: both)")
    p.add_argument("--n_cluster_cell", type=int, default=None,
                   help="override cluster count for unseen_cell (default: config=6)")
    p.add_argument("--n_cluster_drug", type=int, default=None,
                   help="override cluster count for unseen_drug (default: config=8)")
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"),
                   help="recipient for notifications (or set EMAIL_TO)")
    p.add_argument("--email-per", default="stage",
                   choices=["stage", "experiment", "none"])
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (few epochs/folds) to validate end-to-end")
    return p.parse_args()


def main():
    args = parse_args()
    modes = [args.only] if args.only else UNSEEN_MODES
    email_to = None if args.email_per == "none" else args.email_to

    # load the (merged, 231-drug) dataset once; reused across both regimes.
    raw = load_raw(args.dataset_path)

    for mode in modes:
        overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root,
                         split_mode=mode)
        if args.n_cluster_cell is not None:
            overrides["n_cluster_cell"] = args.n_cluster_cell
        if args.n_cluster_drug is not None:
            overrides["n_cluster_drug"] = args.n_cluster_drug
        if args.smoke:
            overrides.update(num_epochs=2, outer_folds=2, patience=2,
                             n_cluster_cell=6, n_cluster_drug=6)

        print(f"\n########## SPLIT REGIME: {mode} ##########")
        _, stages, baseline_tag = build_stage1_stages(**overrides)
        run_pipeline(stages, baseline_tag, raw, out_root=args.out_root,
                     email_to=email_to, email_per=args.email_per)


if __name__ == "__main__":
    main()
