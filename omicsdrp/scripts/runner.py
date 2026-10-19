#!/usr/bin/env python
"""Staged Stage-1 runner: BASELINE -> FEATURE -> ATTENTION -> DRUG, in order.

Visual progress to stdout + <out_root>/progress.md (watch with `cat progress.md`
or `tail -f` the sweep log). Emails after each stage via mailx.

Examples
--------
    python runner.py --dataset_path ../../data --out_root ./Results \
        --email-to jmj3078@gmail.com
    python runner.py --smoke                     # tiny end-to-end check
    python runner.py --split_mode unseen_drug    # run the whole ladder unseen
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.data import load_raw                       # noqa: E402
from omicsdrp.pipeline import build_stage1_stages, run_pipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./Results")
    p.add_argument("--split_mode", default="mixed",
                   choices=["mixed", "unseen_cell", "unseen_drug"],
                   help="split regime the whole ladder runs under")
    p.add_argument("--drug_encoder", default="morgan",
                   help="reference drug encoder the whole ladder runs with")
    p.add_argument("--stages", nargs="*", default=None,
                   help="run only these stages, e.g. 0_baseline 1_feature (default: all)")
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"),
                   help="recipient for mailx notifications (or set EMAIL_TO)")
    p.add_argument("--email-per", default="stage",
                   choices=["stage", "experiment", "none"])
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (few epochs/folds) to validate end-to-end")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root,
                     split_mode=args.split_mode, drug_encoder=args.drug_encoder)
    if args.smoke:
        overrides.update(num_epochs=2, outer_folds=2, patience=2,
                         n_cluster_cell=6, n_cluster_drug=6)

    ref, stages, baseline_tag = build_stage1_stages(**overrides)
    if args.stages:
        stages = type(stages)((k, v) for k, v in stages.items() if k in args.stages)
    email_to = None if args.email_per == "none" else args.email_to

    raw = load_raw(args.dataset_path)
    run_pipeline(stages, baseline_tag, raw, out_root=args.out_root,
                 email_to=email_to, email_per=args.email_per)


if __name__ == "__main__":
    main()
