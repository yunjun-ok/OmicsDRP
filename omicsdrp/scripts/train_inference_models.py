#!/usr/bin/env python
"""Train ensemble inference models for the 12 mixed-split Stage-1 conditions.

Separate track from the nested-CV sweep (Results/): produces K=5 deployable
models per condition (plain 5-fold CV, held-out fold = early stopping) and saves
them, with their per-fold scalers, to a SEPARATE folder (default
./InferenceModels/) so nothing collides with the nested-CV Results/.

Usage:
    conda activate omicsdrp
    cd omicsdrp/scripts
    python train_inference_models.py --dataset_path ../../data \
        --out_root ./InferenceModels --email-to jmj3078@gmail.com
    python train_inference_models.py --smoke        # tiny end-to-end check

Idempotent / resumable: a condition whose 5 fold_*.pt already exist is skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import torch

from omicsdrp.data import load_raw
from omicsdrp.pipeline import build_stage1_stages, config_label
from omicsdrp.inference_models import train_inference_models
from omicsdrp.notify import send_email


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./InferenceModels")
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"))
    p.add_argument("--email-per", choices=["condition", "all", "none"], default="all")
    p.add_argument("--smoke", action="store_true",
                   help="tiny epochs/folds for a fast end-to-end check")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        args.out_root = os.path.join(args.out_root, "_smoke")
    overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root,
                     split_mode="mixed")
    if args.smoke:
        overrides.update(num_epochs=2, outer_folds=2, patience=2, num_workers=0)

    # the 12 mixed conditions == the mixed Stage-1 grid (baseline + 3 ablation axes)
    _, stages, baseline_tag = build_stage1_stages(**overrides)
    configs = [c for cfgs in stages.values() for c in cfgs]
    total = len(configs)

    os.makedirs(args.out_root, exist_ok=True)
    print(f"\n=== Inference-model training: {total} mixed conditions "
          f"(K={overrides.get('outer_folds', 5)} models each) -> {args.out_root} ===")
    for i, c in enumerate(configs, 1):
        print(f"  {i:2d}. {config_label(c)}"
              + ("  *(baseline)*" if c.tag() == baseline_tag else ""))

    send_email(f"[OmicsDRP] Inference-model training started ({total} conditions)",
               "\n".join(config_label(c) for c in configs), args.email_to)

    raw = load_raw(args.dataset_path)

    t0 = time.time()
    for i, c in enumerate(configs, 1):
        c = replace(c, out_root=args.out_root)
        print(f"\n[{i}/{total}] {config_label(c)}  (tag={c.tag()})")
        ct0 = time.time()
        cond_dir = train_inference_models(raw, c, out_root=args.out_root, device=device)
        dur = time.time() - ct0
        print(f"[{i}/{total}] done in {dur:.0f}s -> {cond_dir}")
        if args.email_per == "condition":
            send_email(f"[OmicsDRP] Inference model {i}/{total} done",
                       f"{config_label(c)}\ntag={c.tag()}\n{dur:.0f}s\n{cond_dir}",
                       args.email_to)

    total_dur = time.time() - t0
    print(f"\nAll {total} conditions done in {total_dur/60:.1f} min -> {args.out_root}")
    if args.email_per in ("all", "condition"):
        send_email(f"[OmicsDRP] Inference-model training COMPLETE ({total} conditions)",
                   f"out_root={args.out_root}\nelapsed={total_dur/60:.1f} min",
                   args.email_to)


if __name__ == "__main__":
    main()
