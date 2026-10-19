#!/usr/bin/env python
"""Gene-set control: repeat the 3 evaluation tracks on RANDOM 909-gene sets.

Shows that the curated 909 PGKB pharmacogenes carry signal that an arbitrary
size-matched gene set does not. Architecture and parameter count are identical
(909 genes x 4 omics either way), so the only thing that changes is WHICH genes.

For each seed: mixed nested CV + unseen_cell + unseen_drug, written to the same
``Results/`` tree as everything else (tags are prefixed ``random<seed>__``).
Idempotent -- completed experiments are reused, so a killed run resumes.

    conda activate omicsdrp
    cd omicsdrp/scripts
    python run_gene_resample.py --seeds 1 2 3 4 5 --email-to jmj3078@gmail.com
    python run_gene_resample.py --smoke            # fast end-to-end check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.ablations import reference_config          # noqa: E402
from omicsdrp.data import load_raw                       # noqa: E402
from omicsdrp.experiment import run_experiment           # noqa: E402
from omicsdrp.notify import send_email                   # noqa: E402
from omicsdrp.pipeline import _fmt_dur                   # noqa: E402

SPLITS = ["mixed", "unseen_cell", "unseen_drug"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./Results")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"))
    p.add_argument("--email-per", default="seed", choices=["seed", "experiment", "none"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    email_to = None if args.email_per == "none" else args.email_to
    overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root)
    if args.smoke:
        overrides.update(num_epochs=2, outer_folds=2, patience=2)

    total = len(args.seeds) * len(args.splits)
    print(f"\n=== gene-set control: {len(args.seeds)} random draws x "
          f"{len(args.splits)} splits = {total} experiments ===")
    send_email(f"[OmicsDRP] gene-resample started ({total} experiments)",
               f"seeds={args.seeds}\nsplits={args.splits}\nout_root={args.out_root}",
               email_to)

    t0 = time.time()
    done = 0
    for seed in args.seeds:
        gene_set = f"random:{seed}"
        # built once per seed (parses ~900 MB of CSV), then cached on disk
        raw = load_raw(args.dataset_path, gene_set=gene_set)
        lines = []
        for split in args.splits:
            cfg = reference_config(name=f"gene_{gene_set}", gene_set=gene_set,
                                   split_mode=split, **overrides)
            done += 1
            print(f"\n[{done}/{total}] {gene_set} · split={split}")
            et0 = time.time()
            s = run_experiment(cfg, raw=raw)
            dur = time.time() - et0 if s.get("status") != "cached" else 0.0
            line = (f"- {gene_set} {split}: {s.get('status')} "
                    f"RMSE={s.get('rmse_mean', float('nan')):.4f} "
                    f"R2={s.get('r2_mean', float('nan')):.4f} "
                    f"Pearson={s.get('pearson_mean', float('nan')):.4f} "
                    f"[{_fmt_dur(dur) if dur else 'cached'}]")
            print("   " + line)
            lines.append(line)
            if args.email_per == "experiment":
                send_email(f"[OmicsDRP] {gene_set} {split} done", line, email_to)
        del raw
        if args.email_per == "seed":
            send_email(f"[OmicsDRP] gene-resample seed {seed} done ({done}/{total})",
                       "\n".join(lines), email_to)

    msg = f"All {total} experiments in {_fmt_dur(time.time()-t0)} -> {args.out_root}"
    print(f"\n=== {msg} ===")
    send_email(f"[OmicsDRP] gene-resample COMPLETE ({total} experiments)", msg, email_to)


if __name__ == "__main__":
    main()
