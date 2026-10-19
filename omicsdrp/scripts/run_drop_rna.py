#!/usr/bin/env python
"""Complete the drop-style omics lattice: the 4 subsets WITHOUT RNA.

The original ablation always kept RNA, so "what happens when the transcriptome
is gone" was never measured. This runs the missing corner (SNP+MET, SNP+CNV,
MET+CNV, SNP+MET+CNV) across all three split regimes, using the ORIGINAL
column-removal semantics -- no noise substitution.

Read alongside the noise ladder: `drop RNA` and `noise=RNA` remove the same
information, but only the latter holds parameter count constant, so the pair
separates "lost information" from "smaller model".

    conda activate omicsdrp
    cd omicsdrp/scripts
    python run_drop_rna.py --email-to jmj3078@gmail.com     # ~12 GPU-hours
    python run_drop_rna.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omicsdrp.ablations import reference_config, DROP_OMICS_SETS_NO_RNA  # noqa: E402
from omicsdrp.data import load_raw                                       # noqa: E402
from omicsdrp.experiment import run_experiment                           # noqa: E402
from omicsdrp.notify import send_email                                   # noqa: E402
from omicsdrp.pipeline import _fmt_dur, _omics_label                     # noqa: E402

SPLITS = ["mixed", "unseen_cell", "unseen_drug"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../../data")
    p.add_argument("--out_root", default="./Results")
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--email-to", default=os.environ.get("EMAIL_TO"))
    p.add_argument("--email-per", default="omics", choices=["omics", "experiment", "none"])
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    email_to = None if args.email_per == "none" else args.email_to
    overrides = dict(dataset_path=args.dataset_path, out_root=args.out_root)
    if args.smoke:
        overrides.update(num_epochs=2, outer_folds=2, patience=2)

    raw = load_raw(args.dataset_path)
    total = len(DROP_OMICS_SETS_NO_RNA) * len(args.splits)
    print(f"\n=== drop-style RNA-less lattice: {total} experiments ===")
    send_email(f"[OmicsDRP] drop-RNA lattice started ({total} experiments)",
               f"omics sets={DROP_OMICS_SETS_NO_RNA}\nsplits={args.splits}", email_to)

    t0, done = time.time(), 0
    for omics in DROP_OMICS_SETS_NO_RNA:
        label = _omics_label(omics)
        lines = []
        for split in args.splits:
            cfg = reference_config(name=f"drop_{label}", omics=omics,
                                   split_mode=split, **overrides)
            done += 1
            print(f"\n[{done}/{total}] omics={label} · split={split}")
            et0 = time.time()
            s = run_experiment(cfg, raw=raw)
            dur = time.time() - et0 if s.get("status") != "cached" else 0.0
            line = (f"- {label} {split}: {s.get('status')} "
                    f"RMSE={s.get('rmse_mean', float('nan')):.4f} "
                    f"R2={s.get('r2_mean', float('nan')):.4f} "
                    f"[{_fmt_dur(dur) if dur else 'cached'}]")
            print("   " + line)
            lines.append(line)
        if args.email_per == "omics":
            send_email(f"[OmicsDRP] drop {label} done ({done}/{total})",
                       "\n".join(lines), email_to)

    msg = f"All {total} experiments in {_fmt_dur(time.time()-t0)} -> {args.out_root}"
    print(f"\n=== {msg} ===")
    send_email("[OmicsDRP] drop-RNA lattice COMPLETE", msg, email_to)


if __name__ == "__main__":
    main()
