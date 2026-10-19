#!/usr/bin/env python
"""Stage-1b sequential driver: noise-based feature ablation + gene-set control.

Runs, in order, in ONE resumable process (so ``run_sweep.sh`` can drive the whole
~57 GPU-hour block detached):

  1. ``runner.py``            -- mixed nested CV, feature stage = 15 noise subsets
  2. ``run_unseen.py``        -- the same ladder under unseen_cell + unseen_drug
  3. ``run_gene_resample.py`` -- 5 random 909-gene draws x 3 splits

Everything is tag-addressed and idempotent, so already-finished experiments
(baseline / cell-encoder / drug-encoder stages from earlier sweeps) are reused
rather than retrained, and a crash resumes at fold granularity.

    RUNNER=run_stage1b.py EMAIL_TO=... ./run_sweep.sh
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner                # noqa: E402
import run_unseen            # noqa: E402
import run_gene_resample     # noqa: E402

STEPS = [
    ("1/3 feature-noise ladder, mixed nested CV", runner, []),
    ("2/3 feature-noise ladder, unseen_cell + unseen_drug", run_unseen, []),
    ("3/3 gene-set control, 5 random 909-gene draws x 3 splits",
     run_gene_resample, ["--seeds", "1", "2", "3", "4", "5"]),
]


def main() -> None:
    passthrough = sys.argv[1:]          # --dataset_path / --out_root from run_sweep.sh
    for label, mod, extra in STEPS:
        print(f"\n\n{'#' * 78}\n### {label}\n{'#' * 78}\n", flush=True)
        sys.argv = ["run_stage1b"] + passthrough + extra
        mod.main()


if __name__ == "__main__":
    main()
