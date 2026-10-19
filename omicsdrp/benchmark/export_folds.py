"""Freeze the cross-validation folds once, so every competitor sees the same split.

Run this ONCE in the ``omicsdrp`` env before any adapter. It calls the project's
own ``load_raw`` + ``build_folds`` and writes ``exports/<split_mode>.npz``. The
adapters — which live in three different conda envs with different sklearn
builds — then only ever *read* those arrays. Recomputing ``build_folds`` inside
each env would risk silently divergent KMeans results and would invalidate any
comparison against the OmicsDRP numbers we already published.

The canonical config is the Stage-1 baseline (all four omics, attention cell
encoder, Morgan drug encoder, seed 2024, 5 outer folds). That matters for the
unseen splits: their clusters are derived from the omics feature matrix, so a
different omics subset would produce different folds.

Every export is verified against the stored fold predictions of the matching
baseline run under ``scripts/Results/``. A mismatch aborts — the whole point is
that these indices are identical to the ones behind our reported metrics.

Usage:
    conda activate omicsdrp
    python export_folds.py                    # all three split modes
    python export_folds.py --split mixed      # just one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "omicsdrp" / "src"))

from omicsdrp.config import ExperimentConfig          # noqa: E402
from omicsdrp.data import load_raw, merge_duplicate_drugs  # noqa: E402
from omicsdrp.splits import build_folds               # noqa: E402

EXPORT_DIR = HERE / "exports"
RESULTS_DIR = REPO_ROOT / "omicsdrp" / "scripts" / "Results"
DATA_DIR = REPO_ROOT / "data"

# The Stage-1 baseline: this is the run whose folds our published numbers use.
BASELINE_TAG_PREFIX = "SNP+MET+CNV+RNA__attention__morgan"


def _canonical_config(split_mode: str) -> ExperimentConfig:
    """Baseline config — defaults everywhere except the split mode."""
    return ExperimentConfig(split_mode=split_mode, outer_folds=5)


def _drug_and_cell_labels():
    """Recover the row/column labels behind ``RawData.ic50``.

    ``load_raw`` drops the DataFrame after converting to a tensor, so we redo the
    same read + duplicate merge to get the cell Model_IDs and drug IDs that the
    integer indices in ``pairs`` refer to.
    """
    ic50_df = pd.read_csv(DATA_DIR / "IC50_GDSC2.csv", index_col=0)
    drug_meta = pd.read_csv(
        DATA_DIR / "TargetDrugs_with_MorganFingerprint_GDSC2_512.txt", sep="\t")
    ic50_df, merged_meta = merge_duplicate_drugs(ic50_df, drug_meta)

    cell_model_ids = np.array([str(x) for x in ic50_df.index])
    drug_ids = np.array([str(x) for x in ic50_df.columns])

    # merged_meta rows are aligned to ic50_df columns by construction
    name_col = next((c for c in ("DRUG_NAME", "Drug_Name", "NAME")
                     if c in merged_meta.columns), None)
    smile_col = next((c for c in ("SMILE", "SMILES", "Smiles")
                      if c in merged_meta.columns), None)
    if smile_col is None:
        raise RuntimeError(f"no SMILES column in drug meta: {list(merged_meta.columns)}")

    drug_names = (np.array([str(x) for x in merged_meta[name_col]])
                  if name_col else drug_ids.copy())
    drug_smiles = np.array([str(x) for x in merged_meta[smile_col]])

    if len(drug_smiles) != len(drug_ids):
        raise RuntimeError(
            f"drug meta ({len(drug_smiles)}) not aligned to IC50 columns ({len(drug_ids)})")

    # COSMIC IDs — DeepTTA and PaccMann index their native tables by COSMIC.
    cell_meta = pd.read_csv(DATA_DIR / "Cell_line_meta.csv")
    model_to_cosmic = {
        str(r["Model_ID"]): int(r["COSMIC_ID"])
        for _, r in cell_meta.iterrows() if not pd.isna(r["COSMIC_ID"])
    }
    cosmic = np.array([model_to_cosmic.get(m, -1) for m in cell_model_ids], dtype=np.int64)

    return cell_model_ids, cosmic, drug_ids, drug_names, drug_smiles


def _reference_test_pairs(split_mode: str, n_folds: int):
    """Load (sample_idx, drug_idx) test sets from the published baseline run.

    Returns ``None`` when that run isn't on disk, in which case verification is
    skipped with a loud warning rather than silently passing.
    """
    matches = sorted(RESULTS_DIR.glob(f"{BASELINE_TAG_PREFIX}__{split_mode}__*"))
    if not matches:
        return None
    folds_dir = matches[0] / "folds"
    ref = {}
    for k in range(1, n_folds + 1):
        p = folds_dir / f"fold_{k}_predictions.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        ref[k] = set(zip(df["sample_idx"].astype(int), df["drug_idx"].astype(int)))
    return matches[0].name, ref


def export(split_mode: str, verify: bool = True) -> None:
    config = _canonical_config(split_mode)
    raw = load_raw(str(DATA_DIR))
    folds = build_folds(raw, config)

    cell_model_ids, cosmic, drug_ids, drug_names, drug_smiles = _drug_and_cell_labels()
    if len(cell_model_ids) != raw.n_cell or len(drug_ids) != raw.n_drug:
        raise RuntimeError("label recovery does not match RawData shape")

    pairs = np.asarray(raw.pairs, dtype=np.int64)
    ic50 = raw.ic50.cpu().numpy()
    y = ic50[pairs[:, 0], pairs[:, 1]].astype(np.float32)

    arrays = {
        "pairs": pairs,
        "y": y,
        "cell_model_ids": cell_model_ids,
        "cell_cosmic_ids": cosmic,
        "drug_ids": drug_ids,
        "drug_names": drug_names,
        "drug_smiles": drug_smiles,
    }
    for f in folds:
        arrays[f"fold{f.fold}_train"] = np.asarray(f.train_pair_idx, dtype=np.int64)
        arrays[f"fold{f.fold}_val"] = np.asarray(f.val_pair_idx, dtype=np.int64)
        arrays[f"fold{f.fold}_test"] = np.asarray(f.test_pair_idx, dtype=np.int64)

    # ---- verification against the published baseline folds -----------------
    verdict = "skipped (baseline run not found on disk)"
    if verify:
        ref = _reference_test_pairs(split_mode, config.outer_folds)
        if ref is not None:
            tag, ref_folds = ref
            for f in folds:
                mine = set(zip(pairs[f.test_pair_idx, 0].tolist(),
                               pairs[f.test_pair_idx, 1].tolist()))
                if mine != ref_folds[f.fold]:
                    raise SystemExit(
                        f"FOLD MISMATCH [{split_mode} fold {f.fold}] vs {tag}: "
                        f"{len(mine ^ ref_folds[f.fold])} pairs differ. "
                        "Refusing to export — benchmark folds must equal the "
                        "folds behind our reported numbers.")
            verdict = f"verified identical to {tag}"
        else:
            print(f"  !! WARNING: no baseline run for {split_mode}; folds UNVERIFIED")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXPORT_DIR / f"{split_mode}.npz", **arrays)

    meta = {
        "split_mode": split_mode,
        "outer_folds": config.outer_folds,
        "seed": config.seed,
        "inner_val_frac": config.inner_val_frac,
        "n_cluster_cell": config.n_cluster_cell,
        "n_cluster_drug": config.n_cluster_drug,
        "n_cell": int(raw.n_cell),
        "n_drug": int(raw.n_drug),
        "n_pairs": int(len(pairs)),
        "baseline_tag_prefix": BASELINE_TAG_PREFIX,
        "verification": verdict,
        "fold_sizes": {
            str(f.fold): {"train": int(len(f.train_pair_idx)),
                          "val": int(len(f.val_pair_idx)),
                          "test": int(len(f.test_pair_idx))}
            for f in folds
        },
    }
    (EXPORT_DIR / f"{split_mode}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[{split_mode}] {len(pairs)} pairs, {raw.n_cell} cells x {raw.n_drug} drugs "
          f"-> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["mixed", "unseen_cell", "unseen_drug", "all"],
                    default="all")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the check against published baseline folds")
    args = ap.parse_args()

    modes = (["mixed", "unseen_cell", "unseen_drug"]
             if args.split == "all" else [args.split])
    for m in modes:
        export(m, verify=not args.no_verify)


if __name__ == "__main__":
    main()
