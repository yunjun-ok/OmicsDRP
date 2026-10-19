"""Stage-1 ablation grid.

One-factor-at-a-time (OFAT) around a reference config, which is the standard way
to attribute a performance change to a single design choice. Groups:

  * ``omics``   -- omics ablation by NOISE substitution: all 15 non-empty subsets
                   of the 4 modalities, capacity held constant (see below).
  * ``encoder`` -- attention vs. plain MLP cell encoder.
  * ``drug``    -- drug representation. The default grid compares only the
                   **frozen-representation** family (morgan baseline + the 4
                   pretrained encoders): every one is "fixed representation ->
                   trained projection head", an apples-to-apples comparison.
                   The end-to-end GNNs (gin/gcn) are intentionally EXCLUDED --
                   they train the drug encoder jointly, so "how much of the gap
                   is representation vs. extra trainable capacity" is ill-defined
                   as a fair ablation. (They remain implemented and can still be
                   run manually via an explicit config.)
  * ``split``   -- mixed / unseen-cell / unseen-drug evaluation regimes.

``build_grid(groups=...)`` lets the runner pick which groups to execute so the
"skeleton today, run sequentially later" workflow can be scoped per session.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from .config import (ExperimentConfig, DRUG_ENCODERS, DRUG_ENCODER_FAMILY,
                     OMICS_ORDER)

# Fair drug-representation ablation = frozen-representation encoders only
# (baseline morgan + pretrained). GNNs are excluded (end-to-end, not comparable).
ABLATION_DRUG_ENCODERS = [d for d in DRUG_ENCODERS
                          if DRUG_ENCODER_FAMILY[d] != "scratch_graph"]

# Feature ablation = NOISE substitution, not modality removal. Every condition
# keeps all 4 omics columns (identical input_dim / parameter count) and replaces
# the ablated modalities' scaled values with N(0,1) noise. Two reasons this beats
# dropping columns: (a) network capacity is held constant, so a metric drop is
# attributable to lost information rather than to a smaller model; (b) single
# modality ablations become possible (dropping to one column collapsed the cell
# branch). Hence the full power set: all 15 non-empty noise subsets of the 4
# modalities (noise=none is the intact baseline == the reference config).
FEATURE_NOISE_SETS = [[o] for o in OMICS_ORDER] + [
    ["SNP", "MET"], ["SNP", "CNV"], ["SNP", "RNA"],
    ["MET", "CNV"], ["MET", "RNA"], ["CNV", "RNA"],
    ["SNP", "MET", "CNV"], ["SNP", "MET", "RNA"],
    ["SNP", "CNV", "RNA"], ["MET", "CNV", "RNA"],
    ["SNP", "MET", "CNV", "RNA"],    # all-noise control (no real omics at all)
]


# The ORIGINAL drop-style ablation (column removed, not noised) was RNA-anchored,
# so the RNA-less corner of the lattice was never tested -- exactly the corner
# that asks "is RNA the only modality that matters?". These 4 subsets complete
# it: together with the 7 already-run RNA-anchored sets, all 11 subsets of size
# >= 2 exist. Size-1 sets stay out (a single column collapses the cell branch).
# NOTE: dropping columns shrinks input_dim/parameters -- that confound is why the
# main ablation uses noise substitution; these runs are a cross-check, and each
# has a capacity-matched twin in the noise ladder (e.g. drop RNA <-> noise=RNA).
DROP_OMICS_SETS_NO_RNA = [
    ["SNP", "MET"], ["SNP", "CNV"], ["MET", "CNV"], ["SNP", "MET", "CNV"],
]


def reference_config(**overrides) -> ExperimentConfig:
    base = ExperimentConfig(
        name="ref",
        omics=["SNP", "MET", "CNV", "RNA"],
        cell_encoder="attention",
        drug_encoder="morgan",
        split_mode="mixed",
    )
    return replace(base, **overrides)


def build_grid(groups: List[str] = None, **base_overrides) -> List[ExperimentConfig]:
    groups = groups or ["omics", "encoder", "drug", "split"]
    ref = reference_config(**base_overrides)
    grid: Dict[str, ExperimentConfig] = {}

    # reference always included
    grid[ref.tag()] = replace(ref, name="ref")

    if "omics" in groups:
        for noise in FEATURE_NOISE_SETS:
            c = replace(ref, name="omics", noise_omics=noise)
            grid[c.tag()] = c

    if "encoder" in groups:
        for enc in ("attention", "mlp"):
            c = replace(ref, name="encoder", cell_encoder=enc)
            grid[c.tag()] = c

    if "drug" in groups:
        for drug in ABLATION_DRUG_ENCODERS:   # morgan + 4 pretrained (no GNNs)
            c = replace(ref, name="drug", drug_encoder=drug)
            grid[c.tag()] = c

    if "split" in groups:
        for split in ("mixed", "unseen_cell", "unseen_drug"):
            c = replace(ref, name="split", split_mode=split)
            grid[c.tag()] = c

    return list(grid.values())
