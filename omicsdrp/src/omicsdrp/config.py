"""Experiment configuration for the OmicsDRP ablation skeleton.

A single ``ExperimentConfig`` fully specifies one run along every ablation axis
of Stage 1:

  1. which omics modalities feed the cell branch,
  2. the cell encoder architecture (attention vs. plain MLP),
  3. the drug representation method,
  4. the data-splitting regime (mixed / unseen-cell / unseen-drug),
  5. the nested-CV / training hyper-parameters.

The orchestrator (:mod:`run_ablations`) materialises a grid of these configs
and executes them sequentially, so adding a new ablated variant is just adding
an entry to a list -- no changes to the training engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import hashlib
import json


# Column order inside every per-gene tensor is fixed by preprocessing
# (see data/raw_data/Data_Preprocessing.ipynb, cell 9):
#     np.column_stack((snp, met, cnv, rna))
OMICS_ORDER = ["SNP", "MET", "CNV", "RNA"]
OMICS_TO_INDEX = {name: i for i, name in enumerate(OMICS_ORDER)}


def omics_to_indices(omics: List[str]) -> List[int]:
    """Map modality names (e.g. ``["RNA", "MET"]``) to column indices, kept in
    canonical ``OMICS_ORDER`` so the model input layout is deterministic."""
    unknown = [o for o in omics if o not in OMICS_TO_INDEX]
    if unknown:
        raise ValueError(f"Unknown omics {unknown}; valid = {OMICS_ORDER}")
    return sorted(OMICS_TO_INDEX[o] for o in omics)


# Recognised choices -- the registries validate against these.
CELL_ENCODERS = ("attention", "mlp")
# Drug-representation candidates (Stage-1 ablation axis 3). Three families:
#   baseline     : morgan                 -- 512-bit fingerprint -> MLP (current model)
#   from-scratch : gin, gcn               -- 2D molecular graph, trained end-to-end
#   pretrained   : chemberta, molformer   -- SMILES language models (frozen embedding)
#                  graphormer, unimol     -- graph/3D transformers (frozen embedding)
# Pretrained encoders use FROZEN, pre-extracted embeddings as fixed input; only a
# projection head is trained. (Uni-Mol replaces GROVER: better maintained via
# `unimol_tools`, stronger drug-discovery context. See notes in drug_encoders.py.)
DRUG_ENCODERS = ("morgan", "gin", "gcn", "chemberta", "molformer", "graphormer", "unimol")
DRUG_ENCODER_FAMILY = {
    "morgan": "baseline",
    "gin": "scratch_graph", "gcn": "scratch_graph",
    "chemberta": "pretrained_lm", "molformer": "pretrained_lm",
    "graphormer": "pretrained_graph", "unimol": "pretrained_graph",
}
SPLIT_MODES = ("mixed", "unseen_cell", "unseen_drug")


@dataclass
class ExperimentConfig:
    # --- identity ---
    name: str = "default"

    # --- ablation axis 1: omics combination ---
    # Single omics collapses the cell branch (input_dim == 1), so the smallest
    # meaningful baselines are 2-modality, e.g. ["RNA", "MET"] or ["RNA", "CNV"].
    omics: List[str] = field(default_factory=lambda: ["SNP", "MET", "CNV", "RNA"])

    # --- ablation axis 1b: noise ablation ---
    # Modalities whose (already-scaled) columns are REPLACED by N(0,1) noise
    # instead of being dropped. Keeps input_dim / parameter count identical to
    # the full model, so a drop in performance is attributable to the lost
    # information rather than to a smaller network -- and lets a *single*
    # modality be ablated (dropping down to one column collapses the branch).
    # Must be a subset of ``omics``; empty == no noise (the intact baseline).
    noise_omics: List[str] = field(default_factory=list)

    # --- ablation axis 2: cell encoder ---
    cell_encoder: str = "attention"          # one of CELL_ENCODERS

    # --- ablation axis 3: drug representation ---
    drug_encoder: str = "morgan"             # one of DRUG_ENCODERS

    # --- control axis: which 909 genes feed the cell branch ---
    # "pgkb"        -- the curated 909 pharmacogenes (the paper's gene set)
    # "random:<s>"  -- 909 genes sampled (seed <s>) from the NaN-free, non-
    #                  degenerate all-gene pool EXCLUDING the PGKB genes; a
    #                  size-matched negative control that keeps the architecture
    #                  and parameter count identical. See data.build_gene_dict.
    gene_set: str = "pgkb"

    # --- ablation axis 4: split regime ---
    split_mode: str = "mixed"                # one of SPLIT_MODES
    # For unseen splits: number of clusters used to stratify groups across folds
    # so a whole similarity-cluster never lands entirely in the test fold.
    # Defaults kept low so every cluster has >= outer_folds members (valid
    # stratification); raise only after inspecting scripts/inspect_clusters.py
    # output -- the k-sweep flags where singleton clusters start appearing.
    n_cluster_cell: int = 6
    n_cluster_drug: int = 8

    # --- nested CV (outer-test / inner-val) ---
    outer_folds: int = 5
    inner_val_frac: float = 0.15             # fraction of outer-train held for early stopping
    seed: int = 2024

    # --- model dims ---
    embedding_dim: int = 128
    gene_embed_dim: int = 8
    num_heads: int = 2
    dropout: float = 0.1

    # --- optimisation ---
    batch_size: int = 128
    num_workers: int = 4                     # DataLoader worker processes
    num_epochs: int = 100
    lr: float = 0.01
    weight_decay: float = 1e-4
    patience: int = 7

    # --- io ---
    dataset_path: str = "../data"
    out_root: str = "./Results"

    def omics_indices(self) -> List[int]:
        return omics_to_indices(self.omics)

    def noise_indices(self) -> List[int]:
        """Column indices (canonical OMICS_ORDER) to fill with noise."""
        extra = [o for o in self.noise_omics if o not in self.omics]
        if extra:
            raise ValueError(f"noise_omics {extra} not in omics {self.omics}")
        return omics_to_indices(self.noise_omics)

    @property
    def n_omics(self) -> int:
        return len(self.omics)

    def tag(self) -> str:
        """Short, filesystem-safe experiment tag = the EXPERIMENT IDENTITY.

        Deliberately independent of the cosmetic ``name`` (and ``out_root``): two
        configs that differ only by their stage label map to the SAME tag, so a
        shared model (e.g. the baseline that also appears as "feature=all",
        "cell=attention", "drug=morgan") is trained exactly once and reused
        everywhere via the resume/cache path -- no duplicate training.
        """
        omics_tag = "+".join(o for o in OMICS_ORDER if o in self.omics)
        noise_tag = "+".join(o for o in OMICS_ORDER if o in self.noise_omics)
        if noise_tag:
            omics_tag += f"__noise-{noise_tag}"
        if self.gene_set != "pgkb":
            omics_tag = f"{self.gene_set.replace(':', '')}__{omics_tag}"
        base = f"{omics_tag}__{self.cell_encoder}__{self.drug_encoder}__{self.split_mode}"
        ident = {k: v for k, v in asdict(self).items() if k not in ("name", "out_root")}
        # canonicalise omics so list ORDER doesn't change identity (["RNA","SNP"]
        # and ["SNP","RNA"] are the same model -> same tag -> trained once).
        ident["omics"] = self.omics_indices()
        # keep the hash backward-compatible: a NEUTRAL new field is dropped from
        # the identity so previously-trained experiments still resolve to their
        # original tag (and are reused via resume) instead of retraining.
        if self.noise_omics:
            ident["noise_omics"] = self.noise_indices()
        else:
            ident.pop("noise_omics", None)
        if self.gene_set == "pgkb":
            ident.pop("gene_set", None)
        h = hashlib.md5(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:6]
        return f"{base}__{h}"

    def to_dict(self) -> dict:
        return asdict(self)
