"""Drug-branch encoders (ablation axis 3: drug representation method).

Common contract
---------------
Every encoder is an ``nn.Module`` that owns its per-drug representation (built
from ``drug_meta``) and maps a batch of **drug indices** to an
``embedding_dim`` vector::

    forward(drug_idx: LongTensor[B]) -> Tensor[B, embedding_dim]

Indexing by drug id (rather than passing features through the DataLoader) lets a
fingerprint encoder hold a dense table while a graph encoder holds a list of
molecular graphs -- without changing the training loop or the Dataset.

Finalised Stage-1 candidate list (7 = 1 baseline + 6 test candidates)
--------------------------------------------------------------------
Implemented today:
  * ``morgan``  -- 512-bit Morgan fingerprint -> MLP (the current model).

From-scratch graph family (trained end-to-end, gradient flows to the encoder):
  * ``gin`` / ``gcn`` -- 2D molecular graphs from SMILES (rdkit + torch-geometric).
    These are intentionally *heterogeneous* from the pretrained ones: the encoder
    holds a list of ``torch_geometric.Data`` graphs and batches them in forward.

Pretrained family (FROZEN pre-extracted embedding -> trained projection head only):
  * ``chemberta`` / ``molformer`` -- SMILES language models (HuggingFace).
  * ``graphormer`` / ``unimol``   -- graph / 3D transformers.
    Uni-Mol replaces GROVER (GROVER is unmaintained + not on HF; Uni-Mol is
    actively maintained via ``unimol_tools`` and has stronger drug-discovery
    context). NOTE: HuggingFace dropped its Graphormer integration, so Graphormer
    needs a pinned版本/standalone repo -- more friction than the LM/Uni-Mol paths.

All pretrained embeddings are produced ONCE, offline, by a per-model precompute
script -> ``data/drug_embeddings/<model>.npy`` aligned to drug index order; the
training loop only loads the table and trains a small projection. This keeps the
heavy/dirty deps (Uni-Mol conformers, pinned Graphormer) out of the train env.

Every non-morgan encoder is currently a typed stub with fixed interface, so the
orchestrator can enumerate + skip them until each is implemented.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEncoderNotImplemented(NotImplementedError):
    """Raised by stubbed encoders so the runner can skip + record, not crash."""


class BaseDrugEncoder(nn.Module):
    output_dim: int  # == embedding_dim, set by subclass

    def forward(self, drug_idx: torch.LongTensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def all_drug_embeddings(self) -> torch.Tensor:
        """Embeddings for every drug (for interpretability / caching)."""
        device = next(self.parameters()).device
        idx = torch.arange(self._n_drug, device=device)
        return self.forward(idx)


class MorganFPEncoder(BaseDrugEncoder):
    def __init__(self, drug_meta: pd.DataFrame, embedding_dim: int, dropout: float):
        super().__init__()
        fps = np.asarray(
            [[float(b) for b in str(fp).split(",")]
             for fp in drug_meta["Morgan_Fingerprint"]], dtype=np.float32)
        # non-persistent: rebuilt fresh from drug_meta at construction time, so
        # loading a checkpoint onto a different drug set (e.g. external-data
        # inference) doesn't hit a shape mismatch trying to restore this table.
        self.register_buffer("fp_table", torch.from_numpy(fps), persistent=False)  # [n_drug, 512]
        self._n_drug, input_dim = fps.shape
        self.output_dim = embedding_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.batchnorm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_idx: torch.LongTensor) -> torch.Tensor:
        x = self.fp_table[drug_idx]
        x = self.dropout(F.relu(self.batchnorm(self.fc1(x))))
        x = self.dropout(self.layernorm(self.fc2(x)))
        return x


class _StubDrugEncoder(BaseDrugEncoder):
    """Placeholder that documents what each unimplemented encoder needs."""

    # kind -> (family, frozen?, checkpoint/source, implementation note)
    REQUIREMENTS = {
        # from-scratch graph (end-to-end; NOT frozen) --------------------------
        "gin": "[scratch_graph] deps: rdkit + torch-geometric. Build 2D graphs from "
               "drug_meta['SMILE'] (atom/bond features, GraphDRP-style ~9 atom feats). "
               "GINConv stack + global mean pool -> embedding_dim. Encoder holds 241 "
               "torch_geometric.Data and batches via Batch.from_data_list in forward.",
        "gcn": "[scratch_graph] same pipeline as gin but GCNConv layers (shares ~90% "
               "of the code; only the conv type differs).",
        # pretrained language models (FROZEN embedding -> projection) ----------
        "chemberta": "[pretrained_lm, FROZEN] deps: transformers. ckpt "
                     "'DeepChem/ChemBERTa-77M-MLM'. Precompute: tokenize SMILES, "
                     "mean/CLS-pool last hidden -> data/drug_embeddings/chemberta.npy.",
        "molformer": "[pretrained_lm, FROZEN] deps: transformers (trust_remote_code). "
                     "ckpt 'ibm/MoLFormer-XL-both-10pct'. Precompute pooled embedding "
                     "-> data/drug_embeddings/molformer.npy.",
        # pretrained graph/3D transformers (FROZEN embedding -> projection) ----
        "graphormer": "[pretrained_graph, FROZEN] ckpt 'clefourrier/graphormer-base-"
                      "pcqm4mv2'. NOTE: HF removed Graphormer; pin an older transformers "
                      "or use the standalone microsoft/Graphormer repo + OGB featurizer. "
                      "Precompute graph embedding -> data/drug_embeddings/graphormer.npy.",
        "unimol": "[pretrained_graph, FROZEN] deps: unimol_tools (+ rdkit for ETKDG "
                  "conformers; 241 mols is trivial). Replaces GROVER. Extract the "
                  "molecular representation -> data/drug_embeddings/unimol.npy.",
    }

    def __init__(self, kind: str, drug_meta: pd.DataFrame, embedding_dim: int, dropout: float):
        super().__init__()
        self.kind = kind
        self._n_drug = len(drug_meta)
        self.output_dim = embedding_dim
        # keep a param so .parameters()/.to(device) work if ever constructed
        self._noop = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, drug_idx):  # pragma: no cover
        raise DrugEncoderNotImplemented(
            f"Drug encoder '{self.kind}' is a stub. To implement: {self.REQUIREMENTS[self.kind]}")


PRETRAINED_KINDS = ("chemberta", "molformer", "graphormer", "unimol")
SCRATCH_GRAPH_KINDS = ("gin", "gcn")


def build_drug_encoder(kind: str, drug_meta: pd.DataFrame, embedding_dim: int,
                       dropout: float, dataset_path: str = "../data") -> BaseDrugEncoder:
    if kind == "morgan":
        return MorganFPEncoder(drug_meta, embedding_dim, dropout)

    if kind in SCRATCH_GRAPH_KINDS:
        # end-to-end GNN, implemented in graph_drug_encoders by the subagent
        from .graph_drug_encoders import build_graph_drug_encoder
        return build_graph_drug_encoder(kind, drug_meta, embedding_dim, dropout)

    if kind in PRETRAINED_KINDS:
        # frozen pre-extracted embedding table -> projection head
        from .pretrained_drug_encoder import (embedding_path, embedding_available,
                                              PretrainedEmbeddingDrugEncoder)
        if embedding_available(dataset_path, kind):
            source_rows = (drug_meta["_source_row"].tolist()
                           if "_source_row" in drug_meta.columns else None)
            return PretrainedEmbeddingDrugEncoder(
                embedding_path(dataset_path, kind), embedding_dim, dropout,
                n_drug_expected=len(drug_meta), source_rows=source_rows)
        return _StubDrugEncoder(kind, drug_meta, embedding_dim, dropout)

    raise ValueError(f"Unknown drug_encoder {kind!r}")


def is_implemented(kind: str, dataset_path: str = "../data") -> bool:
    """True when the encoder can actually run right now.

    * ``morgan`` always.
    * pretrained kinds once ``data/drug_embeddings/<kind>.npy`` exists.
    * ``gin`` / ``gcn`` once the graph module is implemented.
    """
    if kind == "morgan":
        return True
    if kind in PRETRAINED_KINDS:
        from .pretrained_drug_encoder import embedding_available
        return embedding_available(dataset_path, kind)
    if kind in SCRATCH_GRAPH_KINDS:
        try:
            from .graph_drug_encoders import GRAPH_IMPLEMENTED
            return bool(GRAPH_IMPLEMENTED)
        except Exception:
            return False
    return False
