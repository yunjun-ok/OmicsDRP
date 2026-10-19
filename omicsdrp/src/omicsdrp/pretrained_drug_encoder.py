"""Generic consumer of pre-extracted, FROZEN pretrained drug embeddings.

Shared contract for all pretrained drug encoders (chemberta / molformer /
graphormer / unimol): a per-model offline script writes a ready-to-train table

    data/drug_embeddings/<model>.npy   float32, shape [n_drug, D], row i == drug_idx i

and this encoder loads it as a frozen buffer and trains only a small projection
head to ``embedding_dim``. Because the table is frozen, no gradient flows into
the pretrained model and the heavy extraction deps never touch the train env.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .drug_encoders import BaseDrugEncoder


def embedding_path(dataset_path: str, kind: str) -> str:
    return os.path.join(dataset_path, "drug_embeddings", f"{kind}.npy")


def embedding_available(dataset_path: str, kind: str) -> bool:
    return os.path.isfile(embedding_path(dataset_path, kind))


class PretrainedEmbeddingDrugEncoder(BaseDrugEncoder):
    def __init__(self, emb_path: str, embedding_dim: int, dropout: float,
                 n_drug_expected: int | None = None, source_rows=None):
        super().__init__()
        table = np.load(emb_path).astype(np.float32)
        if table.ndim != 2:
            raise ValueError(f"{emb_path}: expected 2D [n_drug, D], got {table.shape}")
        # Embedding tables are stored in the ORIGINAL (pre-merge) drug order. When
        # duplicate drugs were merged at the data level, select the representative
        # rows so the table aligns with the merged drug_idx order.
        if source_rows is not None and table.shape[0] != n_drug_expected:
            table = table[np.asarray(source_rows, dtype=int)]
        if n_drug_expected is not None and table.shape[0] != n_drug_expected:
            raise ValueError(
                f"{emb_path}: has {table.shape[0]} rows but dataset has "
                f"{n_drug_expected} drugs; table must be aligned to drug_idx order.")
        # non-persistent: rebuilt fresh from drug_meta at construction time, so
        # loading a checkpoint onto a different drug set (e.g. external-data
        # inference) doesn't hit a shape mismatch trying to restore this table.
        self.register_buffer("emb_table", torch.from_numpy(table), persistent=False)
        self._n_drug, input_dim = table.shape
        self.output_dim = embedding_dim

        # projection head (the only trained part), mirroring the Morgan head so
        # the drug branches are comparable across representations.
        self.fc1 = nn.Linear(input_dim, 256)
        self.batchnorm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_idx: torch.LongTensor) -> torch.Tensor:
        x = self.emb_table[drug_idx]
        x = self.dropout(F.relu(self.batchnorm(self.fc1(x))))
        x = self.dropout(self.layernorm(self.fc2(x)))
        return x


def load_meta(dataset_path: str, kind: str) -> dict:
    p = embedding_path(dataset_path, kind).replace(".npy", ".meta.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}
