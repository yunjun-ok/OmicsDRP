"""Composable drug-response model.

``DRPModel`` = cell encoder (attention|mlp) + drug encoder (morgan|...) + a shared
response-prediction head. Every ablation axis is injected here, so a config maps
1:1 to a model without touching the training engine.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cell_encoders import build_cell_encoder
from .drug_encoders import build_drug_encoder, BaseDrugEncoder


class ResponsePredictionLayer(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_embedding, drug_embedding):
        x = torch.cat((cell_embedding, drug_embedding), dim=1)
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


class DRPModel(nn.Module):
    def __init__(self, genes, drug_meta, config):
        super().__init__()
        n_gene = len(genes)
        self.cell_encoder = build_cell_encoder(
            kind=config.cell_encoder, genes=genes, n_gene=n_gene,
            gene_embed_dim=config.gene_embed_dim, num_heads=config.num_heads,
            embedding_dim=config.embedding_dim, dropout=config.dropout,
            input_dim=config.n_omics)
        self.drug_encoder: BaseDrugEncoder = build_drug_encoder(
            kind=config.drug_encoder, drug_meta=drug_meta,
            embedding_dim=config.embedding_dim, dropout=config.dropout,
            dataset_path=config.dataset_path)
        self.response = ResponsePredictionLayer(config.embedding_dim, config.dropout)

    def forward(self, gene_data: dict, drug_idx: torch.LongTensor):
        cell_emb = self.cell_encoder(gene_data)
        drug_emb = self.drug_encoder(drug_idx)
        return self.response(cell_emb, drug_emb)


def count_parameters(model: nn.Module) -> int:
    """Total trainable params -- reported for Stage-2 complexity comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if isinstance(m, nn.Linear) and m.weight is not None:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        if fan_in != 0 and fan_out != 0:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
