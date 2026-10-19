"""Cell-branch encoders (ablation axis 2: attention vs. plain MLP).

Both encoders share a **vectorised** per-gene embedding front-end and produce a
fixed ``embedding_dim`` cell representation, so they are drop-in interchangeable
in :class:`DRPModel`.

Design notes
------------
* Input is a single tensor ``[B, n_gene, n_omics]`` (not a 909-key dict), so the
  DataLoader does a cheap slice instead of building a dict per sample.
* Per-gene weights are kept (each gene has its own Linear), but computed as one
  batched einsum (:class:`PerGeneLinear`) instead of a 909-iteration Python loop
  -- same math, one GPU kernel, ~orders of magnitude faster dispatch.
* Normalisation is **LayerNorm per gene token** (batch-independent), replacing the
  original per-gene BatchNorm. For an attention-over-gene-tokens model LayerNorm
  is the standard, more stable choice; it also removes BN's train/eval
  running-stat mismatch (important for the unseen-cell split) and batch-size
  fragility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerGeneLinear(nn.Module):
    """Independent Linear per gene, vectorised.

    weight ``[n_gene, in, out]``, bias ``[n_gene, out]`` -- equivalent to n_gene
    separate ``nn.Linear(in, out)`` but evaluated as a single einsum.
    """

    def __init__(self, n_gene: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_gene, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_gene, out_dim))
        for g in range(n_gene):  # per-gene xavier, one-time at construction
            nn.init.xavier_uniform_(self.weight[g])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, G, in] -> [B, G, out]
        return torch.einsum("bgi,gio->bgo", x, self.weight) + self.bias


class GeneFrontEnd(nn.Module):
    """[B, n_gene, n_omics] -> [B, n_gene, gene_embed_dim].

    Per-gene 2-layer embedding with LayerNorm (batch-independent). Mirrors the
    original block structure (fc1 -> norm -> relu -> fc2 -> norm) with BN replaced
    by LN and the 909-way Python loop replaced by :class:`PerGeneLinear`.
    """

    def __init__(self, genes, gene_embed_dim: int, input_dim: int):
        super().__init__()
        self.genes = list(genes)
        self.n_gene = len(self.genes)
        self.fc1 = PerGeneLinear(self.n_gene, input_dim, gene_embed_dim)
        self.ln1 = nn.LayerNorm(gene_embed_dim)
        self.fc2 = PerGeneLinear(self.n_gene, gene_embed_dim, gene_embed_dim)
        self.ln2 = nn.LayerNorm(gene_embed_dim)

    def forward(self, gene_data: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(gene_data)))
        x = self.ln2(self.fc2(x))
        return x


class AttentionCellEncoder(nn.Module):
    def __init__(self, genes, n_gene, gene_embed_dim, num_heads, embedding_dim,
                 dropout, input_dim):
        super().__init__()
        self.front = GeneFrontEnd(genes, gene_embed_dim, input_dim)
        self.attention = nn.MultiheadAttention(gene_embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(gene_embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(gene_embed_dim, gene_embed_dim), nn.ReLU(),
            nn.Linear(gene_embed_dim, int(0.5 * embedding_dim)))
        self.norm2 = nn.LayerNorm(int(0.5 * embedding_dim))
        self.fc = nn.Linear(int(0.5 * embedding_dim) * n_gene, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # cached for interpretability (Stage 3)

    def forward(self, gene_data):
        g = self.front(gene_data)                      # [B, n_gene, gene_embed_dim]
        attn_out, attn_w = self.attention(g, g, g)
        x = self.norm1(g + attn_out)
        x = self.norm2(self.ff(x))
        x = x.flatten(start_dim=1)
        x = self.dropout(self.ln(self.fc(x)))
        self.last_attn = attn_w
        return x


class MLPCellEncoder(nn.Module):
    """Attention-free baseline: flatten gene embeddings -> MLP -> embedding_dim."""

    def __init__(self, genes, n_gene, gene_embed_dim, embedding_dim, dropout, input_dim):
        super().__init__()
        self.front = GeneFrontEnd(genes, gene_embed_dim, input_dim)
        self.net = nn.Sequential(
            nn.Linear(gene_embed_dim * n_gene, embedding_dim), nn.ReLU(),
            nn.LayerNorm(embedding_dim), nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout))
        self.last_attn = None

    def forward(self, gene_data):
        g = self.front(gene_data)          # [B, n_gene, gene_embed_dim]
        x = g.flatten(start_dim=1)
        return self.net(x)


def build_cell_encoder(kind: str, genes, n_gene, gene_embed_dim, num_heads,
                       embedding_dim, dropout, input_dim: int):
    """Factory. ``input_dim`` = number of selected omics modalities."""
    if kind == "attention":
        return AttentionCellEncoder(genes, n_gene, gene_embed_dim, num_heads,
                                    embedding_dim, dropout, input_dim)
    if kind == "mlp":
        return MLPCellEncoder(genes, n_gene, gene_embed_dim, embedding_dim,
                              dropout, input_dim)
    raise ValueError(f"Unknown cell_encoder {kind!r}")
