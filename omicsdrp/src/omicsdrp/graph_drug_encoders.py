"""From-scratch graph drug encoders (gin / gcn), trained END-TO-END.

Unlike the pretrained family, these build a 2D molecular graph per drug from
``drug_meta['SMILE']`` and learn the representation jointly with the DRP model
(gradient flows into the GNN). To keep the index-based Dataset/loop unchanged,
the encoder holds the 241 pre-built graphs and batches them inside ``forward``:

    forward(drug_idx: LongTensor[B]) -> Tensor[B, embedding_dim]
        batch = Batch.from_data_list([self.graphs[i] for i in drug_idx])
        ... GIN/GCN convs + global pooling -> project to embedding_dim

Graphs are built once in ``__init__`` (GraphDRP-style atom features + bonds as
undirected edges). Any SMILES that fails to parse falls back to a minimal
single-node graph (logged, never crashes). ``GRAPH_IMPLEMENTED`` is True.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Batch

# RDKit emits a C++ deprecation warning per atom for GetImplicitValence; silence
# the rdApp logger so building 241 graphs doesn't flood stderr.
RDLogger.DisableLog("rdApp.*")
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool

from .drug_encoders import BaseDrugEncoder, DrugEncoderNotImplemented

logger = logging.getLogger(__name__)

# Flipped to True now that gin/gcn are implemented + tested.
GRAPH_IMPLEMENTED = True

# GraphDRP-style atom feature construction ---------------------------------
# The set of atom symbols we one-hot; anything else collapses into "Unknown".
_ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
]


def _one_hot(value, choices, allow_unknown: bool = False):
    """One-hot encode ``value`` against ``choices``, clamping any out-of-vocab
    value to the last slot (GraphDRP behaviour -- clamp rather than raise)."""
    if value not in choices:
        value = choices[-1]
    return [1 if value == c else 0 for c in choices]


def _atom_features(atom) -> List[float]:
    """~GraphDRP atom features: symbol one-hot + degree + #H + implicit valence
    + aromaticity flag. Length is fixed across all atoms."""
    feats: List[float] = []
    feats += _one_hot(atom.GetSymbol(), _ATOM_SYMBOLS, allow_unknown=True)
    feats += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    feats += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    feats += _one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    return feats


# Feature dimension implied by _atom_features (44 + 11 + 11 + 11 + 1).
ATOM_FEATURE_DIM = len(_ATOM_SYMBOLS) + 11 + 11 + 11 + 1


def _smiles_to_data(smiles: str) -> Data:
    """Build a 2D molecular graph (torch_geometric ``Data``) from a SMILES.

    Nodes carry ``ATOM_FEATURE_DIM`` features; bonds become undirected edges
    (both directions). Returns ``None`` for an unparseable SMILES (caller falls
    back to a single-node graph).
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = np.asarray([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # single/disconnected atoms with no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=torch.from_numpy(x), edge_index=edge_index)


def _minimal_graph() -> Data:
    """A single-node, no-edge graph used when a SMILES cannot be parsed."""
    return Data(
        x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )


class GraphDrugEncoder(BaseDrugEncoder):
    """Shared GIN/GCN encoder; only the conv layer type differs by ``kind``.

    Pre-builds one molecular graph per drug (row order == drug_idx order) and,
    in ``forward``, batches the requested subset with ``Batch.from_data_list``,
    runs the conv stack + global mean pooling, and projects to ``embedding_dim``.
    """

    def __init__(self, kind: str, drug_meta: pd.DataFrame, embedding_dim: int,
                 dropout: float, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        if kind not in ("gin", "gcn"):
            raise DrugEncoderNotImplemented(kind)
        self.kind = kind
        self.output_dim = embedding_dim
        self._n_drug = len(drug_meta)

        # --- pre-build 241 graphs in drug_idx (row) order ---
        smiles_list = list(drug_meta["SMILE"])
        graphs: List[Data] = []
        self.failed_idx: List[int] = []
        for i, smi in enumerate(smiles_list):
            data = _smiles_to_data(smi)
            if data is None:
                self.failed_idx.append(i)
                logger.warning("drug_idx %d: SMILES failed to parse (%r); "
                               "using minimal single-node graph.", i, smi)
                data = _minimal_graph()
            graphs.append(data)
        self.graphs = graphs
        # Track which device the graph tensors currently live on so we only
        # move them when the model moves.
        self._graph_device = torch.device("cpu")

        in_dim = ATOM_FEATURE_DIM

        # --- conv stack ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer in range(num_layers):
            din = in_dim if layer == 0 else hidden_dim
            if kind == "gin":
                mlp = nn.Sequential(
                    nn.Linear(din, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            else:  # gcn
                self.convs.append(GCNConv(din, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # --- projection head to embedding_dim ---
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _graphs_to_device(self, device: torch.device) -> None:
        if self._graph_device != device:
            self.graphs = [
                Data(x=g.x.to(device), edge_index=g.edge_index.to(device))
                for g in self.graphs
            ]
            self._graph_device = device

    def forward(self, drug_idx: torch.LongTensor) -> torch.Tensor:
        device = drug_idx.device
        self._graphs_to_device(device)

        batch = Batch.from_data_list([self.graphs[int(i)] for i in drug_idx.tolist()])
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch_vec)          # [B, hidden_dim]
        x = self.dropout(self.layernorm(self.proj(x)))  # [B, embedding_dim]
        return x


def build_graph_drug_encoder(kind: str, drug_meta: pd.DataFrame,
                             embedding_dim: int, dropout: float) -> BaseDrugEncoder:
    """Return a trained-from-scratch GIN or GCN drug encoder.

    ``kind`` in {"gin", "gcn"}; graphs are built from ``drug_meta['SMILE']`` in
    row order (== drug_idx order).
    """
    if not GRAPH_IMPLEMENTED:
        raise DrugEncoderNotImplemented(
            f"graph drug encoder '{kind}' not implemented yet "
            f"(needs rdkit + torch-geometric; see graph_drug_encoders.py).")
    if kind not in ("gin", "gcn"):
        raise DrugEncoderNotImplemented(kind)
    return GraphDrugEncoder(kind, drug_meta, embedding_dim, dropout)
