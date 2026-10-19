"""DeepTTA adapter — native inputs, our folds.

Native cell input : GDSC1000 U219 RMA basal expression, all 17,737 genes
                    (``Cell_line_RMA_proc_basalExp.txt``), indexed by COSMIC ID.
Native drug input : ESPF subword tokenisation of SMILES (max 50 tokens), fed to
                    DeepTTA's own 8-layer transformer.
Label             : ln(IC50), the same target every model in this benchmark is
                    trained on so the metric is directly comparable.

Run inside the ``benchmark_deeptta`` env.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402
import common  # noqa: E402

UPSTREAM = common.VENDOR_DIR / "DeepTTA"
sys.path.insert(0, str(UPSTREAM))

import Step3_model  # noqa: E402
from Step2_DataEncoding import DataEncoding  # noqa: E402
from Step3_model import MLP, Classifier, transformer  # noqa: E402

N_GENES = 17737


class _DS(Dataset):
    def __init__(self, tokens, masks, expr, y):
        self.tokens, self.masks, self.expr, self.y = tokens, masks, expr, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tokens[i], self.masks[i], self.expr[i], self.y[i]


def _collate(batch):
    t = torch.from_numpy(np.stack([b[0] for b in batch])).long()
    m = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    e = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    y = torch.from_numpy(np.asarray([b[3] for b in batch])).float()
    return t, m, e, y


class DeepTTAAdapter(_base.BaseAdapter):
    name = "deeptta"
    default_lr = 1e-4
    default_batch_size = 64

    def load_native(self, export):
        # Step3_model hardcodes a module-level `device` and calls .to(device)
        # inside its forwards; point it at the device we were asked to use.
        Step3_model.device = self.device

        expr = pd.read_csv(UPSTREAM / "GDSC_data" / "Cell_line_RMA_proc_basalExp.txt",
                           sep="\t")
        expr = expr.drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS")
        cols = [c for c in expr.columns if c.startswith("DATA.")]
        expr = expr[cols].rename(columns={c: int(c.split(".")[1]) for c in cols})
        expr = expr.T                       # rows = cells (COSMIC), cols = genes
        # A handful of COSMIC IDs appear twice in the GDSC table; keep the first
        # so .loc returns exactly one row per cell.
        expr = expr[~expr.index.duplicated(keep="first")]
        if expr.shape[1] != N_GENES:
            raise RuntimeError(f"expected {N_GENES} genes, got {expr.shape[1]}")

        # Align to our cell ordering; rows we have no COSMIC match for stay NaN
        # and their pairs get dropped by usable_pairs().
        cosmic = export.cell_cosmic_ids
        self.cell_ok = np.array([c in expr.index for c in cosmic])
        mat = np.zeros((len(cosmic), N_GENES), dtype=np.float32)
        present = np.where(self.cell_ok)[0]
        mat[present] = expr.loc[cosmic[present]].values.astype(np.float32)
        self.expr = mat

        # ESPF-encode our SMILES with DeepTTA's own vocabulary.
        enc = DataEncoding(str(UPSTREAM))
        tok, msk, ok = [], [], []
        for s in export.drug_smiles:
            try:
                i, m = enc._drug2emb_encoder(str(s))
                tok.append(np.asarray(i, dtype=np.int64))
                msk.append(np.asarray(m, dtype=np.float32))
                ok.append(True)
            except Exception:
                tok.append(np.zeros(50, dtype=np.int64))
                msk.append(np.zeros(50, dtype=np.float32))
                ok.append(False)
        self.tokens = np.stack(tok)
        self.masks = np.stack(msk)
        self.drug_ok = np.asarray(ok)
        print(f"  cells with expression: {self.cell_ok.sum()}/{len(cosmic)} | "
              f"drugs encoded: {self.drug_ok.sum()}/{len(export.drug_smiles)}")

    def usable_pairs(self, export, pair_idx):
        p = export.pairs[pair_idx]
        keep = self.cell_ok[p[:, 0]] & self.drug_ok[p[:, 1]]
        return pair_idx[keep]

    def fit_scaler(self, export, fit_idx):
        # Upstream DeepTTA feeds raw RMA values straight into the MLP —
        # Step1_getData.getRna only slices columns, it never standardises.
        # Adding a scaler here would change the model, so we don't.
        return None

    def make_loader(self, export, pair_idx, scaler, *, batch_size, shuffle):
        p = export.pairs[pair_idx]
        expr = self.expr[p[:, 0]]
        ds = _DS(self.tokens[p[:, 1]], self.masks[p[:, 1]], expr, export.y[pair_idx])
        # drop_last=False, matching Step3_model.train's DataLoader params.
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=_collate, drop_last=False)

    def build_model(self):
        return Classifier(transformer(), MLP())

    def forward(self, model, batch):
        t, m, e, y = batch
        t, m, e, y = (t.to(self.device), m.to(self.device),
                      e.to(self.device), y.to(self.device))
        return model((t, m), e).squeeze(-1), y


if __name__ == "__main__":
    _base.run(DeepTTAAdapter())
