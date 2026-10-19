"""PaccMann (MCA) adapter — native inputs, our folds.

Native cell input : the 2,128-gene panel PaccMann ships as ``2128_genes.pkl``
                    (network-propagated STRING neighbours of drug targets),
                    standardised — pytoda's ``gene_expression_standardize=True``,
                    fit on the training fold and reused for the held-out fold
                    exactly as ``train_paccmann.py`` does.
Native drug input : SMILES tokenised with PaccMann's own SMILES language
                    (``smiles_language_chembl_gdsc_ccle.pkl``), encoded by the
                    contextual-attention MCA model.
Label             : pytoda min-max scales IC50 natively; predictions are mapped
                    back to ln(IC50) with the *stored* processing parameters
                    before scoring, so the metric matches the other models.

Everything is driven through the upstream ``DrugSensitivityDataset`` and
``MODEL_FACTORY`` rather than reimplemented, so the model sees exactly what it
would in ``examples/IC50/train_paccmann.py``.

Data caveat: our cells are GDSC U219 microarray RMA, not the RNA-seq table
PaccMann V2 was trained on, and 2,089 of the 2,128 genes exist in it (the rest
are HGNC renames). Both facts are recorded in each fold's config.json.

Run inside the ``benchmark_paccmann`` env.
"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _base  # noqa: E402
import common  # noqa: E402

UPSTREAM = common.VENDOR_DIR / "PaccMann"
sys.path.insert(0, str(UPSTREAM))

from paccmann_predictor.models import MODEL_FACTORY  # noqa: E402
from pytoda.datasets import DrugSensitivityDataset  # noqa: E402
from pytoda.smiles.smiles_language import SMILESTokenizer  # noqa: E402

DATASETS = UPSTREAM / "datasets"
PARAMS_FILE = UPSTREAM / "examples" / "IC50" / "paccmann_v2_params.json"
RMA_FILE = common.VENDOR_DIR / "our_data" / "Cell_line_RMA_proc_basalExp.txt"


class PaccMannAdapter(_base.BaseAdapter):
    name = "paccmann"

    def load_native(self, export):
        self.params = json.loads(PARAMS_FILE.read_text())
        self.default_lr = float(self.params["lr"])
        self.default_batch_size = int(self.params["batch_size"])

        with open(DATASETS / "2128_genes.pkl", "rb") as fh:
            panel = list(pickle.load(fh))

        # Cell expression restricted to PaccMann's panel, indexed by our cell ids.
        expr = pd.read_csv(RMA_FILE, sep="\t")
        expr = expr.drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS")
        expr = expr[~expr.index.duplicated(keep="first")]
        cols = [c for c in expr.columns if c.startswith("DATA.")]
        expr = expr[cols].rename(columns={c: int(c.split(".")[1]) for c in cols}).T
        expr = expr[~expr.index.duplicated(keep="first")]

        self.gene_list = [g for g in panel if g in expr.columns]
        self.n_genes_missing = len(panel) - len(self.gene_list)

        cosmic = export.cell_cosmic_ids
        self.cell_ok = np.array([c in expr.index for c in cosmic])
        self.cell_labels = np.array([f"cell_{i}" for i in range(len(cosmic))])
        rows = np.zeros((len(cosmic), len(self.gene_list)), dtype=np.float32)
        present = np.where(self.cell_ok)[0]
        rows[present] = expr.loc[cosmic[present], self.gene_list].values.astype(np.float32)
        self._expr_matrix = rows  # kept for fit-fold-only standardization below

        self.workdir = Path(tempfile.mkdtemp(prefix="paccmann_bench_"))
        self.gep_path = self.workdir / "gene_expression.csv"
        pd.DataFrame(rows, index=self.cell_labels, columns=self.gene_list).to_csv(
            self.gep_path)

        # SMILES file in PaccMann's .smi format: <smiles>\t<id>
        self.drug_labels = np.array([f"drug_{i}" for i in range(len(export.drug_smiles))])
        self.smi_path = self.workdir / "drugs.smi"
        self.smi_path.write_text("\n".join(
            f"{s}\t{lbl}" for s, lbl in zip(export.drug_smiles, self.drug_labels)))

        # PaccMann's pretrained SMILES language, loaded exactly as
        # train_paccmann.py:94 does. (The legacy .pkl in the data bundle drops
        # its special tokens under pytoda 1.1.7 — use the published folder.)
        p = self.params
        self.smiles_language = SMILESTokenizer.from_pretrained(
            str(DATASETS / "smiles_language"))
        self.smiles_language.set_encoding_transforms(
            add_start_and_stop=p.get("add_start_and_stop", True),
            padding=p.get("padding", True),
            padding_length=p.get("smiles_padding_length", None),
        )
        self.test_smiles_language = deepcopy(self.smiles_language)
        self.smiles_language.set_smiles_transforms(
            augment=p.get("augment_smiles", False),
            canonical=p.get("smiles_canonical", False),
            kekulize=p.get("smiles_kekulize", False),
            all_bonds_explicit=p.get("smiles_bonds_explicit", False),
            all_hs_explicit=p.get("smiles_all_hs_explicit", False),
            remove_bonddir=p.get("smiles_remove_bonddir", False),
            remove_chirality=p.get("smiles_remove_chirality", False),
            selfies=p.get("selfies", False),
            sanitize=p.get("selfies", False),
        )
        self.test_smiles_language.set_smiles_transforms(
            augment=False,
            canonical=p.get("test_smiles_canonical", True),
            kekulize=p.get("smiles_kekulize", False),
            all_bonds_explicit=p.get("smiles_bonds_explicit", False),
            all_hs_explicit=p.get("smiles_all_hs_explicit", False),
            remove_bonddir=p.get("smiles_remove_bonddir", False),
            remove_chirality=p.get("smiles_remove_chirality", False),
            selfies=p.get("selfies", False),
            sanitize=p.get("selfies", False),
        )
        # test_smiles_language is deterministic (augment=False), unlike
        # self.smiles_language which upstream deliberately randomises every
        # call (augment_smiles=True is real SMILES augmentation, not
        # incidental cost — must not be cached). The early-stop set is
        # re-tokenised from scratch on every epoch of every fold with only
        # 231 unique drugs, so caching this path only is a pure speedup.
        _tokenize = self.test_smiles_language.smiles_to_token_indexes
        _cache: dict = {}

        def _cached_tokenize(smiles, _cache=_cache, _tokenize=_tokenize):
            cached = _cache.get(smiles)
            if cached is None:
                cached = _tokenize(smiles)
                _cache[smiles] = cached
            return cached.clone()

        self.test_smiles_language.smiles_to_token_indexes = _cached_tokenize

        self.drug_ok = np.ones(len(export.drug_smiles), dtype=bool)
        print(f"  cells with expression: {self.cell_ok.sum()}/{len(cosmic)} | "
              f"panel genes available: {len(self.gene_list)}/{len(panel)}")

    def usable_pairs(self, export, pair_idx):
        p = export.pairs[pair_idx]
        return pair_idx[self.cell_ok[p[:, 0]]]

    def fit_scaler(self, export, fit_idx):
        # gene_expression.csv holds every cell (pytoda needs one file for all
        # folds), so pytoda's own fit-from-file would see val/test cells too.
        # Compute mean/std over the fit fold's cells ourselves and force
        # pytoda to use them via gene_expression_processing_parameters,
        # matching the leakage boundary used everywhere else in this repo.
        train_cells = export.train_cells(fit_idx)
        X = self._expr_matrix[train_cells]
        mean = X.mean(axis=0).astype(np.float64)
        std = X.std(axis=0).astype(np.float64)
        self._gep_processing = {"mean": mean.tolist(), "std": std.tolist()}
        return mean.astype(np.float32), std.astype(np.float32)

    def _write_sensitivity(self, export, pair_idx, name):
        p = export.pairs[pair_idx]
        path = self.workdir / f"{name}.csv"
        pd.DataFrame({
            "drug": self.drug_labels[p[:, 1]],
            "cell_line": self.cell_labels[p[:, 0]],
            "IC50": export.y[pair_idx],
        }).to_csv(path)  # pytoda reads this file with index_col=0
        return path

    def _dataset(self, path, language, processing=None):
        return DrugSensitivityDataset(
            drug_sensitivity_filepath=str(path),
            smi_filepath=str(self.smi_path),
            gene_expression_filepath=str(self.gep_path),
            smiles_language=language,
            gene_list=self.gene_list,
            drug_sensitivity_min_max=self.params.get("drug_sensitivity_min_max", True),
            drug_sensitivity_processing_parameters=(
                {} if processing is None else processing["drug_sensitivity"]),
            gene_expression_standardize=self.params.get(
                "gene_expression_standardize", True),
            gene_expression_min_max=self.params.get("gene_expression_min_max", False),
            gene_expression_processing_parameters=self._gep_processing,
            iterate_dataset=False,
        )

    def make_loader(self, export, pair_idx, scaler, *, batch_size, shuffle):
        name = "fit" if shuffle else f"eval_{len(pair_idx)}"
        path = self._write_sensitivity(export, pair_idx, name)

        if shuffle:
            # The training fold defines the standardisation, exactly as upstream.
            ds = self._dataset(path, self.smiles_language)
            # Size the embedding from the language the dataset actually used —
            # upstream reads smiles_language.number_of_tokens the same way.
            self._vocab_size = ds.smiles_dataset.smiles_language.number_of_tokens
            self._processing = {
                "drug_sensitivity": ds.drug_sensitivity_processing_parameters,
            }
        else:
            ds = self._dataset(path, self.test_smiles_language, self._processing)

        # drop_last=True on training only (upstream setting). Held-out loaders
        # must keep every pair and their order so predictions stay aligned with
        # the fold indices we save.
        #
        # num_workers=0 for eval/early-stop: this loader is re-iterated once
        # per epoch for early stopping, and the SMILES tokenization cache
        # above only pays off if it lives in *this* process — a forked
        # worker rebuilds (and discards) its own copy every iteration since
        # persistent_workers isn't set, silently erasing the speedup.
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
            num_workers=1 if shuffle else 0)

    def build_model(self):
        params = dict(self.params)
        params["number_of_genes"] = len(self.gene_list)
        params["smiles_vocabulary_size"] = self._vocab_size
        return MODEL_FACTORY["mca"](params)

    def forward(self, model, batch):
        smiles, gep, y = batch
        smiles = smiles.to(self.device)
        gep = gep.to(self.device)
        y = y.to(self.device).view(-1)
        pred, _ = model(smiles, gep)
        return pred.view(-1), y

    def to_native_label(self, values):
        """Undo pytoda's min-max so metrics are reported in ln(IC50)."""
        p = self._processing["drug_sensitivity"]["parameters"]
        lo, hi = float(p["min"]), float(p["max"])
        return np.asarray(values) * (hi - lo) + lo


if __name__ == "__main__":
    _base.run(PaccMannAdapter())
