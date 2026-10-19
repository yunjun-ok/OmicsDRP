"""Run a single ``ExperimentConfig`` end-to-end (one row of the ablation grid)."""
from __future__ import annotations

import json
import os

import torch

from .config import ExperimentConfig
from .data import load_raw
from .drug_encoders import is_implemented
from .recorder import ExperimentRecorder, is_experiment_complete
from .nested_cv import run_nested_cv, _free_gpu


def run_experiment(config: ExperimentConfig, raw=None, device: str = None,
                   resume: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = config.tag()

    # experiment-level resume: if every outer fold is already on disk, don't
    # rebuild/re-run -- just return the existing summary. Makes a killed sweep
    # restartable from exactly where it stopped.
    if resume and is_experiment_complete(config.out_root, tag, config.outer_folds):
        summary_path = os.path.join(config.out_root, tag, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {"tag": tag}
        summary["status"] = "cached"
        print(f"[skip] {tag}: already complete ({config.outer_folds} folds), reusing")
        return summary

    recorder = ExperimentRecorder(config.out_root, tag, config.to_dict())

    if not is_implemented(config.drug_encoder, config.dataset_path):
        recorder.event("skipped", reason="drug_encoder_not_implemented",
                       drug_encoder=config.drug_encoder)
        summary = recorder.finalize()
        summary["status"] = "skipped"
        print(f"[skip] {tag}: drug encoder '{config.drug_encoder}' not implemented yet")
        return summary

    if raw is None:
        raw = load_raw(config.dataset_path, gene_set=config.gene_set)
    print(f"[run ] {tag}  (genes={config.gene_set}, omics={config.omics}, noise={config.noise_omics or None}, cell={config.cell_encoder}, "
          f"drug={config.drug_encoder}, split={config.split_mode})")
    summary = run_nested_cv(raw, config, recorder, device)
    summary["status"] = "done"
    _free_gpu()
    return summary
