"""Self-check for the noise-substitution feature ablation.

    conda activate omicsdrp && python omicsdrp/tests/test_noise_ablation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from omicsdrp.config import ExperimentConfig
from omicsdrp.data import apply_noise_omics
from omicsdrp.pipeline import build_stage1_stages

cfg = ExperimentConfig(noise_omics=["MET", "SNP"])
assert cfg.noise_indices() == [0, 1]                      # canonical SNP,MET order
assert "noise-SNP+MET" in cfg.tag()
# order of the list must not change identity; noise must change it
assert cfg.tag() == ExperimentConfig(noise_omics=["SNP", "MET"]).tag()
assert cfg.tag() != ExperimentConfig().tag()
try:
    ExperimentConfig(omics=["RNA", "CNV"], noise_omics=["SNP"]).noise_indices()
    raise AssertionError("noise_omics outside omics must raise")
except ValueError:
    pass

x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
y = apply_noise_omics(x, [0, 1, 2, 3], [1, 3], seed=7)
assert x.shape == y.shape                                  # input_dim preserved
assert torch.equal(y[:, :, [0, 2]], x[:, :, [0, 2]])       # untouched columns
assert not torch.equal(y[:, :, 1], x[:, :, 1])             # noised columns
assert torch.equal(y, apply_noise_omics(x, [0, 1, 2, 3], [1, 3], seed=7))    # deterministic
assert not torch.equal(y, apply_noise_omics(x, [0, 1, 2, 3], [1, 3], seed=8))
assert torch.equal(x, apply_noise_omics(x, [0, 1, 2, 3], [], seed=7))        # no-op
# sub-selected omics: noise index maps to its position, not its raw column
z = apply_noise_omics(x[:, :, :2], [0, 1], [1], seed=7)
assert torch.equal(z[:, :, 0], x[:, :, 0])

_, stages, _ = build_stage1_stages()
assert len(stages["1_feature"]) == 15, len(stages["1_feature"])
tags = [c.tag() for cs in stages.values() for c in cs]
assert len(tags) == len(set(tags))                         # baseline scheduled once
print("ok:", {k: len(v) for k, v in stages.items()}, "total", len(tags))
