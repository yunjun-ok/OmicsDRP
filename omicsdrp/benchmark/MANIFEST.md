# Competitor benchmark — provenance and protocol

Re-trains published DRP models on our GDSC2 cell/drug universe under one fixed
evaluation protocol, so Table 3 can report numbers we produced rather than
numbers copied from papers with incompatible splits.

## Upstream clones

Cloned under `Benchmark/<Model>/` (gitignored — code lives upstream, our
adapters live here). Pinned commits:

| model | repo | commit |
|---|---|---|
| DeepTTA | https://github.com/jianglikun/DeepTTC | `a657d5698f84c693c17109351abbc5bfed55a55c` |
| GraphDRP | https://github.com/hauldhut/GraphDRP | `ad250651f3db95d9cd7d885740e8636d609b960f` |
| PaccMann | https://github.com/PaccMann/paccmann_predictor | `088e5598ec8a6e01ec8698470a5686eb2b9e2d33` |

Upstream sources are imported unmodified; every change lives in `adapters/`.

## The three benchmark rules

1. **Each model keeps its own native input.** No model is forced onto our
   909-gene tensor.
2. **Fixed across models:** the data split (nested CV; mixed / unseen_cell /
   unseen_drug; plus the ensemble retraining regime) and the evaluation metric.
3. **Parameter count is recorded** alongside the metrics.

## Native inputs

| model | cell input | drug input | scaling |
|---|---|---|---|
| DeepTTA | U219 RMA basal expression, 17,737 genes | ESPF subword tokens (max 50) | **none** — `Step1_getData.getRna` only slices columns |
| GraphDRP | 735 binary mutation/CNV features | SMILES → molecular graph, 78-dim atom features | **none** — binary indicators fed raw |
| PaccMann | 2,128-gene panel (`2128_genes.pkl`) | SMILES via PaccMann's pretrained SMILES language (`datasets/smiles_language/`) | pytoda `gene_expression_standardize=True`, fit on the training fold and reused for the held-out fold (`train_paccmann.py:139,170`) |

No preprocessing step was added that the original model does not perform.
Hyperparameters come from upstream defaults (DeepTTA `lr=1e-4, batch=64,
drop_last=False`; GraphDRP `lr=1e-4, batch=1024`; PaccMann from
`paccmann_v2_params.json`).

## Frozen folds

`export_folds.py` runs **once in the `omicsdrp` env** and writes
`exports/<split_mode>.npz`. Adapters only read it — they never call
`build_folds`, because the three adapter envs ship different sklearn builds and
a recomputed KMeans could silently diverge.

The export is generated from the Stage-1 baseline config (all four omics,
attention encoder, Morgan drug encoder, seed 2024, 5 outer folds) and is
**verified pair-for-pair against the stored predictions** of
`scripts/Results/SNP+MET+CNV+RNA__attention__morgan__<split>__*`. A mismatch
aborts the export. Current status:

```
[mixed]       186948 pairs, 873 cells x 231 drugs -> verified identical
[unseen_cell] 186948 pairs, 873 cells x 231 drugs -> verified identical
[unseen_drug] 186948 pairs, 873 cells x 231 drugs -> verified identical
```

Coverage of our universe by each model's native tables: DeepTTA 873/873 cells,
231/231 drugs · GraphDRP 873/873 cells, 231/231 graphs · PaccMann 873/873 cells,
2,089/2,128 panel genes. Any pair a model has no features for is dropped and the
count recorded in that fold's `config.json`.

## Regimes

| regime | trains on | early stops on | evaluated on |
|---|---|---|---|
| `nested` | inner-train | inner-val | outer-test, once |
| `ensemble` | outer-train pool (train+val) | outer-test | — |

`ensemble` mirrors `inference_models.py`: it produces deployable weights for
external inference. **Its fold scores are early-stopping-optimistic and are not
performance numbers** (`config.json` records `scores_are_performance: false`).

## Per-fold outputs

`BenchmarkResults/<model>/<regime>__<split>/fold_<k>/`

| file | contents |
|---|---|
| `model.pt` | best-early-stop state dict |
| `scaler.npz` | fold scaler (only for models that natively scale) |
| `config.json` | hyperparameters, seed used, upstream env, fold signature, dropped-pair counts |
| `predictions.parquet` | `sample_idx, drug_idx, true, pred` in ln(IC50) |
| `metrics.json` | RMSE/MAE/R²/PCC/SCC + `n_params_total` |

Metrics come from `omicsdrp.metrics.regression_metrics`, loaded by path so all
three envs score identically. Models that rescale the target internally
(PaccMann min-max) are mapped back to ln(IC50) before scoring.

Seeding replicates `nested_cv.py`: `set_seed(seed + fold)` with
`cudnn.deterministic=True, benchmark=False`.

## Running

```bash
conda run -n omicsdrp python export_folds.py          # once, verifies folds
python run_benchmark.py --smoke --device cpu          # fast end-to-end check
python run_benchmark.py --email-to you@example.com    # full sweep
python run_benchmark.py --models deeptta --splits unseen_drug
```

Jobs are fold-level resumable (a fold with `metrics.json` + `model.pt` is
skipped); `--overwrite` forces recompute. Progress is written to
`BenchmarkResults/progress.{md,json}`.
