# **OmicsDRP**
Precision medicine aims to identify optimal treatments based on the molecular profiles of individual patients. However, predicting drug response remains a major challenge due to intratumoral heterogeneity. To fully capture this biological complexity, integrating comprehensive multi-omics profiles has become essential, driving the necessity for methodologies that can leverage complex biological and chemical data modalities. In this study, we propose OmicsDRP, a dual-branch deep learning model that predicts drug response by integrating cellular multi-omics profiles and drug structures. This architecture enables each branch to learn informative representations from distinct data modalities, thereby effectively capturing cross-modality interactions governing drug sensitivity. Through cross-validation on the Genomics of Drug Sensitivity in Cancer (GDSC) dataset, OmicsDRP demonstrates consistent improvements in predictive performance over conventional machine learning methods and exhibits competitive predictive accuracy compared to existing drug response prediction models. Furthermore, comprehensive interpretability analyses—including latent embedding visualizations and feature-level SHAP analysis—reveal biologically meaningful insights aligned with known drug targets and mechanisms of action, validating the model's reliability at the molecular level. In summary, OmicsDRP presents a reliable framework for precision oncology by combining robust predictive performance with biological interpretability.
<img width="3284" height="1753" alt="Image" src="https://github.com/user-attachments/assets/54a075e6-8b3c-4354-bea8-7018ef0d4656" />

## Running Step
###### 1. Training 5-fold cross-validation
```text
python Train_5fold_CV.py \
    --batch 128 \
    --num_epochs 100 \
    --embedding_dim 128 \
    --gene_embed_dim 8 \
    --num_heads 2 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --dataset_path ${dataset_path}
```

###### 2. Save results
```text
python Save_Results.py \
    --batch 128 \
    --num_epochs 100 \
    --embedding_dim 128 \
    --gene_embed_dim 8 \
    --num_heads 2 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --dataset_path ${dataset_path}
```


## Final Framework: OmicsDRP (MoLFormer) & Inference
The final OmicsDRP framework uses all four omics (SNP, MET, CNV, RNA), the attention-based cell encoder, and frozen MoLFormer drug embeddings (`ibm/MoLFormer-XL-both-10pct`). It is a 5-fold ensemble trained on GDSC2; the weights are in [`OmicsDRP_final/weights/`](OmicsDRP_final/weights). Each fold checkpoint also stores the gene-wise scaler (mean/std from GDSC training cells) used to normalize new data.

`OmicsDRP_final/predict.py` runs the full pipeline: raw omics → per-fold normalization with the saved GDSC scaler → fold-model prediction → ensemble average.

###### Example: external validation on DepMap (CCLE omics + PRISM-MTS response)
```text
cd OmicsDRP_final
python predict.py --input example/depmap_prism_mts --out depmap_predictions.parquet
```
The example contains 666 DepMap cell lines × 172 PRISM drugs (39,433 measured pairs) and reproduces the reported external result (Pearson 0.546, Spearman 0.455).

To run on your own data, use the same layout as `example/depmap_prism_mts/`:
| File | Content |
|---|---|
| `omics.pth` | `{"cell_ids": [N], "gene_data": {gene: float32 tensor [N, 4]}}`, columns `[SNP, MET, CNV, RNA]`, same 909 PGKB genes as `data/gene_list.txt` |
| `drugs.csv` | `drug_name, smiles` (row order = embedding rows) |
| `drug_embeddings/molformer.npy` | float32 `[n_drug, 768]`, MoLFormer mean-pooled token embeddings |
| `ic50.csv` (optional) | `cell_id, drug_name, ln_ic50`; if given, only these pairs are predicted and metrics are printed, otherwise all cell × drug pairs are predicted |

The DepMap omics in the example were batch-corrected to GDSC with ComBat (R `sva`, via `rpy2`) before normalization.

## Computational Environment
* GPU: NVIDIA RTX 4090, CUDA 12 (PyTorch 2.3.0+cu121)
* Deterministic cuDNN operations were enabled during training
* Drug representations: MoLFormer `ibm/MoLFormer-XL-both-10pct` (Hugging Face)
* Batch correction: ComBat from the R `sva` package, called via `rpy2`
* PPI network analysis: STRING-DB v12.0

## Dependencies
* `python` = 3.9.19
* `pytorch` = 2.3.0
* `pandas` = 1.5.3
* `numpy` = 1.26.4
* `scipy` = 1.13.1
* `matplotlib` = 3.9.2
