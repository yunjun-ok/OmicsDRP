# OmicsDRP Stage-1 sweep progress

`[############################] 2/2`  elapsed 31m39s

| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |
|---|-------|--------|--------|------|----|---------|-------|-----|
| 1 | 0_baseline | omics=SNP+MET+CNV+RNA · noise=none · cell=attention · drug=molformer · split=mixed *(baseline)* | cached | 0.9688±0.0074 | 0.8826 | 0.9398 | 5 | - |
| 2 | 2_attention | omics=SNP+MET+CNV+RNA · noise=none · cell=mlp · drug=molformer · split=mixed | done | 1.0143±0.0195 | 0.8712 | 0.9334 | 5 | 31m34s |
