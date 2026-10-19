#!/bin/bash
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition CNV+RNA__attention__morgan__mixed__47a229
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition MET+CNV+RNA__attention__morgan__mixed__6589e6
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition MET+RNA__attention__morgan__mixed__b45cf1
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+CNV+RNA__attention__morgan__mixed__ad3e70
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__attention__chemberta__mixed__516575
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__attention__graphormer__mixed__672988
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__attention__molformer__mixed__6548ee
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__attention__morgan__mixed__c94ea3
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__attention__unimol__mixed__c79b1c
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+CNV+RNA__mlp__morgan__mixed__444b73
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+MET+RNA__attention__morgan__mixed__021640
conda run -n omicsdrp python ccle_infer_omicsdrp.py --split mts --condition SNP+RNA__attention__morgan__mixed__075302

conda run -n benchmark_deeptta   python ccle_infer_deeptta.py --split mts
conda run -n benchmark_paccmann  python ccle_infer_paccmann.py --split mts
