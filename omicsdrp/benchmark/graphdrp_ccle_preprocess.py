from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_alias  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = REPO_ROOT / "data" / "ccle_processed" / "graphdrp_prep"
DEPMAP_DIR = PREP_DIR / "depmap_raw"
GDSC_FEATURE_FILE = REPO_ROOT / "BenchSources" / "GraphDRP" / "data" / "PANCANCER_Genetic_feature.csv"
CNA_REGIONS_FILE = PREP_DIR / "gdsc_cnaPANCAN_regions.csv"
RNA_COMBAT = REPO_ROOT / "data" / "ccle_processed" / "RNA_combat.csv"
OUT_FILE = PREP_DIR / "CCLE_PANCANCER_Genetic_feature_binary.csv"

# DECISIONS.md #1 -- non-silent coding consequence classes (VEP VariantInfo
# tokens). VariantInfo can be an "&"-joined list of consequences; a variant
# counts if ANY token is in this set.
NONSILENT_CONSEQUENCES = {
    "missense_variant", "frameshift_variant", "stop_gained", "stop_lost",
    "start_lost", "inframe_deletion", "inframe_insertion",
    "protein_altering_variant", "splice_acceptor_variant", "splice_donor_variant",
}

# DECISIONS.md #3 -- PureCN callAlterations() default cutoffs = c(0.5, 6, 7).
CNV_DEL_MAX = 0.5      # CN < this -> deletion hit
CNV_AMP_MIN_FOCAL = 6  # CN >= this -> amplification hit (focal)
CNV_AMP_MIN_BROAD = 7  # CN >= this -> amplification hit (broad); kept separate
                        # from FOCAL for documentation -- callable as one "any" check


def _canonical_cell_order() -> list:
    idx = pd.read_csv(RNA_COMBAT, index_col=0, usecols=[0]).index
    return list(idx)


def _gdsc_735_columns() -> list:
    """The exact 735 column names/order GDSC's own pivot_table produces
    (graphdrp_adapter.py:63-67 sorts columns alphabetically by default)."""
    feat = pd.read_csv(GDSC_FEATURE_FILE)
    wide = feat.pivot_table(index="cosmic_sample_id", columns="genetic_feature",
                             values="is_mutated", aggfunc="max")
    return list(wide.columns)


def _split_feature_groups(columns: list):
    """298 single-gene mutation genes, 10 fusion pairs, 425 CNA region ids.

    HLA-A_mut / HLA-B_mut are real hyphenated gene symbols, not fusions --
    excluded from the fusion set by name (verified against GraphDRP's actual
    310 mutation features: exactly 12 contain '-', 2 of which are HLA genes).
    """
    mut_genes, fusions, cna_regions = [], [], []
    for c in columns:
        if c.startswith("cnaPANCAN"):
            cna_regions.append(c)
        else:
            gene = c[:-4]  # strip "_mut"
            if "-" in gene and gene not in ("HLA-A", "HLA-B"):
                fusions.append(gene)
            else:
                mut_genes.append(gene)
    return mut_genes, fusions, cna_regions


def _build_mutation_matrix(mut_genes: list, cell_order: list) -> pd.DataFrame:
    print(f"[mutation] resolving {len(mut_genes)} genes against DepMap HugoSymbol …")
    usecols = ["ModelID", "HugoSymbol", "VariantInfo"]
    chunks = pd.read_csv(DEPMAP_DIR / "OmicsSomaticMutations.csv", usecols=usecols,
                          chunksize=500_000)
    hugo_symbols = set()
    hits = set()  # {(ModelID, HugoSymbol)} with a qualifying non-silent variant
    n_rows = 0
    for chunk in chunks:
        n_rows += len(chunk)
        hugo_symbols.update(chunk["HugoSymbol"].dropna().unique())
        nonsilent = chunk["VariantInfo"].fillna("").apply(
            lambda v: any(tok in NONSILENT_CONSEQUENCES for tok in v.split("&")))
        sub = chunk[nonsilent]
        hits.update(zip(sub["ModelID"], sub["HugoSymbol"]))
    print(f"  scanned {n_rows} variant rows, {len(hits)} (cell,gene) non-silent hits")

    resolved, missing, ambiguous = gene_alias.resolve_genes(mut_genes, hugo_symbols)
    print(f"  gene resolution: {len(resolved)} resolved, {len(missing)} missing, "
          f"{len(ambiguous)} ambiguous")
    if missing:
        print(f"  missing genes: {missing}")

    mat = pd.DataFrame(0, index=cell_order, columns=[g + "_mut" for g in mut_genes],
                        dtype=np.float32)
    for gene, ccle_symbol in resolved.items():
        col = gene + "_mut"
        for cell in cell_order:
            if (cell, ccle_symbol) in hits:
                mat.loc[cell, col] = 1.0
    for gene in missing:
        mat[gene + "_mut"] = np.nan
    for entry in ambiguous:
        mat[entry["trained_symbol"] + "_mut"] = np.nan
    return mat, {"resolved": len(resolved), "missing": missing, "ambiguous": ambiguous}

FUSION_GENE_TYPOS = {"EWRS1": "EWSR1"}
FUSION_WILDCARDS = {"EWSR1-X"}

def _build_fusion_matrix(fusions: list, cell_order: list) -> pd.DataFrame:
    print(f"[fusion] matching {len(fusions)} fusion features against OmicsFusionFiltered.csv …")
    df = pd.read_csv(DEPMAP_DIR / "OmicsFusionFiltered.csv",
                      usecols=["ModelID", "LeftGene", "RightGene"])

    def _base_symbol(s):
        # DepMap fusion gene fields look like "BCR (ENSG...)" or "BCR"
        return re.sub(r"\s*\(.*\)$", "", str(s)).strip()

    df["LeftGene"] = df["LeftGene"].map(_base_symbol)
    df["RightGene"] = df["RightGene"].map(_base_symbol)
    fusion_symbols = set(df["LeftGene"]) | set(df["RightGene"])
    pair_hits = set(zip(df["ModelID"], df["LeftGene"], df["RightGene"]))
    pair_hits_rev = set(zip(df["ModelID"], df["RightGene"], df["LeftGene"]))

    mat = pd.DataFrame(0, index=cell_order, columns=[f + "_mut" for f in fusions],
                        dtype=np.float32)
    matched, wildcarded = [], []
    for fusion in fusions:
        if fusion in FUSION_WILDCARDS:
            wildcarded.append(fusion)
            mat[fusion + "_mut"] = np.nan
            continue
        genes = fusion.split("-")
        if len(genes) != 2:
            continue
        genes = [FUSION_GENE_TYPOS.get(g, g) for g in genes]
        resolved, _, _ = gene_alias.resolve_genes(genes, fusion_symbols)
        g1 = resolved.get(genes[0], genes[0])
        g2 = resolved.get(genes[1], genes[1])
        hit_cells = {c for c, a, b in pair_hits if a == g1 and b == g2}
        hit_cells |= {c for c, a, b in pair_hits_rev if a == g1 and b == g2}
        if hit_cells:
            matched.append(fusion)
        for c in hit_cells:
            if c in mat.index:
                mat.loc[c, fusion + "_mut"] = 1.0
    print(f"  matched {len(matched)}/{len(fusions)} fusion names in DepMap data: {matched}")
    print(f"  {len(wildcarded)} wildcard feature(s) left as NaN (not matchable): {wildcarded}")
    return mat


def _build_cnv_matrix(cell_order: list) -> pd.DataFrame:
    print("[cnv] loading OmicsAbsoluteCNGene.csv …")
    cn = pd.read_csv(DEPMAP_DIR / "OmicsAbsoluteCNGene.csv", index_col=0)
    cn.columns = [re.sub(r"\s*\(\d+\)$", "", c) for c in cn.columns]
    cn = cn.loc[cn.index.isin(cell_order)]
    cn_gene_set = set(cn.columns)

    regions = pd.read_csv(CNA_REGIONS_FILE)
    mat = pd.DataFrame(0, index=cell_order,
                        columns=[f"cna{r}" if not str(r).startswith("cna") else r
                                 for r in regions["identifier"]],
                        dtype=np.float32)
    mat.columns = regions["identifier"].tolist()

    n_resolved_regions, n_unresolved = 0, 0
    all_region_genes = set()
    for genes_str in regions["contained_genes"].dropna():
        all_region_genes.update(g.strip() for g in genes_str.split(","))
    resolved, missing, ambiguous = gene_alias.resolve_genes(sorted(all_region_genes), cn_gene_set)
    print(f"[cnv] region-gene resolution: {len(resolved)} resolved, {len(missing)} missing, "
          f"{len(ambiguous)} ambiguous")

    for _, row in regions.iterrows():
        region_id = row["identifier"]
        direction = row["direction"]
        genes = [g.strip() for g in str(row["contained_genes"]).split(",") if g.strip()]
        ccle_genes = [resolved[g] for g in genes if g in resolved]
        if not ccle_genes:
            n_unresolved += 1
            mat[region_id] = np.nan
            continue
        n_resolved_regions += 1
        sub = cn[ccle_genes]
        if direction == "Amplification":
            hit = (sub >= CNV_AMP_MIN_FOCAL).any(axis=1)
        else:  # "Deletion"
            hit = (sub < CNV_DEL_MAX).any(axis=1)
        mat.loc[hit.index, region_id] = hit.astype(np.float32).values

    print(f"  {n_resolved_regions}/{len(regions)} regions had >=1 resolvable gene "
          f"({n_unresolved} left as NaN)")
    return mat


def main():
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    cell_order = _canonical_cell_order()
    print(f"CCLE cell order: {len(cell_order)} cells (from RNA_combat.csv)")

    columns = _gdsc_735_columns()
    assert len(columns) == 735, f"expected 735 GDSC feature columns, got {len(columns)}"
    mut_genes, fusions, cna_regions = _split_feature_groups(columns)
    print(f"feature split: {len(mut_genes)} mutation genes, {len(fusions)} fusions, "
          f"{len(cna_regions)} CNA regions (total {len(mut_genes) + len(fusions) + len(cna_regions)})")
    assert len(mut_genes) + len(fusions) == 310
    assert len(cna_regions) == 425

    mut_mat, mut_report = _build_mutation_matrix(mut_genes, cell_order)
    fusion_mat = _build_fusion_matrix(fusions, cell_order)
    cnv_mat = _build_cnv_matrix(cell_order)

    full = pd.concat([mut_mat, fusion_mat, cnv_mat], axis=1)
    full = full[columns]  # exact GDSC column order
    full.to_csv(OUT_FILE)

    n_all_missing_cols = full.columns[full.isna().all()].tolist()
    n_nan_cells = int(full.isna().sum().sum())
    print("\n=== coverage summary ===")
    print(f"matrix shape: {full.shape}")
    print(f"fully-missing columns: {len(n_all_missing_cols)} / {full.shape[1]}")
    print(f"NaN cells: {n_nan_cells} / {full.size} ({100 * n_nan_cells / full.size:.2f}%)")
    print(f"mean feature density (fraction of 1s among non-NaN): "
          f"{np.nanmean(full.values):.4f}")
    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
