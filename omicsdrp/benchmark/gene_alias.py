"""HGNC gene-symbol alias resolution for cross-cohort (CCLE) gene matching.

GDSC's RMA / PaccMann gene lists were fixed years ago and use older
HGNC-approved symbols for a meaningful fraction of genes; CCLE's harmonized
tables use current symbols. A raw string match under-counts real coverage --
this module resolves a trained gene list against a target column set via
HGNC's own prev_symbol/alias_symbol history, trusting a rename only when it
is unambiguous. A historical alias reused by more than one gene over HGNC's
history (e.g. a withdrawn symbol later reassigned to an unrelated gene) is
left unresolved rather than silently guessed -- a prior pass that trusted any
match found 'HGF' spuriously resolving to 'SOS1'.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ccle_processed"
ALIAS_MAP_CACHE = CACHE_DIR / "hgnc_alias_map.json"


def build_alias_map(hgnc_tsv_path) -> dict:
    """Parse HGNC's complete gene set into {symbol_or_alias: [current_approved_symbols]}."""
    alias_map: dict = {}
    with open(hgnc_tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            alias_map.setdefault(sym, set()).add(sym)
            for field in ("prev_symbol", "alias_symbol"):
                for tok in (row.get(field) or "").replace('"', "").split("|"):
                    tok = tok.strip()
                    if tok:
                        alias_map.setdefault(tok, set()).add(sym)
    return {k: sorted(v) for k, v in alias_map.items()}


def load_alias_map(cache_path=ALIAS_MAP_CACHE, hgnc_url=HGNC_URL, force=False) -> dict:
    """Load the cached alias map, building it from a fresh HGNC download if absent.

    Not committed to git (data/ is gitignored for *.json) -- regenerated once
    per machine and reused after that.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())
    tmp_tsv = cache_path.with_suffix(".tsv.tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(hgnc_url, tmp_tsv)
    alias_map = build_alias_map(tmp_tsv)
    cache_path.write_text(json.dumps(alias_map))
    tmp_tsv.unlink(missing_ok=True)
    return alias_map


def resolve_genes(trained_genes, target_columns, alias_map=None):
    """Map each trained gene symbol onto a column in ``target_columns``.

    Returns (resolved, missing, ambiguous):
      resolved  -- {trained_symbol: target_column}, identity or unambiguous rename.
      missing   -- [trained_symbol, ...] with no candidate at all.
      ambiguous -- [{'trained_symbol':..., 'candidates':[...]}] -- the historical
                   alias was reused by more than one gene and at least one
                   candidate exists in target_columns; not auto-trusted.
    """
    if alias_map is None:
        alias_map = load_alias_map()
    target_set = set(target_columns)
    resolved, missing, ambiguous = {}, [], []
    for g in trained_genes:
        if g in target_set:
            resolved[g] = g
            continue
        candidates = alias_map.get(g, [])
        hits = [c for c in candidates if c in target_set]
        if len(candidates) == 1 and hits:
            resolved[g] = hits[0]
        elif hits:
            ambiguous.append({"trained_symbol": g, "candidates": candidates})
        else:
            missing.append(g)
    return resolved, missing, ambiguous
