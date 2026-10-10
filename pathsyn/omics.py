"""DepMap omics loading + pathway-level feature construction.

DepMap matrix files have gene columns named like  'TP53 (7157)'.
Rows are cell lines (ModelID, e.g. ACH-000001).
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .utils import norm_name

GENE_RE = re.compile(r"^(.+?) \((\d+)\)$")
CODING_RE = r"MISSENSE|NONSENSE|FRAME|SPLICE|NONSTOP|START_CODON|TRANSLATION"


def read_depmap_matrix(path: str) -> pd.DataFrame:
    """Read a DepMap wide matrix (expression / CN) -> DataFrame[ModelID x gene symbol]."""
    df = pd.read_csv(path, low_memory=False)
    if "IsDefaultEntryForModel" in df.columns:
        flag = df["IsDefaultEntryForModel"].astype(str).str.lower()
        df = df[flag.isin(["yes", "true"])]
    id_col = "ModelID" if "ModelID" in df.columns else df.columns[0]
    df = df.set_index(id_col)
    df = df[~df.index.duplicated()]
    gene_cols = [c for c in df.columns if GENE_RE.match(str(c))]
    df = df[gene_cols]
    df.columns = [GENE_RE.match(c).group(1) for c in gene_cols]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.astype(np.float32)


def read_mutation_matrix(path: str):
    """Long-format DepMap OmicsSomaticMutations.csv -> (binary matrix, set of sequenced ModelIDs)."""
    want = {"ModelID", "HugoSymbol", "VariantInfo", "VariantType"}
    df = pd.read_csv(path, usecols=lambda c: c in want, low_memory=False)
    sequenced = set(df["ModelID"].unique())
    if "VariantInfo" in df.columns:
        df = df[df["VariantInfo"].astype(str).str.upper().str.contains(CODING_RE, regex=True)]
    mat = pd.crosstab(df["ModelID"], df["HugoSymbol"]).clip(upper=1).astype(np.float32)
    return mat, sequenced


def load_models(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, low_memory=False)
    keep = [c for c in ["ModelID", "CellLineName", "StrippedCellLineName", "CCLEName", "SangerModelID",
                        "COSMICID", "OncotreeLineage", "OncotreePrimaryDisease"] if c in m.columns]
    return m[keep]


def build_alias_map(models: pd.DataFrame) -> dict:
    """normalised alias -> ModelID (resolves names / Sanger IDs / COSMIC IDs used by GDSC & DrugComb)."""
    amap = {}
    cols = [c for c in ["ModelID", "CellLineName", "StrippedCellLineName", "CCLEName", "SangerModelID", "COSMICID"]
            if c in models.columns]
    for _, r in models.iterrows():
        for c in cols:
            v = r[c]
            if pd.notna(v):
                amap.setdefault(norm_name(v), r["ModelID"])
    return amap


def read_gmt(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                out[parts[0]] = [g for g in parts[2:] if g]
    return out


def find_gmt(directory: str, pattern: str):
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    return hits[0] if hits else None


def select_pathways(collections: dict, universe: set, score_var: dict,
                    min_size=10, max_size=500, n_kegg=75, n_reactome=75):
    """collections: {'HALLMARK': {name: genes}, 'KEGG': {...}, 'REACTOME': {...}}.
    Keep all Hallmark + top-variance KEGG / Reactome sets (default ~200 tokens)."""
    chosen = {}
    for src, sets in collections.items():
        cand = {}
        for name, genes in sets.items():
            g = [x for x in genes if x in universe]
            if min_size <= len(g) <= max_size:
                cand[name] = g
        if src == "HALLMARK":
            keep = list(cand)
        else:
            k = n_kegg if src == "KEGG" else n_reactome
            keep = sorted(cand, key=lambda n: -score_var.get(n, 0.0))[:k]
        for n in keep:
            chosen[n] = (src, cand[n])
    return chosen


def build_cache(depmap_dir, msigdb_dir, out_dir,
                expr_file="OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                cn_file="OmicsCNGene.csv", mut_file="OmicsSomaticMutations.csv",
                model_file="Model.csv", hallmark_glob="h.all.*.symbols.gmt",
                kegg_glob="c2.cp.kegg_*.symbols.gmt", reactome_glob="c2.cp.reactome.*.symbols.gmt",
                n_kegg=75, n_reactome=75, n_raw_genes=2000, min_size=10, max_size=500):
    os.makedirs(out_dir, exist_ok=True)
    print("[omics] reading DepMap files ...")
    models = load_models(os.path.join(depmap_dir, model_file))
    expr = read_depmap_matrix(os.path.join(depmap_dir, expr_file))
    cn = read_depmap_matrix(os.path.join(depmap_dir, cn_file))
    mut, sequenced = read_mutation_matrix(os.path.join(depmap_dir, mut_file))

    cells = sorted(set(expr.index) & set(cn.index) & sequenced)
    print(f"[omics] cells with RNA+CNV+mutation: {len(cells)}")
    expr, cn = expr.loc[cells], cn.loc[cells]
    genes = [g for g in expr.columns]
    universe = set(genes)

    # ---- gene-level normalisation -------------------------------------------------------
    sd = expr.std(0).replace(0, 1.0)
    Z = ((expr - expr.mean(0)) / sd).fillna(0).values.astype(np.float32)            # (C,G)
    cnv = cn.reindex(columns=genes)
    # DepMap CN = log2(relative copy number + 1): diploid == 1.0 -> centre at 0
    if np.nanmedian(cnv.values) > 0.5:
        cnv = cnv - 1.0
    CN = np.clip(np.nan_to_num(cnv.values, nan=0.0), -2, 2).astype(np.float32)
    MU = mut.reindex(index=cells, columns=genes).fillna(0).values.astype(np.float32)

    # ---- pathway collections -------------------------------------------------------------
    cols = {}
    for src, pat in [("HALLMARK", hallmark_glob), ("KEGG", kegg_glob), ("REACTOME", reactome_glob)]:
        p = find_gmt(msigdb_dir, pat)
        if p is None:
            print(f"[omics] WARNING: no GMT matching {pat} in {msigdb_dir}; skipping {src}")
            continue
        print(f"[omics] {src}: {os.path.basename(p)}")
        cols[src] = read_gmt(p)
    if not cols:
        raise FileNotFoundError("No MSigDB .gmt files found")

    gidx = {g: i for i, g in enumerate(genes)}
    # variance of the mean-expression score, used to rank KEGG/Reactome sets
    score_var = {}
    for src, sets in cols.items():
        for name, gl in sets.items():
            ii = [gidx[g] for g in gl if g in gidx]
            if min_size <= len(ii) <= max_size:
                score_var[name] = float(Z[:, ii].mean(1).var())
    chosen = select_pathways(cols, universe, score_var, min_size, max_size, n_kegg, n_reactome)
    names = list(chosen)
    P = len(names)
    print(f"[omics] pathway tokens: {P}")

    M = np.zeros((len(genes), P), np.float32)
    for j, n in enumerate(names):
        ii = [gidx[g] for g in chosen[n][1]]
        M[ii, j] = 1.0 / len(ii)

    feats = np.stack([Z @ M, CN @ M, MU @ M], axis=-1)                               # (C,P,3)
    mu_, sd_ = feats.mean(0, keepdims=True), feats.std(0, keepdims=True)
    feats = ((feats - mu_) / np.where(sd_ == 0, 1.0, sd_)).astype(np.float32)

    # ---- raw top-variance gene features for the 'no pathway mapping' baseline ------------
    top = np.argsort(-Z.var(0))[:n_raw_genes]
    raw = np.concatenate([Z[:, top], CN[:, top], MU[:, top]], axis=1).astype(np.float32)

    meta = models.set_index("ModelID").reindex(cells)
    lineage = (meta["OncotreeLineage"].fillna("Unknown").values if "OncotreeLineage" in meta
               else np.array(["Unknown"] * len(cells)))
    np.savez_compressed(os.path.join(out_dir, "omics.npz"), cell_ids=np.array(cells), feats=feats, raw=raw,
                        lineage=lineage, pathway_names=np.array(names),
                        pathway_source=np.array([chosen[n][0] for n in names]),
                        n_raw_genes=np.array(len(top)))
    with open(os.path.join(out_dir, "pathway_genes.json"), "w") as f:
        json.dump({n: chosen[n][1] for n in names}, f)
    cellset = set(cells)
    with open(os.path.join(out_dir, "alias_map.json"), "w") as f:
        json.dump({k: v for k, v in build_alias_map(models).items() if v in cellset}, f)
    print(f"[omics] saved cache -> {out_dir}")
