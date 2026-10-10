"""Build all cached tensors:  omics.npz (+pathway_genes.json, alias_map.json) and drugs.npz

python -m pathsyn.preprocess --depmap-dir data/raw/depmap --msigdb-dir data/raw/msigdb \
    --synergy data/processed/gdsc2.csv data/processed/drugcomb.csv \
    --drug-table data/processed/drugs.csv --out data/cache
"""
import argparse
import json
import os

import pandas as pd

from .drugs import build_drug_cache
from .omics import build_cache


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depmap-dir", required=True)
    ap.add_argument("--msigdb-dir", required=True)
    ap.add_argument("--synergy", nargs="+", required=True, help="canonical synergy CSV(s); drugs are the union")
    ap.add_argument("--drug-table", required=True)
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--n-kegg", type=int, default=75)
    ap.add_argument("--n-reactome", type=int, default=75)
    ap.add_argument("--n-raw-genes", type=int, default=2000)
    ap.add_argument("--expr-file", default="OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    ap.add_argument("--cn-file", default="OmicsCNGene.csv")
    ap.add_argument("--mut-file", default="OmicsSomaticMutations.csv")
    ap.add_argument("--model-file", default="Model.csv")
    ap.add_argument("--hallmark-glob", default="h.all.*.symbols.gmt")
    ap.add_argument("--kegg-glob", default="c2.cp.kegg_*.symbols.gmt")
    ap.add_argument("--reactome-glob", default="c2.cp.reactome.*.symbols.gmt")
    ap.add_argument("--chemberta", action="store_true")
    a = ap.parse_args(argv)

    build_cache(a.depmap_dir, a.msigdb_dir, a.out, a.expr_file, a.cn_file, a.mut_file, a.model_file,
                a.hallmark_glob, a.kegg_glob, a.reactome_glob, a.n_kegg, a.n_reactome, a.n_raw_genes)
    drugs = set()
    for f in a.synergy:
        t = pd.read_csv(f)
        drugs |= set(t["drug_a"]) | set(t["drug_b"])
    with open(os.path.join(a.out, "pathway_genes.json")) as f:
        pg = json.load(f)
    build_drug_cache(a.drug_table, pg, os.path.join(a.out, "drugs.npz"), drugs, a.chemberta)


if __name__ == "__main__":
    main()
