"""Omics ablations + raw-feature baseline.
python -m pathsyn.ablation --cache data/cache --table data/processed/gdsc2.csv --split cell --out runs/ablation
"""
import argparse
import os

import pandas as pd

from . import train as T

SETTINGS = {
    "all": ["--mods", "rna,cnv,mut"],
    "rna_only": ["--mods", "rna"], "mut_only": ["--mods", "mut"], "cnv_only": ["--mods", "cnv"],
    "rna_mut": ["--mods", "rna,mut"], "rna_cnv": ["--mods", "rna,cnv"],
    "baseline_raw_all": ["--model", "baseline", "--mods", "rna,cnv,mut"],
    "baseline_raw_rna": ["--model", "baseline", "--mods", "rna"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--only", nargs="*", help="subset of settings", default=list(SETTINGS))
    a, rest = ap.parse_known_args()          # remaining args (--cache, --table, --split, ...) go to train
    summ = {}
    for name in a.only:
        print(f"\n######## {name} ########")
        out = os.path.join(a.out, name)
        T.main(rest + SETTINGS[name] + ["--out", out])
        f = os.path.join(out, "summary.csv")
        if os.path.exists(f):
            summ[name] = pd.read_csv(f, index_col=0)["mean"]
    if summ:
        t = pd.DataFrame(summ).T
        t.to_csv(os.path.join(a.out, "ablation_summary.csv"))
        cols = [c for c in ["bliss_pearson", "bliss_rmse", "hsa_pearson", "auroc", "auprc"] if c in t]
        print("\n", t[cols].round(4))


if __name__ == "__main__":
    main()
