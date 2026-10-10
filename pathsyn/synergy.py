"""Synergy tables.

Canonical schema used everywhere in this project (CSV):
    cell_line, drug_a, drug_b, bliss, hsa [, mono_a, mono_b]
  * cell_line : any name / ModelID / SangerModelID / COSMIC id that DepMap Model.csv can resolve
  * mono_a/b  : OPTIONAL monotherapy sensitivity (e.g. ln IC50) of drug_a / drug_b in that cell line (aux task)

`standardize` converts a raw GDSC2-combination or DrugComb table into that schema.
`bliss_hsa_from_matrix` computes Bliss/HSA excess from a raw dose-response inhibition matrix.

CLI example (DrugComb):
    python -m pathsyn.synergy --in data/raw/drugcomb/summary_table_v1.4.csv \
        --out data/processed/drugcomb.csv --preset drugcomb
CLI example (GDSC2 -- first run `head -3 file` and set the column names that exist in YOUR download):
    python -m pathsyn.synergy --in data/raw/gdsc2/combos.csv --out data/processed/gdsc2.csv \
        --cell-col SANGER_MODEL_ID --drug-a-col ANCHOR_NAME --drug-b-col LIBRARY_NAME \
        --bliss-col BLISS_EXCESS --hsa-col HSA_EXCESS
"""
import argparse

import numpy as np
import pandas as pd

from .utils import norm_name

PRESETS = {
    "drugcomb": dict(cell_col="cell_line_name", drug_a_col="drug_row", drug_b_col="drug_col",
                     bliss_col="synergy_bliss", hsa_col="synergy_hsa", mono_a_col=None, mono_b_col=None),
}


def bliss_hsa_from_matrix(inhib: np.ndarray):
    """inhib: (nA+1, nB+1) fractional inhibition in [0,1]; row 0 / col 0 are the zero-dose
    (single-agent) conditions, inhib[0,0] ~ 0. Returns (mean Bliss excess, mean HSA excess) in %-points."""
    a = inhib[1:, 0][:, None]          # drug A alone at each dose
    b = inhib[0, 1:][None, :]          # drug B alone
    obs = inhib[1:, 1:]
    bliss_exp = a + b - a * b
    hsa_exp = np.maximum(a, b)
    return float(np.mean(obs - bliss_exp) * 100), float(np.mean(obs - hsa_exp) * 100)


def standardize(df, cell_col, drug_a_col, drug_b_col, bliss_col, hsa_col, mono_a_col=None, mono_b_col=None):
    out = pd.DataFrame({"cell_line": df[cell_col], "drug_a": df[drug_a_col], "drug_b": df[drug_b_col],
                        "bliss": pd.to_numeric(df[bliss_col], errors="coerce"),
                        "hsa": pd.to_numeric(df[hsa_col], errors="coerce")})
    if mono_a_col and mono_b_col:
        out["mono_a"] = pd.to_numeric(df[mono_a_col], errors="coerce")
        out["mono_b"] = pd.to_numeric(df[mono_b_col], errors="coerce")
    out = out.dropna(subset=["cell_line", "drug_a", "drug_b", "bliss", "hsa"])
    out = out[out["drug_a"].map(norm_name) != out["drug_b"].map(norm_name)]
    # order drugs canonically so (A,B) and (B,A) rows are the same experiment; swap mono columns with them
    swap = out["drug_a"].map(norm_name) > out["drug_b"].map(norm_name)
    out.loc[swap, ["drug_a", "drug_b"]] = out.loc[swap, ["drug_b", "drug_a"]].values
    if "mono_a" in out:
        out.loc[swap, ["mono_a", "mono_b"]] = out.loc[swap, ["mono_b", "mono_a"]].values
    out["_c"], out["_a"], out["_b"] = (out["cell_line"].map(norm_name), out["drug_a"].map(norm_name),
                                       out["drug_b"].map(norm_name))
    first = out.groupby(["_c", "_a", "_b"])[["cell_line", "drug_a", "drug_b"]].first()
    num = out.groupby(["_c", "_a", "_b"]).mean(numeric_only=True)        # average replicates/blocks
    return pd.concat([first, num], axis=1).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=list(PRESETS))
    for k in ["cell-col", "drug-a-col", "drug-b-col", "bliss-col", "hsa-col", "mono-a-col", "mono-b-col"]:
        ap.add_argument("--" + k)
    ap.add_argument("--sep", default=",")
    a = ap.parse_args()
    kw = dict(PRESETS.get(a.preset, {}))
    for k in ["cell_col", "drug_a_col", "drug_b_col", "bliss_col", "hsa_col", "mono_a_col", "mono_b_col"]:
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    df = pd.read_csv(a.inp, sep=a.sep, low_memory=False)
    print("columns in input:", list(df.columns))
    out = standardize(df, **kw)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows, {out.cell_line.nunique()} cell lines, "
          f"{len(set(out.drug_a) | set(out.drug_b))} drugs -> {a.out}")


if __name__ == "__main__":
    main()
