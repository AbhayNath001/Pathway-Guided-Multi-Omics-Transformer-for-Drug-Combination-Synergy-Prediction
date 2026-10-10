"""Generate a small fake dataset in the SAME file formats as DepMap / MSigDB / canonical synergy CSV,
so you can smoke-test the whole pipeline before downloading anything.

python scripts/make_synthetic_data.py --out data/synthetic
python -m pathsyn.preprocess --depmap-dir data/synthetic/depmap --msigdb-dir data/synthetic/msigdb \
    --synergy data/synthetic/synergy.csv --drug-table data/synthetic/drugs.csv --out data/synthetic/cache
python -m pathsyn.train --cache data/synthetic/cache --table data/synthetic/synergy.csv --split random --epochs 5 --out runs/synth
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

SMILES = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "CC(=O)NC1=CC=C(O)C=C1", "C1=CC=C(C=C1)C(=O)O", "CCO", "CC(C)(C)NCC(O)C1=CC(O)=CC(O)=C1",
          "C1CCC(CC1)N", "OC(=O)C1=CC=CC=C1O", "CN(C)C(=N)N=C(N)N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--cells", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    G, C, K, BS = 600, a.cells, 12, 30
    dm, ms = os.path.join(a.out, "depmap"), os.path.join(a.out, "msigdb")
    os.makedirs(dm, exist_ok=True); os.makedirs(ms, exist_ok=True)
    genes = [f"GENE{i}" for i in range(G)]
    gcol = [f"{g} ({1000 + i})" for i, g in enumerate(genes)]
    cells = [f"ACH-{i + 1:06d}" for i in range(C)]
    f = rng.randn(C, K)                                   # latent pathway activity per cell

    expr = 5 + rng.randn(C, G)
    for k in range(K):
        expr[:, k * BS:(k + 1) * BS] += f[:, [k]]
    cn = 1.0 + 0.1 * rng.randn(C, G)
    meta = {"IsDefaultEntryForModel": "Yes"}
    for name, M in [("OmicsExpressionProteinCodingGenesTPMLogp1.csv", expr), ("OmicsCNGene.csv", cn)]:
        d = pd.DataFrame(M, columns=gcol); d.insert(0, "IsDefaultEntryForModel", "Yes"); d.insert(0, "ModelID", cells)
        d.to_csv(os.path.join(dm, name), index=False)
    mu = []
    for c in range(C):
        for g in np.where(rng.rand(G) < 0.02)[0]:
            mu.append((cells[c], genes[g], "MISSENSE"))
    mu.append((cells[0], genes[0], "SILENT"))
    pd.DataFrame(mu, columns=["ModelID", "HugoSymbol", "VariantInfo"]).to_csv(os.path.join(dm, "OmicsSomaticMutations.csv"), index=False)
    lin = np.array(["Breast", "Lung", "Colon", "Pancreas"])[rng.randint(0, 4, C)]
    pd.DataFrame({"ModelID": cells, "CellLineName": [f"Cell-{i}" for i in range(C)],
                  "StrippedCellLineName": [f"CELL{i}" for i in range(C)],
                  "SangerModelID": [f"SIDM{i:05d}" for i in range(C)], "COSMICID": 900000 + np.arange(C),
                  "OncotreeLineage": lin, "OncotreePrimaryDisease": lin}).to_csv(os.path.join(dm, "Model.csv"), index=False)

    def gmt(fn, sets):
        with open(os.path.join(ms, fn), "w") as h:
            for n, g in sets.items():
                h.write(n + "\thttp://x\t" + "\t".join(g) + "\n")
    gmt("h.all.v0.Hs.symbols.gmt", {f"HALLMARK_SET{k}": genes[k * BS:(k + 1) * BS] for k in range(K)})
    gmt("c2.cp.kegg_legacy.v0.Hs.symbols.gmt", {f"KEGG_SET{k}": genes[k * 40:k * 40 + 25] for k in range(12)})
    gmt("c2.cp.reactome.v0.Hs.symbols.gmt", {f"REACTOME_SET{k}": list(rng.choice(genes, 20, replace=False)) for k in range(12)})

    nd = len(SMILES)
    drugs = pd.DataFrame({"drug_name": [f"Drug{i}" for i in range(nd)], "smiles": SMILES,
                          "targets": [";".join(genes[(i % K) * BS:(i % K) * BS + 5]) for i in range(nd)]})
    drugs.to_csv(os.path.join(a.out, "drugs.csv"), index=False)

    rows = []
    for i, j in itertools.combinations(range(nd), 2):
        for c in rng.choice(C, int(C * 0.8), replace=False):
            bl = 6 * f[c, i % K] * f[c, j % K] + 2.0 * (f[c, i % K] + f[c, j % K]) + rng.randn() * 0.2
            rows.append((f"Cell-{c}", f"Drug{i}", f"Drug{j}", bl, 0.8 * bl + rng.randn() * 0.2,
                         -2 * f[c, i % K] + rng.randn() * 0.1, -2 * f[c, j % K] + rng.randn() * 0.1))
    pd.DataFrame(rows, columns=["cell_line", "drug_a", "drug_b", "bliss", "hsa", "mono_a", "mono_b"]).to_csv(
        os.path.join(a.out, "synergy.csv"), index=False)
    print(f"synthetic data in {a.out}: {len(rows)} synergy rows")


if __name__ == "__main__":
    main()
