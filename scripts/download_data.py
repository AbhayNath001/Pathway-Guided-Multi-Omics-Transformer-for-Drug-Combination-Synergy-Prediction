"""Download what can be downloaded automatically (MSigDB) and print manual steps for the rest.

python scripts/download_data.py --msigdb-version 2024.1.Hs
Check https://www.gsea-msigdb.org/gsea/msigdb for the newest release label and pass it via --msigdb-version.
"""
import argparse
import os
import urllib.request

FILES = ["h.all.v{v}.symbols.gmt", "c2.cp.kegg_legacy.v{v}.symbols.gmt",
         "c2.cp.kegg_medicus.v{v}.symbols.gmt", "c2.cp.reactome.v{v}.symbols.gmt"]

MANUAL = """
MANUAL DOWNLOADS (these sites need a browser / click-through; file names can change between releases):

1) DepMap (https://depmap.org/portal/data_page/  -> choose a release, e.g. the latest "DepMap Public 2xQy")
   Put these in data/raw/depmap/ :
     Model.csv
     OmicsExpressionProteinCodingGenesTPMLogp1.csv
     OmicsCNGene.csv
     OmicsSomaticMutations.csv
   (If your release uses different names, pass --expr-file/--cn-file/--mut-file/--model-file to pathsyn.preprocess.)

2) GDSC2 combinations (https://gdsc-combinations.depmap.sanger.ac.uk/  -> Downloads)
   Put the combination synergy table(s) in data/raw/gdsc2/ . Open the header, then convert with pathsyn.synergy
   (see its docstring). Monotherapy IC50s are in the same portal / cancerrxgene.org bulk downloads.
   Drug target annotation: cancerrxgene.org -> Downloads -> "screened_compounds" (columns DRUG_NAME, TARGET, TARGET_PATHWAY).

3) DrugComb (https://drugcomb.org/download/ or the Zenodo record linked there)
   Put summary_table*.csv in data/raw/drugcomb/ .
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msigdb-version", default="2024.1.Hs")
    ap.add_argument("--out", default="data/raw/msigdb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{a.msigdb_version}/"
    for f in FILES:
        name = f.format(v=a.msigdb_version)
        dst = os.path.join(a.out, name)
        if os.path.exists(dst):
            print("have", name)
            continue
        try:
            urllib.request.urlretrieve(base + name, dst)
            print("downloaded", name)
        except Exception as e:  # noqa
            print(f"FAILED {name}: {e}\n   -> download manually from {base}")
    print(MANUAL)


if __name__ == "__main__":
    main()
