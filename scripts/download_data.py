"""Data download helper and instructions for the PATH-SYN DrugComb v1.5 workflow."""
import argparse
import os
import urllib.request

FILES = [
    "h.all.v{version}.symbols.gmt",
    "c2.cp.kegg_legacy.v{version}.symbols.gmt",
    "c2.cp.reactome.v{version}.symbols.gmt",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msigdb-version", default="2024.1.Hs")
    parser.add_argument("--out", default="data/real/msigdb")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = (
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
        f"{args.msigdb_version}/"
    )
    for template in FILES:
        filename = template.format(version=args.msigdb_version)
        destination = os.path.join(args.out, filename)
        if os.path.exists(destination):
            print("already present:", destination)
            continue
        try:
            urllib.request.urlretrieve(base + filename, destination)
            print("downloaded:", destination)
        except Exception as exc:
            print(f"Could not download {filename}: {exc}")
            print("Download it manually from:", base)

    print(
        "\nOther required inputs:\n"
        "1. DepMap Public 24Q2: Model.csv, OmicsExpressionProteinCodingGenesTPMLogp1.csv, "
        "OmicsCNGene.csv and OmicsSomaticMutations.csv.\n"
        "2. DrugComb v1.5 summary response table (this is the manuscript's primary response source).\n"
        "3. The verified drug-target annotation CSV used in the experiment. The helper supports "
        "screened_compounds-style DRUG_NAME/SYNONYMS/TARGET columns.\n"
        "Do not replace DrugComb synergy responses with GDSC2 responses. A GDSC/CancerRxGene "
        "compound-annotation file, if used, is only the target-annotation input."
    )


if __name__ == "__main__":
    main()
