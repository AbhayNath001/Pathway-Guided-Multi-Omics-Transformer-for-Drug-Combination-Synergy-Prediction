"""Build a drug table with PubChem SMILES and supplied drug-target gene annotations.

Canonical synergy inputs must contain: cell_line, drug_a, drug_b, bliss, hsa.
The target annotation CSV is an explicit input; it is not downloaded or silently inferred.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsyn.utils import norm_name  # noqa: E402


def pubchem_smiles(name: str, retries: int = 3) -> str | None:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
    )
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            properties = response.json()["PropertyTable"]["Properties"][0]
            return next((value for key, value in properties.items() if "SMILES" in key.upper()), None)
        except Exception:
            if attempt + 1 < retries:
                time.sleep(1.5)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synergy", nargs="+", required=True, help="One or more canonical response CSVs")
    parser.add_argument("--targets-csv", required=True, help="Verified drug-target annotation CSV")
    parser.add_argument("--tgt-name-col", default="DRUG_NAME")
    parser.add_argument("--tgt-syn-col", default="SYNONYMS")
    parser.add_argument("--tgt-col", default="TARGET")
    parser.add_argument("--out", default="data/processed/drugs.csv")
    args = parser.parse_args()

    targets_path = Path(args.targets_csv)
    if not targets_path.is_file():
        raise FileNotFoundError(f"Drug-target annotation file not found: {targets_path}")

    names: set[str] = set()
    for source in args.synergy:
        table = pd.read_csv(source)
        missing = {"drug_a", "drug_b"} - set(table.columns)
        if missing:
            raise ValueError(f"{source} is not a canonical synergy table; missing columns: {sorted(missing)}")
        names.update(table["drug_a"].dropna().astype(str))
        names.update(table["drug_b"].dropna().astype(str))

    annotation = pd.read_csv(targets_path, low_memory=False)
    for col in (args.tgt_name_col, args.tgt_col):
        if col not in annotation.columns:
            raise ValueError(f"Target annotation file is missing required column {col!r}")

    target_map: dict[str, str] = {}
    for _, row in annotation.iterrows():
        if pd.isna(row.get(args.tgt_col)):
            continue
        aliases = [row[args.tgt_name_col]]
        if args.tgt_syn_col in annotation.columns and pd.notna(row.get(args.tgt_syn_col)):
            aliases.extend(re.split(r"[;,|]", str(row[args.tgt_syn_col])))
        genes = str(row[args.tgt_col])
        for alias in aliases:
            if pd.notna(alias) and str(alias).strip():
                target_map.setdefault(norm_name(alias), genes)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(output).to_dict("records") if output.exists() else []
    row_index = {norm_name(row["drug_name"]): i for i, row in enumerate(existing)}
    records = existing

    for drug in sorted(names):
        key = norm_name(drug)
        target_genes = target_map.get(key)
        old = records[row_index[key]] if key in row_index else {}
        smiles = old.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            smiles = pubchem_smiles(drug)
            time.sleep(0.25)
        record = {"drug_name": drug, "smiles": smiles, "targets": target_genes}
        if key in row_index:
            records[row_index[key]] = record
        else:
            row_index[key] = len(records)
            records.append(record)
        print(f"{drug:32s} SMILES={'OK' if smiles else 'MISSING':8s} targets={'OK' if target_genes else 'MISSING'}")

    result = pd.DataFrame(records, columns=["drug_name", "smiles", "targets"])
    result.to_csv(output, index=False)
    print(
        f"\nWrote {output}: {result['smiles'].notna().sum()}/{len(result)} with SMILES; "
        f"{result['targets'].notna().sum()}/{len(result)} with target annotations."
    )
    missing_targets = result.loc[result["targets"].isna(), "drug_name"].tolist()
    if missing_targets:
        print(f"WARNING: {len(missing_targets)} drugs have no target annotation and will receive zero target-pathway features.")
        print("Review these entries before treating the feature cache as equivalent to the manuscript experiment.")


if __name__ == "__main__":
    main()
