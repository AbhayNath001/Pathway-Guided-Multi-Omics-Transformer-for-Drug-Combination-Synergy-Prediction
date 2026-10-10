"""Create data/processed/drugs.csv (drug_name, smiles, targets) for every drug in your synergy tables.
SMILES come from PubChem (by name); targets from a GDSC compound-annotation file (optional).
Anything PubChem can't resolve is written with an empty smiles -> fill it by hand, then re-run (existing rows are kept).

python scripts/build_drug_table.py --synergy data/processed/gdsc2.csv data/processed/drugcomb.csv \
    --targets-csv data/raw/gdsc2/screened_compounds.csv --out data/processed/drugs.csv
"""
import argparse
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsyn.utils import norm_name  # noqa: E402


def pubchem_smiles(name, retries=3):
    import requests
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                for k, v in props.items():
                    if "SMILES" in k.upper():
                        return v
                return None
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synergy", nargs="+", required=True)
    ap.add_argument("--out", default="data/processed/drugs.csv")
    ap.add_argument("--targets-csv")
    ap.add_argument("--tgt-name-col", default="DRUG_NAME")
    ap.add_argument("--tgt-syn-col", default="SYNONYMS")
    ap.add_argument("--tgt-col", default="TARGET")
    a = ap.parse_args()

    names = set()
    for f in a.synergy:
        t = pd.read_csv(f)
        names |= set(t["drug_a"].astype(str)) | set(t["drug_b"].astype(str))
    existing = pd.read_csv(a.out) if os.path.exists(a.out) else pd.DataFrame(columns=["drug_name", "smiles", "targets"])
    have = {norm_name(n): i for i, n in enumerate(existing["drug_name"])}

    tgt = {}
    if a.targets_csv:
        ann = pd.read_csv(a.targets_csv)
        for _, r in ann.iterrows():
            if pd.isna(r.get(a.tgt_col)):
                continue
            keys = [r[a.tgt_name_col]] + (re.split(r"[;,]", str(r[a.tgt_syn_col])) if a.tgt_syn_col in ann and pd.notna(r[a.tgt_syn_col]) else [])
            for k in keys:
                tgt.setdefault(norm_name(k), str(r[a.tgt_col]))

    rows = existing.to_dict("records")
    for n in sorted(names):
        k = norm_name(n)
        if k in have and pd.notna(rows[have[k]].get("smiles")):
            continue
        smi = pubchem_smiles(n)
        time.sleep(0.25)                                   # PubChem: <= 5 requests / s
        print(f"{n:30s} {'OK ' if smi else 'MISSING'}")
        rec = {"drug_name": n, "smiles": smi, "targets": tgt.get(k)}
        if k in have:
            rows[have[k]] = rec
        else:
            rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    out = pd.DataFrame(rows)
    print(f"wrote {a.out}: {out.smiles.notna().sum()}/{len(out)} with SMILES, {out.targets.notna().sum()} with targets")


if __name__ == "__main__":
    main()
