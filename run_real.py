"""Run PATH-SYN on DrugComb v1.5 with DepMap 24Q2 multi-omics.

This entry point deliberately requires an explicit drug-target annotation file.
It does not substitute GDSC2 response records for DrugComb v1.5.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], description: str) -> None:
    print(f"\n{'=' * 72}\n{description}\n{'=' * 72}", flush=True)
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_summary(folder: Path) -> Path:
    candidates = sorted(
        p for p in folder.glob("*.csv")
        if "summary" in p.name.lower() or "drugcomb" in p.name.lower()
    )
    if len(candidates) != 1:
        names = "\n".join(f"  - {p}" for p in candidates) or "  (none found)"
        raise FileNotFoundError(
            f"Expected exactly one DrugComb summary CSV in {folder}; found {len(candidates)}:\n{names}\n"
            "Pass --drugcomb-csv to select the intended file."
        )
    return candidates[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-aligned PATH-SYN DrugComb v1.5 workflow")
    p.add_argument("--data-dir", default="data/real")
    p.add_argument("--out", default="runs/drugcomb_v1_5")
    p.add_argument("--drugcomb-csv", default=None, help="Raw DrugComb v1.5 summary CSV; auto-detected if omitted")
    p.add_argument("--targets-csv", default=None, help="Required target-annotation file used for drug->pathway mapping")
    p.add_argument("--target-name-col", default="DRUG_NAME")
    p.add_argument("--target-syn-col", default="SYNONYMS")
    p.add_argument("--target-col", default="TARGET")
    p.add_argument("--extra-synergy", nargs="*", default=[],
                   help="Additional canonical response tables used only to build drug features (e.g. external compounds)")
    p.add_argument("--msigdb-version", default="2024.1.Hs")
    p.add_argument("--n-kegg", type=int, default=75)
    p.add_argument("--n-reactome", type=int, default=75)
    p.add_argument("--split", choices=["random", "cell", "pair", "cancer_type", "full"], default="random")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    a = p.parse_args()

    root = Path(a.data_dir)
    depmap = root / "depmap"
    msigdb = root / "msigdb"
    raw_drugcomb = root / "drugcomb"
    processed = root / "processed"
    cache = root / "cache"
    processed.mkdir(parents=True, exist_ok=True)

    if not a.skip_download:
        run([
            sys.executable, "scripts/setup_real_data.py",
            "--out", str(root),
            "--msigdb-version", a.msigdb_version,
        ], "Download/setup DepMap 24Q2, DrugComb v1.5 and MSigDB")

    drugcomb_raw = Path(a.drugcomb_csv) if a.drugcomb_csv else find_summary(raw_drugcomb)
    if not drugcomb_raw.is_file():
        raise FileNotFoundError(f"DrugComb summary CSV not found: {drugcomb_raw}")

    drugcomb_table = processed / "drugcomb_v1_5.csv"
    if not a.skip_preprocess:
        if not a.targets_csv:
            raise ValueError(
                "--targets-csv is required for the manuscript workflow. Supply the verified drug-target "
                "annotation file used for the study; target vectors must not be silently omitted."
            )
        target_file = Path(a.targets_csv)
        if not target_file.is_file():
            raise FileNotFoundError(f"Drug-target annotation file not found: {target_file}")

        run([
            sys.executable, "-m", "pathsyn.synergy",
            "--in", str(drugcomb_raw),
            "--out", str(drugcomb_table),
            "--preset", "drugcomb",
        ], "Convert DrugComb v1.5 to the canonical response schema")

        drug_table = processed / "drugs.csv"
        run([
            sys.executable, "scripts/build_drug_table.py",
            "--synergy", str(drugcomb_table),
            *[str(x) for x in a.extra_synergy],
            "--targets-csv", str(target_file),
            "--tgt-name-col", a.target_name_col,
            "--tgt-syn-col", a.target_syn_col,
            "--tgt-col", a.target_col,
            "--out", str(drug_table),
        ], "Retrieve PubChem structures and map supplied drug-target annotations")

        synergy_inputs = [str(drugcomb_table), *[str(x) for x in a.extra_synergy]]
        for item in synergy_inputs:
            if not Path(item).is_file():
                raise FileNotFoundError(f"Canonical synergy table not found: {item}")
        run([
            sys.executable, "-m", "pathsyn.preprocess",
            "--depmap-dir", str(depmap),
            "--msigdb-dir", str(msigdb),
            "--synergy", *synergy_inputs,
            "--drug-table", str(drug_table),
            "--out", str(cache),
            "--n-kegg", str(a.n_kegg),
            "--n-reactome", str(a.n_reactome),
        ], "Build DepMap pathway features and drug feature cache")
    else:
        if not cache.is_dir():
            raise FileNotFoundError(f"--skip-preprocess was supplied but cache directory is missing: {cache}")
        if not drugcomb_table.is_file():
            raise FileNotFoundError(f"--skip-preprocess was supplied but canonical table is missing: {drugcomb_table}")

    command = [
        sys.executable, "-m", "pathsyn.train",
        "--cache", str(cache),
        "--table", str(drugcomb_table),
        "--split", a.split,
        "--epochs", str(a.epochs),
        "--bs", str(a.batch_size),
        "--folds", str(a.folds),
        "--seed", str(a.seed),
        "--out", str(Path(a.out) / a.split),
    ]
    if a.max_folds is not None:
        command.extend(["--max-folds", str(a.max_folds)])
    if a.device:
        command.extend(["--device", a.device])
    run(command, f"Train/evaluate PATH-SYN ({a.split} split)")
    print(f"\nCompleted. Results are under: {Path(a.out) / a.split}")


if __name__ == "__main__":
    main()
