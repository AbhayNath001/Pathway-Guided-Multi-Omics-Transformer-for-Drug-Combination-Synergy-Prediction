"""Run the full PathSyn pipeline on REAL data (DepMap + DrugComb).

Usage:
    python run_real.py                   # full pipeline: download -> preprocess -> train
    python run_real.py --skip-download   # if you already ran setup_real_data.py
    python run_real.py --epochs 30       # custom epochs
    python run_real.py --split cell      # cold-start splits

This script handles everything:
  1. Data setup (download DepMap + DrugComb + gene sets)
  2. Synergy table standardisation (DrugComb -> canonical CSV)
  3. Drug table construction (SMILES from PubChem + targets)
  4. Omics + drug caching (pathway tokens, fingerprints)
  5. Model training with cross-validation
"""
import argparse
import os
import subprocess
import sys


def run(cmd, desc=None, allow_fail=False):
    """Run a command and raise on failure."""
    if desc:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0 and not allow_fail:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description="Full real-data PathSyn pipeline")
    ap.add_argument("--data-dir", default="data/real", help="Root data directory")
    ap.add_argument("--out", default="runs/real", help="Output directory for results")
    ap.add_argument("--split", default="random", choices=["random", "cell", "pair", "cancer_type", "full"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=None, help="Limit folds for faster testing")
    ap.add_argument("--skip-download", action="store_true", help="Skip data download step")
    ap.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing (use existing cache)")
    ap.add_argument("--device", default=None, help="Force device (cpu/cuda)")
    ap.add_argument("--depmap-repo",
                    default=r"C:\Users\user\Downloads\depmap_omics-master\depmap_omics-master")
    a = ap.parse_args()

    depmap_dir = os.path.join(a.data_dir, "depmap")
    msigdb_dir = os.path.join(a.data_dir, "msigdb")
    drugcomb_dir = os.path.join(a.data_dir, "drugcomb")
    processed_dir = os.path.join(a.data_dir, "processed")
    cache_dir = os.path.join(a.data_dir, "cache")

    # ---- Step 1: Download Data ----
    if not a.skip_download:
        run(f'python scripts/setup_real_data.py --out {a.data_dir} --depmap-repo "{a.depmap_repo}"',
            "Step 1: Download & Setup Real Data")
    else:
        print("\n[skip] Data download")

    # ---- Step 2: Standardise synergy table ----
    synergy_csv = os.path.join(processed_dir, "drugcomb.csv")
    if not os.path.exists(synergy_csv) and not a.skip_preprocess:
        # Find the DrugComb summary CSV
        drugcomb_raw = None
        for f in os.listdir(drugcomb_dir):
            if f.endswith(".csv") and "summary" in f.lower():
                drugcomb_raw = os.path.join(drugcomb_dir, f)
                break
        if drugcomb_raw is None:
            print("[ERROR] No DrugComb summary CSV found. Run data setup first.")
            sys.exit(1)

        os.makedirs(processed_dir, exist_ok=True)
        run(f'python -m pathsyn.synergy --in "{drugcomb_raw}" --out "{synergy_csv}" --preset drugcomb',
            "Step 2: Standardise DrugComb Synergy Table")
    else:
        print(f"\n[skip] Synergy table already exists: {synergy_csv}" if os.path.exists(synergy_csv) else "\n[skip] Preprocessing skipped")

    # ---- Step 3: Build drug table ----
    drugs_csv = os.path.join(processed_dir, "drugs.csv")
    if not os.path.exists(drugs_csv) and not a.skip_preprocess:
        os.makedirs(processed_dir, exist_ok=True)
        run(f'python scripts/build_drug_table.py --synergy "{synergy_csv}" --out "{drugs_csv}"',
            "Step 3: Build Drug Table (PubChem SMILES lookup — this may take a while)")
    else:
        print(f"\n[skip] Drug table already exists: {drugs_csv}" if os.path.exists(drugs_csv) else "\n[skip] Preprocessing skipped")

    # ---- Step 4: Preprocess (cache omics + drugs) ----
    if not a.skip_preprocess:
        # Determine the correct GMT glob patterns based on what files exist
        gmt_files = os.listdir(msigdb_dir) if os.path.isdir(msigdb_dir) else []

        hallmark_glob = "h.all.*.symbols.gmt"
        kegg_glob = "c2.cp.kegg_*.symbols.gmt"
        reactome_glob = "c2.cp.reactome.*.symbols.gmt"

        # If using depmap_omics repo files (non-standard names), use custom globs
        if any("hallmark" in f.lower() for f in gmt_files):
            hallmark_glob = "hallmark.gmt"
        if any("kegg" in f.lower() and not f.startswith("c2") for f in gmt_files):
            kegg_glob = "KEGG*.gmt"
        if any("reactome" in f.lower() and not f.startswith("c2") for f in gmt_files):
            reactome_glob = "Reactome*.gmt"

        # Check required DepMap files
        depmap_reqs = ["Model.csv", "OmicsExpressionProteinCodingGenesTPMLogp1.csv", "OmicsCNGene.csv", "OmicsSomaticMutations.csv"]
        missing_or_empty = [f for f in depmap_reqs if not os.path.exists(os.path.join(depmap_dir, f)) or os.path.getsize(os.path.join(depmap_dir, f)) < 1000]
        if missing_or_empty:
            print(f"[ERROR] DepMap directory is missing valid files: {missing_or_empty}")
            print("Please re-run data download: python scripts/setup_real_data.py --out data/real")
            sys.exit(1)

        run(f'python -m pathsyn.preprocess '
            f'--depmap-dir "{depmap_dir}" '
            f'--msigdb-dir "{msigdb_dir}" '
            f'--synergy "{synergy_csv}" '
            f'--drug-table "{drugs_csv}" '
            f'--out "{cache_dir}" '
            f'--hallmark-glob "{hallmark_glob}" '
            f'--kegg-glob "{kegg_glob}" '
            f'--reactome-glob "{reactome_glob}"',
            "Step 4: Preprocess Omics + Drugs -> Cache")
    else:
        print("\n[skip] Preprocessing")

    # ---- Step 5: Train model ----
    out_dir = os.path.join(a.out, a.split)
    train_cmd = (f'python -m pathsyn.train '
                 f'--cache "{cache_dir}" '
                 f'--table "{synergy_csv}" '
                 f'--split {a.split} '
                 f'--epochs {a.epochs} '
                 f'--folds {a.folds} '
                 f'--out "{out_dir}"')
    if a.max_folds:
        train_cmd += f" --max-folds {a.max_folds}"
    if a.device:
        train_cmd += f" --device {a.device}"

    run(train_cmd, f"Step 5: Train PathSyn Model (split={a.split}, epochs={a.epochs})")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {os.path.abspath(out_dir)}")
    print("  - config.json       : training configuration")
    print("  - splits.csv        : train/val/test indices per fold")
    print("  - fold_metrics.csv  : per-fold test metrics")
    print("  - summary.csv       : mean ± std across folds")
    print("  - predictions.csv   : all test predictions")
    if a.split == "full":
        print("  - model.pt          : trained model checkpoint")

    # Print summary if it exists
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        print("\n--- Results Summary ---")
        import pandas as pd
        print(pd.read_csv(summary_path, index_col=0).round(4).to_string())


if __name__ == "__main__":
    main()
