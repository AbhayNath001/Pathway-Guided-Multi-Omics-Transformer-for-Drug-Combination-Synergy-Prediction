import os
import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Step 1: Generating Synthetic Data ===")
    run("python scripts/make_synthetic_data.py --out data/synthetic")
    
    print("\n=== Step 2: Preprocessing ===")
    run("python -m pathsyn.preprocess --depmap-dir data/synthetic/depmap --msigdb-dir data/synthetic/msigdb --synergy data/synthetic/synergy.csv --drug-table data/synthetic/drugs.csv --out data/synthetic/cache")
    
    print("\n=== Step 3: Training Model (Smoke Test) ===")
    run("python -m pathsyn.train --cache data/synthetic/cache --table data/synthetic/synergy.csv --split random --epochs 30 --max-folds 3 --out runs/synth")
    
    print("\n=== All steps completed successfully! ===")
    print("Results are in the 'runs/synth' directory.")



# test: {'bliss_rmse': 1.5141, 'bliss_pearson': 0.9723, 'bliss_spearman': 0.9714, 'hsa_rmse': 1.229, 'hsa_pearson': 0.9714, 'hsa_spearman': 0.9683, 'auroc': 0.9878, 'auprc': 0.9848, 'brier': 0.0413}