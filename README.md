# PATH-SYN — pathway-guided multi-omics transformer for drug-combination synergy

```
pathsyn/
  omics.py              DepMap loading, MSigDB pathway tokens ([RNA, CNV, MUT] per pathway), raw baseline features
  drugs.py              Morgan fingerprints (RDKit) + drug-target -> pathway vectors (optional ChemBERTa)
  synergy.py            canonical synergy-table schema, GDSC2/DrugComb adapters, Bliss/HSA from raw dose matrices
  preprocess.py         builds data/cache/{omics.npz, drugs.npz, pathway_genes.json, alias_map.json}
  data.py               tensor dataset + splits (random / cell / pair / cancer_type cold-start, leakage-free)
  model.py              PathSyn (cross-attention) and RawMLP baseline
  train.py              training, CV, early stopping, MC-dropout, metrics, `--split full` final model
  ablation.py           omics ablations + raw-feature baseline
  evaluate_external.py  DrugComb validation (overlap filtering, leakage removal)
  explain.py            pathway attributions (attention + gradient x input)
scripts/
  download_data.py  build_drug_table.py  make_synthetic_data.py
```

## 0. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install torch per https://pytorch.org for your CUDA version
```

## 1. Smoke test (no downloads, ~minutes on CPU)
```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m pathsyn.preprocess --depmap-dir data/synthetic/depmap --msigdb-dir data/synthetic/msigdb \
   --synergy data/synthetic/synergy.csv --drug-table data/synthetic/drugs.csv --out data/synthetic/cache
python -m pathsyn.train --cache data/synthetic/cache --table data/synthetic/synergy.csv \
   --split random --epochs 5 --max-folds 1 --out runs/synth
```

## 2. Real data
```bash
python scripts/download_data.py --msigdb-version 2024.1.Hs     # then follow the printed manual steps (DepMap, GDSC2, DrugComb)

# (a) convert raw synergy tables to the canonical schema  (cell_line, drug_a, drug_b, bliss, hsa[, mono_a, mono_b])
python -m pathsyn.synergy --in data/raw/drugcomb/summary_table_v1.4.csv --out data/processed/drugcomb.csv --preset drugcomb
python -m pathsyn.synergy --in data/raw/gdsc2/<file>.csv --out data/processed/gdsc2.csv \
   --cell-col <..> --drug-a-col <..> --drug-b-col <..> --bliss-col <..> --hsa-col <..> [--mono-a-col <..> --mono-b-col <..>]

# (b) SMILES (PubChem) + targets (GDSC compound annotation)
python scripts/build_drug_table.py --synergy data/processed/gdsc2.csv data/processed/drugcomb.csv \
   --targets-csv data/raw/gdsc2/screened_compounds.csv --out data/processed/drugs.csv
#     -> open drugs.csv, fill any missing SMILES by hand, re-run to resume

# (c) cache omics + pathways + drugs
python -m pathsyn.preprocess --depmap-dir data/raw/depmap --msigdb-dir data/raw/msigdb \
   --synergy data/processed/gdsc2.csv data/processed/drugcomb.csv --drug-table data/processed/drugs.csv --out data/cache
```

## 3. Experiments
```bash
T="--cache data/cache --table data/processed/gdsc2.csv"
python -m pathsyn.train $T --split random      --out runs/random          # stratified 5-fold CV
python -m pathsyn.train $T --split cell        --out runs/cell_cold       # unseen cell lines
python -m pathsyn.train $T --split pair        --out runs/pair_cold       # unseen drug pairs
python -m pathsyn.train $T --split cancer_type --out runs/loco            # leave-one-lineage-out
python -m pathsyn.ablation $T --split cell --out runs/ablation            # rna/cnv/mut combos + raw baseline
python -m pathsyn.train $T --split full --out runs/final                  # final model on all data
python -m pathsyn.evaluate_external --model runs/final/model.pt --cache data/cache \
   --table data/processed/drugcomb.csv --train-table data/processed/gdsc2.csv --out runs/external
python -m pathsyn.explain --model runs/final/model.pt --cache data/cache --cell MCF7 --drug-a <A> --drug-b <B>
```
Outputs per run: `config.json, splits.csv` (row indices per fold), `fold_metrics.csv, summary.csv, predictions.csv`
(with MC-dropout std), and `model.pt` for `--split full`.

## Design notes / things to know
* **Pathway tokens**: per cell line and pathway, 3 features = mean z-scored expression, mean centred CN, fraction of
  member genes mutated; z-scored across cell lines. Hallmark (all) + top-variance KEGG/Reactome sets (default 50+75+75).
* **Drug-target alignment**: pathways hit by a drug's targets get a learned "targeted by A/B" embedding added to their
  token, and the drug token also receives a projection of its target-pathway vector.
* **Cross-attention**: queries = [CLS, drugA, drugB]; keys/values = pathway tokens. Auxiliary monotherapy pass uses
  [CLS, drug] through the same network (needs `mono_a/mono_b`; skipped if absent).
* Drug order is randomly swapped in training and predictions are averaged over both orders (symmetry).
* Splits are seeded (42); validation for early stopping is grouped like the test split, so cold-start stays cold-start.
* Synergy label = `bliss > --syn-threshold` (default 0). Many papers use a stricter cut (e.g. 5 or 10) — tune it.

## Caveats
* The Python modules were written to match the *documented* DepMap / MSigDB / DrugComb layouts, but I could not download
  real data in the build environment. **Check your headers**: file names/columns change between DepMap releases and
  the GDSC2 combination download (hence the explicit `--*-col` flags and `--expr-file` etc.).
* PyTorch training code was not executed where this was written (no torch there); the data pipeline (omics/pathway
  caching, drug cache, splits, leakage checks, adapters) was run on synthetic data. Run step 1 first and report any error.
* DrugComb vs GDSC2 Bliss values are on different scales; trust correlations/AUROC more than RMSE externally.
