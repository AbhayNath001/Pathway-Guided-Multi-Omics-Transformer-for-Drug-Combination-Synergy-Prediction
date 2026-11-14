# PATH-SYN: Pathway-Guided Multi-Omics Transformer for Drug-Combination Synergy Prediction

This repository contains the PATH-SYN implementation and data-processing utilities. The primary experimental workflow described in the manuscript uses **DrugComb v1.5 response records** with **DepMap Public 24Q2** molecular profiles. GDSC combination-response measurements are not substituted for DrugComb in the manuscript workflow. A GDSC/CancerRxGene compound-annotation file may be supplied separately for drug-target annotations; it is not used as the drug-combination response table.

## 1. Repository layout

```text
pathsyn/
  data.py                 Dataset construction and data splits
  drugs.py                RDKit Morgan fingerprints and target-to-pathway mapping
  evaluate_external.py    Frozen-model evaluation on a canonical external response table
  model.py                PATH-SYN and baseline architectures
  omics.py                DepMap parsing and pathway selection/feature construction
  preprocess.py           Builds cached omics and drug features
  synergy.py              Converts source response tables to the canonical schema
  train.py                Training, cross-validation, prediction and metrics
  utils.py                Shared utilities and metrics
scripts/
  build_drug_table.py     PubChem SMILES + supplied drug-target annotation table
  download_data.py        Download instructions and MSigDB helper
  make_synthetic_data.py  Generates a clearly labelled synthetic smoke-test dataset
  setup_real_data.py      Downloads DepMap 24Q2, DrugComb v1.5 and MSigDB files
run_real.py               Configurable real-data preprocessing/training entry point
run_all.py                Synthetic smoke test
requirements.txt
```

## 2. Environment

Python 3.10 or 3.11 is recommended. Install PyTorch from the official selector for the CPU/CUDA environment, then install the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

RDKit is required for the reported Morgan fingerprints. The code stops with an informative error if RDKit is unavailable; it does not silently substitute hashed SMILES features.

## 3. Required real-data inputs

### 3.1 Drug-combination response data
- DrugComb v1.5 summary response table.
- The default converter preset expects the column names encoded in `pathsyn/synergy.py` (`cell_line_name`, `drug_row`, `drug_col`, `synergy_bliss`, `synergy_hsa`).
- If your downloaded DrugComb v1.5 file uses different headers, inspect the CSV header and supply explicit column mappings to `pathsyn.synergy`; do not silently use a GDSC2 synergy table as a substitute.

### 3.2 DepMap Public 24Q2
The setup helper downloads these files where provider endpoints are available:
- `Model.csv`
- `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
- `OmicsCNGene.csv`
- `OmicsSomaticMutations.csv`

### 3.3 MSigDB 2024.1.Hs
The setup helper downloads:
- Hallmark: `h.all.v2024.1.Hs.symbols.gmt`
- KEGG Legacy: `c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt`
- Reactome: `c2.cp.reactome.v2024.1.Hs.symbols.gmt`

### 3.4 Drug-target annotations
The drug table builder requires an annotation CSV supplied by the user. The repository helper supports a GDSC/CancerRxGene `screened_compounds.csv`-style file with a drug-name column, an optional synonyms column, and a target-gene column (defaults: `DRUG_NAME`, `SYNONYMS`, `TARGET`). Specify the exact file/release used for the study when running the pipeline. This is a separate source from PubChem: PubChem is queried for compound SMILES, while the supplied annotation table provides target gene symbols. Target genes are mapped to the selected pathway gene sets to create the drug-to-pathway vector.

If your target file has different headers, use `--target-name-col`, `--target-syn-col`, and `--target-col`. The run script requires the target-annotation file path so target information is not silently omitted. Unmatched compounds are reported and receive an all-zero target-pathway vector; inspect that report before accepting the cache.

## 4. Pathway selection implemented in the code

The pathway selector in `pathsyn/omics.py`:
1. Intersects each MSigDB gene set with the protein-coding gene universe available in the DepMap expression matrix.
2. Excludes candidate sets with fewer than 10 or more than 500 overlapping genes.
3. Retains all eligible Hallmark sets.
4. For each eligible KEGG and Reactome set, calculates the mean standardized expression of its member genes for each available cell line, then calculates the variance of that pathway-level score across cell lines.
5. Selects the 75 highest-variance KEGG sets and 75 highest-variance Reactome sets by default.

Thus, the default is up to 50 Hallmark + 75 KEGG + 75 Reactome tokens; the actual total depends on the number of eligible Hallmark sets and candidates in the supplied GMT files. The selected pathway names and their gene membership are saved in the cache (`omics.npz` and `pathway_genes.json`). To reproduce an existing model, use the same MSigDB release, DepMap inputs and selector parameters as the experiment that generated that model.

**Preprocessing scope:** expression standardization and variance ranking are currently calculated over the cell lines loaded from the supplied DepMap files before the downstream response split is generated. Grouped evaluation splits hold out response records/groups, but this preprocessing is not a fully inductive, fold-specific pathway-selection pipeline. Interpret cold-start results accordingly; do not describe preprocessing as fold-isolated unless it has been rerun separately within each training fold.

## 5. Prepare and run the DrugComb v1.5 workflow

### 5.1 Download the public inputs

```bash
python scripts/setup_real_data.py --out data/real --msigdb-version 2024.1.Hs
```

The script attempts to download DepMap 24Q2, DrugComb v1.5 and MSigDB gene sets. Some provider files may require manual access. If so, place the downloaded file in the printed directory and rerun with the appropriate `--skip-...` flag.

### 5.2 Supply the target annotation file

Obtain the exact target annotation release used for your study and save it locally, for example:

```text
data/external/screened_compounds.csv
```

Do not assume that the example filename establishes the annotation release. Record the actual provider, release/access date and file in the manuscript/supplementary information.

### 5.3 Run preprocessing and training

```bash
python run_real.py \
  --data-dir data/real \
  --targets-csv data/external/screened_compounds.csv \
  --split random \
  --epochs 30 \
  --batch-size 32 \
  --folds 5 \
  --seed 42 \
  --n-kegg 75 \
  --n-reactome 75 \
  --out runs/drugcomb_v1_5
```

The run script:
1. Locates/downloads the DrugComb v1.5 summary file.
2. Converts it to the canonical schema (`cell_line, drug_a, drug_b, bliss, hsa`).
3. Retrieves compound SMILES from PubChem and reads target genes from the supplied annotation file.
4. Builds DepMap/MSigDB pathway features and drug fingerprints/target-pathway features.
5. Runs the requested evaluation split and writes configuration, fold metrics, predictions and split assignments.

If the DrugComb header differs from the default preset, convert it manually:

```bash
python -m pathsyn.synergy \
  --in data/real/drugcomb/<downloaded-file>.csv \
  --out data/real/processed/drugcomb.csv \
  --cell-col <CELL_LINE_COLUMN> \
  --drug-a-col <DRUG_A_COLUMN> \
  --drug-b-col <DRUG_B_COLUMN> \
  --bliss-col <BLISS_COLUMN> \
  --hsa-col <HSA_COLUMN>
```

Then use `run_real.py --skip-download --drugcomb-csv data/real/processed/drugcomb.csv ...`.

### 5.4 Evaluation splits

Supported splits are:
- `random`: stratified five-fold row-level evaluation.
- `cell`: grouped cell-line cold-start.
- `pair`: grouped unordered drug-pair cold-start.
- `cancer_type`: leave-one-lineage-out.
- `full`: trains a checkpoint with a within-training validation subset for early stopping and does not report a held-out test score. This checkpoint can be used for external evaluation.

Example final checkpoint run:

```bash
python run_real.py \
  --data-dir data/real \
  --targets-csv data/external/screened_compounds.csv \
  --split full \
  --epochs 30 \
  --batch-size 32 \
  --folds 5 \
  --seed 42 \
  --out runs/drugcomb_final
```

The model checkpoint is saved under the selected output directory (see the command output).

## 6. External evaluation with a frozen checkpoint

External response tables must first be converted to the same canonical schema. To represent external compounds in the feature cache, pass their canonical response tables through `--extra-synergy` during preprocessing; these files are used to build compound features and are **not** used as training response labels.

Example:

```bash
python run_real.py \
  --data-dir data/real \
  --targets-csv data/external/screened_compounds.csv \
  --extra-synergy data/real/external/nci_almanac.csv data/real/external/az_sanger.csv \
  --split full \
  --out runs/drugcomb_final
```

Then evaluate each external table without updating the checkpoint:

```bash
python -m pathsyn.evaluate_external \
  --model runs/drugcomb_final/full/model.pt \
  --cache data/real/cache \
  --table data/real/external/nci_almanac.csv \
  --train-table data/real/processed/drugcomb_v1_5.csv \
  --out runs/external/nci_almanac

python -m pathsyn.evaluate_external \
  --model runs/drugcomb_final/full/model.pt \
  --cache data/real/cache \
  --table data/real/external/az_sanger.csv \
  --train-table data/real/processed/drugcomb_v1_5.csv \
  --out runs/external/az_sanger
```

The evaluator reports the number of usable external rows after feature matching and exact (cell-line, unordered-drug-pair) overlap removal. By default it permits cell lines available in the DepMap feature cache, including cell lines not represented in the training response table, and requires external drugs to have cached feature vectors. Use `--cell-policy seen` or `--require any|both` only when that restriction is intended. The retained count can therefore be smaller than the raw external manifest.

For genuinely independent validation, audit source-study/record overlap between DrugComb v1.5 and each external source before running this command. A frozen checkpoint alone does not establish source independence if the same experimental records were used in model development.

## 7. Synthetic smoke test

`run_all.py` is deliberately limited to a small synthetic pipeline smoke test.

```bash
python run_all.py --out runs/synthetic-smoke-test
```

Use this only to check that the code modules can read the synthetic file formats and execute a short training pass.

## 8. Outputs

For a cross-validation run, the output directory contains:
- `config.json`: exact command-line configuration.
- `splits.csv`: row indices assigned to train/validation/test for each fold.
- `fold_metrics.csv` and `summary.csv`: metrics calculated from the run.
- `predictions.csv`: held-out observations and prediction outputs.

For a `full` run, the output includes `model.pt` and training configuration/scaler information. External runs write `external_predictions.csv` and `external_metrics.json`.
