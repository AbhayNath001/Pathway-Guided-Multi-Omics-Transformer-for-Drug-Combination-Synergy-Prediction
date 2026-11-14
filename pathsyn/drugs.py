"""Drug featurisation: RDKit Morgan fingerprints + target-to-pathway multi-hot vectors."""
import re

import numpy as np
import pandas as pd

from .utils import norm_name


def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Return a chemically meaningful RDKit Morgan fingerprint.

    RDKit is required. A hashed-character fallback is intentionally not used because
    it is not a molecular fingerprint and would change the model input definition.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to generate PATH-SYN Morgan fingerprints. "
            "Install the rdkit package from requirements.txt."
        ) from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(molecule).astype(np.float32)


def chemberta_embed(smiles_list, model_name="seyonec/ChemBERTa-zinc-base-v1"):
    """Optional pretrained SMILES transformer embedding (mean pooled, 768-d)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    outputs = []
    with torch.no_grad():
        for smiles in smiles_list:
            encoded = tokenizer(smiles, return_tensors="pt", truncation=True, max_length=256)
            outputs.append(model(**encoded).last_hidden_state.mean(1).squeeze(0).numpy())
    return np.stack(outputs).astype(np.float32)


def build_drug_cache(drug_csv: str, pathway_genes: dict, out_npz: str, drug_names_needed=None, chemberta=False):
    """Read drug_name/smiles/targets CSV and construct drug features."""
    df = pd.read_csv(drug_csv)
    required = {"drug_name", "smiles", "targets"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Drug table is missing required columns: {sorted(missing_columns)}")

    df["key"] = df["drug_name"].map(norm_name)
    df = df.drop_duplicates("key")
    if drug_names_needed is not None:
        needed = {norm_name(name) for name in drug_names_needed}
        missing = needed - set(df["key"])
        if missing:
            print(
                f"[drugs] WARNING: {len(missing)} required drugs have no row in {drug_csv}; "
                f"examples: {sorted(missing)[:10]}"
            )
        df = df[df["key"].isin(needed)]

    pathway_names = list(pathway_genes)
    gene_to_pathways: dict[str, list[int]] = {}
    for index, pathway in enumerate(pathway_names):
        for gene in pathway_genes[pathway]:
            gene_to_pathways.setdefault(gene.upper(), []).append(index)

    keys, fingerprints, target_vectors, smiles_used = [], [], [], []
    for _, row in df.iterrows():
        if pd.isna(row.get("smiles")):
            print(f"[drugs] missing SMILES for {row['drug_name']}; skipped")
            continue
        fingerprint = morgan_fp(str(row["smiles"]))
        if fingerprint is None:
            print(f"[drugs] invalid SMILES for {row['drug_name']}; skipped")
            continue

        target_vector = np.zeros(len(pathway_names), dtype=np.float32)
        if pd.notna(row.get("targets")):
            for gene in re.split(r"[;,|/]", str(row["targets"])):
                for pathway_index in gene_to_pathways.get(gene.strip().upper(), []):
                    target_vector[pathway_index] = 1.0
        keys.append(row["key"])
        fingerprints.append(fingerprint)
        target_vectors.append(target_vector)
        smiles_used.append(str(row["smiles"]))

    if not fingerprints:
        raise ValueError("No valid drug SMILES were converted into Morgan fingerprints.")

    drug_features = np.stack(fingerprints)
    if chemberta:
        drug_features = np.concatenate([drug_features, chemberta_embed(smiles_used)], axis=1)
    target_features = np.stack(target_vectors)
    np.savez_compressed(out_npz, keys=np.array(keys), x=drug_features, target_pw=target_features)
    print(
        f"[drugs] {len(keys)} drugs; feature dimension={drug_features.shape[1]}; "
        f"{int((target_features.sum(1) > 0).sum())} drugs with >=1 mapped target pathway"
    )
