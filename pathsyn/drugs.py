"""Drug featurisation: Morgan fingerprints (RDKit) + target->pathway multi-hot vectors."""
import re
import zlib

import numpy as np
import pandas as pd

from .utils import norm_name


def _hash_fp(smiles: str, n_bits=2048, k=3):
    """Fallback if RDKit missing: hashed character k-grams of the SMILES (NOT chemically meaningful)."""
    v = np.zeros(n_bits, np.float32)
    for i in range(max(1, len(smiles) - k + 1)):
        v[zlib.crc32(smiles[i:i + k].encode()) % n_bits] = 1.0
    return v


def morgan_fp(smiles: str, radius=2, n_bits=2048):
    try:
        from rdkit import Chem
    except ImportError:
        return _hash_fp(smiles, n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr


def chemberta_embed(smiles_list, model_name="seyonec/ChemBERTa-zinc-base-v1"):
    """Optional pretrained SMILES transformer embedding (mean pooled, 768-d)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, mdl = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for s in smiles_list:
            enc = tok(s, return_tensors="pt", truncation=True, max_length=256)
            out.append(mdl(**enc).last_hidden_state.mean(1).squeeze(0).numpy())
    return np.stack(out).astype(np.float32)


def build_drug_cache(drug_csv: str, pathway_genes: dict, out_npz: str, drug_names_needed=None, chemberta=False):
    """drug_csv columns: drug_name, smiles, targets (gene symbols separated by ; , | /)."""
    df = pd.read_csv(drug_csv)
    df["key"] = df["drug_name"].map(norm_name)
    df = df.drop_duplicates("key")
    if drug_names_needed is not None:
        need = {norm_name(x) for x in drug_names_needed}
        missing = need - set(df["key"])
        if missing:
            print(f"[drugs] WARNING: {len(missing)} drugs have no entry in {drug_csv} (rows using them are dropped): "
                  f"{sorted(missing)[:10]}...")
        df = df[df["key"].isin(need)]
    pnames = list(pathway_genes)
    gene2p = {}
    for j, p in enumerate(pnames):
        for g in pathway_genes[p]:
            gene2p.setdefault(g.upper(), []).append(j)

    keys, fps, tgts, smis = [], [], [], []
    for _, r in df.iterrows():
        if pd.isna(r.get("smiles")):
            continue
        fp = morgan_fp(str(r["smiles"]))
        if fp is None:
            print(f"[drugs] invalid SMILES for {r['drug_name']}, skipped")
            continue
        t = np.zeros(len(pnames), np.float32)
        if pd.notna(r.get("targets")):
            for g in re.split(r"[;,|/]", str(r["targets"])):
                for j in gene2p.get(g.strip().upper(), []):
                    t[j] = 1.0
        keys.append(r["key"]); fps.append(fp); tgts.append(t); smis.append(str(r["smiles"]))
    x = np.stack(fps)
    if chemberta:
        x = np.concatenate([x, chemberta_embed(smis)], axis=1)
    tg = np.stack(tgts)
    np.savez_compressed(out_npz, keys=np.array(keys), x=x, target_pw=tg)
    print(f"[drugs] {len(keys)} drugs, feature dim {x.shape[1]}, {int((tg.sum(1) > 0).sum())} with >=1 target pathway")
