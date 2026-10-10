"""Dataset object (all tensors live on one device) + split generators."""
import json
import os

import numpy as np
import pandas as pd
import torch

from .utils import norm_name

MODS = ["rna", "cnv", "mut"]


class PathSynData:
    def __init__(self, cache_dir, table, syn_threshold=0.0, device="cpu", verbose=True):
        om = np.load(os.path.join(cache_dir, "omics.npz"), allow_pickle=True)
        dr = np.load(os.path.join(cache_dir, "drugs.npz"), allow_pickle=True)
        with open(os.path.join(cache_dir, "alias_map.json")) as f:
            alias = json.load(f)
        self.device = device
        self.cell_ids = [str(c) for c in om["cell_ids"]]
        self.cell_index = {c: i for i, c in enumerate(self.cell_ids)}
        self.pathway_names = [str(x) for x in om["pathway_names"]]
        self.n_path = len(self.pathway_names)
        self.n_raw_genes = int(om["n_raw_genes"])
        self.drug_keys = [str(k) for k in dr["keys"]]
        self.drug_index = {k: i for i, k in enumerate(self.drug_keys)}
        self.drug_dim = dr["x"].shape[1]

        df = pd.read_csv(table) if isinstance(table, (str, os.PathLike)) else table.copy()
        n0 = len(df)
        df["cell_id"] = df["cell_line"].map(lambda s: alias.get(norm_name(s)))
        df["ia"] = df["drug_a"].map(lambda s: self.drug_index.get(norm_name(s)))
        df["ib"] = df["drug_b"].map(lambda s: self.drug_index.get(norm_name(s)))
        df = df.dropna(subset=["cell_id", "ia", "ib"]).reset_index(drop=True)
        if verbose:
            print(f"[data] {len(df)}/{n0} rows usable (cell line in omics cache & both drugs featurised)")
        for c in ["bliss", "hsa", "mono_a", "mono_b"]:
            if c not in df:
                df[c] = np.nan
        df["ci"] = df["cell_id"].map(self.cell_index).astype(int)
        df["ia"], df["ib"] = df["ia"].astype(int), df["ib"].astype(int)
        df["y_cls"] = (df["bliss"] > syn_threshold).astype(np.float32)
        lin = dict(zip(om["cell_ids"], om["lineage"]))
        df["lineage"] = df["cell_id"].map(lambda c: str(lin[c]))
        df["pair_key"] = [f"{min(a, b)}_{max(a, b)}" for a, b in zip(df.ia, df.ib)]
        self.df = df
        self.has_mono = bool(np.isfinite(df[["mono_a", "mono_b"]].values).any())

        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.cell_feats = T(om["feats"])                       # (C,P,3)
        self.cell_raw = T(om["raw"])                           # (C,3G)
        self.drug_x = T(dr["x"])                               # (D,F)
        self.drug_t = T(dr["target_pw"])                       # (D,P)
        self.ci, self.ia, self.ib = (T(df[c].values, torch.long) for c in ["ci", "ia", "ib"])
        self.y = {c: T(df[c].values) for c in ["bliss", "hsa", "y_cls", "mono_a", "mono_b"]}

    def __len__(self):
        return len(self.df)

    def batch(self, idx, swap=None):
        """idx: LongTensor/array of row indices. swap: optional bool tensor -> exchange drug A/B."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        ia, ib = self.ia[idx], self.ib[idx]
        ma, mb = self.y["mono_a"][idx], self.y["mono_b"][idx]
        if swap is not None:
            ia, ib = torch.where(swap, ib, ia), torch.where(swap, self.ia[idx], ib)
            ma, mb = torch.where(swap, mb, ma), torch.where(swap, self.y["mono_a"][idx], mb)
        c = self.ci[idx]
        return dict(feats=self.cell_feats[c], raw=self.cell_raw[c],
                    xa=self.drug_x[ia], xb=self.drug_x[ib], ta=self.drug_t[ia], tb=self.drug_t[ib],
                    bliss=self.y["bliss"][idx], hsa=self.y["hsa"][idx], y_cls=self.y["y_cls"][idx],
                    mono_a=ma, mono_b=mb)


# --------------------------------------------------------------------------------------------
def _group_folds(groups, n_splits, seed):
    """Seeded group k-fold: whole groups go to one fold."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def make_splits(df, scheme, n_splits=5, seed=42, min_test=50):
    """Returns list of (train_idx, test_idx).
    random       : stratified (by synergy label) K-fold over rows
    cell         : cell-line cold start (group K-fold on cell line)
    pair         : drug-pair cold start (group K-fold on unordered pair)
    cancer_type  : leave-one-lineage-out (one fold per lineage with >= min_test rows)
    """
    if scheme == "random":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(df)), df["y_cls"].values))
    if scheme == "cell":
        return _group_folds(df["cell_id"].values, n_splits, seed)
    if scheme == "pair":
        return _group_folds(df["pair_key"].values, n_splits, seed)
    if scheme == "cancer_type":
        out = []
        for lin, n in df["lineage"].value_counts().items():
            if n >= min_test:
                m = (df["lineage"] == lin).values
                out.append((np.where(~m)[0], np.where(m)[0]))
        return out
    raise ValueError(scheme)


def inner_val_split(df, train_idx, scheme, frac=0.1, seed=42):
    """Hold out `frac` of the training rows for early stopping, respecting the split's grouping
    (so e.g. cell-cold-start validation also uses unseen cell lines)."""
    rng = np.random.RandomState(seed)
    col = {"cell": "cell_id", "pair": "pair_key", "cancer_type": "cell_id"}.get(scheme)
    if col is None:                                   # random / full: row-level holdout
        p = rng.permutation(train_idx)
        k = int(len(p) * frac)
        return p[k:], p[:k]
    g = df[col].values[train_idx]
    u = np.unique(g)
    rng.shuffle(u)
    val_g = set(u[: max(1, int(len(u) * frac))])
    m = np.array([x in val_g for x in g])
    return train_idx[~m], train_idx[m]
