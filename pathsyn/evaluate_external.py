"""External validation on DrugComb (or any canonical-schema table) with a model trained by `--split full`.

python -m pathsyn.evaluate_external --model runs/final/model.pt --cache data/cache \
    --table data/processed/drugcomb.csv --train-table data/processed/gdsc2.csv --out runs/external

Rows are kept only if the cell line was in the training set and (>=1 | both) drugs were seen in training
(--require any|both); exact (cell line, drug pair) combos that are in the training table are REMOVED to avoid leakage.
NOTE: DrugComb and GDSC2 use different dose ranges / curve-fitting, so absolute Bliss values are not on the same scale;
rank metrics (Pearson/Spearman/AUROC/AUPRC) are the meaningful ones, RMSE is only indicative.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .data import PathSynData
from .train import build_model, pred_frame, predict, score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--table", required=True)
    ap.add_argument("--train-table", required=True)
    ap.add_argument("--out", default="runs/external")
    ap.add_argument("--require", choices=["any", "both"], default="any")
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.model, map_location=a.device)
    cfg, sc = ck["cfg"], ck["scalers"]
    tr = PathSynData(a.cache, a.train_table, cfg["syn_threshold"], "cpu", verbose=False)
    ext = PathSynData(a.cache, a.table, cfg["syn_threshold"], a.device)

    seen_cells, seen_drugs = set(tr.df.ci), set(tr.df.ia) | set(tr.df.ib)
    train_keys = set(zip(tr.df.ci, tr.df.pair_key))
    d = ext.df
    n_seen = d.ia.isin(seen_drugs).astype(int) + d.ib.isin(seen_drugs).astype(int)
    keep = d.ci.isin(seen_cells) & (n_seen >= (1 if a.require == "any" else 2))
    keep &= ~pd.Series([(c, p) in train_keys for c, p in zip(d.ci, d.pair_key)], index=d.index)
    idx = np.where(keep.values)[0]
    print(f"[external] {len(idx)}/{len(d)} rows kept (seen cell line, drugs: {a.require}, not in training)")
    if len(idx) == 0:
        raise SystemExit("no overlapping rows")

    model = build_model(cfg, ext).to(a.device)
    model.load_state_dict(ck["state"])
    pred = predict(model, ext, idx, sc, a.mc)
    res = {"overall": score(ext, idx, pred), "n": int(len(idx))}
    for k in (1, 2):
        sub = np.where(n_seen.values[idx] == k)[0]
        if len(sub) > 10:
            res[f"drugs_seen_{k}"] = score(ext, idx[sub], {n: v[sub] for n, v in pred.items()})
            res[f"drugs_seen_{k}"]["n"] = int(len(sub))
    pred_frame(ext, idx, pred).to_csv(os.path.join(a.out, "external_predictions.csv"), index=False)
    with open(os.path.join(a.out, "external_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
