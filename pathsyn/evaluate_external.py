"""Frozen-model evaluation on an external canonical-schema response table.

External tables must contain cell_line, drug_a, drug_b, bliss and hsa. External
cell lines and drugs must be represented in the DepMap/chemical feature cache.
Exact training (cell-line, unordered-drug-pair) records are removed.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--table", required=True, help="External canonical response table")
    parser.add_argument("--train-table", required=True, help="Canonical response table used for model development")
    parser.add_argument("--out", default="runs/external")
    parser.add_argument("--cell-policy", choices=["available", "seen"], default="available",
                         help="available: any cell line in the omics cache; seen: only training cell lines")
    parser.add_argument("--require", choices=["none", "any", "both"], default="none",
                         help="Require none/any/both drugs to have appeared in the training response table")
    parser.add_argument("--mc", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    checkpoint = torch.load(args.model, map_location=args.device)
    config, scalers = checkpoint["cfg"], checkpoint["scalers"]
    train_data = PathSynData(args.cache, args.train_table, config["syn_threshold"], "cpu", verbose=False)
    external_data = PathSynData(args.cache, args.table, config["syn_threshold"], args.device)

    seen_cells = set(train_data.df["cell_id"])
    seen_drugs = set(train_data.df["ia"]) | set(train_data.df["ib"])
    train_pairs = set(zip(train_data.df["cell_id"], train_data.df["pair_key"]))

    external_frame = external_data.df
    drug_seen_count = (
        external_frame["ia"].isin(seen_drugs).astype(int)
        + external_frame["ib"].isin(seen_drugs).astype(int)
    )
    keep = np.ones(len(external_frame), dtype=bool)
    if args.cell_policy == "seen":
        keep &= external_frame["cell_id"].isin(seen_cells).to_numpy()
    if args.require == "any":
        keep &= drug_seen_count.to_numpy() >= 1
    elif args.require == "both":
        keep &= drug_seen_count.to_numpy() == 2

    exact_overlap = np.array([
        (cell, pair) in train_pairs
        for cell, pair in zip(external_frame["cell_id"], external_frame["pair_key"])
    ])
    keep &= ~exact_overlap
    indices = np.flatnonzero(keep)

    print(
        f"[external] usable after feature matching: {len(external_frame)}; "
        f"retained after overlap/policy filters: {len(indices)}"
    )
    if len(indices) == 0:
        raise SystemExit("No eligible external records remain after feature matching and overlap filters.")

    model = build_model(config, external_data).to(args.device)
    model.load_state_dict(checkpoint["state"])
    predictions = predict(model, external_data, indices, scalers, args.mc)
    metrics = {
        "overall": score(external_data, indices, predictions),
        "n_retained": int(len(indices)),
        "n_feature_matched": int(len(external_frame)),
        "cell_policy": args.cell_policy,
        "drug_policy": args.require,
        "exact_training_pair_overlap_removed": int(exact_overlap.sum()),
    }

    pred_frame(external_data, indices, predictions).to_csv(
        os.path.join(args.out, "external_predictions.csv"), index=False
    )
    with open(os.path.join(args.out, "external_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
