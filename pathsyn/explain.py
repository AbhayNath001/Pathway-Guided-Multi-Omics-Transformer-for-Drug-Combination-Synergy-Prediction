"""Pathway attributions for one (cell line, drug A, drug B) prediction.

python -m pathsyn.explain --model runs/final/model.pt --cache data/cache --cell MCF7 --drug-a Olaparib --drug-b Cisplatin
Two scores per pathway: cross-attention weight (last layer, mean over query tokens) and gradient x input
(effect of that pathway's [RNA,CNV,MUT] features on the predicted Bliss excess).
"""
import argparse

import numpy as np
import pandas as pd
import torch

from .data import MODS, PathSynData
from .train import build_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--drug-a", required=True)
    ap.add_argument("--drug-b", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    ck = torch.load(a.model, map_location=a.device)
    cfg, sc = ck["cfg"], ck["scalers"]
    if cfg["model"] != "pathsyn":
        raise SystemExit("explanations only exist for the pathway model")
    one = pd.DataFrame({"cell_line": [a.cell], "drug_a": [a.drug_a], "drug_b": [a.drug_b], "bliss": [0.0], "hsa": [0.0]})
    data = PathSynData(a.cache, one, cfg["syn_threshold"], a.device)
    if len(data) != 1:
        raise SystemExit("cell line or drug not found in cache")
    model = build_model(cfg, data).to(a.device)
    model.load_state_dict(ck["state"])
    model.eval()

    b = data.batch([0])
    b["feats"] = b["feats"].clone().requires_grad_(True)
    out = model(b, return_attn=True)
    out["bliss"].sum().backward()
    grad_x_in = (b["feats"].grad * b["feats"]).detach()[0]            # (P,3)
    attn = out["attn"].detach()[0].mean(0).cpu().numpy()               # (P,)
    pred = out["bliss"].item() * sc["bliss"][1] + sc["bliss"][0]
    print(f"predicted Bliss excess {pred:.2f} | P(synergy) {torch.sigmoid(out['logit']).item():.2f}")

    df = pd.DataFrame({"pathway": data.pathway_names, "attention": attn,
                       "grad_x_input": grad_x_in.sum(-1).cpu().numpy(),
                       **{f"gxi_{m}": grad_x_in[:, i].cpu().numpy() for i, m in enumerate(MODS)},
                       "targeted_by_A": data.drug_t[data.ia[0]].cpu().numpy(),
                       "targeted_by_B": data.drug_t[data.ib[0]].cpu().numpy()})
    df["abs_gxi"] = df["grad_x_input"].abs()
    print("\nTop pathways by |gradient x input|:")
    print(df.sort_values("abs_gxi", ascending=False).head(a.top).round(4).to_string(index=False))
    print("\nTop pathways by attention:")
    print(df.sort_values("attention", ascending=False).head(a.top).round(4).to_string(index=False))
    if a.out:
        df.drop(columns="abs_gxi").to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
