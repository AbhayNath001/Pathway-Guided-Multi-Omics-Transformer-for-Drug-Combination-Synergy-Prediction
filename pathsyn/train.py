"""Train / cross-validate PATH-SYN.

Examples
  python -m pathsyn.train --cache data/cache --table data/processed/gdsc2.csv --split random      --out runs/random
  python -m pathsyn.train ... --split cell | pair | cancer_type
  python -m pathsyn.train ... --split full --out runs/final      # fit on all data -> runs/final/model.pt (for external eval)
  python -m pathsyn.train ... --mods rna,mut                      # omics ablation
  python -m pathsyn.train ... --model baseline                    # raw-omics MLP baseline
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import MODS, PathSynData, inner_val_split, make_splits
from .model import PathSyn, RawMLP
from .utils import classification_metrics, regression_metrics, set_seed


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--split", default="random", choices=["random", "cell", "pair", "cancer_type", "full"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=None, help="run only the first k folds (debug)")
    ap.add_argument("--model", default="pathsyn", choices=["pathsyn", "baseline"])
    ap.add_argument("--mods", default="rna,cnv,mut")
    ap.add_argument("--syn-threshold", type=float, default=0.0, help="Bliss > thr => synergistic")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of classification loss")
    ap.add_argument("--mu", type=float, default=0.5, help="weight of monotherapy auxiliary loss")
    ap.add_argument("--mc", type=int, default=30, help="MC-dropout samples at test time (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def build_model(cfg, data):
    mods = tuple(int(m in cfg["mods"].split(",")) for m in MODS)
    if cfg["model"] == "baseline":
        return RawMLP(data.n_raw_genes, data.drug_dim, mods, cfg["d"], cfg["dropout"])
    return PathSyn(data.n_path, data.drug_dim, cfg["d"], cfg["heads"], cfg["layers"], cfg["dropout"], mods)


def fit_scalers(data, idx):
    d = data.df.iloc[idx]
    mono = np.concatenate([d["mono_a"].values, d["mono_b"].values])
    mono = mono[np.isfinite(mono)]
    f = lambda v: (float(np.mean(v)), float(np.std(v)) or 1.0)
    return {"bliss": f(d["bliss"].values), "hsa": f(d["hsa"].values),
            "mono": f(mono) if len(mono) else (0.0, 1.0)}


def mono_loss(model, b, sc):
    k = model.key
    inp, x, t = torch.cat([b[k], b[k]]), torch.cat([b["xa"], b["xb"]]), torch.cat([b["ta"], b["tb"]])
    y = torch.cat([b["mono_a"], b["mono_b"]])
    m = torch.isfinite(y)
    if m.sum() == 0:
        return None
    pred = model.forward_mono(inp[m], x[m], t[m])
    return F.mse_loss(pred, (y[m] - sc["mono"][0]) / sc["mono"][1])


def run_epoch(model, data, idx, sc, cfg, opt=None):
    train = opt is not None
    model.train(train)
    order = np.random.permutation(idx) if train else np.asarray(idx)
    tot, n = 0.0, 0
    for s in range(0, len(order), cfg["bs"]):
        bi = order[s:s + cfg["bs"]]
        swap = (torch.rand(len(bi), device=data.device) < 0.5) if train else None   # A/B order augmentation
        b = data.batch(bi, swap)
        with torch.set_grad_enabled(train):
            out = model(b)
            loss = (F.mse_loss(out["bliss"], (b["bliss"] - sc["bliss"][0]) / sc["bliss"][1])
                    + F.mse_loss(out["hsa"], (b["hsa"] - sc["hsa"][0]) / sc["hsa"][1])
                    + cfg["lam"] * F.binary_cross_entropy_with_logits(out["logit"], b["y_cls"]))
            if cfg["mu"] > 0 and data.has_mono:
                lm = mono_loss(model, b, sc)
                if lm is not None:
                    loss = loss + cfg["mu"] * lm
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * len(bi)
        n += len(bi)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, data, idx, sc, mc=0, bs=256):
    """Returns dict of arrays: bliss/hsa/prob (mean) and *_sd (MC-dropout std). A/B-order symmetrised."""
    idx = np.asarray(idx)
    means, sds = [], []
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]
        flip = torch.ones(len(bi), dtype=torch.bool, device=data.device)
        samples = []
        for _ in range(max(mc, 1)):
            model.train(mc > 0)                                # MC dropout keeps dropout active
            ps = []
            for sw in (None, flip):
                o = model(data.batch(bi, sw))
                ps.append(torch.stack([o["bliss"] * sc["bliss"][1] + sc["bliss"][0],
                                       o["hsa"] * sc["hsa"][1] + sc["hsa"][0],
                                       torch.sigmoid(o["logit"])]))
            samples.append(torch.stack(ps).mean(0))
        S = torch.stack(samples)                               # (n,3,B)
        means.append(S.mean(0).cpu())
        sds.append(S.std(0).cpu() if S.size(0) > 1 else torch.zeros_like(S[0]).cpu())
    m, sd = torch.cat(means, 1).numpy(), torch.cat(sds, 1).numpy()
    return {"bliss": m[0], "hsa": m[1], "prob": m[2], "bliss_sd": sd[0], "hsa_sd": sd[1], "prob_sd": sd[2]}


def score(data, idx, pred):
    d = data.df.iloc[idx]
    r = {}
    for k in ["bliss", "hsa"]:
        r.update({f"{k}_{a}": b for a, b in regression_metrics(d[k].values, pred[k]).items()})
    r.update(classification_metrics(d["y_cls"].values, pred["prob"]))
    return r


def pred_frame(data, idx, pred):
    d = data.df.iloc[idx]
    out = d[["cell_line", "drug_a", "drug_b", "lineage", "bliss", "hsa", "y_cls"]].reset_index(drop=True)
    for k, v in pred.items():
        out["p_" + k] = v
    return out


def train_one(cfg, data, tr, va, sc, log=print):
    model = build_model(cfg, data).to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        l_tr = run_epoch(model, data, tr, sc, cfg, opt)
        l_va = run_epoch(model, data, va, sc, cfg)
        log(f"  epoch {ep:02d}  train {l_tr:.4f}  val {l_va:.4f}  ({time.time() - t0:.1f}s)")
        if l_va < best - 1e-4:
            best, best_state, bad = l_va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("  early stopping")
                break
    model.load_state_dict(best_state)
    return model


def main(argv=None):
    cfg = vars(get_parser().parse_args(argv))
    os.makedirs(cfg["out"], exist_ok=True)
    set_seed(cfg["seed"])
    data = PathSynData(cfg["cache"], cfg["table"], cfg["syn_threshold"], cfg["device"])
    df = data.df
    print(f"[train] {len(df)} samples | {df.cell_id.nunique()} cell lines | "
          f"{len(set(df.ia) | set(df.ib))} drugs | {df.pair_key.nunique()} drug pairs | "
          f"synergistic fraction {df.y_cls.mean():.2f} | monotherapy labels: {data.has_mono}")
    with open(os.path.join(cfg["out"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    if cfg["split"] == "full":
        folds = [(np.arange(len(df)), np.array([], int))]
    else:
        folds = make_splits(df, cfg["split"], cfg["folds"], cfg["seed"])
    if cfg["max_folds"]:
        folds = folds[:cfg["max_folds"]]

    rows, all_preds, split_rows = [], [], []
    for k, (tr_all, te) in enumerate(folds):
        tr, va = inner_val_split(df, tr_all, cfg["split"], 0.1, cfg["seed"])
        for nm, ii in [("train", tr), ("val", va), ("test", te)]:
            split_rows.append(pd.DataFrame({"fold": k, "set": nm, "row": ii}))
        print(f"[fold {k}] train {len(tr)}  val {len(va)}  test {len(te)}")
        set_seed(cfg["seed"] + k)
        sc = fit_scalers(data, tr)
        model = train_one(cfg, data, tr, va, sc)
        if len(te):
            pred = predict(model, data, te, sc, cfg["mc"])
            m = score(data, te, pred)
            if cfg["split"] == "cancer_type":
                m["held_out"] = df.lineage.values[te][0]
            m["fold"] = k
            rows.append(m)
            fr = pred_frame(data, te, pred)
            fr["fold"] = k
            all_preds.append(fr)
            print("  test:", {a: round(b, 4) for a, b in m.items() if isinstance(b, float)})
        else:
            torch.save({"state": model.state_dict(), "cfg": cfg, "scalers": sc, "n_path": data.n_path,
                        "drug_dim": data.drug_dim, "n_raw_genes": data.n_raw_genes},
                       os.path.join(cfg["out"], "model.pt"))
            print(f"[train] saved {cfg['out']}/model.pt")

    pd.concat(split_rows).to_csv(os.path.join(cfg["out"], "splits.csv"), index=False)
    if rows:
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(cfg["out"], "fold_metrics.csv"), index=False)
        pd.concat(all_preds).to_csv(os.path.join(cfg["out"], "predictions.csv"), index=False)
        num = res.select_dtypes("number").drop(columns=["fold", "n"], errors="ignore")
        summ = pd.DataFrame({"mean": num.mean(), "std": num.std()})
        summ.to_csv(os.path.join(cfg["out"], "summary.csv"))
        print("\n=== summary (mean +- std over folds) ===")
        print(summ.round(4))
    return cfg["out"]


if __name__ == "__main__":
    main()
