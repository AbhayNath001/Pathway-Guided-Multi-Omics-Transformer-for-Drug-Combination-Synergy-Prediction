"""Shared helpers: seeding, name normalisation, metrics."""
import os
import random
import re

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def norm_name(x) -> str:
    """Normalise cell-line / drug names so 'MCF-7', 'mcf7', 'MCF 7' all match."""
    if isinstance(x, float) and np.isfinite(x) and x == int(x):
        x = int(x)
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()


def regression_metrics(y, p) -> dict:
    from scipy.stats import pearsonr, spearmanr
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3 or np.std(y[m]) == 0 or np.std(p[m]) == 0:
        return {"rmse": np.nan, "pearson": np.nan, "spearman": np.nan, "n": int(m.sum())}
    y, p = y[m], p[m]
    return {
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p)[0]),
        "n": int(m.sum()),
    }


def classification_metrics(y, prob) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, prob = np.asarray(y, float), np.asarray(prob, float)
    m = np.isfinite(y) & np.isfinite(prob)
    y, prob = y[m], prob[m]
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan}
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
