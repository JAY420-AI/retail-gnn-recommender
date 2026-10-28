"""Tiny grid search (lr x reg) per trainable model, selected on VALIDATION NDCG@10 only.
Writes results/tuning.csv and prints the best setting; the test set is never touched."""
import itertools

import pandas as pd

from . import config as C
from .train import train

GRID = {"lr": [1e-3, 5e-3], "reg": [1e-4, 1e-3, 1e-2]}

if __name__ == "__main__":
    rows = []
    for name in ("mf_bpr", "lightgcn"):
        for lr, reg in itertools.product(GRID["lr"], GRID["reg"]):
            hist = train(name, lr=lr, reg=reg)
            rows.append({"model": name, "lr": lr, "reg": reg, "epochs": len(hist),
                         "best_val_ndcg@10": max(h["val_ndcg@10"] for h in hist)})
    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS_DIR / "tuning.csv", index=False)
    print(df.to_string(index=False))
    print(df.loc[df.groupby("model")["best_val_ndcg@10"].idxmax()].to_string(index=False))
