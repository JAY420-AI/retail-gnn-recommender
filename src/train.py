"""Train MF-BPR / LightGCN with BPR loss, negative sampling, L2 reg, Adam and
early stopping on validation NDCG@10."""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import graph as G
from .evaluate import evaluate
from .models import LightGCN, MFBPR, Popularity


def set_seed(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_negatives(users, pos_keys, n_items, rng):
    """One uniform random negative per positive; resample once if it hit a known positive."""
    neg = rng.integers(0, n_items, len(users))
    for _ in range(3):
        bad = np.isin(users * n_items + neg, pos_keys)
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, bad.sum())
    return neg


def bpr_loss(model, u, i, j, reg):
    """-log sigmoid(s_ui - s_uj) + L2 on the *initial* (layer-0) embeddings of the batch."""
    pos, neg = model(u, i, j)
    loss = -torch.nn.functional.logsigmoid(pos - neg).mean()
    eu, ei, ej = model.base_embeddings(u, i, j)
    l2 = (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / (2 * len(u))
    return loss + reg * l2, loss


def make_model(name, train_df, nu, ni, dim=C.EMB_DIM, layers=C.N_LAYERS):
    if name == "popularity":
        return Popularity(train_df.item.values, ni)
    if name == "mf_bpr":
        return MFBPR(nu, ni, dim)
    adj = G.build_norm_adj(train_df.user.values, train_df.item.values, nu, ni)
    return LightGCN(nu, ni, adj, dim, layers)


def load_trained(name, train_df, nu, ni, dim=C.EMB_DIM, layers=C.N_LAYERS):
    """Rebuild a model and load its best checkpoint (popularity needs none)."""
    model = make_model(name, train_df, nu, ni, dim, layers)
    if name != "popularity":
        model.load_state_dict(torch.load(C.MODEL_DIR / f"{name}.pt"))
        model.eval()
    return model


def train(name, dim=C.EMB_DIM, layers=C.N_LAYERS, lr=C.LR, reg=C.REG, batch=C.BATCH,
          max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE):
    set_seed()
    nu, ni = G.sizes()
    tr, va = G.load_split("train"), G.load_split("val")
    model = make_model(name, tr, nu, ni, dim, layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    users, items = tr.user.values, tr.item.values
    pos_keys = np.sort(users.astype(np.int64) * ni + items)
    rng = np.random.default_rng(C.SEED)
    best, bad, hist = -1.0, 0, []
    C.MODEL_DIR.mkdir(exist_ok=True)
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        negs = sample_negatives(users.astype(np.int64), pos_keys, ni, rng)
        perm = rng.permutation(len(users))
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            u, i, j = (torch.as_tensor(a[idx]) for a in (users, items, negs))
            loss, bpr = bpr_loss(model, u, i, j, reg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += bpr.item() * len(idx)
        model.eval()
        v = evaluate(model, va, [tr], ni, ks=(10,))["ndcg@10"]
        hist.append({"epoch": ep, "train_bpr_loss": tot / len(perm), "val_ndcg@10": v})
        print(f"[{name}] epoch {ep:3d}  loss {tot/len(perm):.4f}  val NDCG@10 {v:.4f}  ({time.time()-t0:.1f}s)")
        if v > best:
            best, bad = v, 0
            torch.save(model.state_dict(), C.MODEL_DIR / f"{name}.pt")
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
    C.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(hist).to_csv(C.RESULTS_DIR / f"history_{name}.csv", index=False)
    print(f"[{name}] best val NDCG@10 = {best:.4f}")
    return hist


def plot_curves(names=("mf_bpr", "lightgcn")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in names:
        h = pd.read_csv(C.RESULTS_DIR / f"history_{n}.csv")
        ax[0].plot(h.epoch, h.train_bpr_loss, label=n)
        ax[1].plot(h.epoch, h["val_ndcg@10"], label=n)
    ax[0].set(title="Training BPR loss", xlabel="epoch")
    ax[1].set(title="Validation NDCG@10", xlabel="epoch")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mf_bpr", "lightgcn", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--dim", type=int, default=C.EMB_DIM)
    ap.add_argument("--layers", type=int, default=C.N_LAYERS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--reg", type=float, default=C.REG)
    a = ap.parse_args()
    for n in (["mf_bpr", "lightgcn"] if a.model == "all" else [a.model]):
        train(n, a.dim, a.layers, a.lr, a.reg, max_epochs=a.epochs)
    if a.model == "all":
        plot_curves()
