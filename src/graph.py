"""Bipartite user-item graph and its symmetric-normalised adjacency matrix."""
import numpy as np
import pandas as pd
import warnings

import torch

warnings.filterwarnings("ignore", message="Sparse invariant")

from . import config as C


def load_split(name: str, proc_dir=C.PROC_DIR) -> pd.DataFrame:
    return pd.read_csv(proc_dir / f"{name}.csv")


def sizes(proc_dir=C.PROC_DIR):
    """Return (n_users, n_items) from the ID-mapping files."""
    return len(pd.read_csv(proc_dir / "users.csv")), len(pd.read_csv(proc_dir / "items.csv"))


def build_norm_adj(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """A_hat = D^-1/2 A D^-1/2 for A = [[0, R], [R^T, 0]] (sparse, (U+I)x(U+I)).

    Item node j is stored at index n_users + j. Only training edges are used.
    """
    u = torch.as_tensor(np.array(users), dtype=torch.long)
    i = torch.as_tensor(items, dtype=torch.long) + n_users
    rows = torch.cat([u, i])
    cols = torch.cat([i, u])
    n = n_users + n_items
    deg = torch.zeros(n).scatter_add_(0, rows, torch.ones(len(rows)))
    inv_sqrt = deg.pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0.0          # isolated nodes
    vals = inv_sqrt[rows] * inv_sqrt[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()


def graph_stats(users, items, n_users, n_items) -> dict:
    e = len(users)
    return {"users": n_users, "items": n_items, "edges": e,
            "density": e / (n_users * n_items),
            "avg_user_degree": e / n_users, "avg_item_degree": e / n_items}


if __name__ == "__main__":
    tr = load_split("train")
    nu, ni = sizes()
    for k, v in graph_stats(tr.user.values, tr.item.values, nu, ni).items():
        print(f"{k:16s}: {v:.5f}" if isinstance(v, float) else f"{k:16s}: {v}")
    A = build_norm_adj(tr.user.values, tr.item.values, nu, ni)
    print("normalised adjacency:", tuple(A.shape), "nnz =", A._nnz())
