"""Three recommenders sharing one scoring interface: model.score(user_idx) -> [B, n_items]."""
import numpy as np
import torch
import torch.nn as nn

from . import config as C


class Popularity:
    """Non-personalised baseline: rank items by training interaction count."""
    name = "popularity"

    def __init__(self, train_items: np.ndarray, n_items: int):
        self.counts = torch.as_tensor(np.bincount(train_items, minlength=n_items), dtype=torch.float)

    def score(self, users: torch.Tensor) -> torch.Tensor:
        return self.counts.unsqueeze(0).repeat(len(users), 1)


class MFBPR(nn.Module):
    """Matrix factorisation: score = <user vector, item vector>, trained with BPR."""
    name = "mf_bpr"

    def __init__(self, n_users, n_items, dim=C.EMB_DIM):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def final_embeddings(self):
        return self.user_emb.weight, self.item_emb.weight

    def base_embeddings(self, u, i, j):
        return self.user_emb(u), self.item_emb(i), self.item_emb(j)

    def forward(self, u, i, j):
        U, I = self.final_embeddings()
        return (U[u] * I[i]).sum(-1), (U[u] * I[j]).sum(-1)

    @torch.no_grad()
    def score(self, users):
        U, I = self.final_embeddings()
        return U[users] @ I.T


class LightGCN(nn.Module):
    """LightGCN: e^(k+1) = A_hat e^(k) (no weights, no non-linearity);
    final embedding = mean of e^(0..K)."""
    name = "lightgcn"

    def __init__(self, n_users, n_items, adj: torch.Tensor, dim=C.EMB_DIM, n_layers=C.N_LAYERS):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.adj = adj
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self._cache = None

    def final_embeddings(self):
        x = self.emb.weight
        layers = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            layers.append(x)
        out = torch.stack(layers).mean(0)             # layer combination: mean
        return out[: self.n_users], out[self.n_users:]

    def base_embeddings(self, u, i, j):
        return self.emb(u), self.emb(i + self.n_users), self.emb(j + self.n_users)

    def forward(self, u, i, j):
        U, I = self.final_embeddings()
        return (U[u] * I[i]).sum(-1), (U[u] * I[j]).sum(-1)

    @torch.no_grad()
    def score(self, users):
        U, I = self.final_embeddings()
        return U[users] @ I.T
