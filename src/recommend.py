"""Inference API: recommend for a known customer, or build a cold-start profile."""
from functools import lru_cache
from typing import Iterable, Union

import pandas as pd
import torch

from . import config as C
from . import graph as G
from .train import load_trained


class Recommender:
    def __init__(self):
        self.users = pd.read_csv(C.PROC_DIR / "users.csv")
        self.items = pd.read_csv(C.PROC_DIR / "items.csv")
        self.train = G.load_split("train")
        self.val = G.load_split("val")
        nu, ni = len(self.users), len(self.items)
        self.model = load_trained("lightgcn", self.train, nu, ni)
        with torch.no_grad():                       # propagate once, reuse for every query
            self.U, self.I = self.model.final_embeddings()
        self.uidx = {u: k for k, u in enumerate(self.users.user_id)}
        self.aidx = {a: k for k, a in enumerate(self.items.asin)}
        self.pop = self.train.item.value_counts()
        # history = everything the customer is known to have bought (train + val)
        self.hist = pd.concat([self.train, self.val]).groupby("user").item.apply(list).to_dict()

    # ---------------------------------------------------------------- helpers
    def _user(self, user: Union[str, int]) -> int:
        if isinstance(user, str) and user in self.uidx:
            return self.uidx[user]
        if isinstance(user, int) and 0 <= user < len(self.users):
            return user
        raise KeyError(f"unknown customer: {user!r}")

    def _table(self, idx, scores) -> pd.DataFrame:
        df = self.items.iloc[idx][["asin", "title", "category", "price"]].reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        df["score"] = [float(s) for s in scores]
        return df

    def history(self, user: Union[str, int]) -> pd.DataFrame:
        """Items the customer already interacted with (most recent first)."""
        u = self._user(user)
        ts = pd.concat([self.train, self.val])
        ids = ts[ts.user == u].sort_values("timestamp", ascending=False).item.tolist()
        return self.items.iloc[ids][["asin", "title", "category", "price"]].reset_index(drop=True)

    # ------------------------------------------------------------------- API
    @torch.no_grad()
    def recommend(self, user_id: Union[str, int], k: int = 10) -> pd.DataFrame:
        """Top-k unseen items for a known customer (raw Amazon user_id or index)."""
        u = self._user(user_id)
        scores = self.U[u] @ self.I.T
        scores[self.hist.get(u, [])] = -float("inf")        # never re-recommend owned items
        top = scores.topk(k)
        return self._table(top.indices.tolist(), top.values.tolist())

    @torch.no_grad()
    def recommend_from_profile(self, product_ids: Iterable[str], k: int = 10) -> pd.DataFrame:
        """Cold start: average the embeddings of the picked products, rank by dot product."""
        idx = [self.aidx[p] for p in product_ids if p in self.aidx]
        if not idx:
            raise ValueError("no known products in profile")
        profile = self.I[idx].mean(0)
        scores = self.I @ profile
        scores[idx] = -float("inf")
        top = scores.topk(k)
        return self._table(top.indices.tolist(), top.values.tolist())


@lru_cache(maxsize=1)
def _default() -> Recommender:
    return Recommender()


def recommend(user_id, k=10) -> pd.DataFrame:
    return _default().recommend(user_id, k)


def recommend_from_profile(product_ids, k=10) -> pd.DataFrame:
    return _default().recommend_from_profile(product_ids, k)


if __name__ == "__main__":
    r = _default()
    uid = r.users.user_id[0]
    print("history:\n", r.history(uid).head(5).to_string(index=False))
    print("recommendations:\n", r.recommend(uid, 5).to_string(index=False))
    pick = r.items.asin[r.pop.index[:2]].tolist()
    print("profile", pick, "\n", r.recommend_from_profile(pick, 5).to_string(index=False))
