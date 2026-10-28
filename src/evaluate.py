"""Full-ranking evaluation: every user is ranked against ALL items, with
already-seen items masked out. Metrics: Precision/Recall/NDCG @ K."""
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import graph as G


def topk_metrics(topk: torch.Tensor, truth: torch.Tensor, ks=C.KS) -> dict:
    """Per-user metrics.

    topk:  [B, Kmax] ranked item indices.   truth: [B, n_items] bool relevance.
    Returns {metric_K: tensor[B]} (users with no relevant item must be dropped by caller).
    """
    hits = truth.gather(1, topk).float()                          # [B, Kmax]
    n_rel = truth.sum(1).float()
    disc = 1.0 / torch.log2(torch.arange(topk.shape[1]).float() + 2.0)
    out = {}
    for k in ks:
        h = hits[:, :k]
        dcg = (h * disc[:k]).sum(1)
        ideal_n = torch.clamp(n_rel, max=k).long()                # ideal: all relevant ranked first
        idcg = torch.cumsum(disc[:k], 0)[ideal_n - 1]
        out[f"precision@{k}"] = h.sum(1) / k
        out[f"recall@{k}"] = h.sum(1) / n_rel
        out[f"ndcg@{k}"] = dcg / idcg
    return out


def build_sets(df: pd.DataFrame, n_items: int):
    """user -> list of items."""
    return df.groupby("user").item.apply(list).to_dict()


@torch.no_grad()
def evaluate(model, target: pd.DataFrame, seen_dfs, n_items: int, ks=C.KS, batch=512) -> dict:
    """Mean metrics over users that have >=1 target item.

    target: the split being scored. seen_dfs: splits whose items are masked (train, +val for test).
    """
    tgt = build_sets(target, n_items)
    seen = build_sets(pd.concat(seen_dfs), n_items)
    users = np.array(sorted(tgt))
    acc = {}
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        scores = model.score(torch.as_tensor(ub)).clone()
        truth = torch.zeros(len(ub), n_items, dtype=torch.bool)
        for r, u in enumerate(ub):
            scores[r, seen.get(u, [])] = -float("inf")            # mask history
            truth[r, tgt[u]] = True
        topk = scores.topk(max(ks), dim=1).indices
        for k, v in topk_metrics(topk, truth, ks).items():
            acc.setdefault(k, []).append(v)
    res = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    res["n_eval_users"] = len(users)
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .train import load_trained
    nu, ni = G.sizes()
    tr, va, te = (G.load_split(s) for s in ("train", "val", "test"))
    rows = []
    for name in ("popularity", "mf_bpr", "lightgcn"):
        model = load_trained(name, tr, nu, ni)
        res = evaluate(model, te, [tr, va], ni)           # test: mask train + val history
        print(name, {k: round(v, 4) for k, v in res.items()})
        rows.append({"model": name, **res})
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"popularity": "#9aa5b1", "mf_bpr": "#4c78a8", "lightgcn": "#e8743b"}
    for ax, metric in zip(axes, ("precision", "recall", "ndcg")):
        cols = [f"{metric}@{k}" for k in C.KS]
        x = np.arange(len(C.KS))
        for j, r in df.iterrows():
            ax.bar(x + (j - 1) * 0.27, [r[c] for c in cols], 0.27, label=r["model"], color=colors[r["model"]])
        ax.set_xticks(x, [f"K={k}" for k in C.KS])
        ax.set_title(metric.upper() if metric == "ndcg" else metric.capitalize())
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Test-set comparison (full ranking, history masked)")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "model_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
