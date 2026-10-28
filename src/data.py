"""Data pipeline: download -> filter -> subsample -> ID mapping -> processed files.

The Amazon Reviews 2023 benchmark already ships 5-core interaction files that
are split by *global timestamp* (train < valid < test), so there is no
temporal leakage between the splits. We only keep positive interactions
(rating >= 4) and map raw string IDs to dense integer indices.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- download
def download(raw_dir: Path = C.RAW_DIR, category: str = C.CATEGORY) -> None:
    """Fetch the 5-core split CSVs and the item metadata from Hugging Face."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {f"{category}.{s}.csv": f"{C.HF_BASE}/benchmark/5core/timestamp/{category}.{s}.csv"
             for s in ("train", "valid", "test")}
    files[f"meta_{category}.jsonl"] = f"{C.HF_BASE}/raw/meta_categories/meta_{category}.jsonl"
    for name, url in files.items():
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)


# -------------------------------------------------------------- synthetic
def make_synthetic(raw_dir: Path, category: str = C.CATEGORY, n_users=3000, n_items=1500,
                   seed: int = C.SEED) -> None:
    """Fallback ONLY: tiny fake data with the same schema (latent taste clusters).

    Metrics computed on it say nothing about real-world performance.
    """
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_cl = 12
    u_cl, i_cl = rng.integers(0, n_cl, n_users), rng.integers(0, n_cl, n_items)
    pop = rng.pareto(1.2, n_items) + 1
    rows = []
    for u in range(n_users):
        p = np.where(i_cl == u_cl[u], 8.0, 1.0) * pop
        items = rng.choice(n_items, size=int(rng.integers(8, 40)), replace=False, p=p / p.sum())
        ts = np.sort(rng.integers(1_500_000_000_000, 1_690_000_000_000, len(items)))
        for i, t in zip(items, ts):
            rows.append((f"U{u:05d}", f"P{i:05d}", float(rng.choice([3, 4, 5], p=[.15, .35, .5])), t))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    q1, q2 = df.timestamp.quantile([.8, .9])
    df[df.timestamp <= q1].to_csv(raw_dir / f"{category}.train.csv", index=False)
    df[(df.timestamp > q1) & (df.timestamp <= q2)].to_csv(raw_dir / f"{category}.valid.csv", index=False)
    df[df.timestamp > q2].to_csv(raw_dir / f"{category}.test.csv", index=False)
    with open(raw_dir / f"meta_{category}.jsonl", "w") as f:
        for i in range(n_items):
            f.write(json.dumps({"parent_asin": f"P{i:05d}", "title": f"Synthetic Game {i}",
                                "categories": ["Video Games", f"Genre {i_cl[i]}"],
                                "price": round(float(rng.uniform(5, 60)), 2)}) + "\n")


# --------------------------------------------------------------- filtering
def filter_and_sample(train, valid, test, n_users, seed):
    """Keep positives, sample users, then iteratively enforce min-degree on train."""
    pos = lambda d: d[d.rating >= C.POS_RATING][["user_id", "parent_asin", "timestamp"]].copy()
    train, valid, test = pos(train), pos(valid), pos(test)
    users = train.user_id.unique()
    if len(users) > n_users:
        # Random users would almost never have val/test activity (most 5-core users
        # finish in train), leaving a uselessly small evaluation set. So we first
        # keep users that appear in val/test, then fill up with random others.
        # This only picks WHICH users to model; no val/test edge enters the graph.
        rng = np.random.default_rng(seed)
        later = set(valid.user_id) | set(test.user_id)
        prio = np.array(sorted(set(users) & later))
        rest = np.array(sorted(set(users) - later))
        if len(prio) > n_users:
            keep = rng.choice(prio, n_users, replace=False)
        else:
            keep = np.concatenate([prio, rng.choice(rest, n_users - len(prio), replace=False)])
        train = train[train.user_id.isin(keep)]
    while True:  # alternate user/item degree filters until stable
        ic = train.parent_asin.value_counts()
        train = train[train.parent_asin.isin(ic.index[ic >= C.MIN_ITEM_TRAIN])]
        uc = train.user_id.value_counts()
        train = train[train.user_id.isin(uc.index[uc >= C.MIN_USER_TRAIN])]
        if (train.parent_asin.value_counts() >= C.MIN_ITEM_TRAIN).all():
            break
    u_set, i_set = set(train.user_id), set(train.parent_asin)
    sel = lambda d: d[d.user_id.isin(u_set) & d.parent_asin.isin(i_set)]
    return train, sel(valid), sel(test)


def load_meta(path: Path, asins: set) -> pd.DataFrame:
    """Stream the (large) metadata file, keeping only the items we need."""
    rows = {}
    with open(path) as f:
        for line in f:
            if '"parent_asin"' not in line:
                continue
            d = json.loads(line)
            a = d.get("parent_asin")
            if a in asins and a not in rows:
                cats = d.get("categories") or []
                rows[a] = {"asin": a, "title": d.get("title") or a,
                           "category": " > ".join(cats[1:]) if len(cats) > 1 else (d.get("main_category") or ""),
                           "price": pd.to_numeric(d.get("price"), errors="coerce")}
    meta = pd.DataFrame(list(rows.values()), columns=["asin", "title", "category", "price"])
    missing = sorted(asins - set(meta.asin))
    if missing:  # item has no metadata record: keep it, title falls back to the ASIN
        meta = pd.concat([meta, pd.DataFrame({"asin": missing, "title": missing, "category": "", "price": np.nan})])
    return meta


# -------------------------------------------------------------------- main
def build(raw_dir=C.RAW_DIR, out_dir=C.PROC_DIR, category=C.CATEGORY,
          n_users=C.N_USERS_SAMPLE, seed=C.SEED):
    rd = lambda s: pd.read_csv(raw_dir / f"{category}.{s}.csv")
    train, valid, test = filter_and_sample(rd("train"), rd("valid"), rd("test"), n_users, seed)
    users = sorted(train.user_id.unique())
    items = sorted(train.parent_asin.unique())
    umap, imap = {u: k for k, u in enumerate(users)}, {a: k for k, a in enumerate(items)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train), ("val", valid), ("test", test)):
        enc = pd.DataFrame({"user": d.user_id.map(umap), "item": d.parent_asin.map(imap),
                            "timestamp": d.timestamp.values})
        enc.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(enc):7d} interactions, {enc.user.nunique():6d} users")
    pd.DataFrame({"user": range(len(users)), "user_id": users}).to_csv(out_dir / "users.csv", index=False)
    meta = load_meta(raw_dir / f"meta_{category}.jsonl", set(items)).set_index("asin").loc[items].reset_index()
    meta.insert(0, "item", range(len(items)))
    meta.to_csv(out_dir / "items.csv", index=False)
    print(f"saved processed files to {out_dir} ({len(users)} users, {len(items)} items)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="download real data first")
    ap.add_argument("--synthetic", action="store_true", help="use fake data (pipeline check only)")
    ap.add_argument("--raw-dir", type=Path, default=C.RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.PROC_DIR)
    ap.add_argument("--n-users", type=int, default=C.N_USERS_SAMPLE)
    a = ap.parse_args()
    if a.synthetic:
        print("WARNING: generating SYNTHETIC data - results are not real.")
        make_synthetic(a.raw_dir)
    elif a.download or not (a.raw_dir / f"{C.CATEGORY}.train.csv").exists():
        try:
            download(a.raw_dir)
        except Exception as e:  # network failure -> be explicit, never silently fake
            raise SystemExit(f"Download failed ({e}). Re-run later, or use --synthetic for a pipeline smoke test.")
    build(a.raw_dir, a.out_dir, n_users=a.n_users)
