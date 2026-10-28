"""Builds notebooks/walkthrough.ipynb (then executed with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("# Retail Product Recommendation using LightGCN - walkthrough\n"
   "End-to-end run on Amazon Reviews 2023 (Video_Games, subsample). "
   "Run `python -m src.data`, `python -m src.train`, `python -m src.evaluate` first; "
   "this notebook loads the saved checkpoints so numbers match `results/metrics.csv`."),
code("import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')\n"
     "import pandas as pd, matplotlib.pyplot as plt\n"
     "from src import config as C, graph as G\n"
     "from src.evaluate import evaluate\n"
     "from src.train import load_trained\n"
     "pd.set_option('display.max_colwidth', 70); pd.set_option('display.width', 200)"),
md("## 1. Data\nTemporal split (train < valid < test), positives = rating >= 4."),
code("tr, va, te = (G.load_split(s) for s in ('train','val','test'))\n"
     "items = pd.read_csv(C.PROC_DIR/'items.csv')\n"
     "print({n: len(d) for n, d in zip(('train','val','test'), (tr, va, te))})\n"
     "print('time ranges (ms):', [(int(d.timestamp.min()), int(d.timestamp.max())) for d in (tr, va, te)])\n"
     "items.head()"),
md("## 2. Graph (training edges only)"),
code("nu, ni = G.sizes()\n"
     "print(G.graph_stats(tr.user.values, tr.item.values, nu, ni))\n"
     "A = G.build_norm_adj(tr.user.values, tr.item.values, nu, ni)\n"
     "print('normalised adjacency', tuple(A.shape), 'nnz', A._nnz())\n"
     "deg = tr.item.value_counts()\n"
     "plt.figure(figsize=(5,3.2)); plt.hist(deg, bins=50, log=True); plt.xlabel('item degree'); plt.ylabel('#items (log)'); plt.title('Long-tailed item degree'); plt.show()"),
md("## 3. Training curves (from the saved training run)"),
code("fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))\n"
     "for n in ('mf_bpr', 'lightgcn'):\n"
     "    h = pd.read_csv(C.RESULTS_DIR/f'history_{n}.csv')\n"
     "    ax[0].plot(h.epoch, h.train_bpr_loss, label=n); ax[1].plot(h.epoch, h['val_ndcg@10'], label=n)\n"
     "ax[0].set_title('train BPR loss'); ax[1].set_title('val NDCG@10'); ax[0].legend(); plt.show()"),
md("## 4. Test evaluation (full ranking, train+val history masked)"),
code("rows = []\n"
     "for n in ('popularity', 'mf_bpr', 'lightgcn'):\n"
     "    m = load_trained(n, tr, nu, ni)\n"
     "    rows.append({'model': n, **evaluate(m, te, [tr, va], ni)})\n"
     "res = pd.DataFrame(rows).set_index('model'); res.round(4)"),
code("res[[f'ndcg@{k}' for k in C.KS]].T.plot.bar(figsize=(6, 3.4), rot=0, title='NDCG@K on test'); plt.show()"),
md("## 5. Recommendations"),
code("from src.recommend import Recommender\n"
     "rec = Recommender()\n"
     "uid = rec.users.user_id[0]\n"
     "print('Past purchases:'); display(rec.history(uid).head(6))\n"
     "print('Top-10 recommendations:'); display(rec.recommend(uid, 10))"),
md("## 6. Cold start - profile from picked products"),
code("picked = items[items.title.str.contains('Zelda: Breath of the Wild', case=False)].asin.head(2).tolist()\n"
     "print(picked); display(rec.recommend_from_profile(picked, 10))"),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/walkthrough.ipynb")
