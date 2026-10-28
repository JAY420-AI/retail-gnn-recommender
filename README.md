# Retail Product Recommendation Using GNN (LightGCN)

A graph-based product recommender. Customers and products are the two node types of a bipartite graph; a positive review (rating >= 4) is an edge. **LightGCN**, written from scratch in plain PyTorch with a sparse normalised adjacency, learns node embeddings, and products are ranked for a customer by dot product. It is compared with a popularity baseline and MF-BPR. A Streamlit app serves recommendations for a customer ID or a "pick some products" cold-start profile.

**Data:** Amazon Reviews 2023, category `Video_Games` (real data, downloaded from Hugging Face; no synthetic data was used for any reported number).

## Results (test period, full ranking, history masked, 2,111 test customers)

| Model | P@10 | R@10 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| Popularity | 0.0033 | 0.0224 | 0.0078 | 0.0113 | 0.0157 |
| MF-BPR | 0.0014 | 0.0083 | 0.0026 | 0.0042 | 0.0060 |
| **LightGCN** | **0.0046** | **0.0296** | **0.0125** | **0.0167** | **0.0217** |

Full table (K = 5, 10, 20) in `results/metrics.csv`; chart in `results/model_comparison.png`. Absolute values are low (sparse data: about 6.6 training edges per customer, test months after training); this is a single seed and ~2.1k evaluation users, so small gaps are indicative, not conclusive.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run each step (from the project root)

| Step | Command |
|---|---|
| Download + preprocess data (~485 MB download once) | `python -m src.data` |
| Graph statistics | `python -m src.graph` |
| (optional) tiny hyper-parameter grid on validation | `python -m src.tune` |
| Train MF-BPR and LightGCN (~1-2 min on CPU) | `python -m src.train --model all` |
| Evaluate on test, write `results/metrics.csv` + chart | `python -m src.evaluate` |
| Unit tests for the metrics | `python -m pytest -q tests` |
| Sample recommendations in the terminal | `python -m src.recommend` |
| Demo app | `streamlit run app.py` |
| Everything above except tune/app | `./run_all.sh` |

Rebuild deliverables: `python notebooks/make_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb`, `python report/build_report.py`, `python presentation/build_slides.py`.

If the download is impossible (offline), `python -m src.data --synthetic` makes a small fake dataset with the same schema so the code can be smoke-tested. Its metrics mean nothing. `python -m src.data --download` forces a re-download.

## Project layout

```
app.py                  Streamlit demo
run_all.sh              full pipeline
src/config.py           paths, seeds, hyper-parameter defaults
src/data.py             download, filter, subsample, ID mapping, split files
src/graph.py            bipartite graph, normalised adjacency, statistics
src/models.py           Popularity, MF-BPR, LightGCN
src/train.py            BPR training, early stopping, checkpoints, curves
src/tune.py             lr x L2 grid selected on validation
src/evaluate.py         full-ranking metrics, results table and chart
src/recommend.py        recommend(user_id, k), recommend_from_profile(items, k)
tests/test_metrics.py   hand-checked metric tests
notebooks/walkthrough.ipynb   executed end-to-end notebook
report/                 Micro_Project_Report.docx (+ builder script)
presentation/slides.pptx      11 slides (+ builder script)
VIVA_QA.md              viva questions and answers
results/, models/, data/      outputs, checkpoints, raw + processed data
```

## Design decisions

- **Category and files:** Video_Games, using the dataset's official 5-core files, which are already split by global timestamp (train < valid < test). Item metadata streamed from the raw meta file (only needed items kept).
- **Implicit feedback:** rating >= 4 is positive; lower ratings are dropped, not used as negatives.
- **Subsampling:** up to 30,000 users. Users who also appear in valid/test are kept first; otherwise random users almost never have future activity and the test set would be tiny (we saw 326 users that way). This only decides *which users are modelled*; no validation/test edge is used to build the graph. It does make the user pool somewhat more active than the population. Then iterative filtering: users >= 3 and items >= 5 training positives. Final: 22,269 users, 7,811 items, 147,013 training edges.
- **No leakage:** graph and popularity counts use training edges only; hyper-parameters and early stopping use validation only; validation/test rows are restricted to users and items seen in training. Test ranking masks train + validation items. Validation/test edges are *not* added to the graph at test time (simplest strict protocol; a production system would refresh the graph).
- **Model:** LightGCN with 3 layers, 64-d embeddings, mean layer combination. BPR loss with one uniform negative per positive (collisions resampled), L2 on layer-0 embeddings, Adam (lr 0.005, L2 0.001), batch 4096, early stopping (patience 6) on validation NDCG@10. lr and L2 chosen from a 2 x 3 grid per model on validation (`results/tuning.csv`). MF-BPR got the same tuning budget.
- **Evaluation:** full ranking against all items (no sampled negatives), history masked, averaged over users with at least one target item.
- **Cold start:** average of the picked products' final LightGCN embeddings, dot-product ranking. Works for new customers, not for items absent from the graph.
- **Reproducibility:** seed 42 (NumPy, PyTorch, subsampling, negative sampling), CPU only. A repeat of the full pipeline reproduced `metrics.csv` exactly in our run.
- **Interpretation caution:** the report explains the MF-BPR result as overfitting from sparse per-user data; that is a hypothesis supported by the training curves, not an ablation result.

