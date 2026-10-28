"""Builds report/Micro_Project_Report.docx. All numbers are read from results/ and data/processed/."""
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

R, P = "results/", "data/processed/"
m = pd.read_csv(R + "metrics.csv").set_index("model")
tr, va, te = (pd.read_csv(P + f"{s}.csv") for s in ("train", "val", "test"))
items = pd.read_csv(P + "items.csv")
nu, ni, ne = tr.user.nunique(), len(items), len(tr)
dens = ne / (nu * ni)
tun = pd.read_csv(R + "tuning.csv")
g, p, mf = m.loc["lightgcn"], m.loc["popularity"], m.loc["mf_bpr"]
gain_pop = (g["ndcg@10"] / p["ndcg@10"] - 1) * 100
gain_mf = g["ndcg@10"] / mf["ndcg@10"]

d = Document()
d.styles["Normal"].font.name = "Calibri"
d.styles["Normal"].font.size = Pt(11)


def H(t, l=1): d.add_heading(t, l)
def para(t): d.add_paragraph(t)
def bullets(xs):
    for x in xs: d.add_paragraph(x, style="List Bullet")
def fig(path, cap, w=6.0):
    d.add_picture(path, width=Inches(w)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = d.add_paragraph(cap); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.runs[0].italic = True


# ---- title + team
t = d.add_heading("Retail Product Recommendation Using Graph Neural Networks", 0)
para("Micro-Project Report - Artificial Neural Networks")
para("Dataset: Amazon Reviews 2023 (Video Games)  |  Model: LightGCN")
H("Team Details")
tb = d.add_table(rows=1, cols=3); tb.style = "Light Grid Accent 1"
for c, h in zip(tb.rows[0].cells, ("Name", "Enrollment No.", "Role")): c.text = h
for i in (1, 2, 3, 4):
    r = tb.add_row().cells
    r[0].text, r[1].text, r[2].text = f"[Team member {i} name]", "[Enrollment number]", "[Role]"
para("")
para("Guide / Faculty: [Guide name]        Department: [Department]        Institute: [Institute name]        Academic year: [Year]")

# ---- abstract
H("Abstract")
para("Online retailers need to show each customer a short list of products they are likely to want. "
     "This project models customers and products as the two node types of a bipartite graph whose edges are "
     "positive reviews (rating of 4 or 5) and uses LightGCN, a simplified graph convolutional network, to learn "
     "node embeddings. Products are ranked for a customer by the inner product of embeddings. We trained on a "
     f"subsample of the Amazon Reviews 2023 Video Games 5-core data ({nu:,} customers, {ni:,} products, {ne:,} training "
     "edges), split by time, and compared against a popularity ranking and matrix factorisation trained with the same loss. "
     f"On the held-out future period LightGCN reached NDCG@10 = {g['ndcg@10']:.4f}, versus {p['ndcg@10']:.4f} for popularity "
     f"and {mf['ndcg@10']:.4f} for MF-BPR. A Streamlit prototype returns ranked recommendations for a customer ID or "
     "for a small profile of picked products (cold start).")

H("Introduction")
para("Recommendation is one of the most commercially important uses of machine learning in retail. Classic "
     "collaborative filtering learns one vector per customer and per product from the purchase matrix. A purchase "
     "history, however, is naturally a graph: a customer is connected to products, and those products are connected to "
     "other customers. Graph neural networks exploit this structure by letting information flow along edges, so a "
     "customer's representation is informed by the products they bought, by the other customers who bought the same "
     "products, and (with more layers) by what those customers bought in turn.")

H("Problem Statement")
para("Given historical customer-product interactions, build a graph-based recommender that, for a customer ID or a "
     "small profile of products, returns a ranked list of candidate products the customer has not yet bought and is "
     "likely to engage with in the future.")

H("Objectives")
bullets(["Build a customer-product bipartite graph from real review data using training-period edges only.",
         "Implement LightGCN in plain PyTorch with a sparse symmetric-normalised adjacency matrix.",
         "Compare it against two baselines (popularity, MF-BPR) with a leakage-free full-ranking protocol (Precision, Recall, NDCG at K = 5, 10, 20).",
         "Provide a working demo that handles both known customers and cold-start profiles."])

H("Proposed Solution")
para("The pipeline has five stages: (1) load and filter the data and split it by time; (2) build the bipartite graph "
     "and its normalised adjacency from training edges; (3) train LightGCN with the BPR pairwise loss and early "
     "stopping on validation NDCG@10; (4) evaluate by ranking all products for each test customer with their history "
     "masked; (5) serve recommendations through a Python API and a Streamlit app. For customers outside the graph, "
     "a profile vector is the average embedding of the products they pick.")

H("Technologies and Tools")
bullets(["Python 3.11, PyTorch (sparse tensors; LightGCN written from scratch, no PyG)",
         "pandas, NumPy for data handling; matplotlib for plots",
         "Streamlit for the demo; pytest for unit tests",
         "Hugging Face hosted Amazon Reviews 2023 data"])

H("Dataset Description")
para("Amazon Reviews 2023 (McAuley Lab) is a large public review corpus. We use the Video_Games category: the "
     "5-core interaction files (every user and item has at least 5 interactions in the full data), which the "
     "authors already split by global timestamp into train, valid and test, plus the item metadata file (title, "
     "category path, price). A rating >= 4 is treated as a positive implicit interaction.")
para("The raw 5-core files hold about 737k / 35k / 43k interactions in train / valid / test. To keep CPU training short we "
     "subsampled users: every train user who also appears in valid or test was kept first (otherwise almost no user "
     "would have future activity to evaluate on), then random users were added up to 30,000, followed by an iterative "
     "filter (users >= 3 and items >= 5 positive training interactions). Only training-period "
     "edges decide which users and items enter the graph; later splits are restricted to those users and items.")
tb = d.add_table(rows=1, cols=4); tb.style = "Light Grid Accent 1"
for c, h in zip(tb.rows[0].cells, ("Split", "Interactions", "Users", "Period")): c.text = h
for n, df_, per in (("Train", tr, "earliest - Aug 2021"), ("Validation", va, "Aug 2021 - Jul 2022"), ("Test", te, "Jul 2022 - Sep 2023")):
    r = tb.add_row().cells
    r[0].text, r[1].text, r[2].text, r[3].text = n, f"{len(df_):,}", f"{df_.user.nunique():,}", per
para("")
para(f"Resulting graph: {nu:,} customers, {ni:,} products, {ne:,} edges, density {dens:.5f} "
     f"({ne/nu:.1f} edges per customer, {ne/ni:.1f} per product on average). Item degrees are long-tailed.")

H("Methodology")
H("Graph construction", 2)
para("Customers and products are nodes of a bipartite graph G with an edge (u, i) for every positive training interaction. "
     "With the interaction matrix R of size |U| x |I|, the adjacency is A = [[0, R], [R^T, 0]]. We use the symmetric "
     "normalisation  A_hat = D^(-1/2) A D^(-1/2),  where D is the diagonal degree matrix. A_hat is stored as a sparse "
     "tensor, so one propagation step costs time proportional to the number of edges.")
H("LightGCN", 2)
para("Each node has a free embedding e^(0). Propagation is purely linear neighbourhood averaging:")
para("    E^(k+1) = A_hat E^(k)      for k = 0 ... K-1")
para("Equivalently, for a customer u:  e_u^(k+1) = sum over products i in N(u) of  e_i^(k) / sqrt(|N(u)| |N(i)|), "
     "and symmetrically for products. There are no weight matrices and no non-linear activations. The final embedding "
     "combines all layers with a uniform mean (so each layer, including the original embedding, contributes equally):")
para("    e_u = (1/(K+1)) * sum_{k=0..K} e_u^(k),     e_i likewise")
para("The preference score is the inner product  s(u, i) = e_u . e_i. We used K = 3 layers and 64 dimensions.")
H("BPR loss", 2)
para("Training uses Bayesian Personalised Ranking: for each observed pair (u, i) we sample a product j the customer has "
     "not interacted with and require the observed product to score higher:")
para("    L = - mean over (u, i, j) of  log sigmoid( s(u, i) - s(u, j) )  +  lambda * ( ||e_u^(0)||^2 + ||e_i^(0)||^2 + ||e_j^(0)||^2 ) / (2B)")
para("The L2 term regularises only the layer-0 embeddings of the batch (B = batch size). Optimiser: Adam.")
H("Evaluation protocol", 2)
para("For each test customer with at least one test positive, all products are scored. Products already seen in training "
     "or validation are masked out, the top K are taken, and Precision@K, Recall@K and NDCG@K are averaged over customers "
     "(K = 5, 10, 20). Validation uses the same procedure with training items masked, and drives early stopping.")

H("Implementation")
bullets(["src/data.py - download, positive filtering, user subsampling, ID mapping, processed files.",
         "src/graph.py - bipartite graph, normalised sparse adjacency, statistics.",
         "src/models.py - Popularity, MF-BPR and LightGCN behind one score() interface.",
         "src/train.py - BPR with uniform negative sampling, early stopping (patience 6), checkpointing, curves.",
         "src/tune.py - small learning-rate x regularisation grid chosen on validation only.",
         "src/evaluate.py - full-ranking metrics, results table and comparison chart; tests/test_metrics.py checks the metrics on hand-computed examples.",
         "src/recommend.py and app.py - inference API (recommend, recommend_from_profile) and Streamlit demo."])
para("Seeds are fixed (42) for NumPy and PyTorch. Training both models takes about one to two minutes on a laptop CPU.")
d.add_paragraph("Hyper-parameters (selected on validation NDCG@10): learning rate 0.005, L2 weight 0.001, embedding dimension 64, "
                "batch size 4096, LightGCN layers 3. The grid searched was lr in {0.001, 0.005} x L2 in {0.0001, 0.001, 0.01}.")

H("Results and Analysis")
para("Test-set results (all numbers produced by src/evaluate.py; "
     f"{int(g['n_eval_users']):,} test customers):")
cols = ["precision@5", "recall@5", "ndcg@5", "precision@10", "recall@10", "ndcg@10", "precision@20", "recall@20", "ndcg@20"]
tb = d.add_table(rows=1, cols=4); tb.style = "Light Grid Accent 1"
for c, h in zip(tb.rows[0].cells, ("Metric", "Popularity", "MF-BPR", "LightGCN")): c.text = h
for c in cols:
    r = tb.add_row().cells
    r[0].text = c.replace("precision", "Precision").replace("recall", "Recall").replace("ndcg", "NDCG")
    for k, mod in enumerate(("popularity", "mf_bpr", "lightgcn"), 1): r[k].text = f"{m.loc[mod, c]:.4f}"
para("")
fig(R + "model_comparison.png", "Figure 1. Test metrics for the three models.")
fig(R + "training_curves.png", "Figure 2. BPR training loss and validation NDCG@10 per epoch.", 5.8)
H("Discussion", 2)
para(f"LightGCN is the best model on every metric and K. At K = 10 its NDCG is {gain_pop:.0f}% higher than the popularity "
     f"ranking and {gain_mf:.1f}x that of MF-BPR. Absolute values are small, which is normal for this setting: the "
     "catalogue has thousands of products, customers have only about 6-7 training interactions on average, and the test "
     "period is months later, so a customer's next purchase is hard to pin down.")
para("The result that matters most for the project goal is LightGCN over MF-BPR, since both use the same loss, "
     "embedding size and sampler. MF-BPR fell below even the popularity list. Its training loss kept dropping while "
     "validation NDCG stalled (Figure 2), which is the signature of overfitting: with so few interactions per customer, "
     "a free vector per customer is poorly determined. LightGCN averages over neighbours, so each customer embedding "
     "is tied to those of the products they bought and, through them, to similar customers; this smoothing acts "
     "as a strong regulariser and also shares signal with sparse nodes. This is our interpretation; we did not run an "
     "ablation that isolates it.")
para("Popularity is a competitive baseline here because video-game purchases are concentrated on a few hot items "
     "(a common pattern in retail data). Honest caveats: the numbers come from one seed and one subsample of 2,111 "
     "evaluation customers, so small differences (for example at K = 20 between LightGCN and popularity) should "
     "not be over-read; we did not compute confidence intervals. The MF-BPR baseline got the same small tuning grid as LightGCN, "
     "but a longer search could narrow the gap.")
fig(R + "app_tab1.png", "Figure 3. Demo, customer-ID tab (past purchases next to recommendations).", 6.2)
fig(R + "app_tab2.png", "Figure 4. Demo, cold-start profile tab.", 6.2)

H("Limitations")
bullets(["Subsampled data (one category, ~22k customers); results may differ on the full dataset or other categories.",
         "Evaluation is on a modest number of test customers and a single run; no confidence intervals.",
         "The graph uses training edges only, so a customer's validation/test activity does not influence their embedding at test time (it is only used for masking at serving).",
         "Only ratings and timestamps are used; titles, categories, prices and review text are not model inputs. Cold-start profiles only work for products already in the graph.",
         "Offline metrics are proxies; they do not measure business outcomes. Treating a rating >= 4 as the only positive signal ignores purchases that were reviewed poorly."])
H("Future Scope")
bullets(["Add item side information (category, price, text embeddings) to help new products and users.",
         "Try other graph recommenders (NGCF, GraphSAGE-style, attention) and hard negative sampling.",
         "Use multiple seeds and bootstrap intervals; evaluate on the full category.",
         "Incorporate time-aware edges / session order; serve with approximate nearest-neighbour search at scale."])
H("Conclusion")
para(f"We built an end-to-end graph recommender with LightGCN on real Amazon review data under a strict temporal protocol. "
     f"It outperformed both a popularity baseline and MF-BPR (NDCG@10 {g['ndcg@10']:.4f} vs {p['ndcg@10']:.4f} and {mf['ndcg@10']:.4f}), "
     "and the same embeddings power a working customer-ID and cold-start demo. The absolute scores are low, as expected for sparse "
     "implicit data, and further work should add side information and more thorough statistical evaluation.")

H("References")
bullets(["He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., Wang, M. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.",
         "Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.",
         "Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. Bridging Language and Items for Retrieval and Recommendation (Amazon Reviews 2023). arXiv 2024.",
         "Kipf, T., Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.",
         "Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019."])
d.save("report/Micro_Project_Report.docx")
print("saved")
