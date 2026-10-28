# Viva Q&A - Retail Product Recommendation Using LightGCN

**1. What is the project in one sentence?**
A recommender that treats customers and products as nodes of a bipartite graph, learns embeddings with LightGCN, and ranks unseen products for a customer (or for a profile of picked products).

**2. How is a GNN different from ordinary collaborative filtering (CF)?**
Plain CF (e.g. matrix factorisation) learns one free vector per user and item and only uses an interaction when it is a direct training example. A GNN also passes information along edges, so a user's final vector is built from their items, those items' other buyers, and so on. Higher-order neighbours are used explicitly, which helps most for users with few interactions.

**3. Why a graph at all?** Purchase data is relational: "customer bought product" links. The graph makes "customers like me" and "products bought together" a structural property rather than something the model must discover from a flat table.

**4. What does LightGCN do?** It repeatedly multiplies the embeddings by the normalised adjacency (`E^(k+1) = A_hat E^(k)`), i.e. averages neighbours' embeddings, then averages the layer outputs `E^(0..K)` into the final embedding. The score is a dot product. The only trainable parameters are the initial embeddings.

**5. Why does LightGCN drop non-linearity and weight matrices?** In recommendation the nodes have no rich input features, only IDs. Feature-transform matrices and activations (from GCN) add parameters and overfitting risk without useful features to transform; the LightGCN authors showed removing them helped. Here it also keeps the model simple and fast: one sparse matrix product per layer.

**6. Why symmetric normalisation `D^-1/2 A D^-1/2`?** It keeps embedding magnitudes stable and stops very popular items (high degree) or very active users from dominating the sum; each edge is scaled by `1/sqrt(deg_u * deg_i)`.

**7. Why combine layers by a mean?** Each layer captures a different neighbourhood radius (layer 0 = itself, layer 1 = direct neighbours, ...). Averaging them preserves self-information and counters over-smoothing from deep propagation. Our config is `K = 3`, mean weights.

**8. What is BPR loss and why use it?** Bayesian Personalised Ranking: for a triple (user, positive item, sampled negative item) minimise `-log sigmoid(s_pos - s_neg)`. Our task is ranking and we only have implicit positives, so a pairwise "positive above negative" objective matches the goal better than predicting a rating.

**9. How are negatives chosen and what is the risk?** One uniformly random item per positive (resampled if it collides with a known positive). Unobserved does not mean disliked, so some negatives are false negatives; uniform sampling is simple but not "hard". Future work: hard negative sampling.

**10. How do you avoid data leakage?** The data is split by time (train < valid < test); the graph and popularity counts come only from training edges; hyper-parameters and early stopping use validation only; the test set is evaluated once at the end. Validation/test users and items are restricted to those present in training.

**11. Describe the evaluation protocol.** Full ranking: for every test user, score all items, mask items seen in training (and validation for the test run), take the top K, compare with the user's test items. Metrics Precision@K, Recall@K, NDCG@K for K = 5, 10, 20, averaged over users who have at least one test item. No sampled negatives, so numbers are not inflated.

**12. What do the metrics mean?** Precision@K = fraction of the K suggestions that are hits. Recall@K = fraction of the user's relevant items retrieved. NDCG@K additionally rewards putting hits near the top (discount `1/log2(rank+1)`, normalised by the ideal ranking).

**13. What were your results and are they good?** On the test period LightGCN had the best numbers for every metric (NDCG@10 about 0.0167 vs 0.0113 popularity and 0.0042 MF-BPR; exact table in `results/metrics.csv`). Absolute values are small because the data is sparse (about 7 training interactions per customer) and the test is months ahead; this is typical for Amazon-style benchmarks. It's one seed and about 2,100 evaluation users, so small gaps should not be over-interpreted.

**14. Why did MF-BPR do worse than popularity?** With so few interactions per customer, a free vector per customer overfits: the training loss kept falling while validation NDCG stalled. Graph propagation ties each customer to their items' embeddings and acts as a regulariser. (This is our interpretation; we did not run an ablation to prove it.)

**15. How do you handle cold start?** A brand-new customer is not in the graph. `recommend_from_profile` averages the embeddings of the products they pick and ranks all other products by dot product with that vector. It works for new users but not for new items (those have no edges, hence no embedding), where content features would be needed.

**16. (Bonus) Limitations and next steps?** One category and a subsample; no confidence intervals; no use of text/price/category as model inputs. Next: side-information features, hard negatives, multiple seeds, full data, other GNNs.

**17. (Bonus) Why not PyTorch Geometric?** The model is just sparse matrix multiplication, so plain PyTorch is enough, easier to explain and has fewer dependencies.
