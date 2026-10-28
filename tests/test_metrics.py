"""Hand-checked examples for the ranking metrics."""
import math

import torch

from src.evaluate import topk_metrics


def test_known_values():
    # 1 user, 6 items, relevant = {0, 3}; ranking = [3, 1, 0, 2, 4]
    truth = torch.zeros(1, 6, dtype=torch.bool)
    truth[0, [0, 3]] = True
    topk = torch.tensor([[3, 1, 0, 2, 4]])
    m = topk_metrics(topk, truth, ks=(2, 5))
    assert m["precision@2"].item() == 0.5            # 1 hit in top 2
    assert m["recall@2"].item() == 0.5               # 1 of 2 relevant found
    assert math.isclose(m["precision@5"].item(), 2 / 5, rel_tol=1e-6)
    assert m["recall@5"].item() == 1.0
    dcg2 = 1 / math.log2(2)                           # hit at rank 1 only
    idcg2 = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(m["ndcg@2"].item(), dcg2 / idcg2, rel_tol=1e-6)
    dcg5 = 1 / math.log2(2) + 1 / math.log2(4)        # hits at ranks 1 and 3
    assert math.isclose(m["ndcg@5"].item(), dcg5 / idcg2, rel_tol=1e-6)


def test_perfect_and_empty_hits():
    truth = torch.zeros(2, 4, dtype=torch.bool)
    truth[0, 1] = True
    truth[1, 2] = True
    topk = torch.tensor([[1, 0], [0, 3]])             # user0 perfect, user1 misses
    m = topk_metrics(topk, truth, ks=(2,))
    assert m["ndcg@2"].tolist() == [1.0, 0.0]
    assert m["recall@2"].tolist() == [1.0, 0.0]
