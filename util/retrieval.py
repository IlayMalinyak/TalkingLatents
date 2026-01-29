import numpy as np
import torch
from typing import Tuple, Optional, Union


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def cosine_knn(
    queries: Union[np.ndarray, torch.Tensor],
    database: Union[np.ndarray, torch.Tensor],
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-k nearest neighbors by cosine similarity.

    Args:
      queries: shape [Q, D]
      database: shape [N, D]
      k: number of neighbors to return

    Returns:
      (indices, scores):
        indices: [Q, k] integer array of database indices
        scores:  [Q, k] cosine similarity scores in descending order
    """
    Q = _to_numpy(queries).astype(np.float32, copy=False)
    X = _to_numpy(database).astype(np.float32, copy=False)

    # Normalize to make cosine = dot product
    Qn = l2_normalize(Q)
    Xn = l2_normalize(X)

    # Similarity matrix [Q, N]
    sims = Qn @ Xn.T

    # Top-k per row
    if k >= sims.shape[1]:
        idx = np.argsort(-sims, axis=1)
        top_idx = idx
        top_scores = np.take_along_axis(sims, idx, axis=1)
    else:
        # partial sort for efficiency
        part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        top_scores = np.take_along_axis(part_scores, order, axis=1)

    return top_idx[:, :k], top_scores[:, :k]


def recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute Recall@k given retrieved neighbor indices and ground-truth item indices.

    Args:
      retrieved_indices: [Q, kmax] array with ranked neighbor indices
      ground_truth_indices: [Q] array with true indices (or best matches)
      k: cutoff for recall

    Returns:
      Recall@k in [0,1]
    """
    k = min(k, retrieved_indices.shape[1])
    hits = 0
    for q in range(retrieved_indices.shape[0]):
        if ground_truth_indices[q] in retrieved_indices[q, :k]:
            hits += 1
    return hits / max(1, retrieved_indices.shape[0])


def parse_neighbors_text(text: str) -> Optional[list]:
    """
    Parse a neighbors answer like: "Neighbors: 154001003, 154001005, 154001008"
    Returns list of ints or None if no match.
    """
    if not isinstance(text, str):
        return None
    low = text.strip()
    if not low.lower().startswith('neighbors'):
        return None
    # Split at ':' and take right side
    if ':' in low:
        right = low.split(':', 1)[1]
    else:
        right = low
    tokens = [t.strip().strip('[]') for t in right.split(',') if t.strip()]
    ids = []
    for t in tokens:
        # remove any trailing punctuation
        t = t.rstrip('.;')
        try:
            ids.append(int(t))
        except ValueError:
            continue
    return ids if ids else None


def parse_order_text(text: str) -> Optional[list]:
    """
    Parse an order answer like: "Order: 154001003, 154001005, 154001008"
    Returns list of ints in order or None if no match.
    """
    if not isinstance(text, str):
        return None
    low = text.strip()
    if not low.lower().startswith('order'):
        return None
    # Split at ':' and take right side
    if ':' in low:
        right = low.split(':', 1)[1]
    else:
        right = low
    tokens = [t.strip().strip('[]') for t in right.split(',') if t.strip()]
    ids = []
    for t in tokens:
        t = t.rstrip('.;')
        try:
            ids.append(int(t))
        except ValueError:
            continue
    return ids if ids else None


if __name__ == "__main__":
    # Simple CLI test: random data where queries == first few db items
    import argparse
    p = argparse.ArgumentParser("Cosine kNN retrieval (utility test)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--q", type=int, default=10)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    db = rng.standard_normal((args.n, args.d)).astype(np.float32)
    # make first Q entries match queries
    queries = db[:args.q].copy()
    # add slight noise to queries for realism
    queries += 0.01 * rng.standard_normal(queries.shape).astype(np.float32)

    idx, scores = cosine_knn(queries, db, k=args.k)
    gt = np.arange(args.q)
    r1 = recall_at_k(idx, gt, k=1)
    rk = recall_at_k(idx, gt, k=args.k)
    print(f"Top-1 recall: {r1:.3f}   Top-{args.k} recall: {rk:.3f}")
