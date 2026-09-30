"""
Explicit MaxST-based Reranker
==============================
Computes true Maximum Spanning Tree weight for each candidate evaluation.
"""

import numpy as np
from typing import List
import heapq


def compute_maxst_weight(nodes: List[int], similarity_matrix: np.ndarray) -> float:
    """
    Compute the weight of Maximum Spanning Tree on given nodes using Prim's algorithm.

    Args:
        nodes: List of node indices (local indices in similarity_matrix)
        similarity_matrix: Full similarity matrix [N x N]

    Returns:
        Total weight of the MaxST
    """
    if len(nodes) <= 1:
        return 0.0

    n = len(nodes)
    visited = [False] * n
    visited[0] = True
    total_weight = 0.0

    # Priority queue: (-weight, from_idx, to_idx)
    pq = []
    for j in range(1, n):
        weight = similarity_matrix[nodes[0], nodes[j]]
        heapq.heappush(pq, (-weight, 0, j))

    edges_added = 0
    while pq and edges_added < n - 1:
        neg_weight, from_idx, to_idx = heapq.heappop(pq)

        if visited[to_idx]:
            continue

        visited[to_idx] = True
        total_weight += (-neg_weight)
        edges_added += 1

        for j in range(n):
            if not visited[j]:
                weight = similarity_matrix[nodes[to_idx], nodes[j]]
                heapq.heappush(pq, (-weight, to_idx, j))

    return total_weight


class ExplicitMaxSTReranker:
    """
    Reranker that uses explicit MaxST weight calculation.
    """

    def __init__(self, lambda_: float = 1.0, top_l_neighbors: int = None, verbose: bool = False):
        self.lambda_ = lambda_
        self.top_l_neighbors = top_l_neighbors
        self.verbose = verbose

    def rerank(
        self,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray,
        candidate_indices: np.ndarray,
        K: int = 25
    ) -> List[int]:
        """
        Rerank candidates using explicit MaxST weight calculation.

        Args:
            query_emb: Query embedding [D]
            candidate_embs: Candidate embeddings [B x D]
            candidate_indices: Array of candidate indices [B]
            K: Number of results to return

        Returns:
            List of selected candidate indices
        """
        # Convert to numpy if needed
        try:
            import torch
            if isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.cpu().numpy()
            if isinstance(candidate_embs, torch.Tensor):
                candidate_embs = candidate_embs.cpu().numpy()
            if isinstance(candidate_indices, torch.Tensor):
                candidate_indices = candidate_indices.cpu().numpy()
        except ImportError:
            pass

        B = len(candidate_indices)
        k = min(K, B)

        # Force L2 normalization to ensure cosine similarity in [-1, 1]
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        candidate_embs = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-12)

        # Compute cosine similarities
        qt_sims = candidate_embs @ query_emb
        tt_sims = candidate_embs @ candidate_embs.T

        # Map from [-1, 1] to [0, 1]
        qt_sims = (qt_sims + 1) / 2
        tt_sims = (tt_sims + 1) / 2

        # Construct extended similarity matrix: include query as node B
        q_node = B
        W = np.zeros((B + 1, B + 1), dtype=tt_sims.dtype)
        W[:B, :B] = tt_sims
        W[q_node, :B] = qt_sims
        W[:B, q_node] = qt_sims
        W[q_node, q_node] = 0.0

        # Apply graph sparsification if specified
        if self.top_l_neighbors is not None:
            L = min(self.top_l_neighbors, B)
            for i in range(B):
                # Keep only top-L neighbors for each candidate
                sims = W[i, :B].copy()
                if L < B:
                    top_l_indices = np.argpartition(sims, -L)[-L:]
                    mask = np.ones(B, dtype=bool)
                    mask[top_l_indices] = False
                    W[i, :B][mask] = 0.0
            # Query connections are not sparsified

        selected_local = []
        selected_set = set()  # For O(1) lookup
        selected_global = []
        current_maxst_weight = 0.0

        for iteration in range(k):
            best_local_idx = -1
            best_gain = -np.inf
            best_new_maxst_weight = 0.0

            for local_idx in range(B):
                if local_idx in selected_set:
                    continue

                relevance = qt_sims[local_idx]

                # Compute MaxST weight on {query} ∪ selected ∪ {candidate}
                nodes = selected_local + [local_idx, q_node]
                new_maxst_weight = compute_maxst_weight(nodes, W)

                coherence_gain = new_maxst_weight - current_maxst_weight
                gain = relevance + self.lambda_ * coherence_gain

                if gain > best_gain:
                    best_gain = gain
                    best_local_idx = local_idx
                    best_new_maxst_weight = new_maxst_weight

            selected_local.append(best_local_idx)
            selected_set.add(best_local_idx)
            selected_global.append(int(candidate_indices[best_local_idx]))
            current_maxst_weight = best_new_maxst_weight

            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}/{k}: "
                      f"MaxST weight = {current_maxst_weight:.4f}")

        return selected_global
