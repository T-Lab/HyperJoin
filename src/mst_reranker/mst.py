"""
MST Core Algorithm for Label-Free Reranking
============================================

Prim-style Maximum Spanning Tree reranker for unsupervised scenarios.

Key features:
- Greedy MST expansion using Prim's algorithm
- Max-based coherence (best edge to current tree)
- Single hyperparameter lambda

Algorithm:
---------
Prim-style MST generation:
1. Initialize with query as root
2. Iteratively add the node with maximum marginal gain:
   Δ(j|S) = w(q,j) + λ · max_{s∈S} w(j,s)
3. Stop when K nodes are selected

This implements a maximum spanning tree where:
- Each new node connects via its strongest edge to the current tree
- Coherence = max edge weight (not sum)
"""

import torch
import numpy as np
import heapq
from typing import List, Dict, Optional


class LabelFreeMSTReranker:
    """
    Prim-style MST-based reranker for Label-Free (unsupervised) scenarios.

    Problem Formulation:
    -------------------
    Given:
        - Query column q
        - Top-B candidate columns C = {c1, c2, ..., cB}

    Goal:
        Build a maximum spanning tree rooted at q, selecting K nodes that maximize:

        G(T) = Σ w(q, ci) + λ · Σ w_MST(ci, cj)
               ci∈T           (ci,cj)∈MST(T)

        where:
        - w(q, ci) = cosine_similarity(q, ci)  [query-target relevance]
        - w_MST(ci, cj) = edge weight in maximum spanning tree
        - λ ≥ 0: controls the importance of tree coherence relative to relevance

    Hyperparameter:
    --------------
    - lambda (λ): The ONLY tunable parameter
        - λ = 0: Pure query relevance (no MST coherence)
        - λ = 0.5: MST coherence weighted at 50% of relevance
        - λ = 1: MST coherence equals query relevance in importance
        - λ > 1: MST coherence dominates

    Algorithm (Prim-style):
    ----------------------
    Greedy maximum spanning tree expansion:
    1. Initialize tree T with query as root
    2. Iteratively add the node with maximum marginal gain:
       Δ(j|T) = w(q,j) + λ · max_{s∈T} w(j,s)

       This is the Prim cut step: the coherence gain is the best edge
       connecting the new node to the current tree.

    3. Stop when K nodes are selected

    Key difference from sum-based approaches:
    - Max-based: Each node connects via its BEST edge to tree (MST property)
    - Sum-based: Each node's score considers ALL edges to tree (density)
    """

    def __init__(self, lambda_: float = 1.0, normalize_coherence: bool = True,
                 top_l_neighbors: Optional[int] = None, verbose: bool = False):
        """
        Initialize Label-Free MST Reranker.

        Args:
            lambda_: Coherence weight (≥ 0). Controls the relative importance of MST coherence.
            normalize_coherence: Deprecated for max-based MST (max is scale-invariant).
                                Kept for API compatibility but ignored.
            top_l_neighbors: If set, each candidate only connects to its top-L most similar neighbors
                           (graph sparsification). If None, uses complete graph (all B² edges).
            verbose: If True, print detailed logs.
        """
        self.lambda_ = lambda_
        self.normalize_coherence = normalize_coherence  # Ignored for max-based
        self.top_l_neighbors = top_l_neighbors
        self.verbose = verbose

        if self.lambda_ < 0:
            raise ValueError(f"lambda must be non-negative, got {self.lambda_}")

    def build_tree_cover(self,
                        query_emb: torch.Tensor,
                        candidate_embs: torch.Tensor,
                        candidate_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Build tree cover: compute all edge weights.

        Args:
            query_emb: Query embedding [D]
            candidate_embs: Candidate embeddings [B, D]
            candidate_indices: Global indices of candidates [B]

        Returns:
            Dictionary containing:
            - qt_weights: Query-target edge weights [B]
            - tt_weights: Target-target edge weights [B, B]
            - candidate_indices: Global indices [B]
        """
        B = len(candidate_indices)

        if self.verbose:
            print(f" Building tree cover (B={B} candidates)...")

        # Query-Target edge weights
        query_emb_norm = query_emb.unsqueeze(0)  # [1, D]
        qt_weights = torch.cosine_similarity(
            query_emb_norm,
            candidate_embs,
            dim=1
        ).cpu().numpy()  # [B]

        # Target-Target edge weights (symmetric matrix)
        # Compute all pairwise similarities first
        all_sims = np.zeros((B, B), dtype=np.float32)
        for i in range(B):
            for j in range(i+1, B):
                emb_sim = torch.cosine_similarity(
                    candidate_embs[i].unsqueeze(0),
                    candidate_embs[j].unsqueeze(0),
                    dim=1
                ).item()
                all_sims[i, j] = emb_sim
                all_sims[j, i] = emb_sim

        # Apply graph sparsification if top_l_neighbors is set
        tt_weights = np.zeros((B, B), dtype=np.float32)
        L = self.top_l_neighbors if self.top_l_neighbors is not None else B

        for i in range(B):
            # Get similarities for node i to all other nodes
            sims = all_sims[i].copy()
            sims[i] = -np.inf  # Exclude self

            # Keep only top-L neighbors
            if L < B:
                top_l_indices = np.argpartition(sims, -L)[-L:]
                for j in top_l_indices:
                    tt_weights[i, j] = all_sims[i, j]
            else:
                # Complete graph
                tt_weights[i] = all_sims[i]
                tt_weights[i, i] = 0.0

        # Make symmetric (keep max of both directions for safety)
        tt_weights = np.maximum(tt_weights, tt_weights.T)

        if self.verbose:
            print(f"   Query-Target edges: {B}")
            if L < B:
                actual_edges = np.count_nonzero(tt_weights) // 2
                print(f"   Target-Target edges: {actual_edges} (sparse, L={L})")
            else:
                print(f"   Target-Target edges: {B*(B-1)//2} (complete graph)")
            avg_qt = np.mean(qt_weights)
            nonzero_tt = tt_weights[tt_weights > 0]
            avg_tt = np.mean(nonzero_tt) if len(nonzero_tt) > 0 else 0.0
            print(f"   Avg Q-T weight: {avg_qt:.4f}")
            print(f"   Avg T-T weight: {avg_tt:.4f}")

        return {
            'qt_weights': qt_weights,
            'tt_weights': tt_weights,
            'candidate_indices': candidate_indices
        }

    def greedy_mst(self,
                   tree_cover: Dict[str, np.ndarray],
                   K: int) -> List[int]:
        """
        Prim-style maximum spanning tree expansion with bounded nodes.

        This implements Prim's algorithm for maximum spanning tree:
        - Maintains a growing tree T (initially just query)
        - Each iteration adds the node with best connection to T
        - Connection quality = relevance + λ * (best edge to T)

        Args:
            tree_cover: Dictionary from build_tree_cover()
            K: Maximum number of nodes to select

        Returns:
            List of selected global indices (length ≤ K)
        """
        qt_weights = tree_cover['qt_weights']
        tt_weights = tree_cover['tt_weights']
        candidate_indices = tree_cover['candidate_indices']
        B = len(candidate_indices)

        if self.verbose:
            print(f"\n Greedy MST selection (K={K}, B={B}, λ={self.lambda_})...")

        # Priority queue: (-priority, local_idx, iteration)
        heap = []
        selected_local = set()  # Local indices of selected nodes
        result_global = []  # Global indices of selected nodes

        # MST tracking
        parent = {}  # parent[node] = which node it connects to in MST
        mst_edges = []  # List of (node, parent, weight) for MST edges
        mst_total_weight = 0.0  # Total MST weight

        # Initialize heap with all candidates
        for i in range(B):
            priority = qt_weights[i]  # Initial priority = Q-T similarity
            heapq.heappush(heap, (-priority, i, -1))

        iteration = 0
        while len(result_global) < K and heap:
            neg_priority, local_idx, source_iter = heapq.heappop(heap)

            # Skip if already selected
            if local_idx in selected_local:
                continue

            # Find best edge to current tree (for MST construction)
            if len(selected_local) > 0:
                best_parent = max(selected_local, key=lambda s: tt_weights[local_idx, s])
                best_edge_weight = tt_weights[local_idx, best_parent]
                parent[local_idx] = best_parent
                mst_edges.append((local_idx, best_parent, best_edge_weight))
                mst_total_weight += best_edge_weight
            else:
                parent[local_idx] = None  # First node, no parent (connects to query)
                best_edge_weight = 0.0

            # Select this node
            selected_local.add(local_idx)
            global_idx = candidate_indices[local_idx]
            result_global.append(int(global_idx))

            if self.verbose and iteration < 5:
                # Show max-based coherence (best edge to tree)
                if len(selected_local) > 1:
                    coherence = best_edge_weight
                    parent_info = f"→ #{parent[local_idx]}" if parent[local_idx] is not None else "→ query"
                else:
                    coherence = 0.0
                    parent_info = "→ query (root)"
                print(f"   Step {iteration+1}: Select #{local_idx} {parent_info}, "
                      f"edge_weight={coherence:.4f}, priority={-neg_priority:.4f}")

            # Update heap: recalculate priorities for remaining candidates
            for j in range(B):
                if j not in selected_local:
                    base_score = qt_weights[j]

                    # Coherence bonus: MAX edge weight to selected nodes (Prim-style MST)
                    if len(selected_local) > 0:
                        coherence_bonus = max(tt_weights[j, s] for s in selected_local)
                    else:
                        coherence_bonus = 0.0

                    # Note: normalize_coherence is ignored for max-based coherence
                    # (max is already scale-invariant to tree size)

                    # Combined priority
                    new_priority = base_score + self.lambda_ * coherence_bonus
                    heapq.heappush(heap, (-new_priority, j, iteration))

            iteration += 1

        if self.verbose:
            print(f" MST selection complete: {len(result_global)}/{K} nodes selected")
            if len(result_global) < K:
                print(f"   ️ Only {len(result_global)} candidates available (< K={K})")
            print(f"   MST edges: {len(mst_edges)}")
            print(f"   MST total weight: {mst_total_weight:.4f}")
            if len(mst_edges) > 0:
                avg_edge_weight = mst_total_weight / len(mst_edges)
                print(f"   MST avg edge weight: {avg_edge_weight:.4f}")

        return result_global

    def rerank(self,
               query_emb: torch.Tensor,
               candidate_embs: torch.Tensor,
               candidate_indices: np.ndarray,
               K: int = 25) -> List[int]:
        """
        Complete reranking pipeline.

        Args:
            query_emb: Query embedding [D]
            candidate_embs: Candidate embeddings [B, D]
            candidate_indices: Global indices of candidates [B]
            K: Number of results to return

        Returns:
            List of K selected global indices
        """
        # Build tree cover
        tree_cover = self.build_tree_cover(query_emb, candidate_embs, candidate_indices)

        # Greedy MST generation
        selected = self.greedy_mst(tree_cover, K)

        return selected

    def rerank_with_trace(self,
                          query_emb: torch.Tensor,
                          candidate_embs: torch.Tensor,
                          candidate_indices: np.ndarray,
                          K: int = 25) -> Dict:
        """
        Same as rerank() but also returns per-step selection trace for case studies.

        Returns dict with:
          - selected: list of K global indices (same as rerank())
          - trace: list of dicts, one per selected item, each containing
              {step, global_idx, rel, coh, parent_global_idx, edge_weight}
            where rel = w(q, item), coh = max edge to current tree
            (0 for the first item), and parent_global_idx is the tree parent.
        """
        tree_cover = self.build_tree_cover(query_emb, candidate_embs, candidate_indices)
        qt = tree_cover['qt_weights']
        tt = tree_cover['tt_weights']
        cand_idx = tree_cover['candidate_indices']
        B = len(cand_idx)
        K = min(K, B)

        selected_local: List[int] = []
        trace: List[Dict] = []
        # priority for first pick = qt only
        # for subsequent picks: qt[j] + lambda * max_{s in selected} tt[j, s]
        for step in range(K):
            if not selected_local:
                # first pick: argmax over qt
                priorities = qt.copy()
                used = np.zeros(B, dtype=bool)
            else:
                # max edge to tree
                coh = tt[:, selected_local].max(axis=1)
                priorities = qt + self.lambda_ * coh
                used = np.zeros(B, dtype=bool)
                used[selected_local] = True
                priorities[used] = -np.inf

            j = int(np.argmax(priorities))
            if priorities[j] == -np.inf:
                break

            if not selected_local:
                parent_local = None
                edge_w = 0.0
                coh_score = 0.0
            else:
                # which selected node maximizes the tt edge
                edge_to_each_selected = tt[j, selected_local]
                best_idx_in_selected = int(np.argmax(edge_to_each_selected))
                parent_local = selected_local[best_idx_in_selected]
                edge_w = float(edge_to_each_selected[best_idx_in_selected])
                coh_score = edge_w

            trace.append({
                'step': step + 1,
                'global_idx': int(cand_idx[j]),
                'rel': float(qt[j]),
                'coh': coh_score,
                'parent_global_idx': int(cand_idx[parent_local]) if parent_local is not None else None,
                'edge_weight': edge_w,
            })
            selected_local.append(j)

        selected_global = [t['global_idx'] for t in trace]
        return {'selected': selected_global, 'trace': trace}

    def rerank_batch(self,
                    query_embs: torch.Tensor,
                    candidate_embs: torch.Tensor,
                    candidate_indices: np.ndarray,
                    K: int = 25) -> List[List[int]]:
        """
        Batch reranking for multiple queries.

        Args:
            query_embs: Query embeddings [N, D]
            candidate_embs: Candidate embeddings [N, B, D]
            candidate_indices: Global indices [N, B]
            K: Number of results per query

        Returns:
            List of N result lists, each containing K global indices
        """
        N = query_embs.shape[0]
        results = []

        for i in range(N):
            selected = self.rerank(
                query_embs[i],
                candidate_embs[i],
                candidate_indices[i],
                K=K
            )
            results.append(selected)

        return results
