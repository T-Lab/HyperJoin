import os
import sys
import torch
import numpy as np
import pandas as pd
import heapq
from pathlib import Path
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from evaluator import DetailedEvaluator

class LabelFreeMSTReranker:

    def __init__(self, lambda_: float = 1.0, normalize_coherence: bool = True,
                 top_l_neighbors: Optional[int] = None, verbose: bool = False):

        self.lambda_ = lambda_
        self.normalize_coherence = normalize_coherence
        self.top_l_neighbors = top_l_neighbors
        self.verbose = verbose

        if self.lambda_ < 0:
            raise ValueError(f"lambda must be non-negative, got {self.lambda_}")

    def build_tree_cover(self,
                        query_emb: torch.Tensor,
                        candidate_embs: torch.Tensor,
                        candidate_indices: np.ndarray) -> Dict[str, np.ndarray]:

        B = len(candidate_indices)

        if self.verbose:
            print(f"[TREE] Building tree cover (B={B} candidates)...")

        # Query-Target edge weights
        query_emb_norm = query_emb.unsqueeze(0)
        qt_weights = torch.cosine_similarity(
            query_emb_norm,
            candidate_embs,
            dim=1
        ).cpu().numpy()

        # Target-Target edge weights
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

        # Apply graph sparsification
        tt_weights = np.zeros((B, B), dtype=np.float32)
        L = self.top_l_neighbors if self.top_l_neighbors is not None else B

        for i in range(B):
            sims = all_sims[i].copy()
            sims[i] = -np.inf

            if L < B:
                top_l_indices = np.argpartition(sims, -L)[-L:]
                for j in top_l_indices:
                    tt_weights[i, j] = all_sims[i, j]
            else:
                tt_weights[i] = all_sims[i]
                tt_weights[i, i] = 0.0

        tt_weights = np.maximum(tt_weights, tt_weights.T)

        return {
            'qt_weights': qt_weights,
            'tt_weights': tt_weights,
            'candidate_indices': candidate_indices
        }

    def greedy_mst(self,
                   tree_cover: Dict[str, np.ndarray],
                   K: int) -> List[int]:

        qt_weights = tree_cover['qt_weights']
        tt_weights = tree_cover['tt_weights']
        candidate_indices = tree_cover['candidate_indices']
        B = len(candidate_indices)

        if self.verbose:
            print(f"\n[SEARCH] Greedy MST selection (K={K}, B={B}, λ={self.lambda_})...")

        heap = []
        selected_local = set()
        result_global = []

        # Initialize heap
        for i in range(B):
            priority = qt_weights[i]
            heapq.heappush(heap, (-priority, i, -1))

        iteration = 0
        while len(result_global) < K and heap:
            neg_priority, local_idx, source_iter = heapq.heappop(heap)

            if local_idx in selected_local:
                continue

            selected_local.add(local_idx)
            global_idx = candidate_indices[local_idx]
            result_global.append(int(global_idx))

            # Update heap
            for j in range(B):
                if j not in selected_local:
                    base_score = qt_weights[j]

                    if len(selected_local) > 0:
                        coherence_bonus = max(tt_weights[j, s] for s in selected_local)
                    else:
                        coherence_bonus = 0.0

                    new_priority = base_score + self.lambda_ * coherence_bonus
                    heapq.heappush(heap, (-new_priority, j, iteration))

            iteration += 1

        if self.verbose:
            print(f"[SUCCESS] MST selection complete: {len(result_global)}/{K} nodes selected")

        return result_global

    def rerank(self,
               query_emb: torch.Tensor,
               candidate_embs: torch.Tensor,
               candidate_indices: np.ndarray,
               K: int = 25) -> List[int]:
        tree_cover = self.build_tree_cover(query_emb, candidate_embs, candidate_indices)
        selected = self.greedy_mst(tree_cover, K)
        return selected

    def rerank_batch(self,
                     query_embs: torch.Tensor,
                     candidate_embs: torch.Tensor,
                     candidate_indices: np.ndarray,
                     K: int = 25) -> List[List[int]]:
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


class GraphBuilderSearch:
    def __init__(self,
                 target_dataset,
                 joinable_pairs_path: str,
                 only_can: bool = True,
                 cache_path: Optional[str] = None):

        self.target_dataset = target_dataset
        self.joinable_pairs_path = joinable_pairs_path
        self.only_can = only_can
        self.cache_path = cache_path

    def _build_table_column_mapping(self) -> Dict[str, int]:
        mapping = {}
        md = getattr(self.target_dataset, 'metadata', None)
        items = []

        if md:
            if isinstance(md, dict) and 'metadata' in md:
                items = md['metadata']
            elif isinstance(md, list):
                items = md

        for idx, meta in enumerate(items):
            table_name = str(meta.get('table_name', f'target_{idx}')).replace('.csv', '')
            column_name = str(meta.get('column_name', f'col_{idx}'))
            key = f"{table_name}.{column_name}"
            mapping[key] = idx

        return mapping

    def build_adjacency_matrix(self, verbose: bool = True) -> np.ndarray:
        
        # Check cache
        if self.cache_path and os.path.exists(self.cache_path):
            if verbose:
                print(f"[INFO] Loading cached adjacency matrix from {self.cache_path}")
            adjacency = np.load(self.cache_path)
            if verbose:
                num_nodes = adjacency.shape[0]
                num_edges = int(np.sum(adjacency) / 2)
                density = num_edges / (num_nodes * (num_nodes - 1) / 2) * 100
                print(f"[SUCCESS] Loaded: {num_nodes} nodes, {num_edges} edges, "
                      f"density={density:.4f}%")
            return adjacency

        # Build from scratch
        if not os.path.exists(self.joinable_pairs_path):
            raise FileNotFoundError(
                f"Joinable pairs file not found: {self.joinable_pairs_path}"
            )

        if verbose:
            print(f"[INFO] Building GT adjacency matrix from {self.joinable_pairs_path}")

        df = pd.read_csv(self.joinable_pairs_path)

        # Filter by dataset
        if self.only_can:
            opendata_prefixes = ('CAN_CSV', 'USA_CSV', 'UK_CSV', 'SG_CSV')
            mask = df['query_table'].astype(str).str.startswith(opendata_prefixes)
            df = df[mask]
            if verbose:
                print(f"  Filtered to OpenData tables: {len(df)} pairs")

        mapping = self._build_table_column_mapping()
        num_nodes = len(self.target_dataset)

        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        edge_count = 0
        skipped_count = 0

        for _, row in df.iterrows():
            src_table = str(row['query_table']).replace('.csv', '')
            dst_table = str(row['candidate_table']).replace('.csv', '')
            src_col = str(row['query_column'])
            dst_col = str(row['candidate_column'])
            src_key = f"{src_table}.{src_col}"
            dst_key = f"{dst_table}.{dst_col}"

            if src_key in mapping and dst_key in mapping:
                u = mapping[src_key]
                v = mapping[dst_key]
                if u != v:
                    adjacency[u, v] = 1.0
                    adjacency[v, u] = 1.0
                    edge_count += 1
            else:
                skipped_count += 1

        if verbose:
            print(f"[INFO] GT Graph Statistics:")
            print(f"  Nodes (columns): {num_nodes}")
            print(f"  Edges (undirected): {edge_count}")
            density = edge_count / (num_nodes * (num_nodes - 1) / 2) * 100 if num_nodes > 1 else 0
            print(f"  Density: {density:.4f}%")
            if skipped_count > 0:
                print(f"  [WARNING] Skipped pairs (not in target): {skipped_count}")

        # Cache
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, adjacency)
            if verbose:
                print(f"[SAVE] Cached adjacency matrix to {self.cache_path}")

        return adjacency


class CoherenceEvaluator(DetailedEvaluator):

    def __init__(self, datasets: str, gt_adjacency: Optional[np.ndarray] = None):

        super().__init__(datasets)
        self.gt_adjacency = gt_adjacency

    def compute_coherence_metrics(self,
                                  search_results: List[List[int]],
                                  ground_truth: Dict[int, List[int]],
                                  k_values: List[int] = [5, 15, 25]) -> Dict[str, float]:

        if self.gt_adjacency is None:
            print("[WARNING] GT adjacency matrix not provided, skipping coherence metrics")
            return {}

        metrics = {}

        for K in k_values:
            coherence_scores = []
            diversity_scores = []
            ccr_scores = []

            for query_idx in range(len(search_results)):
                if query_idx not in ground_truth:
                    continue

                topk = search_results[query_idx][:K]

                if len(topk) == 0:
                    continue

                coherence = self._compute_coherence(topk)
                coherence_scores.append(coherence)

                diversity = self._compute_diversity(topk, K)
                diversity_scores.append(diversity)

                ccr = self._compute_ccr(topk, K)
                ccr_scores.append(ccr)

            metrics[f'Coherence@{K}'] = np.mean(coherence_scores) if coherence_scores else 0.0
            metrics[f'TableDiversity@{K}'] = np.mean(diversity_scores) if diversity_scores else 0.0
            metrics[f'CCR@{K}'] = np.mean(ccr_scores) if ccr_scores else 0.0

        return metrics

    def _compute_coherence(self, indices: List[int]) -> float:
        if len(indices) < 2:
            return 0.0

        num_connected = 0
        num_pairs = 0

        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                num_pairs += 1
                if self.gt_adjacency[idx_i, idx_j] > 0:
                    num_connected += 1

        return num_connected / num_pairs if num_pairs > 0 else 0.0

    def _compute_diversity(self, indices: List[int], K: int) -> float:
        if len(indices) == 0:
            return 0.0

        tables = set()
        for idx in indices:
            meta = self._get_metadata_by_index(idx)
            table_name = meta.get('table_name', f'unknown_{idx}')
            tables.add(table_name)

        return len(tables) / K if K > 0 else 0.0

    def _compute_ccr(self, indices: List[int], K: int) -> float:
        if len(indices) == 0:
            return 0.0

        largest_component = self._find_largest_component(indices)
        return largest_component / K if K > 0 else 0.0

    def _find_largest_component(self, indices: List[int]) -> int:
        if len(indices) == 0:
            return 0

        n = len(indices)
        adj_list = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i+1, n):
                idx_i = indices[i]
                idx_j = indices[j]
                if self.gt_adjacency[idx_i, idx_j] > 0:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        visited = set()
        max_component_size = 0

        for start in range(n):
            if start in visited:
                continue

            queue = [start]
            visited.add(start)
            component_size = 1

            while queue:
                node = queue.pop(0)
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        component_size += 1

            max_component_size = max(max_component_size, component_size)

        return max_component_size

    def _get_metadata_by_index(self, idx: int) -> Dict:
        return {'table_name': f'unknown_{idx}', 'column_name': ''}

    def run_complete_evaluation_with_coherence(self,
                                              search_results: List[List[int]],
                                              k_values: List[int] = [1, 5, 10, 15, 20, 25]) -> Dict[str, float]:

        base_metrics = self.run_complete_evaluation(search_results)

        ground_truth = self.load_ground_truth()

        coherence_k_values = [k for k in [5, 15, 25] if k in k_values]
        coherence_metrics = self.compute_coherence_metrics(
            search_results,
            ground_truth,
            k_values=coherence_k_values
        )

        base_metrics.update(coherence_metrics)

        return base_metrics
