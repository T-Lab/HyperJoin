"""
Extended Evaluator with Coherence Metrics
==========================================

Extends DetailedEvaluator to include coherence-related metrics
that measure the quality of MST reranking.

New Metrics:
-----------
1. Coherence@K: Proportion of pairs in Top-K that are connected in GT
2. Table Diversity@K: Number of distinct tables in Top-K
3. CCR@K (Connected Component Ratio): Size of largest component in Top-K
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from evaluator import DetailedEvaluator


class CoherenceEvaluator(DetailedEvaluator):
    """
    Extended evaluator with coherence metrics for MST reranking.

    Motivation:
    ----------
    Standard IR metrics (Precision, Recall, F1) measure how many correct
    results are returned, but not whether they form a coherent set.

    MST reranking aims to return results that are not only relevant but
    also connected (forming a coherent group of joinable columns).

    New Metrics:
    -----------
    - Coherence@K: Measures density of GT connections in Top-K
    - Table Diversity@K: Measures table spread (lower = more coherent)
    - CCR@K: Measures largest connected component (higher = more coherent)
    """

    def __init__(self, datasets: str, gt_adjacency: Optional[np.ndarray] = None):
        """
        Initialize coherence evaluator.

        Args:
            datasets: Dataset name
            gt_adjacency: GT adjacency matrix [N, N] (optional)
                         If None, coherence metrics will not be computed
        """
        super().__init__(datasets)
        self.gt_adjacency = gt_adjacency

    def compute_coherence_metrics(self,
                                   search_results: List[List[int]],
                                   ground_truth: Dict[int, List[int]],
                                   k_values: List[int] = [5, 15, 25]) -> Dict[str, float]:
        """
        Compute coherence metrics.

        Args:
            search_results: Search results for each query
            ground_truth: Ground truth for each query
            k_values: K values to compute metrics for

        Returns:
            Dictionary with coherence metrics:
                {
                    'Coherence@5': float,
                    'TableDiversity@5': float,
                    'CCR@5': float,
                    ...
                }
        """
        if self.gt_adjacency is None:
            print("️ GT adjacency matrix not provided, skipping coherence metrics")
            return {}

        metrics = {}

        for K in k_values:
            coherence_scores = []
            diversity_scores = []
            ccr_scores = []

            for query_idx in range(len(search_results)):
                if query_idx not in ground_truth:
                    continue

                # Get Top-K predictions
                topk = search_results[query_idx][:K]

                if len(topk) == 0:
                    continue

                # 1. Coherence@K: Connected pairs / Total pairs
                coherence = self._compute_coherence(topk)
                coherence_scores.append(coherence)

                # 2. Table Diversity@K: Distinct tables / K
                diversity = self._compute_diversity(topk, K)
                diversity_scores.append(diversity)

                # 3. CCR@K: Largest component size / K
                ccr = self._compute_ccr(topk, K)
                ccr_scores.append(ccr)

            # Average across all queries
            metrics[f'Coherence@{K}'] = np.mean(coherence_scores) if coherence_scores else 0.0
            metrics[f'TableDiversity@{K}'] = np.mean(diversity_scores) if diversity_scores else 0.0
            metrics[f'CCR@{K}'] = np.mean(ccr_scores) if ccr_scores else 0.0

        return metrics

    def _compute_coherence(self, indices: List[int]) -> float:
        """
        Compute coherence score for a set of indices.

        Coherence = (# connected pairs) / (# total pairs)

        Args:
            indices: List of column indices

        Returns:
            Coherence score in [0, 1]
        """
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
        """
        Compute table diversity score.

        Diversity = (# distinct tables) / K

        Lower diversity means more coherent (columns from fewer tables).

        Args:
            indices: List of column indices
            K: Total number of positions

        Returns:
            Diversity score in [0, 1]
        """
        if len(indices) == 0:
            return 0.0

        # Extract table names
        tables = set()
        for idx in indices:
            meta = self._get_metadata_by_index(idx)
            table_name = meta.get('table_name', f'unknown_{idx}')
            tables.add(table_name)

        return len(tables) / K if K > 0 else 0.0

    def _compute_ccr(self, indices: List[int], K: int) -> float:
        """
        Compute Connected Component Ratio.

        CCR = (size of largest connected component) / K

        Higher CCR means more coherent (larger connected group).

        Args:
            indices: List of column indices
            K: Total number of positions

        Returns:
            CCR score in [0, 1]
        """
        if len(indices) == 0:
            return 0.0

        largest_component = self._find_largest_component(indices)
        return largest_component / K if K > 0 else 0.0

    def _find_largest_component(self, indices: List[int]) -> int:
        """
        Find the size of the largest connected component using BFS.

        Args:
            indices: List of column indices

        Returns:
            Size of largest connected component
        """
        if len(indices) == 0:
            return 0

        n = len(indices)
        # Build adjacency list for subgraph
        adj_list = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i+1, n):
                idx_i = indices[i]
                idx_j = indices[j]
                if self.gt_adjacency[idx_i, idx_j] > 0:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        # BFS to find connected components
        visited = set()
        max_component_size = 0

        for start in range(n):
            if start in visited:
                continue

            # BFS from start
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
        """
        Get metadata for a given target index.

        Args:
            idx: Target column index

        Returns:
            Metadata dictionary
        """
        # target_names is a list of "table.column" strings
        if self.target_names and idx < len(self.target_names):
            full_name = self.target_names[idx]
            # Parse "table_name.column_name"
            if '.' in full_name:
                parts = full_name.rsplit('.', 1)
                return {
                    'table_name': parts[0],
                    'column_name': parts[1] if len(parts) > 1 else ''
                }
            return {'table_name': full_name, 'column_name': ''}
        return {'table_name': f'unknown_{idx}', 'column_name': ''}

    def run_complete_evaluation_with_coherence(self,
                                               search_results: List[List[int]],
                                               k_values: List[int] = [1, 5, 10, 15, 20, 25]) -> Dict[str, float]:
        """
        Run complete evaluation including coherence metrics.

        This is the main evaluation interface for MST reranking experiments.

        Args:
            search_results: Search results for each query
            k_values: K values to evaluate

        Returns:
            Dictionary containing:
                - Standard metrics: Precision@K, Recall@K, F1@K
                - Coherence metrics: Coherence@K, TableDiversity@K, CCR@K

        Example:
            >>> evaluator = CoherenceEvaluator('CAN_ALL', gt_adjacency)
            >>> results = evaluator.run_complete_evaluation_with_coherence(
            ...     search_results
            ... )
            >>> print(f"Coherence@25: {results['Coherence@25']:.4f}")
        """
        # Step 1: Run standard evaluation (P, R, F1)
        base_metrics = self.run_complete_evaluation(search_results)

        # Step 2: Load ground truth
        ground_truth = self.load_ground_truth()

        # Step 3: Compute coherence metrics
        coherence_k_values = [k for k in [5, 15, 25] if k in k_values]
        coherence_metrics = self.compute_coherence_metrics(
            search_results,
            ground_truth,
            k_values=coherence_k_values
        )

        # Step 4: Merge results
        base_metrics.update(coherence_metrics)

        return base_metrics

    def print_coherence_summary(self, metrics: Dict[str, float]):
        """
        Print a formatted summary of coherence metrics.

        Args:
            metrics: Metrics dictionary from run_complete_evaluation_with_coherence
        """
        print("\n" + "="*60)
        print("  Coherence Metrics Summary")
        print("="*60)

        for k in [5, 15, 25]:
            coherence_key = f'Coherence@{k}'
            diversity_key = f'TableDiversity@{k}'
            ccr_key = f'CCR@{k}'

            if coherence_key in metrics:
                print(f"\n  @ K={k}:")
                print(f"    Coherence:       {metrics[coherence_key]:.4f}")
                print(f"    Table Diversity: {metrics[diversity_key]:.4f}")
                print(f"    CCR:             {metrics[ccr_key]:.4f}")

        print("="*60)
        print("\nInterpretation:")
        print("  Coherence@K:       Higher is better (more connected pairs)")
        print("  Table Diversity@K: Lower is better (fewer tables = more coherent)")
        print("  CCR@K:             Higher is better (larger connected component)")
