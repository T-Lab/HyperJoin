"""
Graph Builder for Search Phase
================================

Build GT adjacency matrix for MST reranking.
Similar to graph_builder_train.py but specialized for search-time usage.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional


class GraphBuilderSearch:
    """
    Build GT adjacency matrix for MST reranking.

    This differs from GraphBuilderTrain in that:
    - No need for edge_index tensor (only adjacency matrix)
    - No need for positive_pairs list (only for reranking)
    - Simpler interface focused on search-time requirements
    """

    def __init__(self,
                 target_dataset,
                 joinable_pairs_path: str,
                 only_can: bool = True,
                 cache_path: Optional[str] = None):
        """
        Initialize graph builder.

        Args:
            target_dataset: Target dataset with metadata
            joinable_pairs_path: Path to joinable_pairs.csv
            only_can: Whether to only use CAN/USA/UK/SG data
            cache_path: Optional path to cache the adjacency matrix
        """
        self.target_dataset = target_dataset
        self.joinable_pairs_path = joinable_pairs_path
        self.only_can = only_can
        self.cache_path = cache_path

    def _build_table_column_mapping(self) -> Dict[str, int]:
        """
        Build mapping: "table.column" -> target index.

        This mapping is used to convert joinable pairs (in table.column format)
        to target indices for adjacency matrix construction.

        Returns:
            Dictionary mapping "table_name.column_name" to index
        """
        mapping = {}
        md = getattr(self.target_dataset, 'metadata', None)
        items = []

        # Handle different metadata formats
        if md:
            if isinstance(md, dict) and 'metadata' in md:
                items = md['metadata']
            elif isinstance(md, list):
                items = md

        # Build mapping
        for idx, meta in enumerate(items):
            table_name = str(meta.get('table_name', f'target_{idx}')).replace('.csv', '')
            column_name = str(meta.get('column_name', f'col_{idx}'))
            key = f"{table_name}.{column_name}"
            mapping[key] = idx

        return mapping

    def build_adjacency_matrix(self, verbose: bool = True) -> np.ndarray:
        """
        Build GT adjacency matrix from joinable pairs.

        The adjacency matrix A is:
        - A[i, j] = 1 if columns i and j are joinable (i != j)
        - A[i, j] = 0 otherwise
        - Symmetric: A[i, j] = A[j, i]
        - No self-loops: A[i, i] = 0

        Args:
            verbose: Whether to print statistics

        Returns:
            Adjacency matrix [N, N] of dtype float32

        Raises:
            FileNotFoundError: If joinable_pairs_path does not exist
        """
        # Check cache first
        if self.cache_path and os.path.exists(self.cache_path):
            if verbose:
                print(f" Loading cached adjacency matrix from {self.cache_path}")
            adjacency = np.load(self.cache_path)
            if verbose:
                num_nodes = adjacency.shape[0]
                num_edges = int(np.sum(adjacency) / 2)  # Divide by 2 for undirected
                density = num_edges / (num_nodes * (num_nodes - 1) / 2) * 100
                print(f" Loaded: {num_nodes} nodes, {num_edges} edges, "
                      f"density={density:.4f}%")
            return adjacency

        # Build from scratch
        if not os.path.exists(self.joinable_pairs_path):
            raise FileNotFoundError(
                f"Joinable pairs file not found: {self.joinable_pairs_path}"
            )

        if verbose:
            print(f" Building GT adjacency matrix from {self.joinable_pairs_path}")

        # Load joinable pairs
        df = pd.read_csv(self.joinable_pairs_path)

        # Filter by dataset if needed
        if self.only_can:
            opendata_prefixes = ('CAN_CSV', 'USA_CSV', 'UK_CSV', 'SG_CSV')
            mask = df['query_table'].astype(str).str.startswith(opendata_prefixes)
            df = df[mask]
            if verbose:
                print(f"   Filtered to OpenData tables: {len(df)} pairs")

        # Build table.column -> index mapping
        mapping = self._build_table_column_mapping()
        num_nodes = len(self.target_dataset)

        # Initialize adjacency matrix
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # Populate adjacency matrix
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
                if u != v:  # Exclude self-loops
                    adjacency[u, v] = 1.0
                    adjacency[v, u] = 1.0  # Undirected
                    edge_count += 1
            else:
                skipped_count += 1

        if verbose:
            print(f" GT Graph Statistics:")
            print(f"   Nodes (columns): {num_nodes}")
            print(f"   Edges (undirected): {edge_count}")
            density = edge_count / (num_nodes * (num_nodes - 1) / 2) * 100 if num_nodes > 1 else 0
            print(f"   Density: {density:.4f}%")
            if skipped_count > 0:
                print(f"   ️ Skipped pairs (not in target): {skipped_count}")

        # Cache if path provided
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.save(self.cache_path, adjacency)
            if verbose:
                print(f" Cached adjacency matrix to {self.cache_path}")

        return adjacency

    def get_connected_components_stats(self, adjacency: np.ndarray) -> Dict:
        """
        Compute connected components statistics (for analysis).

        Args:
            adjacency: Adjacency matrix [N, N]

        Returns:
            Dictionary with statistics:
                - num_components: Number of connected components
                - largest_component_size: Size of largest component
                - component_sizes: List of component sizes
        """
        N = adjacency.shape[0]
        visited = np.zeros(N, dtype=bool)
        component_sizes = []

        def bfs(start):
            queue = [start]
            visited[start] = True
            size = 1

            while queue:
                node = queue.pop(0)
                neighbors = np.where(adjacency[node] > 0)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        size += 1
            return size

        # Find all components
        for i in range(N):
            if not visited[i]:
                size = bfs(i)
                component_sizes.append(size)

        component_sizes.sort(reverse=True)

        return {
            'num_components': len(component_sizes),
            'largest_component_size': component_sizes[0] if component_sizes else 0,
            'component_sizes': component_sizes,
            'avg_component_size': np.mean(component_sizes) if component_sizes else 0
        }
