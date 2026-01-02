import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TextEncoder(nn.Module):
    """Encodes table and column names into vectors."""

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(text_ids)
        mask = (text_ids != 0).float().unsqueeze(-1)
        masked_embeddings = embeddings * mask
        lengths = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        pooled = masked_embeddings.sum(dim=1) / lengths.squeeze(-1)
        return pooled


class ContentEncoder(nn.Module):
    """Encodes column content features into vectors."""

    def __init__(self, content_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(content_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        return self.projection(content_features)


def build_dual_hypergraph_from_pairs(num_columns: int,
                                     joinable_pairs: List[Tuple[int, int]],
                                     metadata: List[dict],
                                     use_type1_edges: bool = True,
                                     use_type2_edges: bool = True) -> Tuple[torch.Tensor, int]:
    type1_hyperedges = []
    valid_pairs = 0
    components = {}

    if use_type1_edges:
        parent = list(range(num_columns))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        for col_i, col_j in joinable_pairs:
            if col_i < num_columns and col_j < num_columns:
                union(col_i, col_j)
                valid_pairs += 1

        components = defaultdict(list)
        for col in range(num_columns):
            components[find(col)].append(col)

        for component in components.values():
            if len(component) >= 2:
                type1_hyperedges.append(sorted(component))

    type2_hyperedges = []
    table_groups = {}

    if use_type2_edges:
        table_groups = defaultdict(list)
        for col_idx, meta in enumerate(metadata):
            if col_idx >= num_columns:
                break
            table_name = meta.get('table_name', '')
            if table_name:
                table_groups[table_name].append(col_idx)

        for columns in table_groups.values():
            if len(columns) >= 2:
                type2_hyperedges.append(sorted(columns))

    all_hyperedges = type1_hyperedges + type2_hyperedges
    num_hyperedges = len(all_hyperedges)

    if num_hyperedges == 0:
        print("[WARNING] No hyperedges generated")
        return torch.eye(num_columns), 0

    H = torch.zeros(num_columns, num_hyperedges)
    for edge_idx, nodes in enumerate(all_hyperedges):
        for node in nodes:
            H[node, edge_idx] = 1.0

    print(f"[INFO] Hypergraph: {len(type1_hyperedges)} Type1 + {len(type2_hyperedges)} Type2 = {num_hyperedges} total")

    return H, len(type1_hyperedges)


def build_batch_hypergraph(batch_indices: List[int],
                           global_hypergraph: torch.Tensor) -> torch.Tensor:
    batch_H = global_hypergraph[batch_indices]
    relevant_edges = (batch_H.sum(dim=0) > 0)
    return batch_H[:, relevant_edges]