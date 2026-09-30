"""
Hypergraph Utility Functions
=============================

Utility functions for building and manipulating column-level hypergraphs.
This file contains only the utility functions needed by the enhanced model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from collections import defaultdict


class TextEncoder(nn.Module):
    """Text encoder: encodes table_name and column_name into vectors (identical to the baseline)."""

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: [batch_size, seq_len] text token ids
        Returns:
            [batch_size, embed_dim] mean-pooled text embedding
        """
        embeddings = self.embedding(text_ids)  # [batch_size, seq_len, embed_dim]

        # Build a mask to exclude padding
        mask = (text_ids != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Mean pooling (padding excluded)
        masked_embeddings = embeddings * mask
        lengths = mask.sum(dim=1, keepdim=True)  # [batch_size, 1, 1]
        lengths = torch.clamp(lengths, min=1.0)  # avoid division by zero

        pooled = masked_embeddings.sum(dim=1) / lengths.squeeze(-1)  # [batch_size, embed_dim]
        return pooled


class ContentEncoder(nn.Module):
    """Content encoder: encodes a column's cell values into a vector (identical to the baseline)."""

    def __init__(self, content_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(content_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content_features: [batch_size, content_dim] column content features
        Returns:
            [batch_size, embed_dim] content embedding
        """
        return self.projection(content_features)


def build_dual_hypergraph_from_pairs(num_columns: int,
                                      joinable_pairs: List[Tuple[int, int]],
                                      metadata: List[dict],
                                      use_type1_edges: bool = True,
                                      use_type2_edges: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Build the dual-hyperedge hypergraph.

    Type 1: joinable connected-component hyperedges
        - all columns connected via joinable relations form one hyperedge
        - connected components are found with Union-Find

    Type 2: same-table column hyperedges
        - all columns of the same table form one hyperedge

    Args:
        num_columns: total number of columns
        joinable_pairs: [(col_i, col_j), ...] joinable-pair list
        metadata: [{'table_name': 'xxx', ...}, ...] metadata list
        use_type1_edges: whether to use Type-1 hyperedges (inter-table joins)
        use_type2_edges: whether to use Type-2 hyperedges (intra-table columns)

    Returns:
        H: [num_columns, num_hyperedges] incidence matrix
        num_type1_edges: number of Type-1 hyperedges
    """
    # ========================================
    # Type 1: joinable connected-component hyperedges (Union-Find)
    # ========================================
    type1_hyperedges = []
    valid_pairs = 0
    components = {}

    if use_type1_edges:
        # Initialise the union-find structure
        parent = list(range(num_columns))

        def find(x):
            """Find the root (path compression)."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            """Union two sets."""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Union all joinable relations
        for col_i, col_j in joinable_pairs:
            if col_i < num_columns and col_j < num_columns:
                union(col_i, col_j)
                valid_pairs += 1

        # Extract the connected components
        components = defaultdict(list)
        for col in range(num_columns):
            root = find(col)
            components[root].append(col)

        # Create Type-1 hyperedges (size >= 2)
        for component in components.values():
            if len(component) >= 2:
                type1_hyperedges.append(sorted(component))

    # ========================================
    # Type 2: same-table column hyperedges
    # ========================================
    type2_hyperedges = []
    table_groups = {}

    if use_type2_edges:
        # Group by table_name
        table_groups = defaultdict(list)
        for col_idx, meta in enumerate(metadata):
            if col_idx >= num_columns:
                break
            table_name = meta.get('table_name', '')
            if table_name:
                table_groups[table_name].append(col_idx)

        # Create Type-2 hyperedges (size >= 2)
        for table_name, columns in table_groups.items():
            if len(columns) >= 2:
                type2_hyperedges.append(sorted(columns))

    # ========================================
    # Build the incidence matrix
    # ========================================

    # Merge the two hyperedge types
    all_hyperedges = type1_hyperedges + type2_hyperedges
    num_hyperedges = len(all_hyperedges)

    if num_hyperedges == 0:
        print("  No hyperedges were generated; returning the identity matrix")
        return torch.eye(num_columns), 0

    # Build the incidence matrix H
    H = torch.zeros(num_columns, num_hyperedges)

    for edge_idx, nodes in enumerate(all_hyperedges):
        for node in nodes:
            H[node, edge_idx] = 1.0

    # Statistics
    type1_sizes = [len(he) for he in type1_hyperedges]
    type2_sizes = [len(he) for he in type2_hyperedges]
    all_sizes = type1_sizes + type2_sizes

    column_hyperedge_counts = H.sum(dim=1)
    isolated_columns = (column_hyperedge_counts == 0).sum().item()

    print(f"\n Dual-hyperedge statistics:")
    print(f"  Raw joinable pairs: {len(joinable_pairs)} (valid: {valid_pairs})")
    print(f"  Connected components: {len(components)}")
    print(f"  Tables: {len(table_groups)}")
    print(f"\n   Type 1 (joinable components): {len(type1_hyperedges)} {'' if use_type1_edges else ' (disabled)'}")
    if type1_sizes:
        print(f"     Sizes: min={min(type1_sizes)}, max={max(type1_sizes)}, avg={sum(type1_sizes)/len(type1_sizes):.2f}")
    print(f"   Type 2 (same-table columns): {len(type2_hyperedges)} {'' if use_type2_edges else ' (disabled)'}")
    if type2_sizes:
        print(f"     Sizes: min={min(type2_sizes)}, max={max(type2_sizes)}, avg={sum(type2_sizes)/len(type2_sizes):.2f}")
    print(f"\n   Total hyperedges: {num_hyperedges}")
    print(f"     Hyperedge sizes: min={min(all_sizes)}, max={max(all_sizes)}, avg={sum(all_sizes)/len(all_sizes):.2f}")
    print(f"     Total incidences: {int(H.sum().item())}")
    print(f"     Avg hyperedges per column: {column_hyperedge_counts.mean().item():.2f}")
    print(f"     Isolated columns: {isolated_columns} / {num_columns}")

    density = (H > 0).sum().item() / H.numel() * 100
    print(f"     Hypergraph density: {density:.2f}%")

    return H, len(type1_hyperedges)


def build_batch_hypergraph(batch_indices: List[int],
                            global_hypergraph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the sub-hypergraph relevant to a batch from the global hypergraph.

    Args:
        batch_indices: [batch_size] column indices of the current batch
        global_hypergraph: [total_columns, num_hyperedges] global hypergraph

    Returns:
        batch_hypergraph: [batch_size, num_relevant_edges] batch sub-hypergraph
        relevant_edges:   [num_hyperedges] bool mask of the kept columns.
                          Use ``relevant_edges[:num_type1_edges].sum()`` to get
                          the number of Type-1 edges that survive in the batch;
                          the global ``num_type1_edges`` is NOT valid for the
                          filtered batch hypergraph.
    """
    # Take the rows of the batch indices
    batch_H = global_hypergraph[batch_indices]  # [batch_size, num_hyperedges]

    # Keep the hyperedges that involve at least one batch node
    relevant_edges = (batch_H.sum(dim=0) > 0)

    # Slice out the relevant hyperedges
    batch_H = batch_H[:, relevant_edges]  # [batch_size, num_relevant_edges]

    return batch_H, relevant_edges


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (InfoNCE).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                query_emb: torch.Tensor,
                positive_emb: torch.Tensor,
                negative_embs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query_emb: [batch_size, embed_dim] query embeddings
            positive_emb: [batch_size, embed_dim] positive embeddings
            negative_embs: [batch_size, num_neg, embed_dim] negative embeddings (optional)

        Returns:
            loss: scalar
        """
        batch_size = query_emb.size(0)

        # L2-normalise so dot products are cosine similarities
        query_emb = F.normalize(query_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)

        # Positive similarity (cosine)
        pos_sim = torch.sum(query_emb * positive_emb, dim=1) / self.temperature  # [batch]

        # Negative similarities
        if negative_embs is not None:
            # Normalise the negatives
            negative_embs = F.normalize(negative_embs, p=2, dim=2)
            # Use the provided negatives
            neg_sim = torch.matmul(
                query_emb.unsqueeze(1),  # [batch, 1, dim]
                negative_embs.transpose(1, 2)  # [batch, dim, num_neg]
            ).squeeze(1) / self.temperature  # [batch, num_neg]

            # Concatenate positive and negative logits
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1+num_neg]
        else:
            # Use the other in-batch items as negatives (cosine similarity)
            all_sim = torch.matmul(query_emb, positive_emb.t()) / self.temperature  # [batch, batch]
            logits = all_sim

        # Labels: positive at index 0 (or the diagonal)
        if negative_embs is not None:
            labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
        else:
            labels = torch.arange(batch_size, dtype=torch.long, device=query_emb.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
