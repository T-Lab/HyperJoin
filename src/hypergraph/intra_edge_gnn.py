"""
Intra-Hyperedge GNN Module
===========================

True GNN message passing: local message passing inside each hyperedge.

Design:
- the nodes of each hyperedge form a small fully connected graph
- GNN message passing runs on that local graph
- the updated node features are aggregated into the hyperedge embedding

Advantages:
- real message passing rather than a plain MLP
- local scope (inside a hyperedge) keeps it efficient
- captures node interactions within the hyperedge
"""

import torch
import torch.nn as nn
from typing import Tuple


class IntraHyperedgeGNN(nn.Module):
    """
    GNN message-passing module inside hyperedges.

    Runs GNN message passing among the nodes of each hyperedge.
    """

    def __init__(self,
                 embed_dim: int,
                 num_layers: int = 2,
                 aggregation: str = 'mean',
                 use_edge_type: bool = True):
        """
        Args:
            embed_dim: node-embedding dimension
            num_layers: number of GNN layers
            aggregation: final aggregation method ('mean', 'max', 'sum')
            use_edge_type: whether to distinguish Type-1 and Type-2 hyperedges
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.use_edge_type = use_edge_type

        # GNN message-passing layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GNNLayer(embed_dim))

        # Edge-type-specific output transforms
        if use_edge_type:
            self.type1_output = nn.Linear(embed_dim, embed_dim)
            self.type2_output = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                column_embeddings: torch.Tensor,
                H: torch.Tensor,
                num_type1_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: GNN message passing inside every hyperedge.

        Args:
            column_embeddings: [num_columns, embed_dim] column embeddings
            H: [num_columns, num_hyperedges] hypergraph incidence matrix
            num_type1_edges: number of Type-1 hyperedges

        Returns:
            patch_embeddings_type1: [num_type1_edges, embed_dim]
            patch_embeddings_type2: [num_type2_edges, embed_dim]
        """
        num_columns, num_hyperedges = H.shape
        device = column_embeddings.device

        # Stores the embedding of each hyperedge
        hyperedge_embeddings = []

        # Process each hyperedge
        for j in range(num_hyperedges):
            # Node indices of hyperedge j
            node_mask = H[:, j] > 0  # [num_columns]
            node_indices = torch.where(node_mask)[0]  # [hyperedge_size]
            hyperedge_size = node_indices.size(0)

            if hyperedge_size == 0:
                # Empty hyperedge (should not happen)
                hyperedge_embeddings.append(torch.zeros(self.embed_dim, device=device))
                continue

            # Extract the node features of the hyperedge
            node_features = column_embeddings[node_indices]  # [hyperedge_size, embed_dim]

            # Build the fully connected adjacency inside the hyperedge:
            # every node connects to all other nodes of the hyperedge
            adj_matrix = torch.ones(hyperedge_size, hyperedge_size, device=device)
            # Self-loops are kept (standard GCN practice)

            # Symmetric normalisation: D^{-1/2} A D^{-1/2}
            degree = adj_matrix.sum(dim=1)  # [hyperedge_size]
            degree = torch.clamp(degree, min=1.0)
            degree_inv_sqrt = torch.pow(degree, -0.5)  # D^{-1/2}
            degree_inv_sqrt = degree_inv_sqrt.view(-1, 1)  # [hyperedge_size, 1]
            adj_matrix_norm = degree_inv_sqrt * adj_matrix * degree_inv_sqrt.t()  # symmetric normalisation

            # GNN message passing inside the hyperedge
            h = node_features
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, adj_matrix_norm)  # [hyperedge_size, embed_dim]

            # Aggregate the node features of the hyperedge
            if self.aggregation == 'mean':
                hyperedge_emb = h.mean(dim=0)  # [embed_dim]
            elif self.aggregation == 'max':
                hyperedge_emb = h.max(dim=0)[0]
            elif self.aggregation == 'sum':
                hyperedge_emb = h.sum(dim=0)
            else:
                raise ValueError(f"Unsupported aggregation: {self.aggregation}")

            hyperedge_embeddings.append(hyperedge_emb)

        # Stack all hyperedge embeddings
        hyperedge_embeddings = torch.stack(hyperedge_embeddings, dim=0)  # [num_hyperedges, embed_dim]

        # Split Type-1 and Type-2 hyperedges
        type1_embeddings = hyperedge_embeddings[:num_type1_edges]
        type2_embeddings = hyperedge_embeddings[num_type1_edges:]

        # Type-specific transforms
        if self.use_edge_type:
            type1_embeddings = self.type1_output(type1_embeddings)
            type2_embeddings = self.type2_output(type2_embeddings)
        else:
            type1_embeddings = self.output_proj(type1_embeddings)
            type2_embeddings = self.output_proj(type2_embeddings)

        return type1_embeddings, type2_embeddings


class GNNLayer(nn.Module):
    """
    Single GNN message-passing layer.

    h_v^(l+1) = sigma(W * (h_v^(l) + sum_{u in N(v)} h_u^(l)))
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Message transform
        self.message_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        # Update transform
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # concat(self, aggregated_messages)
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, embed_dim]
            adj_matrix: [num_nodes, num_nodes] normalised adjacency matrix

        Returns:
            updated_features: [num_nodes, embed_dim]
        """
        # Step 1: aggregate neighbour features (includes self via the loops)
        aggregated = torch.mm(adj_matrix, node_features)  # [num_nodes, embed_dim]

        # Step 2: message transform
        messages = self.message_mlp(aggregated)  # [num_nodes, embed_dim]

        # Step 3: update node features (concat raw features and messages)
        combined = torch.cat([node_features, messages], dim=-1)  # [num_nodes, 2*embed_dim]
        updated_features = self.update_mlp(combined)  # [num_nodes, embed_dim]

        # Residual connection (after LayerNorm)
        updated_features = updated_features + node_features

        return updated_features


class IntraHyperedgeGNNBatched(nn.Module):
    """
    Batched intra-hyperedge GNN (more efficient).

    Processes all hyperedges at once via sparse-style matrix ops.
    """

    def __init__(self,
                 embed_dim: int,
                 num_layers: int = 2,
                 aggregation: str = 'mean',
                 use_edge_type: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.use_edge_type = use_edge_type

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GNNLayer(embed_dim))

        # Output transforms
        if use_edge_type:
            self.type1_output = nn.Linear(embed_dim, embed_dim)
            self.type2_output = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                column_embeddings: torch.Tensor,
                H: torch.Tensor,
                num_type1_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward pass.

        Core idea:
        1. Build the intra-hyperedge adjacency (block-diagonal matrix)
        2. Run GNN message passing in one batch
        3. Aggregate to the hyperedge level with H
        """
        num_columns, num_hyperedges = H.shape
        device = column_embeddings.device

        # Build the fully connected graph inside each hyperedge
        # all nodes of a hyperedge are mutually connected
        edge_index_list = []
        for j in range(num_hyperedges):
            node_indices = torch.where(H[:, j] > 0)[0]
            hyperedge_size = node_indices.size(0)

            if hyperedge_size <= 1:
                continue

            # Fully connected: every node links to all other nodes of the hyperedge
            for i in range(hyperedge_size):
                for k in range(hyperedge_size):
                    if i != k:
                        edge_index_list.append([node_indices[i].item(), node_indices[k].item()])

        if len(edge_index_list) == 0:
            # No edges; aggregate directly
            hyperedge_embeddings = torch.mm(H.t(), column_embeddings)
            patch_sizes = H.sum(dim=0, keepdim=True).t()
            hyperedge_embeddings = hyperedge_embeddings / torch.clamp(patch_sizes, min=1.0)
        else:
            # Build the adjacency matrix (with self-loops)
            edge_index = torch.tensor(edge_index_list, device=device).t()  # [2, num_edges]
            adj_matrix = torch.zeros(num_columns, num_columns, device=device)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
            # Add self-loops
            adj_matrix.fill_diagonal_(1.0)

            # Symmetric normalisation: D^{-1/2} A D^{-1/2}
            degree = adj_matrix.sum(dim=1)  # [num_columns]
            degree = torch.clamp(degree, min=1.0)
            degree_inv_sqrt = torch.pow(degree, -0.5)  # D^{-1/2}
            # Symmetric normalisation via broadcasting
            adj_matrix_norm = degree_inv_sqrt.view(-1, 1) * adj_matrix * degree_inv_sqrt.view(1, -1)

            # GNN message passing
            h = column_embeddings
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, adj_matrix_norm)

            # Aggregate to the hyperedge level
            hyperedge_embeddings = torch.mm(H.t(), h)  # [num_hyperedges, embed_dim]
            patch_sizes = H.sum(dim=0, keepdim=True).t()
            hyperedge_embeddings = hyperedge_embeddings / torch.clamp(patch_sizes, min=1.0)

        # Split Type 1 and Type 2
        type1_embeddings = hyperedge_embeddings[:num_type1_edges]
        type2_embeddings = hyperedge_embeddings[num_type1_edges:]

        # Type-specific transforms
        if self.use_edge_type:
            type1_embeddings = self.type1_output(type1_embeddings)
            type2_embeddings = self.type2_output(type2_embeddings)
        else:
            type1_embeddings = self.output_proj(type1_embeddings)
            type2_embeddings = self.output_proj(type2_embeddings)

        return type1_embeddings, type2_embeddings
