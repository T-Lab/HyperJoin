"""
Enhanced Layers
====================================

Brings the Graph ViT / MLP-Mixer ideas into HyperJoin:
1. Three-level positional encoding (ThreeLevelPositionalEncoding)
2. Patch GNN encoder (PatchGNNEncoder)
3. Hypergraph-aware Mixer layer (HypergraphAwareMixer)

Authors: Claude + liushiyuan
Date: 2025-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# ============================================
# 1. Three-level positional encoding
# ============================================

class ThreeLevelPositionalEncoding(nn.Module):
    """
    Three-level positional encoding.

    Purpose: inject structural position information into column embeddings.

    Three levels:
    - Table-level PE: captures the global semantics of the table a column belongs to
    - Column-level PE: captures a column's structural position in the joinable graph
    - (Cell-level PE: not used here; too fine-grained)

    Inspired by:
    - Laplacian Positional Encoding (LapPE) from Graph Transformers
    - uses eigenvectors of the graph Laplacian to encode node positions
    """

    def __init__(self,
                 embed_dim: int,
                 num_tables: int,
                 column_pe_dim: int = 16,
                 use_table_pe: bool = True,
                 use_column_pe: bool = True):
        """
        Initialise the three-level positional encoding module.

        Args:
            embed_dim: dimension of the column embeddings
            num_tables: number of tables in the dataset
            column_pe_dim: dimension of the Column-level PE (top-k Laplacian eigenvectors)
            use_table_pe: whether to use Table-level PE
            use_column_pe: whether to use Column-level PE
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tables = num_tables
        self.column_pe_dim = column_pe_dim
        self.use_table_pe = use_table_pe
        self.use_column_pe = use_column_pe

        # ----------------------------------------
        # Table-level PE: learnable table embeddings
        # ----------------------------------------
        if self.use_table_pe:
            # One learnable embedding per table;
            # captures global semantics (e.g. "user table" vs "order table")
            self.table_embeddings = nn.Embedding(num_tables, embed_dim)
            nn.init.normal_(self.table_embeddings.weight, std=0.02)

        # ----------------------------------------
        # Column-level PE: Laplacian positional encoding
        # ----------------------------------------
        if self.use_column_pe:
            # Project the Laplacian eigenvectors (column_pe_dim) up to embed_dim
            # with a two-layer MLP.
            self.column_pe_encoder = nn.Sequential(
                nn.Linear(column_pe_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )

        # ----------------------------------------
        # Fusion weights: learn how to combine the PE levels
        # ----------------------------------------
        # alpha: weight of the table PE
        # beta: weight of the column PE
        # the raw embedding keeps weight (1 - alpha - beta)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Table PE weight
        self.beta = nn.Parameter(torch.tensor(0.1))   # Column PE weight

    def compute_column_pe(self,
                         adjacency_matrix: torch.Tensor,
                         k: Optional[int] = None) -> torch.Tensor:
        """
        Compute the Column-level positional encoding (Laplacian eigendecomposition).

        Method:
        - Build the joinable graph over columns (joinable pairs become edges)
        - Compute the graph Laplacian L = D - A
        - Eigendecompose L = V A V^T
        - Keep the eigenvectors of the k smallest eigenvalues as the PE

        Intuition:
        - eigenvectors capture each column's "structural role" in the joinable graph,
        - e.g. centrality, bridging, community membership

        Args:
            adjacency_matrix: [num_columns, num_columns] column adjacency;
                              adjacency_matrix[i, j] = 1 iff columns i and j are joinable
            k: number of eigenvectors to use (defaults to self.column_pe_dim)

        Returns:
            column_pe: [num_columns, k] Column-level positional encoding
        """
        if k is None:
            k = self.column_pe_dim

        # Convert to numpy
        A = adjacency_matrix.cpu().numpy()
        num_columns = A.shape[0]

        # Memory optimisation: use sparse matrices for large datasets (e.g. CAN_ALL)
        # Check the sparsity
        num_nonzero = np.count_nonzero(A)
        sparsity = num_nonzero / (num_columns * num_columns)
        use_sparse = (sparsity < 0.1) or (num_columns > 10000)  # sparsity <10% or >10k columns

        if use_sparse:
            print(f"   Using sparse-matrix computation (sparsity: {sparsity*100:.4f}%, non-zeros: {num_nonzero})")

            # Convert to a sparse matrix
            A_sparse = csr_matrix(A)

            # Compute the degree vector
            degrees = np.array(A_sparse.sum(axis=1)).flatten()

            # Avoid division by zero
            degrees_safe = degrees + 1e-8

            # Normalised Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
            # via sparse operations
            from scipy.sparse import eye
            D_inv_sqrt_diag = 1.0 / np.sqrt(degrees_safe)
            D_inv_sqrt = csr_matrix(np.diag(D_inv_sqrt_diag))
            I = eye(num_columns, format='csr')
            L_norm = I - D_inv_sqrt @ A_sparse @ D_inv_sqrt

            # Sparse eigendecomposition: keep the k smallest eigenvalues
            try:
                # eigsh is specialised for sparse symmetric matrices
                # which='SM' selects the smallest eigenvalues
                eigenvalues, eigenvectors = eigsh(
                    L_norm,
                    k=min(k, num_columns - 2),  # eigsh requires k < n-1
                    which='SM',  # Smallest eigenvalues
                    tol=1e-3,    # relaxed tolerance for speed
                    maxiter=1000
                )

                # eigsh returns eigenvalues in ascending order
                column_pe = eigenvectors[:, :k]  # [num_columns, k]

                print(f"   Sparse eigendecomposition done: eigenvalues in [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

            except Exception as e:
                # Fall back to random init if sparse eigendecomposition fails
                print(f"   Warning: sparse Laplacian eigendecomposition failed ({e}); using random init")
                column_pe = np.random.randn(num_columns, k) * 0.01

        else:
            # Dense computation for small / dense matrices
            print(f"   Using dense-matrix computation (sparsity: {sparsity*100:.4f}%)")

            # Compute the degree matrix D
            D = np.diag(A.sum(axis=1))

            # Compute the Laplacian L = D - A
            L = D - A

            # Normalised Laplacian: L_norm = D^{-1/2} L D^{-1/2}
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt

            # Eigendecomposition
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
                column_pe = eigenvectors[:, :k]  # [num_columns, k]

            except np.linalg.LinAlgError:
                print("   Warning: Laplacian eigendecomposition failed; using random init")
                column_pe = np.random.randn(num_columns, k) * 0.01

        # Back to a torch tensor
        column_pe = torch.from_numpy(column_pe).float().to(adjacency_matrix.device)

        return column_pe

    def forward(self,
                column_embeddings: torch.Tensor,
                table_ids: torch.Tensor,
                column_pe: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward: inject positional encoding into the column embeddings.

        Args:
            column_embeddings: [num_columns, embed_dim] raw column embeddings
            table_ids: [num_columns] table ID of each column
            column_pe: [num_columns, column_pe_dim] precomputed Column PE (optional)
            adjacency_matrix: [num_columns, num_columns] column adjacency (if column_pe is not given)

        Returns:
            enhanced_embeddings: [num_columns, embed_dim] PE-augmented embeddings
        """
        num_columns = column_embeddings.shape[0]

        # Start from the raw embeddings
        enhanced_embeddings = column_embeddings

        # ----------------------------------------
        # Step 1: add Table-level PE
        # ----------------------------------------
        if self.use_table_pe:
            # Fetch the table embedding of each column
            table_pe = self.table_embeddings(table_ids)  # [num_columns, embed_dim]

            # Weighted fusion
            enhanced_embeddings = enhanced_embeddings + self.alpha * table_pe

        # ----------------------------------------
        # Step 2: add Column-level PE
        # ----------------------------------------
        if self.use_column_pe:
            # Compute column_pe on the fly when not precomputed
            if column_pe is None:
                if adjacency_matrix is None:
                    raise ValueError("Either column_pe or adjacency_matrix must be provided")
                column_pe = self.compute_column_pe(adjacency_matrix)

            # Encode the column PE up to embed_dim with the MLP
            column_pe_encoded = self.column_pe_encoder(column_pe)  # [num_columns, embed_dim]

            # Weighted fusion
            enhanced_embeddings = enhanced_embeddings + self.beta * column_pe_encoded

        return enhanced_embeddings


# ============================================
# 2. Patch GNN encoder
# ============================================

class PatchGNNEncoder(nn.Module):
    """
    Patch GNN encoder (patch-based graph neural network encoder).

    Purpose: local aggregation inside each patch (hyperedge), like the intra-patch

    Design:
    - each patch corresponds to one hyperedge (Type 1 or Type 2)
    - nodes exchange and aggregate messages inside a patch
    - the output is one aggregated embedding per patch

    Difference from the original BiHMP:
    - BiHMP: Node -> Hyperedge -> Node (two-way propagation)
    - PatchGNN: Node -> Patch only (no propagation back to nodes)
    - the patch-level embeddings feed the Mixer
    """

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 aggregation: str = 'mean',
                 use_edge_type: bool = True):
        """
        Initialise the Patch GNN encoder.

        Args:
            embed_dim: dimension of the column embeddings
            hidden_dim: hidden dimension (defaults to embed_dim)
            num_layers: number of GNN layers
            aggregation: aggregation method ('mean', 'max', 'sum')
            use_edge_type: whether to distinguish Type-1 and Type-2 hyperedges
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.use_edge_type = use_edge_type

        # ----------------------------------------
        # GNN layers: plain MLPs that transform node features
        # ----------------------------------------
        self.node_encoders = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else self.hidden_dim
            out_dim = self.hidden_dim
            self.node_encoders.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim)
            ))

        # ----------------------------------------
        # With edge typing, Type-1 and Type-2 get different aggregation parameters
        # ----------------------------------------
        if use_edge_type:
            # Type 1: joinable columns (emphasises inter-column relatedness)
            self.type1_aggregator = nn.Linear(self.hidden_dim, embed_dim)

            # Type 2: same-table columns (emphasises intra-table coherence)
            self.type2_aggregator = nn.Linear(self.hidden_dim, embed_dim)
        else:
            # Single shared aggregator
            self.aggregator = nn.Linear(self.hidden_dim, embed_dim)

    def aggregate_nodes(self,
                       node_features: torch.Tensor,
                       node_indices: List[int]) -> torch.Tensor:
        """
        Aggregate node features inside one patch.

        Args:
            node_features: [num_columns, hidden_dim] features of all columns
            node_indices: indices of the columns inside the patch

        Returns:
            patch_embedding: [hidden_dim] aggregated patch embedding
        """
        # Extract the node features of the patch
        patch_nodes = node_features[node_indices]  # [patch_size, hidden_dim]

        # Aggregate according to the configured method
        if self.aggregation == 'mean':
            patch_embedding = patch_nodes.mean(dim=0)
        elif self.aggregation == 'max':
            patch_embedding = patch_nodes.max(dim=0)[0]
        elif self.aggregation == 'sum':
            patch_embedding = patch_nodes.sum(dim=0)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")

        return patch_embedding

    def forward(self,
                column_embeddings: torch.Tensor,
                H: torch.Tensor,
                num_type1_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: aggregate column embeddings into patch embeddings.

        Args:
            column_embeddings: [num_columns, embed_dim] column embeddings
            H: [num_columns, num_hyperedges] hypergraph incidence matrix;
               H[i, j] = 1 iff column i belongs to hyperedge j
            num_type1_edges: number of Type-1 hyperedges

        Returns:
            patch_embeddings_type1: [num_type1_edges, embed_dim] Type-1 patch embeddings
            patch_embeddings_type2: [num_type2_edges, embed_dim] Type-2 patch embeddings
        """
        num_columns, num_hyperedges = H.shape
        num_type2_edges = num_hyperedges - num_type1_edges

        # ----------------------------------------
        # Step 1: transform node features through the GNN layers
        # ----------------------------------------
        node_features = column_embeddings
        for encoder in self.node_encoders:
            node_features = encoder(node_features)  # [num_columns, hidden_dim]

        # ----------------------------------------
        # Step 2: aggregate to the patch (hyperedge) level
        # ----------------------------------------
        # Batched aggregation via matrix multiplication: H^T @ node_features
        # H^T: [num_hyperedges, num_columns]
        # Result: [num_hyperedges, hidden_dim]
        patch_embeddings = torch.mm(H.t(), node_features)  # [num_hyperedges, hidden_dim]

        # Normalise by the patch size (node count)
        patch_sizes = H.sum(dim=0, keepdim=True).t()  # [num_hyperedges, 1]
        patch_sizes = torch.clamp(patch_sizes, min=1.0)  # avoid division by zero
        patch_embeddings = patch_embeddings / patch_sizes  # mean aggregation

        # ----------------------------------------
        # Step 3: split Type-1 / Type-2 patches and transform them separately
        # ----------------------------------------
        patch_embeddings_type1_raw = patch_embeddings[:num_type1_edges]  # [num_type1_edges, hidden_dim]
        patch_embeddings_type2_raw = patch_embeddings[num_type1_edges:]  # [num_type2_edges, hidden_dim]

        if self.use_edge_type:
            # Type-specific aggregators
            patch_embeddings_type1 = self.type1_aggregator(patch_embeddings_type1_raw)
            patch_embeddings_type2 = self.type2_aggregator(patch_embeddings_type2_raw)
        else:
            # Shared aggregator
            patch_embeddings_type1 = self.aggregator(patch_embeddings_type1_raw)
            patch_embeddings_type2 = self.aggregator(patch_embeddings_type2_raw)

        return patch_embeddings_type1, patch_embeddings_type2


# ============================================
# 3. Hypergraph-aware Mixer layer
# ============================================

class HypergraphAwareMixer(nn.Module):
    """
    Hypergraph-aware Mixer layer (Hypergraph-Aware MLP-Mixer).

    Purpose: global information exchange across patches while preserving the
    hypergraph structural prior.
    Design:
    - inspired by MLP-Mixer with a Token Mixer and a Channel Mixer
    - Token Mixer: information mixing across patches (attention-like)
    - Channel Mixer: mixing across feature dimensions within a patch
    - key innovation: the Token Mixer is weighted by the hypergraph structure

    Hypergraph structural prior:
    - build the patch adjacency matrix: two patches are adjacent if they share a column
    - the adjacency weights/masks the attention inside the Token Mixer
    - patch interaction therefore follows the hypergraph topology
    """

    def __init__(self,
                 embed_dim: int,
                 num_patches: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_structure_bias: bool = True):
        """
        Initialise the hypergraph-aware Mixer layer.

        Args:
            embed_dim: patch embedding dimension
            num_patches: number of patches (hyperedges)
            mlp_ratio: expansion ratio of the MLP hidden layer
            dropout: dropout rate
            use_structure_bias: whether to weight by the hypergraph structure
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.mlp_ratio = mlp_ratio
        self.use_structure_bias = use_structure_bias

        # ----------------------------------------
        # Token Mixer: information mixing across patches
        # ----------------------------------------
        # Input: [num_patches, embed_dim]
        # the Token Mixer operates along the patch dimension
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        # ----------------------------------------
        # Channel Mixer: mixing across feature dimensions
        # ----------------------------------------
        # the Channel Mixer operates along embed_dim
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        # ----------------------------------------
        # Hypergraph-structure attention (when the structural prior is used)
        # ----------------------------------------
        if use_structure_bias:
            # Learn to blend structure into the Token Mixer
            # via multi-head attention
            self.num_heads = 8
            self.head_dim = embed_dim // self.num_heads
            assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

            self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

            # Structure weighting parameter
            self.structure_weight = nn.Parameter(torch.tensor(0.5))

    def compute_patch_adjacency(self, H: torch.Tensor) -> torch.Tensor:
        """
        Compute the adjacency matrix between patches (hyperedges).

        Method:
        - two patches are adjacent if they share at least one column node
        - A_patch[i, j] = |patch_i ∩ patch_j| (number of shared columns)

        Args:
            H: [num_columns, num_hyperedges] hypergraph incidence matrix

        Returns:
            A_patch: [num_hyperedges, num_hyperedges] patch adjacency matrix
        """
        # H^T @ H counts the shared columns of every edge pair
        # (H^T @ H)[i, j] = sum_k H[k, i] * H[k, j] = |patch_i ∩ patch_j|
        A_patch = torch.mm(H.t(), H)  # [num_hyperedges, num_hyperedges]

        # Zero the diagonal (a patch is not adjacent to itself)
        A_patch = A_patch - torch.diag(torch.diag(A_patch))

        # Normalise to [0, 1]
        max_val = A_patch.max()
        if max_val > 0:
            A_patch = A_patch / max_val

        return A_patch

    def structure_aware_attention(self,
                                 patch_embeddings: torch.Tensor,
                                 adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Structure-aware attention.

        Method:
        - standard self-attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
        - structure-aware:     Attention = softmax(QK^T / sqrt(d) + lambda * A_patch) V
        - A_patch is the patch adjacency matrix; lambda is learnable

        Args:
            patch_embeddings: [num_patches, embed_dim] patch embeddings
            adjacency_matrix: [num_patches, num_patches] patch adjacency matrix

        Returns:
            attended_embeddings: [num_patches, embed_dim] attention-augmented embeddings
        """
        num_patches = patch_embeddings.shape[0]

        # ----------------------------------------
        # Step 1: compute Q, K, V
        # ----------------------------------------
        qkv = self.qkv_proj(patch_embeddings)  # [num_patches, embed_dim * 3]
        qkv = qkv.reshape(num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # [3, num_heads, num_patches, head_dim]

        Q, K, V = qkv[0], qkv[1], qkv[2]  # each [num_heads, num_patches, head_dim]

        # ----------------------------------------
        # Step 2: attention scores (with the structural prior)
        # ----------------------------------------
        # Standard attention scores: QK^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_scores: [num_heads, num_patches, num_patches]

        # Add the structural prior A_patch;
        # the same structure is added to every head
        structure_bias = adjacency_matrix.unsqueeze(0)  # [1, num_patches, num_patches]
        attn_scores = attn_scores + self.structure_weight * structure_bias

        # Softmax normalisation
        attn_weights = F.softmax(attn_scores, dim=-1)  # [num_heads, num_patches, num_patches]

        # ----------------------------------------
        # Step 3: apply the attention weights to V
        # ----------------------------------------
        attended = torch.matmul(attn_weights, V)  # [num_heads, num_patches, head_dim]

        # Concatenate the heads
        attended = attended.permute(1, 0, 2).contiguous()  # [num_patches, num_heads, head_dim]
        attended = attended.reshape(num_patches, self.embed_dim)  # [num_patches, embed_dim]

        # Output projection
        attended_embeddings = self.out_proj(attended)

        return attended_embeddings

    def forward(self,
                patch_embeddings: torch.Tensor,
                H: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward: the Mixer processes patch embeddings.

        Args:
            patch_embeddings: [num_patches, embed_dim] patch embeddings
            H: [num_columns, num_hyperedges] hypergraph incidence (for the adjacency)
            adjacency_matrix: [num_patches, num_patches] precomputed patch adjacency

        Returns:
            output_embeddings: [num_patches, embed_dim] processed patch embeddings
        """
        # ----------------------------------------
        # With the structural prior, compute or fetch the adjacency matrix first
        # ----------------------------------------
        if self.use_structure_bias:
            if adjacency_matrix is None:
                if H is None:
                    raise ValueError("Either H or adjacency_matrix must be provided")
                adjacency_matrix = self.compute_patch_adjacency(H)

            # Structure-aware attention replaces the standard Token Mixer
            token_mixed = self.structure_aware_attention(patch_embeddings, adjacency_matrix)
            output = patch_embeddings + token_mixed  # residual connection
        else:
            # Standard Token Mixer (original MLP-Mixer)
            # Note: transpose needed to operate along the patch dimension
            token_mixed = self.token_mixer(patch_embeddings.t()).t()
            output = patch_embeddings + token_mixed  # residual connection

        # ----------------------------------------
        # Channel Mixer: mixing across feature dimensions
        # ----------------------------------------
        channel_mixed = self.channel_mixer(output)
        output = output + channel_mixed  # residual connection

        return output


# ============================================
# Helper functions
# ============================================

def build_column_adjacency_matrix(joinable_pairs: List[Tuple[int, int]],
                                 num_columns: int) -> torch.Tensor:
    """
    Build the column adjacency matrix from joinable pairs.

    Args:
        joinable_pairs: [(col_i, col_j), ...] list of joinable column pairs
        num_columns: total number of columns

    Returns:
        adjacency_matrix: [num_columns, num_columns] adjacency matrix
    """
    adjacency_matrix = torch.zeros(num_columns, num_columns)

    for i, j in joinable_pairs:
        adjacency_matrix[i, j] = 1.0
        adjacency_matrix[j, i] = 1.0  # undirected graph

    return adjacency_matrix


def visualize_position_encoding(column_pe: torch.Tensor,
                                column_names: List[str],
                                save_path: str = "column_pe_visualization.png"):
    """
    Visualise the Column-level positional encoding.

    For debugging and for understanding the structure captured by the PE.

    Args:
        column_pe: [num_columns, pe_dim] positional encoding
        column_names: list of column names
        save_path: where to save the figure
    """
    try:
        import matplotlib.pyplot as plt

        # 2D visualisation with the first two dimensions
        pe_np = column_pe.cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.scatter(pe_np[:, 0], pe_np[:, 1], alpha=0.6)

        # Add column-name labels
        for i, name in enumerate(column_names):
            plt.annotate(name, (pe_np[i, 0], pe_np[i, 1]),
                        fontsize=8, alpha=0.7)

        plt.xlabel("Position Encoding Dim 1")
        plt.ylabel("Position Encoding Dim 2")
        plt.title("Column-level Positional Encoding Visualization")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Positional-encoding visualisation saved to: {save_path}")

    except ImportError:
        print("Warning: matplotlib not installed; cannot visualise the positional encoding")
