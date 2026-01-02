import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_base import (
    TextEncoder,
    ContentEncoder
)


class ThreeLevelPositionalEncoding(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_tables: int,
                 column_pe_dim: int = 16,
                 use_table_pe: bool = True,
                 use_column_pe: bool = True):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_tables = num_tables
        self.column_pe_dim = column_pe_dim
        self.use_table_pe = use_table_pe
        self.use_column_pe = use_column_pe

        if self.use_table_pe:
            self.table_embeddings = nn.Embedding(num_tables, embed_dim)
            nn.init.normal_(self.table_embeddings.weight, std=0.02)

        if self.use_column_pe:
            self.column_pe_encoder = nn.Sequential(
                nn.Linear(column_pe_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def compute_column_pe(self,
                         adjacency_matrix: torch.Tensor,
                         k: Optional[int] = None) -> torch.Tensor:
        if k is None:
            k = self.column_pe_dim

        A = adjacency_matrix.cpu().numpy()
        num_columns = A.shape[0]

        num_nonzero = np.count_nonzero(A)
        sparsity = num_nonzero / (num_columns * num_columns)
        use_sparse = (sparsity < 0.1) or (num_columns > 10000)

        if use_sparse:

            A_sparse = csr_matrix(A)

            degrees = np.array(A_sparse.sum(axis=1)).flatten()

            degrees_safe = degrees + 1e-8

            from scipy.sparse import eye
            D_inv_sqrt_diag = 1.0 / np.sqrt(degrees_safe)
            D_inv_sqrt = csr_matrix(np.diag(D_inv_sqrt_diag))
            I = eye(num_columns, format='csr')
            L_norm = I - D_inv_sqrt @ A_sparse @ D_inv_sqrt

            try:
                eigenvalues, eigenvectors = eigsh(
                    L_norm,
                    k=min(k, num_columns - 2),
                    which='SM',
                    tol=1e-3,
                    maxiter=1000
                )

                column_pe = eigenvectors[:, :k]

            except Exception as e:
                column_pe = np.random.randn(num_columns, k) * 0.01

        else:

            D = np.diag(A.sum(axis=1))

            L = D - A

            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
                column_pe = eigenvectors[:, :k]

            except np.linalg.LinAlgError:
                column_pe = np.random.randn(num_columns, k) * 0.01

        column_pe = torch.from_numpy(column_pe).float().to(adjacency_matrix.device)

        return column_pe

    def forward(self,
                column_embeddings: torch.Tensor,
                table_ids: torch.Tensor,
                column_pe: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:

        num_columns = column_embeddings.shape[0]

        enhanced_embeddings = column_embeddings

        if self.use_table_pe:
            table_pe = self.table_embeddings(table_ids)
            enhanced_embeddings = enhanced_embeddings + self.alpha * table_pe

        if self.use_column_pe:
            if column_pe is None:
                if adjacency_matrix is None:
                    raise ValueError("必须提供 column_pe 或 adjacency_matrix 之一")
                column_pe = self.compute_column_pe(adjacency_matrix)

            column_pe_encoded = self.column_pe_encoder(column_pe)

            enhanced_embeddings = enhanced_embeddings + self.beta * column_pe_encoded

        return enhanced_embeddings


class PatchGNNEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 2,
                 aggregation: str = 'mean',
                 use_edge_type: bool = True):

        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.use_edge_type = use_edge_type

        self.node_encoders = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else self.hidden_dim
            out_dim = self.hidden_dim
            self.node_encoders.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim)
            ))

        if use_edge_type:
            self.type1_aggregator = nn.Linear(self.hidden_dim, embed_dim)
            self.type2_aggregator = nn.Linear(self.hidden_dim, embed_dim)
        else:
            self.aggregator = nn.Linear(self.hidden_dim, embed_dim)

    def aggregate_nodes(self,
                       node_features: torch.Tensor,
                       node_indices: List[int]) -> torch.Tensor:

        patch_nodes = node_features[node_indices]

        if self.aggregation == 'mean':
            patch_embedding = patch_nodes.mean(dim=0)
        elif self.aggregation == 'max':
            patch_embedding = patch_nodes.max(dim=0)[0]
        elif self.aggregation == 'sum':
            patch_embedding = patch_nodes.sum(dim=0)
        else:
            raise ValueError(f"不支持的聚合方式: {self.aggregation}")

        return patch_embedding

    def forward(self,
                column_embeddings: torch.Tensor,
                H: torch.Tensor,
                num_type1_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:

        num_columns, num_hyperedges = H.shape
        num_type2_edges = num_hyperedges - num_type1_edges

        node_features = column_embeddings
        for encoder in self.node_encoders:
            node_features = encoder(node_features)

        patch_embeddings = torch.mm(H.t(), node_features)
        patch_sizes = H.sum(dim=0, keepdim=True).t()
        patch_sizes = torch.clamp(patch_sizes, min=1.0)
        patch_embeddings = patch_embeddings / patch_sizes

        patch_embeddings = torch.mm(H.t(), node_features)
        patch_sizes = H.sum(dim=0, keepdim=True).t()
        patch_sizes = torch.clamp(patch_sizes, min=1.0)
        patch_embeddings = patch_embeddings / patch_sizes

        patch_embeddings_type1_raw = patch_embeddings[:num_type1_edges]
        patch_embeddings_type2_raw = patch_embeddings[num_type1_edges:]

        if self.use_edge_type:
            patch_embeddings_type1 = self.type1_aggregator(patch_embeddings_type1_raw)
            patch_embeddings_type2 = self.type2_aggregator(patch_embeddings_type2_raw)
        else:
            patch_embeddings_type1 = self.aggregator(patch_embeddings_type1_raw)
            patch_embeddings_type2 = self.aggregator(patch_embeddings_type2_raw)

        return patch_embeddings_type1, patch_embeddings_type2


class HypergraphAwareMixer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_patches: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_structure_bias: bool = True):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.mlp_ratio = mlp_ratio
        self.use_structure_bias = use_structure_bias

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        if use_structure_bias:
            self.num_heads = 8
            self.head_dim = embed_dim // self.num_heads
            assert embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

            self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

            self.structure_weight = nn.Parameter(torch.tensor(0.5))

    def compute_patch_adjacency(self, H: torch.Tensor) -> torch.Tensor:

        A_patch = torch.mm(H.t(), H)  # [num_hyperedges, num_hyperedges]

        A_patch = A_patch - torch.diag(torch.diag(A_patch))

        max_val = A_patch.max()
        if max_val > 0:
            A_patch = A_patch / max_val

        return A_patch

    def structure_aware_attention(self,
                                 patch_embeddings: torch.Tensor,
                                 adjacency_matrix: torch.Tensor) -> torch.Tensor:

        num_patches = patch_embeddings.shape[0]

        qkv = self.qkv_proj(patch_embeddings)
        qkv = qkv.reshape(num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        structure_bias = adjacency_matrix.unsqueeze(0)
        attn_scores = attn_scores + self.structure_weight * structure_bias

        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = torch.matmul(attn_weights, V)

        attended = attended.permute(1, 0, 2).contiguous()
        attended = attended.reshape(num_patches, self.embed_dim)

        attended_embeddings = self.out_proj(attended)

        return attended_embeddings

    def forward(self,
                patch_embeddings: torch.Tensor,
                H: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_structure_bias:
            if adjacency_matrix is None:
                if H is None:
                    raise ValueError("Must provide H or adjacency_matrix")
                adjacency_matrix = self.compute_patch_adjacency(H)

            token_mixed = self.structure_aware_attention(patch_embeddings, adjacency_matrix)
            output = patch_embeddings + token_mixed
        else:
            token_mixed = self.token_mixer(patch_embeddings.t()).t()
            output = patch_embeddings + token_mixed

        channel_mixed = self.channel_mixer(output)
        output = output + channel_mixed

        return output


def build_column_adjacency_matrix(joinable_pairs: List[Tuple[int, int]],
                                 num_columns: int) -> torch.Tensor:

    adjacency_matrix = torch.zeros(num_columns, num_columns)

    for i, j in joinable_pairs:
        adjacency_matrix[i, j] = 1.0
        adjacency_matrix[j, i] = 1.0 

    return adjacency_matrix


def visualize_position_encoding(column_pe: torch.Tensor,
                                column_names: List[str],
                                save_path: str = "column_pe_visualization.png"):

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        pe_np = column_pe.cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.scatter(pe_np[:, 0], pe_np[:, 1], alpha=0.6)

        for i, name in enumerate(column_names):
            plt.annotate(name, (pe_np[i, 0], pe_np[i, 1]),
                        fontsize=8, alpha=0.7)

        plt.xlabel("Position Encoding Dim 1")
        plt.ylabel("Position Encoding Dim 2")
        plt.title("Column-level Positional Encoding Visualization")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Position encoding saved to: {save_path}")


class IntraHyperedgeGNNBatched(nn.Module):

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

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GNNLayer(embed_dim))

        if use_edge_type:
            self.type1_output = nn.Linear(embed_dim, embed_dim)
            self.type2_output = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                column_embeddings: torch.Tensor,
                H: torch.Tensor,
                num_type1_edges: int) -> Tuple[torch.Tensor, torch.Tensor]:

        num_columns, num_hyperedges = H.shape
        device = column_embeddings.device

        edge_index_list = []
        for j in range(num_hyperedges):
            node_indices = torch.where(H[:, j] > 0)[0]
            hyperedge_size = node_indices.size(0)

            if hyperedge_size <= 1:
                continue

            for i in range(hyperedge_size):
                for k in range(hyperedge_size):
                    if i != k:
                        edge_index_list.append([node_indices[i].item(), node_indices[k].item()])

        if len(edge_index_list) == 0:
            hyperedge_embeddings = torch.mm(H.t(), column_embeddings)
            patch_sizes = H.sum(dim=0, keepdim=True).t()
            hyperedge_embeddings = hyperedge_embeddings / torch.clamp(patch_sizes, min=1.0)
        else:
            edge_index = torch.tensor(edge_index_list, device=device).t()  # [2, num_edges]
            adj_matrix = torch.zeros(num_columns, num_columns, device=device)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
            adj_matrix.fill_diagonal_(1.0)

            degree = adj_matrix.sum(dim=1)  # [num_columns]
            degree = torch.clamp(degree, min=1.0)
            degree_inv_sqrt = torch.pow(degree, -0.5)  # D^{-1/2}
            adj_matrix_norm = degree_inv_sqrt.view(-1, 1) * adj_matrix * degree_inv_sqrt.view(1, -1)

            h = column_embeddings
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, adj_matrix_norm)

            hyperedge_embeddings = torch.mm(H.t(), h)  # [num_hyperedges, embed_dim]
            patch_sizes = H.sum(dim=0, keepdim=True).t()
            hyperedge_embeddings = hyperedge_embeddings / torch.clamp(patch_sizes, min=1.0)

        type1_embeddings = hyperedge_embeddings[:num_type1_edges]
        type2_embeddings = hyperedge_embeddings[num_type1_edges:]

        if self.use_edge_type:
            type1_embeddings = self.type1_output(type1_embeddings)
            type2_embeddings = self.type2_output(type2_embeddings)
        else:
            type1_embeddings = self.output_proj(type1_embeddings)
            type2_embeddings = self.output_proj(type2_embeddings)

        return type1_embeddings, type2_embeddings


class EnhancedHyperJoinModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 content_dim: int,
                 num_tables: int,
                 embed_dim: int = 512,
                 column_pe_dim: int = 16,
                 num_patches: int = 100,
                 patch_gnn_layers: int = 2,
                 num_mixer_layers: int = 2,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_table_pe: bool = True,
                 use_column_pe: bool = True,
                 use_structure_bias: bool = True,
                 use_edge_type: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tables = num_tables
        self.column_pe_dim = column_pe_dim
        self.num_patches = num_patches
        self.use_edge_type = use_edge_type

        # Initial encoders
        self.table_encoder = TextEncoder(vocab_size, embed_dim)
        self.column_encoder = TextEncoder(vocab_size, embed_dim)
        self.content_encoder = ContentEncoder(content_dim, embed_dim)

        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.final_projection = nn.Linear(embed_dim, embed_dim)

        # Three-level positional encoding
        self.position_encoding = ThreeLevelPositionalEncoding(
            embed_dim=embed_dim,
            num_tables=num_tables,
            column_pe_dim=column_pe_dim,
            use_table_pe=use_table_pe,
            use_column_pe=use_column_pe
        )

        # Intra-hyperedge GNN encoder
        self.intra_hyperedge_gnn = IntraHyperedgeGNNBatched(
            embed_dim=embed_dim,
            num_layers=patch_gnn_layers,
            aggregation='mean',
            use_edge_type=use_edge_type
        )

        # Hypergraph-aware Mixer layers
        self.mixer_layers = nn.ModuleList([
            HypergraphAwareMixer(
                embed_dim=embed_dim,
                num_patches=num_patches,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_structure_bias=use_structure_bias
            )
            for _ in range(num_mixer_layers)
        ])

        # Patch to column transformation
        if use_edge_type:
            self.patch_to_column_type1 = nn.Linear(embed_dim, embed_dim)
            self.patch_to_column_type2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.patch_to_column = nn.Linear(embed_dim, embed_dim)

        # Output layer
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                table_ids: torch.Tensor,
                column_ids: torch.Tensor,
                content_features: torch.Tensor,
                table_labels: torch.Tensor,
                hypergraph_incidence: torch.Tensor,
                num_type1_edges: int,
                column_pe: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size = table_ids.size(0)
        num_hyperedges = hypergraph_incidence.size(1)

        # Step 1: Encode with three encoders
        table_emb = self.table_encoder(table_ids)
        column_emb = self.column_encoder(column_ids)
        content_emb = self.content_encoder(content_features)

        # Step 2: Weighted fusion
        w_table, w_column, w_content = F.softmax(self.fusion_weights, dim=0)
        fused_embeddings = (
            w_table * table_emb +
            w_column * column_emb +
            w_content * content_emb
        )

        initial_embeddings = self.final_projection(fused_embeddings)

        # Step 3: Inject three-level positional encoding
        enhanced_embeddings = self.position_encoding(
            column_embeddings=initial_embeddings,
            table_ids=table_labels,
            column_pe=column_pe,
            adjacency_matrix=adjacency_matrix
        )

        # Step 4: Intra-hyperedge GNN message passing
        if len(self.mixer_layers) > 0:
            # With PE Mixer architecture
            patch_embeddings_type1, patch_embeddings_type2 = self.intra_hyperedge_gnn(
                column_embeddings=enhanced_embeddings,
                H=hypergraph_incidence,
                num_type1_edges=num_type1_edges
            )

            patch_embeddings = torch.cat([patch_embeddings_type1, patch_embeddings_type2], dim=0)

            # Step 5: Hypergraph-aware Mixer layers
            for mixer_layer in self.mixer_layers:
                patch_embeddings = mixer_layer(
                    patch_embeddings=patch_embeddings,
                    H=hypergraph_incidence,
                    adjacency_matrix=None
                )
        else:
            # Without HIN: traditional hypergraph message passing
            patch_embeddings = torch.mm(hypergraph_incidence.t(), enhanced_embeddings)

            edge_sizes = hypergraph_incidence.sum(dim=0, keepdim=True).t()
            edge_sizes = torch.clamp(edge_sizes, min=1.0)
            patch_embeddings = patch_embeddings / edge_sizes

            patch_embeddings_type1 = patch_embeddings[:num_type1_edges]
            patch_embeddings_type2 = patch_embeddings[num_type1_edges:]

        # Step 6: Propagate back to column level
        patch_embeddings_type1_enhanced = patch_embeddings[:num_type1_edges]
        patch_embeddings_type2_enhanced = patch_embeddings[num_type1_edges:]

        if self.use_edge_type:
            patch_embeddings_type1_transformed = self.patch_to_column_type1(patch_embeddings_type1_enhanced)
            patch_embeddings_type2_transformed = self.patch_to_column_type2(patch_embeddings_type2_enhanced)

            patch_embeddings_transformed = torch.cat([
                patch_embeddings_type1_transformed,
                patch_embeddings_type2_transformed
            ], dim=0)
        else:
            patch_embeddings_transformed = self.patch_to_column(patch_embeddings)

        column_from_patches = torch.mm(hypergraph_incidence, patch_embeddings_transformed)

        column_hyperedge_counts = hypergraph_incidence.sum(dim=1, keepdim=True)
        column_hyperedge_counts = torch.clamp(column_hyperedge_counts, min=1.0)
        column_from_patches = column_from_patches / column_hyperedge_counts

        # Residual connection
        final_embeddings = enhanced_embeddings + column_from_patches

        # Output normalization
        final_embeddings = self.output_norm(final_embeddings)

        # L2 normalization for cosine similarity
        column_embeddings = F.normalize(final_embeddings, p=2, dim=1)

        return column_embeddings

    def encode_without_hypergraph(self,
                                   table_ids: torch.Tensor,
                                   column_ids: torch.Tensor,
                                   content_features: torch.Tensor,
                                   table_labels: torch.Tensor,
                                   column_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference encoding without hypergraph structure."""
        batch_size = table_ids.size(0)

        # Step 1: Encode
        table_emb = self.table_encoder(table_ids)
        column_emb = self.column_encoder(column_ids)
        content_emb = self.content_encoder(content_features)

        # Step 2: Fusion
        w_table, w_column, w_content = F.softmax(self.fusion_weights, dim=0)
        fused_embeddings = (
            w_table * table_emb +
            w_column * column_emb +
            w_content * content_emb
        )

        initial_embeddings = self.final_projection(fused_embeddings)

        # Step 3: Inject positional encoding
        enhanced_embeddings = initial_embeddings

        if self.position_encoding.use_table_pe:
            table_pe = self.position_encoding.table_embeddings(table_labels)
            enhanced_embeddings = enhanced_embeddings + self.position_encoding.alpha * table_pe

        if self.position_encoding.use_column_pe and column_pe is not None:
            column_pe_encoded = self.position_encoding.column_pe_encoder(column_pe)
            enhanced_embeddings = enhanced_embeddings + self.position_encoding.beta * column_pe_encoded

        # Step 4: Normalization
        enhanced_embeddings = self.output_norm(enhanced_embeddings)

        return F.normalize(enhanced_embeddings, p=2, dim=1)

    def get_fusion_weights(self) -> torch.Tensor:
        """Return current fusion weights."""
        return F.softmax(self.fusion_weights, dim=0)


def precompute_column_pe(joinable_pairs: List[Tuple[int, int]],
                        num_columns: int,
                        pe_dim: int = 16,
                        device: str = 'cpu') -> torch.Tensor:
    """Precompute column-level positional encoding."""
    adjacency_matrix = torch.zeros(num_columns, num_columns)
    for i, j in joinable_pairs:
        if i < num_columns and j < num_columns:
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0

    temp_pe_module = ThreeLevelPositionalEncoding(
        embed_dim=512,
        num_tables=1,
        column_pe_dim=pe_dim,
        use_table_pe=False,
        use_column_pe=True
    )

    column_pe = temp_pe_module.compute_column_pe(adjacency_matrix, k=pe_dim)

    return column_pe.to(device)