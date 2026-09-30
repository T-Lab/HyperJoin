"""
Enhanced HyperJoin Model
========================

Enhanced model integrating the Graph ViT / MLP-Mixer ideas

Architecture:
Initial Encoders → Fusion →
Three-level PE -> Patch GNN -> Hypergraph-aware Mixer ->
Column propagation -> Enhanced embeddings

Authors: Claude + liushiyuan
Date: 2025-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original modules
from hypergraph.construction import (
    TextEncoder,
    ContentEncoder
)

# Import the new enhancement modules
from hypergraph.layers import (
    ThreeLevelPositionalEncoding,
    HypergraphAwareMixer,
)

# Import the true intra-hyperedge GNN module
from hypergraph.intra_edge_gnn import IntraHyperedgeGNNBatched


class PairwiseGCNLayer(nn.Module):
    """GCN on collapsed pairwise graph A = H @ H^T (ablation baseline).

    Converts the hypergraph incidence matrix H into a pairwise adjacency
    matrix A = H H^T, applies symmetric normalisation, and runs a single
    GCN propagation step with residual connection.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] node features
            H: [N, E] hypergraph incidence matrix
        Returns:
            out: [N, D] updated node features
        """
        # Collapse hypergraph to pairwise adjacency: A = H @ H^T
        A = torch.mm(H, H.t())  # [N, N]

        # Remove self-loops and add identity
        A = A * (1 - torch.eye(A.size(0), device=A.device))  # zero diagonal
        A = A + torch.eye(A.size(0), device=A.device)         # add I

        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=1).clamp(min=1.0)       # [N]
        deg_inv_sqrt = deg.pow(-0.5)              # [N]
        A_norm = A * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)

        # GCN propagation: A_norm @ x -> Linear -> ReLU -> Dropout -> LayerNorm
        h = torch.mm(A_norm, x)         # [N, D]
        h = self.linear(h)              # [N, D]
        h = F.relu(h)
        h = self.dropout(h)
        h = self.norm(h)

        return h + x  # residual


class EnhancedHyperJoinModel(nn.Module):
    """
    Enhanced HyperJoin model.

    Three major enhancements:
    1. Three-level positional encoding (Table PE + Column PE)
    2. Patch GNN encoder (local aggregation inside hyperedges)
    3. Hypergraph-aware Mixer (global interaction across hyperedges)

    Flow:
    1. Encode initial features with the same encoders as the baseline
    2. Fuse them into the initial column embeddings
    3. [new] Inject three-level positional encoding
    4. [new] Patch GNN: aggregate to the hyperedge level
    5. [new] Hypergraph-aware Mixer: global interaction across hyperedges
    6. [new] Propagate back: distribute the enhanced patch info to columns
    7. Output the enhanced column embeddings
    """

    def __init__(self,
                 vocab_size: int,
                 content_dim: int,
                 num_tables: int,
                 embed_dim: int = 512,
                 column_pe_dim: int = 16,
                 num_patches: int = 100,  # estimated hyperedge count
                 patch_gnn_layers: int = 2,
                 num_mixer_layers: int = 2,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_table_pe: bool = True,
                 use_column_pe: bool = True,
                 use_structure_bias: bool = True,
                 use_edge_type: bool = True,
                 pairwise_mode: bool = False):
        """
        Initialise the enhanced HyperJoin model.

        Args:
            vocab_size: vocabulary size
            content_dim: content-feature dimension
            num_tables: number of tables in the dataset
            embed_dim: embedding dimension
            column_pe_dim: dimension of the Column-level PE
            num_patches: estimated number of patches (hyperedges)
            patch_gnn_layers: number of Patch GNN layers
            num_mixer_layers: number of Mixer layers
            mlp_ratio: expansion ratio of the MLP inside the Mixer
            dropout: dropout rate
            use_table_pe: whether to use Table-level PE
            use_column_pe: whether to use Column-level PE
            use_structure_bias: whether to use the hypergraph structural prior
            use_edge_type: whether to distinguish Type-1 and Type-2 hyperedges
            pairwise_mode: ablation - replace the hypergraph with a GCN on H@H^T
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_tables = num_tables
        self.column_pe_dim = column_pe_dim
        self.num_patches = num_patches
        self.use_edge_type = use_edge_type
        self.pairwise_mode = pairwise_mode

        # ========================================
        # 1. Initial encoders (identical to the baseline)
        # ========================================
        self.table_encoder = TextEncoder(vocab_size, embed_dim)
        self.column_encoder = TextEncoder(vocab_size, embed_dim)
        self.content_encoder = ContentEncoder(content_dim, embed_dim)

        # ========================================
        # 2. Fusion weights (identical to the baseline)
        # ========================================
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.final_projection = nn.Linear(embed_dim, embed_dim)

        # ========================================
        # 3. [new] Three-level positional encoding module
        # ========================================
        self.position_encoding = ThreeLevelPositionalEncoding(
            embed_dim=embed_dim,
            num_tables=num_tables,
            column_pe_dim=column_pe_dim,
            use_table_pe=use_table_pe,
            use_column_pe=use_column_pe
        )

        # ========================================
        # 4. [new] Intra-hyperedge GNN encoder (real GNN message passing)
        # ========================================
        # Local GNN message passing inside each hyperedge, then aggregate to edge level
        self.intra_hyperedge_gnn = IntraHyperedgeGNNBatched(
            embed_dim=embed_dim,
            num_layers=patch_gnn_layers,
            aggregation='mean',
            use_edge_type=use_edge_type
        )

        # ========================================
        # 5. [new] Hypergraph-aware Mixer layers (stacked)
        # ========================================
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

        # ========================================
        # 6. Transformation for propagating back to the column level
        # ========================================
        # With edge typing, the two types get separate transforms
        if use_edge_type:
            self.patch_to_column_type1 = nn.Linear(embed_dim, embed_dim)
            self.patch_to_column_type2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.patch_to_column = nn.Linear(embed_dim, embed_dim)

        # ========================================
        # 7. Output layer
        # ========================================
        self.output_norm = nn.LayerNorm(embed_dim)

        # ========================================
        # 8. [ablation] Pairwise GCN (enabled only in pairwise_mode)
        # ========================================
        if self.pairwise_mode:
            self.pairwise_gcn = PairwiseGCNLayer(embed_dim, dropout)

    def forward(self,
                table_ids: torch.Tensor,
                column_ids: torch.Tensor,
                content_features: torch.Tensor,
                table_labels: torch.Tensor,
                hypergraph_incidence: torch.Tensor,
                num_type1_edges: int,
                column_pe: Optional[torch.Tensor] = None,
                adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            table_ids: [batch_size, seq_len] table-name token ids
            column_ids: [batch_size, seq_len] column-name token ids
            content_features: [batch_size, content_dim] content features
            table_labels: [batch_size] table ID of each column
            hypergraph_incidence: [batch_size, num_hyperedges] hypergraph incidence matrix
            num_type1_edges: number of Type-1 hyperedges
            column_pe: [batch_size, column_pe_dim] precomputed Column PE (optional)
            adjacency_matrix: [batch_size, batch_size] column adjacency (for the Column PE)

        Returns:
            column_embeddings: [batch_size, embed_dim] enhanced column embeddings
        """
        batch_size = table_ids.size(0)
        num_hyperedges = hypergraph_incidence.size(1)

        # ========================================
        # Step 1: encode with the three encoders (identical to the baseline)
        # ========================================
        table_emb = self.table_encoder(table_ids)       # [batch_size, embed_dim]
        column_emb = self.column_encoder(column_ids)    # [batch_size, embed_dim]
        content_emb = self.content_encoder(content_features)  # [batch_size, embed_dim]

        # ========================================
        # Step 2: weighted fusion (identical to the baseline)
        # ========================================
        w_table, w_column, w_content = F.softmax(self.fusion_weights, dim=0)
        fused_embeddings = (
            w_table * table_emb +
            w_column * column_emb +
            w_content * content_emb
        )  # [batch_size, embed_dim]

        # Projection
        initial_embeddings = self.final_projection(fused_embeddings)  # [batch_size, embed_dim]

        # ========================================
        # Step 3: [new] inject three-level positional encoding
        # ========================================
        # Table-level PE + Column-level PE
        enhanced_embeddings = self.position_encoding(
            column_embeddings=initial_embeddings,
            table_ids=table_labels,
            column_pe=column_pe,
            adjacency_matrix=adjacency_matrix
        )  # [batch_size, embed_dim]

        # ========================================
        # [ablation] Pairwise GCN - skip Steps 4-6, run GCN on A = H@H^T
        # ========================================
        if self.pairwise_mode:
            gcn_out = self.pairwise_gcn(enhanced_embeddings, hypergraph_incidence)
            gcn_out = self.output_norm(gcn_out)
            return F.normalize(gcn_out, p=2, dim=1)

        # ========================================
        # Step 4: [new] intra-hyperedge GNN message passing
        # ========================================
        # Check whether the PE Mixer is used (via the layer count)
        if len(self.mixer_layers) > 0:
            # PE-Mixer architecture:
            # local GNN message passing inside each hyperedge, then aggregate to edge level
            patch_embeddings_type1, patch_embeddings_type2 = self.intra_hyperedge_gnn(
                column_embeddings=enhanced_embeddings,
                H=hypergraph_incidence,
                num_type1_edges=num_type1_edges
            )
            # patch_embeddings_type1: [num_type1_edges, embed_dim]
            # patch_embeddings_type2: [num_type2_edges, embed_dim]

            # Concatenate the two patch-embedding types
            patch_embeddings = torch.cat([patch_embeddings_type1, patch_embeddings_type2], dim=0)
            # [num_hyperedges, embed_dim]

            # ========================================
            # Step 5: [new] hypergraph-aware Mixer layers
            # ========================================
            # Global information exchange across patches
            for mixer_layer in self.mixer_layers:
                patch_embeddings = mixer_layer(
                    patch_embeddings=patch_embeddings,
                    H=hypergraph_incidence,
                    adjacency_matrix=None  # computed internally
                )
            # patch_embeddings: [num_hyperedges, embed_dim]
        else:
            # w/o HIN: plain hypergraph message passing (simple aggregation)
            # H^T @ X: aggregate column embeddings to hyperedges
            patch_embeddings = torch.mm(hypergraph_incidence.t(), enhanced_embeddings)
            # [num_hyperedges, embed_dim]

            # Normalise by the number of columns in each hyperedge
            edge_sizes = hypergraph_incidence.sum(dim=0, keepdim=True).t()  # [num_hyperedges, 1]
            edge_sizes = torch.clamp(edge_sizes, min=1.0)
            patch_embeddings = patch_embeddings / edge_sizes

            # Split Type 1 / Type 2 (for later processing)
            patch_embeddings_type1 = patch_embeddings[:num_type1_edges]
            patch_embeddings_type2 = patch_embeddings[num_type1_edges:]

        # ========================================
        # Step 6: [new] propagate back to the column level
        # ========================================
        # Distribute the enhanced patch information back to the columns

        # Split Type-1 and Type-2 patches
        patch_embeddings_type1_enhanced = patch_embeddings[:num_type1_edges]
        patch_embeddings_type2_enhanced = patch_embeddings[num_type1_edges:]

        if self.use_edge_type:
            # Type-specific transforms
            patch_embeddings_type1_transformed = self.patch_to_column_type1(patch_embeddings_type1_enhanced)
            patch_embeddings_type2_transformed = self.patch_to_column_type2(patch_embeddings_type2_enhanced)

            # Concatenate back
            patch_embeddings_transformed = torch.cat([
                patch_embeddings_type1_transformed,
                patch_embeddings_type2_transformed
            ], dim=0)
        else:
            # Shared transform
            patch_embeddings_transformed = self.patch_to_column(patch_embeddings)

        # Propagate via the incidence matrix: H @ patch_embeddings
        # H: [batch_size, num_hyperedges]
        # patch_embeddings_transformed: [num_hyperedges, embed_dim]
        column_from_patches = torch.mm(hypergraph_incidence, patch_embeddings_transformed)
        # [batch_size, embed_dim]

        # Normalise by the number of hyperedges each column joins
        column_hyperedge_counts = hypergraph_incidence.sum(dim=1, keepdim=True)  # [batch_size, 1]
        column_hyperedge_counts = torch.clamp(column_hyperedge_counts, min=1.0)
        column_from_patches = column_from_patches / column_hyperedge_counts

        # Residual connection with the initial embeddings
        final_embeddings = enhanced_embeddings + column_from_patches

        # ========================================
        # Step 7: output normalisation
        # ========================================
        final_embeddings = self.output_norm(final_embeddings)

        # L2 normalisation (for cosine similarity)
        column_embeddings = F.normalize(final_embeddings, p=2, dim=1)

        return column_embeddings

    def encode_without_hypergraph(self,
                                   table_ids: torch.Tensor,
                                   column_ids: torch.Tensor,
                                   content_features: torch.Tensor,
                                   table_labels: torch.Tensor,
                                   column_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference-time encoding (without the hypergraph).

        At inference we may not have the full hypergraph structure,
        so only the initial encoders + positional encoding are used.

        Args:
            table_ids: [batch_size, seq_len]
            column_ids: [batch_size, seq_len]
            content_features: [batch_size, content_dim]
            table_labels: [batch_size] table ID of each column
            column_pe: [batch_size, column_pe_dim] precomputed Column PE (optional)

        Returns:
            column_embeddings: [batch_size, embed_dim]
        """
        batch_size = table_ids.size(0)

        # Step 1: encode
        table_emb = self.table_encoder(table_ids)
        column_emb = self.column_encoder(column_ids)
        content_emb = self.content_encoder(content_features)

        # Step 2: fuse
        w_table, w_column, w_content = F.softmax(self.fusion_weights, dim=0)
        fused_embeddings = (
            w_table * table_emb +
            w_column * column_emb +
            w_content * content_emb
        )

        initial_embeddings = self.final_projection(fused_embeddings)

        # Step 3: inject the positional encoding (Table PE + Column PE)
        enhanced_embeddings = initial_embeddings

        # Add the Table PE
        if self.position_encoding.use_table_pe:
            table_pe = self.position_encoding.table_embeddings(table_labels)
            enhanced_embeddings = enhanced_embeddings + self.position_encoding.alpha * table_pe

        # Add the Column PE (when a precomputed column_pe is provided)
        if self.position_encoding.use_column_pe and column_pe is not None:
            column_pe_encoded = self.position_encoding.column_pe_encoder(column_pe)
            enhanced_embeddings = enhanced_embeddings + self.position_encoding.beta * column_pe_encoded

        # Step 4: normalisation
        enhanced_embeddings = self.output_norm(enhanced_embeddings)

        # L2 normalisation
        return F.normalize(enhanced_embeddings, p=2, dim=1)

    def get_fusion_weights(self) -> torch.Tensor:
        """Return the current fusion weights."""
        return F.softmax(self.fusion_weights, dim=0)


# ============================================
# Helper: precompute the Column-level PE
# ============================================

def precompute_column_pe(joinable_pairs: List[Tuple[int, int]],
                        num_columns: int,
                        pe_dim: int = 16,
                        device: str = 'cpu') -> torch.Tensor:
    """
    Precompute the Column-level positional encoding for all columns.

    Call once during data preprocessing and persist the result;
    load it directly during training/inference to avoid recomputation.

    Args:
        joinable_pairs: [(col_i, col_j), ...] joinable column pairs
        num_columns: total number of columns
        pe_dim: dimension of the positional encoding
        device: device

    Returns:
        column_pe: [num_columns, pe_dim] positional encoding for all columns
    """
    # Build the column adjacency matrix
    adjacency_matrix = torch.zeros(num_columns, num_columns)
    for i, j in joinable_pairs:
        if i < num_columns and j < num_columns:
            adjacency_matrix[i, j] = 1.0
            adjacency_matrix[j, i] = 1.0

    # Use a temporary PE module to compute the encoding
    temp_pe_module = ThreeLevelPositionalEncoding(
        embed_dim=512,  # arbitrary; does not affect the column-PE computation
        num_tables=1,
        column_pe_dim=pe_dim,
        use_table_pe=False,
        use_column_pe=True
    )

    # Compute the Column PE
    column_pe = temp_pe_module.compute_column_pe(adjacency_matrix, k=pe_dim)

    return column_pe.to(device)


# ============================================
# Training helper
# ============================================

def train_step(model: EnhancedHyperJoinModel,
               batch: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer,
               device: str = 'cuda') -> float:
    """
    Single training step.

    Args:
        model: the model
        batch: dict with the following keys:
            - 'table_ids': [batch_size, seq_len]
            - 'column_ids': [batch_size, seq_len]
            - 'content_features': [batch_size, content_dim]
            - 'table_labels': [batch_size]
            - 'hypergraph_incidence': [batch_size, num_hyperedges]
            - 'num_type1_edges': int
            - 'column_pe': [batch_size, column_pe_dim] (optional)
            - 'positive_indices': [batch_size] positive-sample indices
            - 'negative_indices': [batch_size, num_neg] negative-sample indices
        optimizer: the optimizer
        device: device

    Returns:
        loss: the loss value
    """
    model.train()

    # Move tensors to device
    table_ids = batch['table_ids'].to(device)
    column_ids = batch['column_ids'].to(device)
    content_features = batch['content_features'].to(device)
    table_labels = batch['table_labels'].to(device)
    hypergraph_incidence = batch['hypergraph_incidence'].to(device)
    num_type1_edges = batch['num_type1_edges']

    column_pe = batch.get('column_pe', None)
    if column_pe is not None:
        column_pe = column_pe.to(device)

    # Forward pass
    column_embeddings = model(
        table_ids=table_ids,
        column_ids=column_ids,
        content_features=content_features,
        table_labels=table_labels,
        hypergraph_incidence=hypergraph_incidence,
        num_type1_edges=num_type1_edges,
        column_pe=column_pe
    )

    # Contrastive loss
    positive_indices = batch['positive_indices'].to(device)
    negative_indices = batch['negative_indices'].to(device)

    # Extract positive/negative embeddings
    query_emb = column_embeddings  # [batch_size, embed_dim]
    positive_emb = column_embeddings[positive_indices]  # [batch_size, embed_dim]

    # Compute similarities
    pos_sim = torch.sum(query_emb * positive_emb, dim=1)  # [batch_size]

    # Negative similarities
    neg_emb = column_embeddings[negative_indices]  # [batch_size, num_neg, embed_dim]
    neg_sim = torch.bmm(
        query_emb.unsqueeze(1),  # [batch_size, 1, embed_dim]
        neg_emb.transpose(1, 2)   # [batch_size, embed_dim, num_neg]
    ).squeeze(1)  # [batch_size, num_neg]

    # InfoNCE loss
    temperature = 0.1
    logits = torch.cat([pos_sim.unsqueeze(1) / temperature, neg_sim / temperature], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================
# Evaluation helper
# ============================================

@torch.no_grad()
def evaluate(model: EnhancedHyperJoinModel,
            dataloader,
            device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: the model
        dataloader: data loader
        device: device

    Returns:
        metrics: dict of evaluation metrics
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        table_ids = batch['table_ids'].to(device)
        column_ids = batch['column_ids'].to(device)
        content_features = batch['content_features'].to(device)
        table_labels = batch['table_labels'].to(device)
        hypergraph_incidence = batch['hypergraph_incidence'].to(device)
        num_type1_edges = batch['num_type1_edges']

        column_pe = batch.get('column_pe', None)
        if column_pe is not None:
            column_pe = column_pe.to(device)

        # Forward pass
        column_embeddings = model(
            table_ids=table_ids,
            column_ids=column_ids,
            content_features=content_features,
            table_labels=table_labels,
            hypergraph_incidence=hypergraph_incidence,
            num_type1_edges=num_type1_edges,
            column_pe=column_pe
        )

        all_embeddings.append(column_embeddings.cpu())
        all_labels.append(batch['labels'])

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute the similarity matrix
    similarity_matrix = torch.mm(all_embeddings, all_embeddings.t())

    # Compute Recall@K
    metrics = {}
    for k in [1, 5, 10, 25]:
        recall_k = compute_recall_at_k(similarity_matrix, all_labels, k)
        metrics[f'Recall@{k}'] = recall_k

    return metrics


def compute_recall_at_k(similarity_matrix: torch.Tensor,
                       labels: torch.Tensor,
                       k: int) -> float:
    """
    Compute Recall@K.

    Args:
        similarity_matrix: [N, N] similarity matrix
        labels: [N] labels (joinable columns share a label)
        k: the K value

    Returns:
        recall: the Recall@K value
    """
    N = similarity_matrix.size(0)

    # For each query, take the top-K most similar columns
    _, top_k_indices = torch.topk(similarity_matrix, k=k+1, dim=1)  # +1 since the query itself is included

    # Remove the query itself
    top_k_indices = top_k_indices[:, 1:]  # [N, K]

    # Compute recall
    correct = 0
    total = 0

    for i in range(N):
        # All columns joinable with i
        joinable_cols = (labels == labels[i]).nonzero(as_tuple=True)[0]
        joinable_cols = joinable_cols[joinable_cols != i]  # drop the query itself

        if len(joinable_cols) == 0:
            continue

        # Check whether the top-K contains any joinable column
        retrieved = top_k_indices[i]
        hits = torch.isin(retrieved, joinable_cols).sum().item()

        if hits > 0:
            correct += 1
        total += 1

    recall = correct / total if total > 0 else 0.0
    return recall


# ============================================
# Visualisation helper
# ============================================

def visualize_embeddings(embeddings: torch.Tensor,
                        labels: torch.Tensor,
                        save_path: str = "embeddings_visualization.png"):
    """
    Visualise column embeddings with t-SNE.

    Args:
        embeddings: [N, embed_dim] column embeddings
        labels: [N] column labels
        save_path: where to save the figure
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # t-SNE down to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())

        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels.cpu().numpy(), cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Column Embeddings Visualization (t-SNE)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Embedding visualisation saved to: {save_path}")

    except ImportError:
        print("Warning: sklearn or matplotlib not installed; cannot visualise embeddings")
