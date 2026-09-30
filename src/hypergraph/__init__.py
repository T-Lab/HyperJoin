"""
Hypergraph Module - Enhanced Column-Level Hypergraph
====================================================

Enhanced column-level hypergraph with GNN for table joinability discovery
"""

# Import utility functions from construction
from .construction import (
    build_dual_hypergraph_from_pairs,
    build_batch_hypergraph,
    ContrastiveLoss
)

# Import enhanced model components
from .model import EnhancedHyperJoinModel
from .layers import ThreeLevelPositionalEncoding
from .intra_edge_gnn import IntraHyperedgeGNNBatched

__all__ = [
    # Utility functions
    'build_dual_hypergraph_from_pairs',
    'build_batch_hypergraph',
    'ContrastiveLoss',
    # Enhanced model
    'EnhancedHyperJoinModel',
    'ThreeLevelPositionalEncoding',
    'IntraHyperedgeGNNBatched',
]
