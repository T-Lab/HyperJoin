"""
MST Reranker Module for Joinable Table Discovery
=================================================

This module implements Maximum Spanning Tree (MST) based reranking
for joinable table discovery, inspired by OpenMEL's approach to
multimodal entity linking.

Core Idea:
---------
From Top-B candidates (e.g., B=100), select Top-K (e.g., K=25) that:
1. Have high similarity with the query (local relevance)
2. Are connected via joinable relationships (global coherence)

Components:
----------
- mst: Simplified MST for Label-Free (unsupervised) scenarios
- graph_builder: Build GT adjacency matrix for search
- coherence_eval: Extended evaluator with coherence metrics
"""

from .mst import LabelFreeMSTReranker
from .graph_builder import GraphBuilderSearch

__all__ = ['LabelFreeMSTReranker', 'GraphBuilderSearch']
