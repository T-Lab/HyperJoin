"""
Enhanced Column-Level Hypergraph Training Script
=================================================

HyperJoin main training entry point: trains ``EnhancedHyperJoinModel``
(integrating Graph ViT / MLP-Mixer with intra-hyperedge GNN).

Usage::

    python src/hypergraph/train.py \\
        --dataset CAN_ALL \\
        --loss_type triplet --margin 1.0 --hard_neg_ratio 1.0 \\
        --lr 4e-4 --epochs 30 --batch_size 64 \\
        --embed_dim 512 --patch_gnn_layers 2 --num_mixer_layers 2 \\
        --use_residual 1 --dropout 0.05 \\
        --output_dir <output-path>

Pipeline:
  1. Load target/query column vectors and metadata via ``TestDatasetHyperJoin``
  2. Build a shared table/column token vocabulary with ``build_vocabulary`` (from utils)
  3. Wrap positive/negative column pairs in ``ColumnHypergraphDataset``; in Label-Free
     mode pairs are read directly from ``self_supervised_pairs.pkl`` (no manual labels)
  4. ``train_epoch`` runs triplet / contrastive training with hard-negative mining
  5. The best checkpoint is saved to ``<output_dir>/best_model.pth`` for the search stage
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data import TestDatasetHyperJoin
from utils import (
    set_seed,
    get_gpu_memory_mb,
    get_cpu_memory_mb,
    build_vocabulary,
)
from hypergraph.construction import (
    build_dual_hypergraph_from_pairs,
    build_batch_hypergraph,
    ContrastiveLoss
)
# Use the enhanced model
from hypergraph.model import EnhancedHyperJoinModel


class ColumnHypergraphDataset(Dataset):
    """Column-level hypergraph training dataset."""

    def __init__(self,
                 target_dataset: TestDatasetHyperJoin,
                 joinable_pairs_path: str,
                 vocab: Dict[str, int],
                 table_name_to_id: Dict[str, int] = None,
                 negative_ratio: float = 1.0,
                 max_seq_len: int = 10):
        """
        Args:
            target_dataset: Target column dataset
            joinable_pairs_path: Path to the joinable-pairs file
            vocab: Token vocabulary
            table_name_to_id: table-name -> ID mapping (for Table PE)
            negative_ratio: Negative sampling ratio
            max_seq_len: Maximum token sequence length
        """
        self.target_dataset = target_dataset
        self.vocab = vocab
        self.table_name_to_id = table_name_to_id if table_name_to_id is not None else {}
        self.max_seq_len = max_seq_len
        self.negative_ratio = negative_ratio

        # Load metadata
        self.metadata = self._load_metadata()

        # Build the "table.column" -> index mapping
        self.table_col_to_idx = self._build_mapping()

        # Detect Label-Free mode
        self.is_label_free = joinable_pairs_path.endswith('.pkl')

        # Load joinable pairs
        pairs_result = self._load_joinable_pairs(joinable_pairs_path)

        if self.is_label_free:
            # Label-Free: returns (positive_pairs, negative_pairs)
            self.positive_pairs, self.negative_pairs = pairs_result
        else:
            # Supervised: returns positive_pairs; negatives are generated
            self.positive_pairs = pairs_result
            self.negative_pairs = self._generate_negative_pairs()

        # Merge all pairs
        self.all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.all_pairs)

        print(f" Dataset statistics:")
        print(f"   Positive pairs: {len(self.positive_pairs)}")
        print(f"   Negative pairs: {len(self.negative_pairs)}")
        print(f"   Total: {len(self.all_pairs)}")

    def _load_metadata(self):
        """Load metadata."""
        if hasattr(self.target_dataset, 'metadata'):
            metadata = self.target_dataset.metadata
            if isinstance(metadata, dict) and 'metadata' in metadata:
                return metadata['metadata']
            return metadata
        return []

    def _build_mapping(self) -> Dict[str, int]:
        """Build the "table.column" -> index mapping."""
        mapping = {}
        for idx, meta in enumerate(self.metadata):
            table_name = meta.get('table_name', '').replace('.csv', '')
            column_name = meta.get('column_name', '')
            key = f"{table_name}.{column_name}"
            mapping[key] = idx
        print(f" Built mapping for {len(mapping)} columns")
        return mapping

    def _load_joinable_pairs(self, path: str):
        """Load joinable pairs (supports CSV and PKL formats)."""
        # Detect the file format
        if path.endswith('.pkl'):
            # Label-Free mode: load self_supervised_pairs.pkl
            print(f"   Loading Label-Free pairs: {path}")
            with open(path, 'rb') as f:
                pairs_data = pickle.load(f)
                positive_pairs = pairs_data['positive']
                negative_pairs = pairs_data['negative']

            # Convert to the training format
            pairs = []

            # Process positive pairs
            for pair in positive_pairs:
                source_table = pair['table1_name'].replace('.csv', '')
                target_table = pair['table2_name'].replace('.csv', '')
                source_col = pair['col1_name']
                target_col = pair['col2_name']

                source_key = f"{source_table}.{source_col}"
                target_key = f"{target_table}.{target_col}"

                if source_key in self.table_col_to_idx and target_key in self.table_col_to_idx:
                    source_idx = self.table_col_to_idx[source_key]
                    target_idx = self.table_col_to_idx[target_key]
                    if source_idx != target_idx:
                        pairs.append((source_idx, target_idx, 1.0))

            # Process negative pairs
            negative_list = []
            for pair in negative_pairs:
                source_table = pair['table1_name'].replace('.csv', '')
                target_table = pair['table2_name'].replace('.csv', '')
                source_col = pair['col1_name']
                target_col = pair['col2_name']

                source_key = f"{source_table}.{source_col}"
                target_key = f"{target_table}.{target_col}"

                if source_key in self.table_col_to_idx and target_key in self.table_col_to_idx:
                    source_idx = self.table_col_to_idx[source_key]
                    target_idx = self.table_col_to_idx[target_key]
                    if source_idx != target_idx:
                        negative_list.append((source_idx, target_idx, 0.0))

            print(f"   Loaded Label-Free pairs: {len(pairs)} positive, {len(negative_list)} negative")
            return pairs, negative_list  # return a tuple

        else:
            # Supervised mode: load joinable_pairs.csv
            df = pd.read_csv(path)
            pairs = []

            for _, row in df.iterrows():
                source_table = row['query_table'].replace('.csv', '')
                target_table = row['candidate_table'].replace('.csv', '')
                source_col = row['query_column']
                target_col = row['candidate_column']

                source_key = f"{source_table}.{source_col}"
                target_key = f"{target_table}.{target_col}"

                if source_key in self.table_col_to_idx and target_key in self.table_col_to_idx:
                    source_idx = self.table_col_to_idx[source_key]
                    target_idx = self.table_col_to_idx[target_key]

                    if source_idx != target_idx:
                        pairs.append((source_idx, target_idx, 1.0))

            print(f" Loaded joinable pairs: {len(pairs)}")
            return pairs  # positives only

    def _generate_negative_pairs(self) -> List[Tuple[int, int, float]]:
        """Generate negative pairs."""
        num_neg = int(len(self.positive_pairs) * self.negative_ratio)
        negative_pairs = []

        positive_set = {(s, t) for s, t, _ in self.positive_pairs}
        num_columns = len(self.target_dataset)

        attempts = 0
        max_attempts = num_neg * 10

        while len(negative_pairs) < num_neg and attempts < max_attempts:
            source_idx = random.randint(0, num_columns - 1)
            target_idx = random.randint(0, num_columns - 1)

            if source_idx != target_idx and (source_idx, target_idx) not in positive_set:
                negative_pairs.append((source_idx, target_idx, 0.0))

            attempts += 1

        print(f" Generated negative pairs: {len(negative_pairs)}")
        return negative_pairs

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text."""
        tokens = text.replace('.csv', '').split('_')
        ids = []
        for tok in tokens[:self.max_seq_len]:
            ids.append(self.vocab.get(tok, self.vocab.get('<UNK>', 1)))
        while len(ids) < self.max_seq_len:
            ids.append(self.vocab.get('<PAD>', 0))
        return torch.tensor(ids[:self.max_seq_len], dtype=torch.long)

    def _aggregate_content(self, col_data):
        """Aggregate column content features."""
        if isinstance(col_data, torch.Tensor):
            data = col_data
        else:
            data = torch.tensor(col_data, dtype=torch.float32)

        if data.numel() == 0:
            return torch.zeros(1200)

        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Handle single-row columns (torch.std returns NaN for n=1)
        if data.shape[0] == 1:
            # std is undefined for a single row; fill with zeros
            mean_f = data[0]  # (300,)
            std_f = torch.zeros_like(mean_f)  # zeros to avoid NaN
            max_f = data[0]
            min_f = data[0]
            return torch.cat([mean_f, std_f, max_f, min_f], dim=0)

        if data.shape[0] > 0 and data.shape[1] > 0:
            mean_f = torch.mean(data, dim=0)
            std_f = torch.std(data, dim=0)
            max_f = torch.max(data, dim=0)[0]
            min_f = torch.min(data, dim=0)[0]
            return torch.cat([mean_f, std_f, max_f, min_f], dim=0)
        else:
            return torch.zeros(1200)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        source_idx, target_idx, label = self.all_pairs[idx]

        # Source column
        source_meta = self.metadata[source_idx]
        source_table_ids = self._tokenize(source_meta.get('table_name', ''))
        source_column_ids = self._tokenize(source_meta.get('column_name', ''))
        source_content = self._aggregate_content(self.target_dataset[source_idx])
        source_table_name = source_meta.get('table_name', '').replace('.csv', '')
        source_table_label = self.table_name_to_id.get(source_table_name, 0)

        # Target column
        target_meta = self.metadata[target_idx]
        target_table_ids = self._tokenize(target_meta.get('table_name', ''))
        target_column_ids = self._tokenize(target_meta.get('column_name', ''))
        target_content = self._aggregate_content(self.target_dataset[target_idx])
        target_table_name = target_meta.get('table_name', '').replace('.csv', '')
        target_table_label = self.table_name_to_id.get(target_table_name, 0)

        return {
            'source_table_ids': source_table_ids,
            'source_column_ids': source_column_ids,
            'source_content': source_content,
            'source_table_label': torch.tensor(source_table_label, dtype=torch.long),
            'target_table_ids': target_table_ids,
            'target_column_ids': target_column_ids,
            'target_content': target_content,
            'target_table_label': torch.tensor(target_table_label, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'source_idx': source_idx,
            'target_idx': target_idx
        }


def train_epoch(model, train_loader, criterion, optimizer, device,
                global_hypergraph=None, num_type1_edges=None, global_column_pe=None,
                hard_negatives=True, hard_neg_ratio=1.0, hard_topk=5, margin=0.5, loss_type='triplet',
                infonce_temperature=0.1):
    """Train for one epoch.

    Args:
        global_column_pe: [num_columns, column_pe_dim] globally precomputed Column PE
        hard_negatives: Whether to enable hard-negative mining
        hard_neg_ratio: Fraction of negatives drawn from hard mining [0, 1]
        hard_topk: Sample uniformly from the top-k most similar negatives
        margin: Triplet-loss margin
        loss_type: Loss type ('triplet' or 'infonce')
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')

    # Hypergraph stats (printed for the first batch only)
    hypergraph_debug_printed = False

    for batch_idx, batch in enumerate(progress_bar):
        batch_size = batch['source_table_ids'].size(0)

        # Move tensors to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Build the batch sub-hypergraph dynamically
        if global_hypergraph is not None:
            # Merge source and target indices
            all_indices = torch.cat([batch['source_idx'], batch['target_idx']])  # [2*batch_size]

            # Extract the batch sub-hypergraph
            batch_H, relevant_edges = build_batch_hypergraph(all_indices.tolist(), global_hypergraph)
            batch_H = batch_H.to(device)

            # Number of Type-1 edges that survive in this batch; the global
            # count is invalid once irrelevant edges are filtered out.
            batch_num_type1 = int(relevant_edges[:num_type1_edges].sum().item()) \
                if num_type1_edges is not None else None

            # Split the batch hypergraph into source and target parts
            source_hypergraph = batch_H[:batch_size]  # [batch_size, num_relevant_edges]
            target_hypergraph = batch_H[batch_size:]  # [batch_size, num_relevant_edges]

            # Debug: sanity-check the hypergraph (first batch only)
            if not hypergraph_debug_printed:
                print(f"\n Hypergraph debug info (Batch 0):")
                print(f"  Batch size: {batch_size}")
                print(f"  Global hypergraph shape: {global_hypergraph.shape}")
                print(f"  Batch hypergraph shape: {batch_H.shape}")
                print(f"  Source hypergraph shape: {source_hypergraph.shape}")
                print(f"  Target hypergraph shape: {target_hypergraph.shape}")
                print(f"  Source hypergraph non-zeros: {(source_hypergraph > 0).sum().item()} / {source_hypergraph.numel()}")
                print(f"  Target hypergraph non-zeros: {(target_hypergraph > 0).sum().item()} / {target_hypergraph.numel()}")
                print(f"  Source hypergraph density: {(source_hypergraph > 0).sum().item() / source_hypergraph.numel() * 100:.2f}%")
                print(f"  Target hypergraph density: {(target_hypergraph > 0).sum().item() / target_hypergraph.numel() * 100:.2f}%")

                # Count hyperedges per sample
                source_edges_per_sample = (source_hypergraph > 0).sum(dim=1)
                target_edges_per_sample = (target_hypergraph > 0).sum(dim=1)
                print(f"  Source edges per sample: min={source_edges_per_sample.min().item()}, "
                      f"max={source_edges_per_sample.max().item()}, "
                      f"mean={source_edges_per_sample.float().mean().item():.2f}")
                print(f"  Target edges per sample: min={target_edges_per_sample.min().item()}, "
                      f"max={target_edges_per_sample.max().item()}, "
                      f"mean={target_edges_per_sample.float().mean().item():.2f}")

                # Count isolated nodes (samples without any hyperedge)
                source_isolated = (source_edges_per_sample == 0).sum().item()
                target_isolated = (target_edges_per_sample == 0).sum().item()
                print(f"  Source isolated nodes: {source_isolated} / {batch_size}")
                print(f"  Target isolated nodes: {target_isolated} / {batch_size}\n")

                hypergraph_debug_printed = True
        else:
            source_hypergraph = None
            target_hypergraph = None
            batch_num_type1 = None

        # Slice the batch Column PE (when a global PE exists)
        source_column_pe = None
        target_column_pe = None
        if global_column_pe is not None:
            source_indices = batch['source_idx']  # [batch_size]
            target_indices = batch['target_idx']  # [batch_size]
            source_column_pe = global_column_pe[source_indices]  # [batch_size, column_pe_dim]
            target_column_pe = global_column_pe[target_indices]  # [batch_size, column_pe_dim]

        # Forward pass - Source
        source_emb = model(
            table_ids=batch['source_table_ids'],
            column_ids=batch['source_column_ids'],
            content_features=batch['source_content'],
            table_labels=batch['source_table_label'],  #  Table PE
            hypergraph_incidence=source_hypergraph,  # batch hypergraph
            num_type1_edges=batch_num_type1,  # batch-local Type-1 hyperedge count
            column_pe=source_column_pe  #  Column PE
        )

        # Forward pass - Target
        target_emb = model(
            table_ids=batch['target_table_ids'],
            column_ids=batch['target_column_ids'],
            content_features=batch['target_content'],
            table_labels=batch['target_table_label'],  #  Table PE
            hypergraph_incidence=target_hypergraph,  # batch hypergraph
            num_type1_edges=batch_num_type1,  # batch-local Type-1 hyperedge count
            column_pe=target_column_pe  #  Column PE
        )

        # Loss computation (selected by loss_type)
        labels = batch['label']
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        if loss_type in ('triplet', 'soft_triplet') and len(pos_indices) > 0 and len(neg_indices) > 0:
            # ========================================
            # Triplet loss + hard-negative mining (vectorised)
            # 'triplet'      : original hinge max(0, m - pos + neg), hard zero clamp -> margin collapse
            # 'soft_triplet' : softplus log(1 + exp(m - pos + neg)), smooth without clamping
            # Statistically equivalent to the original per-anchor loop (hard-vs-random with prob hard_neg_ratio)
            # ========================================
            cand_neg_targets = target_emb[neg_indices]  # [Nneg, D]
            P = pos_indices.size(0)
            N = cand_neg_targets.size(0)
            dev = source_emb.device

            anchors = source_emb[pos_indices]            # [P, D]
            positives = target_emb[pos_indices]          # [P, D]

            # Positive similarity (identical to the original loop version)
            pos_sims = F.cosine_similarity(anchors, positives, dim=1)  # [P]

            # Per-anchor Bernoulli choice: hard vs random negative (prob = hard_neg_ratio)
            if hard_negatives:
                use_hard = (torch.rand(P, device=dev) < hard_neg_ratio)  # [P]
            else:
                use_hard = torch.zeros(P, dtype=torch.bool, device=dev)

            # ---- Hard branch: one matmul for all anchor x candidate-negative cosine sims ----
            anchors_n = F.normalize(anchors, p=2, dim=1)
            cand_n = F.normalize(cand_neg_targets, p=2, dim=1)
            all_neg_sims = anchors_n @ cand_n.t()                     # [P, N]
            k = min(hard_topk, N)
            topk_vals, topk_idx = torch.topk(all_neg_sims, k=k, dim=1)  # [P, k]
            hard_pick_col = torch.randint(0, k, (P,), device=dev)        # [P]
            hard_neg_local = topk_idx[torch.arange(P, device=dev), hard_pick_col]  # [P], indices into cand_neg_targets

            # ---- Random branch: one uniform negative per anchor ----
            rand_neg_local = torch.randint(0, N, (P,), device=dev)     # [P]

            # Merge the two choices
            chosen_local = torch.where(use_hard, hard_neg_local, rand_neg_local)  # [P]
            negatives = cand_neg_targets[chosen_local]                  # [P, D]

            # Negative similarity
            neg_sims = F.cosine_similarity(anchors, negatives, dim=1)   # [P]

            if loss_type == 'soft_triplet':
                triplet_losses = F.softplus(margin - pos_sims + neg_sims)
            else:
                triplet_losses = torch.clamp(margin - pos_sims + neg_sims, min=0.0)

            loss = triplet_losses.mean()
            correct += int((pos_sims > neg_sims).sum().item())
            total += P

        elif loss_type in ('batch_soft_triplet', 'pairwise_logistic', 'multi_similarity', 'circle') and len(pos_indices) > 0 and len(neg_indices) > 0:
            # ========================================
            # Batch-all pairwise ranking losses.
            # For each positive anchor, compare its paired positive target with
            # all negative targets in the batch instead of sampling one negative.
            # ========================================
            anchors = F.normalize(source_emb[pos_indices], p=2, dim=1)       # [P, D]
            positives = F.normalize(target_emb[pos_indices], p=2, dim=1)     # [P, D]
            neg_targets = F.normalize(target_emb[neg_indices], p=2, dim=1)   # [N, D]

            pos_sims = (anchors * positives).sum(dim=1)                     # [P]
            neg_sims = anchors @ neg_targets.t()                            # [P, N]
            tau = max(infonce_temperature, 1e-3)

            if loss_type == 'batch_soft_triplet':
                neg_smooth_max = tau * torch.logsumexp(neg_sims / tau, dim=1)
                losses = F.softplus(margin - pos_sims + neg_smooth_max)
                loss = losses.mean()

            elif loss_type == 'pairwise_logistic':
                pairwise_gaps = (neg_sims - pos_sims.unsqueeze(1)) / tau
                loss = F.softplus(pairwise_gaps).mean()

            elif loss_type == 'multi_similarity':
                alpha, beta, base = 2.0, 50.0, 0.5
                pos_loss = torch.log1p(torch.exp(-alpha * (pos_sims - base))) / alpha
                neg_loss = torch.log1p(torch.exp(beta * (neg_sims - base)).sum(dim=1)) / beta
                loss = (pos_loss + neg_loss).mean()

            else:  # circle
                circle_margin, gamma = 0.25, 32.0
                ap = torch.clamp_min(-pos_sims.detach() + 1.0 + circle_margin, 0.0)
                an = torch.clamp_min(neg_sims.detach() + circle_margin, 0.0)
                delta_p = 1.0 - circle_margin
                delta_n = circle_margin
                logit_p = -gamma * ap * (pos_sims - delta_p)
                logit_n = gamma * an * (neg_sims - delta_n)
                loss = F.softplus(torch.logsumexp(logit_n, dim=1) + logit_p).mean()

            with torch.no_grad():
                hardest_neg = neg_sims.max(dim=1).values
                correct += int((pos_sims > hardest_neg).sum().item())
                total += pos_sims.size(0)

        elif loss_type == 'sup_contrastive' and len(pos_indices) > 0 and len(neg_indices) > 0:
            # ========================================
            # For each positive anchor, all positives form the positive set and all
            # negatives the negative set; log-sum-exp smoothing (no hard zero clamp),
            # so the objective is insensitive to the margin.
            # ========================================
            anchors = F.normalize(source_emb[pos_indices], p=2, dim=1)            # [P, D]
            pos_targets = F.normalize(target_emb[pos_indices], p=2, dim=1)        # [P, D]
            neg_targets = F.normalize(target_emb[neg_indices], p=2, dim=1)        # [N, D]

            tau = max(infonce_temperature, 1e-3)
            pos_sim = (anchors * pos_targets).sum(dim=1) / tau                    # [P]
            neg_sim = anchors @ neg_targets.t() / tau                             # [P, N]
            # Concatenate pos and neg; softmax is taken over dim 1
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)            # [P, 1+N]
            labels_sc = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels_sc)

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels_sc).sum().item()
                total += logits.size(0)

        elif loss_type == 'infonce':
            # ========================================
            # InfoNCE contrastive loss
            # ========================================
            loss = criterion(source_emb, target_emb)

            # Compute accuracy
            with torch.no_grad():
                sim_matrix = torch.matmul(source_emb, target_emb.t())  # [batch, batch]
                preds = torch.argmax(sim_matrix, dim=1)
                labels_acc = torch.arange(source_emb.size(0), device=device)
                correct += (preds == labels_acc).sum().item()
                total += source_emb.size(0)

        else:
            # Fallback: use InfoNCE when no pos/neg split is available
            loss = criterion(source_emb, target_emb)

            # Compute accuracy
            with torch.no_grad():
                sim_matrix = torch.matmul(source_emb, target_emb.t())  # [batch, batch]
                preds = torch.argmax(sim_matrix, dim=1)
                labels_acc = torch.arange(source_emb.size(0), device=device)
                correct += (preds == labels_acc).sum().item()
                total += source_emb.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update the progress bar
        avg_loss = total_loss / (batch_idx + 1)
        acc = 100 * correct / total if total > 0 else 0
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{avg_loss:.4f}',
            'acc': f'{acc:.1f}%'
        })

    return total_loss / len(train_loader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Column-Level Hypergraph Training')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='datasets/Lake',
                        help='Root directory containing <dataset>/ folders (default: datasets/Lake)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of BiHMP layers (best: 1; avoids over-smoothing on sparse hypergraphs)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='check/hypergraph/column_level_enhanced')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (parallel CPU-side tokenize/content aggregation)')

    # Hard-negative mining parameters (aligned with the baseline)
    parser.add_argument('--hard_negatives', type=int, default=1,
                        help='Enable hard-negative mining (1/0)')
    parser.add_argument('--hard_neg_ratio', type=float, default=1.0,
                        help='Fraction of hard negatives [0,1]')
    parser.add_argument('--hard_topk', type=int, default=5,
                        help='Sample uniformly from the top-k most similar negatives')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Triplet-loss margin')

    # Ablation: whether to use the hypergraph
    parser.add_argument('--use_hypergraph', type=int, default=1,
                        help='Use the hypergraph (1/0) - ablation')
    parser.add_argument('--use_pe_mixer', type=int, default=1,
                        help='Use the PE Mixer (1/0) - w/o HIN ablation')

    parser.add_argument('--use_table_pe', type=int, default=1,
                        help='Use Table-level PE (1/0) - PE ablation')

    # Ablation: independent control of Column PE (to probe shortcut learning).
    # Column PE shares its adjacency source with triplet supervision; this flag
    # disables Column PE only while keeping Table PE / Patch GNN / Mixer / hypergraph on.
    # The other components stay enabled.
    parser.add_argument('--use_column_pe', type=int, default=1,
                        help='Use Column-level Laplacian PE (1/0) - PE ablation')
    parser.add_argument('--column_pe_mode', type=str, default='laplacian',
                        choices=['laplacian', 'random', 'intra_only'],
                        help='Column PE computation: laplacian=eigendecomposition over joinable_pairs (default); '
                             'random=random matrix; intra_only=eigendecomposition over intra-table columns')
    # Edge-masked link-prediction protocol: split pseudo positive_pairs into
    # E_support (PE + hypergraph structure) and E_pred (triplet supervision),
    # disjoint, so the same edge cannot leak from structure into supervision.
    # 0.0 (default) = legacy behaviour where the full set serves as both.
    # The main PE ablation uses 0.2 (hold out 20% for prediction).
    parser.add_argument('--edge_mask_ratio', type=float, default=0.0,
                        help='Fraction of positive pairs held out as E_pred (loss supervision); '
                             'the rest becomes E_support (PE / hypergraph). 0.0 = no split (legacy).')
    # Data-matched control (Variant F): with edge_mask_ratio>0 the loss still uses
    # pred_pairs (same as E), but PE/hypergraph support is forced to the full set
    # to isolate "PE leakage" from "less supervision data".
    # F must share E's seed so pred_pairs is identical for a fair data-matched control.
    parser.add_argument('--pe_use_full_positives', type=int, default=0,
                        help='With edge_mask_ratio>0, whether PE/hypergraph support uses the full positive set '
                             '(1=full set while loss uses pred subset; 0=PE uses the 80% support, default E behaviour)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (for repeated runs)')

    # Ablation: hyperedge-type control
    parser.add_argument('--use_type1_edges', type=int, default=1,
                        help='Use Type-1 hyperedges (inter-table joins) (1/0) - ablation')
    parser.add_argument('--use_type2_edges', type=int, default=1,
                        help='Use Type-2 hyperedges (intra-table columns) (1/0) - ablation')

    # Ablation: pairwise GCN (LLM ablation)
    parser.add_argument('--pairwise_mode', type=int, default=0,
                        help='Pairwise+LLM ablation: GCN on H@H^T instead of the hypergraph (1/0)')

    # Loss function selection
    # Loss function selection
    # 'soft_triplet'   : softplus log(1 + exp(m - pos + neg)); no hard clamp, margin-insensitive
    # 'batch_soft_triplet': batch-all smooth max over all in-batch negatives
    # 'pairwise_logistic': RankNet-style pairwise logistic over all in-batch negatives
    # 'multi_similarity': Multi-Similarity loss with one paired positive per anchor
    # 'circle'         : Circle loss with one paired positive and all in-batch negatives
    # 'infonce'        : InfoNCE contrastive loss; all in-batch items act as negatives, temperature tau
    # 'sup_contrastive': supervised contrastive (multi-positive InfoNCE); one anchor, many positives
    parser.add_argument('--loss_type', type=str, default='triplet',
                        choices=['triplet', 'soft_triplet', 'batch_soft_triplet', 'pairwise_logistic',
                                 'multi_similarity', 'circle', 'infonce', 'sup_contrastive'],
                        help='Loss type (margin-collapse-robust variants)')
    # Robust objective: margin warm-up ramps the margin from 0 to args.margin so
    # hard negatives do not dominate at the start of training.
    parser.add_argument('--margin_warmup_epochs', type=int, default=0,
                        help='Margin warm-up epochs (0 = no warm-up; args.margin is used throughout)')
    parser.add_argument('--collapse_loss_threshold', type=float, default=0.95,
                        help='Loss threshold for the training-collapse detector; <=0 disables it. Default 0.95 keeps legacy behaviour.')

    # Hyperparameters: BiHMP configuration (grid-search tuned)
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='BiHMP dropout rate (best: 0.05)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension (default: 512)')
    parser.add_argument('--use_bn', type=int, default=0,
                        help='Use BatchNorm in BiHMP (1/0) (best: 0; BatchNorm breaks training)')
    parser.add_argument('--use_residual', type=int, default=1,
                        help='Use residual connections in BiHMP (1/0) (best: 1)')

    # Core HIN architecture parameters
    parser.add_argument('--patch_gnn_layers', type=int, default=2,
                        help='Number of Patch GNN layers (default: 2)')
    parser.add_argument('--num_mixer_layers', type=int, default=2,
                        help='Number of Mixer layers (default: 2)')

    args = parser.parse_args()

    # Set the random seed
    set_seed(args.seed)

    # Data paths
    data_dir = f'{args.data_root}/{args.dataset}'
    metadata_path = f'{data_dir}/target_metadata.pkl'
    target_npy_path = f'{data_dir}/target.npy'

    # Detect Label-Free mode (dataset name contains _LabelFree)
    is_label_free = '_LabelFree' in args.dataset
    if is_label_free:
        # Label-Free: use self_supervised_pairs.pkl
        joinable_pairs_path = f'{data_dir}/self_supervised_pairs.pkl'
        print(f" Column-Level Hypergraph Training (dual hyperedge design)")
        print(f"   Training mode: Label-Free (self-supervised)")
        print(f"   Dataset: {args.dataset}")
    else:
        # Supervised: use joinable_pairs.csv
        base_dataset = args.dataset
        joinable_pairs_path = f'datasets/datasets_{base_dataset}/joinable_pairs.csv'
        print(f" Column-Level Hypergraph Training (dual hyperedge design)")
        print(f"   Training mode: Supervised")
        print(f"   Dataset: {args.dataset}")

    print(f"   Epoch: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   BiHMP layers: {args.num_layers}")
    print(f"   Hyperedge types: connected components + intra-table columns")

    # Load data
    print(f"\n Loading data...")
    target_dataset = TestDatasetHyperJoin(target_npy_path, metadata_path)

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata_loaded = pickle.load(f)
        if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
            metadata = metadata_loaded['metadata']
        else:
            metadata = metadata_loaded

    # Build the vocabulary
    print(f"\n Building vocabulary...")
    vocab = build_vocabulary(metadata)
    print(f" Vocabulary size: {len(vocab)}")

    # Build the table_name -> table_id mapping (for Table PE)
    unique_tables = set()
    for meta in metadata:
        table_name = meta.get('table_name', '').replace('.csv', '')
        if table_name:
            unique_tables.add(table_name)
    table_name_to_id = {table_name: idx for idx, table_name in enumerate(sorted(unique_tables))}
    print(f" Table mapping built: {len(table_name_to_id)} tables")

    # Create the training dataset
    print(f"\n Creating dataset...")
    train_dataset = ColumnHypergraphDataset(
        target_dataset=target_dataset,
        joinable_pairs_path=joinable_pairs_path,
        vocab=vocab,
        table_name_to_id=table_name_to_id,
        negative_ratio=1.0
    )

    # Edge-masked link-prediction protocol:
    #   support_pairs : used by PE / hypergraph structure (the "observed" graph)
    #   pred_pairs    : used by triplet supervision (the edges to predict)
    # The two are disjoint; hypergraph construction and the PE adjacency use support_pairs.
    if args.edge_mask_ratio > 0.0 and len(train_dataset.positive_pairs) > 1:
        all_pos = list(train_dataset.positive_pairs)
        n_total = len(all_pos)
        n_pred = max(1, int(round(n_total * args.edge_mask_ratio)))
        # Use a dedicated RNG for the split so the global random stream is not polluted.
        # Variant F (pe_use_full_positives=1) must use the same split_rng so pred_pairs matches E.
        split_rng = random.Random(args.seed + 9173)  # offset keeps the split seed apart from the training seed
        all_pos_shuffled = list(all_pos)  # keep a full copy (used by Variant F)
        split_rng.shuffle(all_pos_shuffled)
        pred_pairs = all_pos_shuffled[:n_pred]
        support_pairs = all_pos_shuffled[n_pred:]
        # train_dataset keeps only pred_pairs as positives (the loss sees pred only)
        train_dataset.positive_pairs = pred_pairs
        train_dataset.all_pairs = list(pred_pairs) + list(train_dataset.negative_pairs)
        random.shuffle(train_dataset.all_pairs)
        # Variant F: force PE/hypergraph support to the full set (same pred_pairs as E, but PE sees everything)
        if args.pe_use_full_positives:
            support_pairs = list(all_pos)  # full set (includes pred_pairs)
            print(f"\nEdge-masked link prediction protocol (F: data-matched control):")
            print(f"   Original positive pairs: {n_total}")
            print(f"   E_support (PE/hypergraph): {len(support_pairs)} (full set, F variant)")
            print(f"   E_pred (triplet supervision): {len(pred_pairs)}")
            print(f"   mask_ratio: {args.edge_mask_ratio} (PE sees the full set, loss uses pred subset)")
        else:
            print(f"\nEdge-masked link prediction protocol:")
            print(f"   Original positive pairs: {n_total}")
            print(f"   E_support (PE/hypergraph): {len(support_pairs)}")
            print(f"   E_pred (triplet supervision): {len(pred_pairs)}")
            print(f"   mask_ratio: {args.edge_mask_ratio}")
    else:
        support_pairs = list(train_dataset.positive_pairs)
        if args.edge_mask_ratio > 0.0:
            print(f"\nedge_mask_ratio={args.edge_mask_ratio} but too few positives ({len(train_dataset.positive_pairs)}); skipping the split")
        else:
            print(f"\nEdge-masked protocol disabled (edge_mask_ratio=0.0); E_support and E_pred both equal the full set (legacy behaviour)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # Reuse the table_name_to_id built above
    num_tables = len(table_name_to_id)

    print(f"\n Dataset statistics:")
    print(f"   Tables: {num_tables}")
    print(f"   Columns: {len(target_dataset)}")

    # Create the enhanced model
    # Ablation: PE Mixer usage controlled by use_pe_mixer
    use_pe_mixer = bool(args.use_pe_mixer)
    if use_pe_mixer:
        print(f"\nCreating enhanced model (Graph ViT/MLP-Mixer)...")
    else:
        print(f"\nCreating base model (w/o HIN - plain hypergraph message passing)...")

    pairwise_mode = bool(args.pairwise_mode)

    # Independent PE control: use_column_pe=0 disables Column PE only;
    # use_table_pe=0 additionally disables Table PE; Mixer/hypergraph still follow use_pe_mixer
    use_table_pe_effective = use_pe_mixer and bool(args.use_table_pe)
    use_column_pe_effective = use_pe_mixer and bool(args.use_column_pe)
    if use_pe_mixer and not bool(args.use_table_pe):
        print(f"   Ablation: Table PE off (Patch GNN / Mixer / hypergraph kept)")
    if use_pe_mixer and not bool(args.use_column_pe):
        print(f"   Ablation: Column PE off (Patch GNN / Mixer / hypergraph kept)")

    model = EnhancedHyperJoinModel(
        vocab_size=len(vocab),
        content_dim=1200,  # mean+std+max+min
        num_tables=num_tables,
        embed_dim=args.embed_dim,
        column_pe_dim=16,
        num_patches=100,  # estimate; adjusted dynamically
        patch_gnn_layers=args.patch_gnn_layers if use_pe_mixer else 0,  # w/o HIN: no Patch GNN
        num_mixer_layers=args.num_mixer_layers if use_pe_mixer else 0,  # w/o HIN: no Mixer
        mlp_ratio=4.0,
        dropout=args.dropout,
        use_table_pe=use_table_pe_effective,
        use_column_pe=use_column_pe_effective,  # independently controllable; see flags above
        use_structure_bias=use_pe_mixer,
        use_edge_type=True,
        pairwise_mode=pairwise_mode
    ).to(args.device)

    if pairwise_mode:
        print(f"   Pairwise+LLM ablation: GCN on A=H@H^T (replaces the hypergraph structure)")

    print(f" Enhanced model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Enhanced features:")
    print(f"    Table-level PE: ")
    print(f"    Column-level PE:  (dim={16})")
    print(f"    Patch GNN layers: {args.patch_gnn_layers}")
    print(f"    Mixer layers: {args.num_mixer_layers}")
    print(f"    Dropout: {args.dropout}")

    # Loss function and optimizer
    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning Rate Scheduler with Warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Ablation: hypergraph usage controlled by flag
    if args.use_hypergraph:
        print(f"\n Building the dual-type hyperedge graph...")
        print(f"   Type 1: joinable connected-component hyperedges {'ON' if args.use_type1_edges else 'OFF'}")
        print(f"   Type 2: intra-table column hyperedges {'ON' if args.use_type2_edges else 'OFF'}")
        _hg_start = time.time()
        # Use E_support rather than all positives: under the edge-masked protocol
        # train_dataset.positive_pairs was replaced by E_pred; hypergraph/PE see support only
        joinable_pairs = [(s, t) for s, t, _ in support_pairs]
        global_hypergraph, num_type1_edges = build_dual_hypergraph_from_pairs(
            num_columns=len(target_dataset),
            joinable_pairs=joinable_pairs,
            metadata=metadata,
            use_type1_edges=bool(args.use_type1_edges),
            use_type2_edges=bool(args.use_type2_edges)
        )
        _hg_time = time.time() - _hg_start
        print(f" Hypergraph built ({_hg_time:.2f}s)")
        print(f"   Columns: {len(target_dataset)}")
        print(f"   Hyperedges: {global_hypergraph.shape[1]}")
        print(f"   Type-1 hyperedges: {num_type1_edges}")
        print(f"   Type-2 hyperedges: {global_hypergraph.shape[1] - num_type1_edges}")
        print(f"   Incidence matrix shape: {global_hypergraph.shape}")

        # Precompute the global Column PE (column-level positional encoding)
        num_columns = len(target_dataset)
        pe_dim = 16
        _pe_time = 0.0

        if not use_column_pe_effective:
            # Ablation: Column PE off (other hypergraph structure kept)
            print(f"\n Skipping Column PE computation (use_column_pe=0)")
            global_column_pe = None
        else:
            pe_mode = args.column_pe_mode
            print(f"\n Precomputing the global column-level PE...")
            print(f"   Mode: {pe_mode}")

            # Cache: use a different key per mode to avoid collisions
            cache_dir = Path("cache/column_pe")
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Compatibility: laplacian + edge_mask_ratio=0 (legacy) keeps the old cache filename;
            # edge_mask_ratio>0 adds a mask suffix (seed included since the split depends on it);
            # Variant F (pe_use_full_positives=1) adds a PE-full tag so it never shares cache with E.
            mask_tag = ""
            if pe_mode == 'laplacian' and args.edge_mask_ratio > 0.0:
                mask_tag = f"_mask{args.edge_mask_ratio:.2f}_s{args.seed}"
                if args.pe_use_full_positives:
                    mask_tag += "_pefull"
            if pe_mode == 'laplacian':
                cache_file = cache_dir / f"{args.dataset}_k{pe_dim}{mask_tag}.pt"
            else:
                cache_file = cache_dir / f"{args.dataset}_k{pe_dim}_{pe_mode}{mask_tag}.pt"

            need_compute = True
            if cache_file.exists():
                print(f"   Loading from cache: {cache_file}")
                cached_data = torch.load(cache_file)
                if cached_data['num_columns'] == num_columns:
                    global_column_pe = cached_data['column_pe'].to(args.device)
                    print(f"   Cache valid; skipping computation")
                    print(f"   Shape: {global_column_pe.shape}")
                    print(f"   Stats: mean={global_column_pe.mean().item():.4f}, std={global_column_pe.std().item():.4f}")
                    need_compute = False
                else:
                    print(f"   Cache invalid (column count mismatch: {cached_data['num_columns']} vs {num_columns}); recomputing")
            else:
                print(f"   No cache found; computing...")

            if need_compute:
                _pe_start = time.time()

                if pe_mode == 'random':
                    # Random PE ablation: tests whether PE is only a leakage channel
                    rng = np.random.RandomState(args.seed)
                    pe_np = rng.randn(num_columns, pe_dim).astype(np.float32) * 0.01
                    global_column_pe_cpu = torch.from_numpy(pe_np)
                    print(f"   Using a random matrix as PE (seed={args.seed})")

                else:
                    # Build the column-level adjacency matrix
                    adjacency_matrix = torch.zeros((num_columns, num_columns), dtype=torch.float32)
                    if pe_mode == 'intra_only':
                        # Intra-table adjacency only; joinable_pairs is not used
                        from collections import defaultdict
                        table_to_cols = defaultdict(list)
                        for col_idx, meta in enumerate(metadata):
                            tname = meta.get('table_name', '').replace('.csv', '')
                            if tname:
                                table_to_cols[tname].append(col_idx)
                        num_intra_edges = 0
                        for tname, cols in table_to_cols.items():
                            for i in range(len(cols)):
                                for j in range(i + 1, len(cols)):
                                    adjacency_matrix[cols[i], cols[j]] = 1.0
                                    adjacency_matrix[cols[j], cols[i]] = 1.0
                                    num_intra_edges += 1
                        print(f"   Using intra-table adjacency: {len(table_to_cols)} tables, {num_intra_edges} edges")
                    else:
                        # laplacian (default): adjacency built from E_support;
                        # edge_mask_ratio > 0: joinable_pairs == support_pairs, disjoint from E_pred;
                        # edge_mask_ratio = 0: legacy behaviour (same source as triplet labels; shortcut risk)
                        for s, t in joinable_pairs:
                            adjacency_matrix[s, t] = 1.0
                            adjacency_matrix[t, s] = 1.0
                        print(f"   Using E_support adjacency: {len(joinable_pairs)} edges "
                              f"(edge_mask_ratio={args.edge_mask_ratio})")

                    print(f"   Adjacency matrix size: {adjacency_matrix.shape} = {adjacency_matrix.numel() * 4 / 1024**3:.2f} GB")

                    # Call the model's compute_column_pe (computed on CPU)
                    global_column_pe_cpu = model.position_encoding.compute_column_pe(
                        adjacency_matrix, k=pe_dim
                    )

                # Save to the cache
                torch.save({
                    'column_pe': global_column_pe_cpu,
                    'num_columns': num_columns,
                    'pe_dim': pe_dim,
                    'pe_mode': pe_mode,
                }, cache_file)
                print(f"   Saved to cache: {cache_file}")

                global_column_pe = global_column_pe_cpu.to(args.device)

                _pe_time = time.time() - _pe_start
                print(f"   Column PE computed ({_pe_time:.2f}s)")
                print(f"   Shape: {global_column_pe.shape}")
                print(f"   Stats: mean={global_column_pe.mean().item():.4f}, std={global_column_pe.std().item():.4f}")

    else:
        print(f"\n Ablation mode: hypergraph disabled")
        print(f"   BiHMP layers will be skipped during training (encoder only)")
        global_hypergraph = None
        num_type1_edges = None
        global_column_pe = None  # Column PE is also disabled when the hypergraph is off
        _hg_time = 0.0
        _pe_time = 0.0

    # Create the output directory
    output_dir = f'{args.output_dir}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)

    # Training
    print(f"\n Start training...\n")
    print(f" Loss function: {args.loss_type.upper()}")
    if args.loss_type in ('triplet', 'soft_triplet', 'batch_soft_triplet',
                          'pairwise_logistic', 'multi_similarity', 'circle'):
        print(f"   Triplet loss")
        print(f"   Hard Negatives: {'ON' if args.hard_negatives else 'OFF'}")
        print(f"   Hard Neg Ratio: {args.hard_neg_ratio}")
        print(f"   Hard TopK: {args.hard_topk}")
        print(f"   Margin: {args.margin}")
        print(f"   Temperature: {args.temperature}")
    else:
        print(f"   InfoNCE contrastive loss")
        print(f"   Temperature: {args.temperature}")
    print(f" Hypergraph: {'ON' if args.use_hypergraph else 'OFF (ablation)'}\n")

    best_loss = float('inf')
    patience = 8  # early-stopping patience
    patience_counter = 0
    min_delta = 0.001  # minimum improvement threshold

    # Scalability timing
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _train_start = time.time()
    _epoch_times = []

    for epoch in range(1, args.epochs + 1):
        _ep_start = time.time()
        # Margin warm-up: ramp the margin from 0 to args.margin over the first epochs
        if args.margin_warmup_epochs > 0 and epoch <= args.margin_warmup_epochs:
            effective_margin = args.margin * (epoch / float(args.margin_warmup_epochs))
        else:
            effective_margin = args.margin
        loss, acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            global_hypergraph, num_type1_edges,  # None when use_hypergraph=0
            global_column_pe,  # precomputed global Column PE
            hard_negatives=bool(args.hard_negatives),
            hard_neg_ratio=args.hard_neg_ratio,
            hard_topk=args.hard_topk,
            margin=effective_margin,
            loss_type=args.loss_type,  # loss type
            infonce_temperature=args.temperature
        )

        _epoch_times.append(time.time() - _ep_start)

        # Update the learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {acc:.2f}%")
        print(f"   Learning Rate: {current_lr:.6f}")

        # Detect training collapse
        if args.collapse_loss_threshold > 0 and loss > args.collapse_loss_threshold and epoch > 3:
            print(f" Warning: loss={loss:.4f} is high; training may be collapsing")
            patience_counter += 1
            if patience_counter >= 3:
                print(f" Training-collapse detector: {patience_counter} consecutive epochs with loss>{args.collapse_loss_threshold}; stopping early")
                break
        else:
            patience_counter = 0

        # Save the best model
        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0  # reset the patience counter
            save_path = f'{output_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': acc,
                'vocab': vocab,
                'model_config': {
                    'vocab_size': len(vocab),
                    'content_dim': 1200,
                    'embed_dim': args.embed_dim,
                    'num_tables': num_tables,
                    'num_bihmp_layers': args.num_layers,
                    'patch_gnn_layers': args.patch_gnn_layers,
                    'num_mixer_layers': args.num_mixer_layers,
                    'use_residual': True,
                    'dropout': args.dropout,
                    'use_bn': False,
                    'pairwise_mode': pairwise_mode
                }
            }, save_path)
            print(f" Saved best model: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stop: no improvement for {patience} epochs; stopping")
                break

    _total_train_time = time.time() - _train_start

    print(f"\n Training complete! Best loss: {best_loss:.4f}")

    # Write timing JSON for scalability experiments
    _timing_data = {
        'dataset': args.dataset,
        'num_columns': len(target_dataset),
        'num_tables': num_tables,
        'hypergraph_construction_time_s': round(_hg_time, 2) if args.use_hypergraph else 0.0,
        'column_pe_time_s': round(_pe_time, 2) if args.use_hypergraph else 0.0,
        'total_training_time_s': round(_total_train_time, 2),
        'num_epochs_completed': len(_epoch_times),
        'avg_epoch_time_s': round(sum(_epoch_times) / len(_epoch_times), 2) if _epoch_times else 0.0,
        'peak_gpu_memory_mb': round(get_gpu_memory_mb(), 1),
        'peak_cpu_memory_mb': round(get_cpu_memory_mb(), 1),
    }
    _timing_path = os.path.join(output_dir, 'training_timing.json')
    with open(_timing_path, 'w') as f:
        json.dump(_timing_data, f, indent=2)
    print(f" Timing saved to: {_timing_path}")


if __name__ == '__main__':
    main()
