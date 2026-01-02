
import os
import sys
import argparse
import pickle
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data import TestDatasetHyperJoin
from utils import set_seed
from model_column_hypergraph import (
    build_dual_hypergraph_from_pairs,
    build_batch_hypergraph,
    ContrastiveLoss
)
from model_enhanced import EnhancedHyperJoinModel


class ColumnHypergraphDataset(Dataset):

    def __init__(self,
                 target_dataset: TestDatasetHyperJoin,
                 joinable_pairs_path: str,
                 vocab: Dict[str, int],
                 table_name_to_id: Dict[str, int] = None,
                 negative_ratio: float = 1.0,
                 max_seq_len: int = 10):

        self.target_dataset = target_dataset
        self.vocab = vocab
        self.table_name_to_id = table_name_to_id if table_name_to_id is not None else {}
        self.max_seq_len = max_seq_len
        self.negative_ratio = negative_ratio

        self.metadata = self._load_metadata()

        self.table_col_to_idx = self._build_mapping()

        self.is_label_free = joinable_pairs_path.endswith('.pkl')

        pairs_result = self._load_joinable_pairs(joinable_pairs_path)

        self.positive_pairs, self.negative_pairs = pairs_result

        self.all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.all_pairs)

    def _load_metadata(self):
        if hasattr(self.target_dataset, 'metadata'):
            metadata = self.target_dataset.metadata
            if isinstance(metadata, dict) and 'metadata' in metadata:
                return metadata['metadata']
            return metadata
        return []

    def _build_mapping(self) -> Dict[str, int]:
        mapping = {}
        for idx, meta in enumerate(self.metadata):
            table_name = meta.get('table_name', '').replace('.csv', '')
            column_name = meta.get('column_name', '')
            key = f"{table_name}.{column_name}"
            mapping[key] = idx
        return mapping

    def _load_joinable_pairs(self, path: str):
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                pairs_data = pickle.load(f)
                positive_pairs = pairs_data['positive']
                negative_pairs = pairs_data['negative']

            pairs = []

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

            return pairs, negative_list

        else:
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

            return pairs

    def _generate_negative_pairs(self) -> List[Tuple[int, int, float]]:
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

        return negative_pairs

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = text.replace('.csv', '').split('_')
        ids = []
        for tok in tokens[:self.max_seq_len]:
            ids.append(self.vocab.get(tok, self.vocab.get('<UNK>', 1)))
        while len(ids) < self.max_seq_len:
            ids.append(self.vocab.get('<PAD>', 0))
        return torch.tensor(ids[:self.max_seq_len], dtype=torch.long)

    def _aggregate_content(self, col_data):
        if isinstance(col_data, torch.Tensor):
            data = col_data
        else:
            data = torch.tensor(col_data, dtype=torch.float32)

        if data.numel() == 0:
            return torch.zeros(1200)

        if data.dim() == 1:
            data = data.unsqueeze(0)

        if data.shape[0] == 1:
            mean_f = data[0]
            std_f = torch.zeros_like(mean_f)
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

        # Source
        source_meta = self.metadata[source_idx]
        source_table_ids = self._tokenize(source_meta.get('table_name', ''))
        source_column_ids = self._tokenize(source_meta.get('column_name', ''))
        source_content = self._aggregate_content(self.target_dataset[source_idx])
        source_table_name = source_meta.get('table_name', '').replace('.csv', '')
        source_table_label = self.table_name_to_id.get(source_table_name, 0)

        # Target
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


def build_vocabulary(metadata):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab_idx = 2

    for meta in metadata:
        table_name = meta.get('table_name', '').replace('.csv', '')
        for token in table_name.split('_'):
            if token and token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1

        column_name = meta.get('column_name', '')
        for token in column_name.split('_'):
            if token and token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1

    return vocab


def train_epoch(model, train_loader, criterion, optimizer, device,
                global_hypergraph=None, num_type1_edges=None, global_column_pe=None,
                hard_negatives=True, hard_neg_ratio=1.0, hard_topk=5, margin=0.5, loss_type='triplet'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')

    hypergraph_debug_printed = False

    for batch_idx, batch in enumerate(progress_bar):
        batch_size = batch['source_table_ids'].size(0)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        if global_hypergraph is not None:
            all_indices = torch.cat([batch['source_idx'], batch['target_idx']])

            batch_H = build_batch_hypergraph(all_indices.tolist(), global_hypergraph)
            batch_H = batch_H.to(device)

            source_hypergraph = batch_H[:batch_size]
            target_hypergraph = batch_H[batch_size:]

        else:
            source_hypergraph = None
            target_hypergraph = None

        source_column_pe = None
        target_column_pe = None
        if global_column_pe is not None:
            source_indices = batch['source_idx']
            target_indices = batch['target_idx']
            source_column_pe = global_column_pe[source_indices]
            target_column_pe = global_column_pe[target_indices]

        source_emb = model(
            table_ids=batch['source_table_ids'],
            column_ids=batch['source_column_ids'],
            content_features=batch['source_content'],
            table_labels=batch['source_table_label'],
            hypergraph_incidence=source_hypergraph,
            num_type1_edges=num_type1_edges,
            column_pe=source_column_pe
        )

        target_emb = model(
            table_ids=batch['target_table_ids'],
            column_ids=batch['target_column_ids'],
            content_features=batch['target_content'],
            table_labels=batch['target_table_label'],
            hypergraph_incidence=target_hypergraph,
            num_type1_edges=num_type1_edges,
            column_pe=target_column_pe
        )

        labels = batch['label']
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        if len(pos_indices) > 0 and len(neg_indices) > 0:
            loss = 0.0
            batch_correct = 0

            cand_neg_targets = target_emb[neg_indices]

            for pos_idx in pos_indices:
                anchor = source_emb[pos_idx]
                positive = target_emb[pos_idx]

                use_hard = hard_negatives and (torch.rand(1).item() < hard_neg_ratio)
                if use_hard and cand_neg_targets.size(0) > 0:
                    sims = F.cosine_similarity(anchor.unsqueeze(0), cand_neg_targets, dim=1)
                    k = min(hard_topk, sims.numel())
                    if k >= 1:
                        topk_vals, topk_idx = torch.topk(sims, k=k, largest=True)
                        pick_local = topk_idx[torch.randint(0, k, (1,)).item()].item()
                        negative = cand_neg_targets[pick_local]
                    else:
                        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
                        negative = target_emb[neg_idx]
                else:
                    neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
                    negative = target_emb[neg_idx]

                pos_sim = F.cosine_similarity(anchor, positive, dim=0)
                neg_sim = F.cosine_similarity(anchor, negative, dim=0)
                triplet_loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
                loss += triplet_loss

                if pos_sim > neg_sim:
                    batch_correct += 1

            loss = loss / len(pos_indices)
            correct += batch_correct
            total += len(pos_indices)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

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
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/enhanced')

    parser.add_argument('--hard_negatives', type=int, default=1)
    parser.add_argument('--hard_neg_ratio', type=float, default=1.0)
    parser.add_argument('--hard_topk', type=int, default=5)
    parser.add_argument('--margin', type=float, default=0.5)

    parser.add_argument('--use_hypergraph', type=int, default=1)
    parser.add_argument('--use_pe_mixer', type=int, default=1)

    parser.add_argument('--use_type1_edges', type=int, default=1)
    parser.add_argument('--use_type2_edges', type=int, default=1)

    parser.add_argument('--loss_type', type=str, default='triplet')

    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--use_residual', type=int, default=1)

    parser.add_argument('--patch_gnn_layers', type=int, default=2)
    parser.add_argument('--num_mixer_layers', type=int, default=2)

    args = parser.parse_args()

    set_seed(2024)
    data_dir = f'datasets/Lake/{args.dataset}'
    metadata_path = f'{data_dir}/target_metadata.pkl'
    target_npy_path = f'{data_dir}/target.npy'

    is_label_free = '_LabelFree' in args.dataset
    joinable_pairs_path = f'{data_dir}/self_supervised_pairs.pkl'

    target_dataset = TestDatasetHyperJoin(target_npy_path, metadata_path)

    with open(metadata_path, 'rb') as f:
        metadata_loaded = pickle.load(f)
        if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
            metadata = metadata_loaded['metadata']
        else:
            metadata = metadata_loaded

    vocab = build_vocabulary(metadata)

    unique_tables = set()
    for meta in metadata:
        table_name = meta.get('table_name', '').replace('.csv', '')
        if table_name:
            unique_tables.add(table_name)
    table_name_to_id = {table_name: idx for idx, table_name in enumerate(sorted(unique_tables))}

    train_dataset = ColumnHypergraphDataset(
        target_dataset=target_dataset,
        joinable_pairs_path=joinable_pairs_path,
        vocab=vocab,
        table_name_to_id=table_name_to_id,
        negative_ratio=1.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    num_tables = len(table_name_to_id)

    use_pe_mixer = bool(args.use_pe_mixer)
    model = EnhancedHyperJoinModel(
        vocab_size=len(vocab),
        content_dim=1200,  # mean+std+max+min
        num_tables=num_tables,
        embed_dim=args.embed_dim,
        column_pe_dim=16,
        num_patches=100, 
        patch_gnn_layers=args.patch_gnn_layers if use_pe_mixer else 0,
        num_mixer_layers=args.num_mixer_layers if use_pe_mixer else 0,
        mlp_ratio=4.0,
        dropout=args.dropout,
        use_table_pe=use_pe_mixer,
        use_column_pe=use_pe_mixer,
        use_structure_bias=use_pe_mixer,
        use_edge_type=True
    ).to(args.device)

    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.use_hypergraph:
        joinable_pairs = [(s, t) for s, t, _ in train_dataset.positive_pairs]
        global_hypergraph, num_type1_edges = build_dual_hypergraph_from_pairs(
            num_columns=len(target_dataset),
            joinable_pairs=joinable_pairs,
            metadata=metadata,
            use_type1_edges=bool(args.use_type1_edges),
            use_type2_edges=bool(args.use_type2_edges)
        )

        num_columns = len(target_dataset)
        pe_dim = 16

        cache_dir = Path("cache/column_pe")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{args.dataset}_k{pe_dim}.pt"

        need_compute = True
        if cache_file.exists():
            cached_data = torch.load(cache_file)

            if cached_data['num_columns'] == num_columns:
                global_column_pe = cached_data['column_pe'].to(args.device)
                need_compute = False
            else:
                need_compute = True
        else:
            need_compute = True

        if need_compute:
            adjacency_matrix = torch.zeros((num_columns, num_columns), dtype=torch.float32)
            for s, t in joinable_pairs:
                adjacency_matrix[s, t] = 1.0
                adjacency_matrix[t, s] = 1.0

            global_column_pe_cpu = model.position_encoding.compute_column_pe(
                adjacency_matrix, k=pe_dim
            )

            cache_file = cache_dir / f"{args.dataset}_k{pe_dim}.pt"
            torch.save({
                'column_pe': global_column_pe_cpu,
                'num_columns': num_columns,
                'pe_dim': pe_dim
            }, cache_file)

            global_column_pe = global_column_pe_cpu.to(args.device)

    else:
        global_column_pe = None
        global_hypergraph = None
        num_type1_edges = None

    output_dir = f'{args.output_dir}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎯 Start training...\n")

    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    min_delta = 0.001

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            global_hypergraph, num_type1_edges,
            global_column_pe,
            hard_negatives=bool(args.hard_negatives),
            hard_neg_ratio=args.hard_neg_ratio,
            hard_topk=args.hard_topk,
            margin=args.margin,
            loss_type=args.loss_type
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {acc:.2f}%")
        print(f"   Learning Rate: {current_lr:.6f}")

        if loss > 0.95 and epoch > 3:
            print(f"⚠️  Warning: Loss={loss:.4f} is too high, training may crash")
            patience_counter += 1
            if patience_counter >= 3:
                print(f"❌ Training crash detected: {patience_counter} epochs Loss>0.95, Stop training")
                break
        else:
            patience_counter = 0

        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0
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
                    'use_bn': False
                }
            }, save_path)
            print(f"💾 Save best model: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stop: {patience} epochs, Stop training")
                break

    print(f"\n✅ Training completed! Best Loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
