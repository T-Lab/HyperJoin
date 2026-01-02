import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data import TestDatasetHyperJoin
from utils import set_seed

from model import EnhancedHyperJoinModel
from evaluator import DetailedEvaluator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mst_reranker'))
from mst_core_labelfree import LabelFreeMSTReranker
from graph_builder_search import GraphBuilderSearch
from evaluator_with_coherence import CoherenceEvaluator


def tokenize_pad(text: str, vocab: dict, max_len: int = 10, vocab_size: int = None) -> torch.Tensor:

    tokens = text.replace('.csv', '').split('_')
    ids = []

    unk_id = vocab.get('<UNK>', 1)
    pad_id = vocab.get('<PAD>', 0)

    if vocab_size is not None:
        if unk_id >= vocab_size:
            unk_id = 1
        if pad_id >= vocab_size:
            pad_id = 0
        unk_id = min(unk_id, vocab_size - 1)
        pad_id = min(pad_id, vocab_size - 1)

    for tok in tokens[:max_len]:
        token_id = vocab.get(tok, unk_id)
        if vocab_size is not None and token_id >= vocab_size:
            token_id = unk_id
        ids.append(token_id)

    while len(ids) < max_len:
        ids.append(pad_id)

    return torch.tensor(ids[:max_len], dtype=torch.long)


def aggregate_content(col_data) -> torch.Tensor:
    if isinstance(col_data, torch.Tensor):
        data = col_data
    else:
        data = torch.tensor(col_data, dtype=torch.float32)

    if data.numel() == 0:
        return torch.zeros(1200)

    if data.dim() == 1:
        data = data.unsqueeze(0)

    if data.shape[0] > 0 and data.shape[1] > 0:
        mean_f = torch.mean(data, dim=0)
        std_f = torch.std(data, dim=0)
        max_f = torch.max(data, dim=0)[0]
        min_f = torch.min(data, dim=0)[0]
        return torch.cat([mean_f, std_f, max_f, min_f], dim=0)
    else:
        return torch.zeros(1200)


def _get_meta_list(dataset):
    if hasattr(dataset, 'metadata'):
        metadata = dataset.metadata
        if isinstance(metadata, dict) and 'metadata' in metadata:
            return metadata['metadata']
        return metadata
    return []


@torch.no_grad()
def encode_dataset(model,
                   dataset: TestDatasetHyperJoin,
                   vocab: dict,
                   table_name_to_id: dict,
                   device: torch.device,
                   vocab_size: int,
                   desc: str = "Encoding") -> torch.Tensor:
    embeddings = []
    metas = _get_meta_list(dataset)

    for i in tqdm(range(len(dataset)), desc=desc):
        meta = metas[i]

        table_ids = tokenize_pad(meta.get('table_name', ''), vocab, vocab_size=vocab_size).unsqueeze(0).to(device)
        column_ids = tokenize_pad(meta.get('column_name', ''), vocab, vocab_size=vocab_size).unsqueeze(0).to(device)

        col_data = torch.tensor(dataset[i], dtype=torch.float32)
        content = aggregate_content(col_data).unsqueeze(0).to(device)

        table_name = meta.get('table_name', '').replace('.csv', '')
        table_label = table_name_to_id.get(table_name, 0)

        if hasattr(model, 'position_encoding') and hasattr(model.position_encoding, 'table_embeddings'):
            max_table_id = model.position_encoding.table_embeddings.weight.shape[0] - 1
            if table_label > max_table_id:
                table_label = 0

        table_labels = torch.tensor([table_label], dtype=torch.long).to(device)

        if hasattr(model, 'encode_without_hypergraph'):
            emb = model.encode_without_hypergraph(table_ids, column_ids, content, table_labels)
        else:
            emb = model(table_ids, column_ids, content, hypergraph_incidence=None)
        embeddings.append(emb.squeeze(0))

    return torch.stack(embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description='Column Hypergraph Search with Optional MST Reranking'
    )

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., CAN_ALL)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--joinable_pairs', type=str, required=False, default=None,
                        help='Path to joinable_pairs.csv (not needed for Label-Free with mst_alpha=0)')
    parser.add_argument('--top_k', type=int, default=25,
                        help='Number of final results to return')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--use_mst', action='store_true',
                        help='Enable MST reranking')
    parser.add_argument('--mst_candidate_size', type=int, default=50,
                        help='Size of candidate pool for MST (B in paper)')
    parser.add_argument('--mst_alpha', type=float, default=0.0,
                        help='Weight for GT joinable relationships (0 for Label-Free mode)')
    parser.add_argument('--mst_beta', type=float, default=0.3,
                        help='Weight for embedding similarity')
    parser.add_argument('--mst_lambda', type=float, default=0.7,
                        help='Weight for global coherence')
    parser.add_argument('--mst_top_l_neighbors', type=int, default=None,
                        help='Graph sparsification: each node connects to top-L neighbors (None=complete graph)')
    parser.add_argument('--mst_cache_graph', action='store_true',
                        help='Cache GT adjacency matrix for faster loading')
    parser.add_argument('--use_explicit_maxst', action='store_true',
                        help='Use explicit MaxST calculation (slower but more accurate)')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Device check
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.device == 'cuda':
        print('⚠️ CUDA not available, using CPU')

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Print configuration
    print(f"\n{'='*60}")
    print(f"  Column Hypergraph Search{'+ MST Reranking' if args.use_mst else ''}")
    print(f"{'='*60}")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Model:           {args.model_path}")
    print(f"  Joinable Pairs:  {args.joinable_pairs}")
    print(f"  Top-K:           {args.top_k}")
    print(f"  Device:          {device}")

    if args.use_mst:
        print(f"\n  MST Configuration:")
        print(f"    Candidate Size (B): {args.mst_candidate_size}")
        print(f"    Alpha (GT weight):  {args.mst_alpha}")
        print(f"    Beta (Emb weight):  {args.mst_beta}")
        print(f"    Lambda (Coherence): {args.mst_lambda}")

    print(f"{'='*60}\n")

    # Data paths
    data_dir = f"datasets/Lake/{args.dataset}"
    target_path = f"{data_dir}/target.npy"
    query_path = f"{data_dir}/query.npy"
    target_metadata_path = f"{data_dir}/target_metadata.pkl"
    query_metadata_path = f"{data_dir}/query_metadata.pkl"

    # Load datasets
    print("📂 Loading datasets...")
    target_dataset = TestDatasetHyperJoin(target_path, target_metadata_path)
    query_dataset = TestDatasetHyperJoin(query_path, query_metadata_path)

    print(f" Dataset statistics:")
    print(f"   Target columns: {len(target_dataset)}")
    print(f"   Query columns:  {len(query_dataset)}")

    # Load model
    print(f"\n📥 Loading Enhanced model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['model_config']
    vocab = checkpoint.get('vocab', None)

    # 🔧  vocab
    if vocab is None:
        raise ValueError("Checkpointvocab，tokenization")

    vocab_size = model_config.get('vocab_size', 10000)
    max_vocab_id = max(vocab.values())
    print(f"\n🔍 Vocab:")
    print(f"   Vocabsize: {len(vocab)} token")
    print(f"   vocab_size: {vocab_size}")
    print(f"   VocabmaxID: {max_vocab_id}")

    #  check vocab ID whetherout of range
    if max_vocab_id >= vocab_size:
        print(f"⚠️  : vocabmaxID ({max_vocab_id}) >= vocab_size ({vocab_size})")
        print(f"   this will causeCUDA device-side assert")
        print(f"   fix: map allout of rangeIDto<UNK>")

        # clean vocab: out of range token remove
        cleaned_vocab = {}
        removed_count = 0
        for token, token_id in vocab.items():
            if token_id < vocab_size:
                cleaned_vocab[token] = token_id
            else:
                removed_count += 1

        print(f"   remove {removed_count} out of rangetoken")
        vocab = cleaned_vocab

    print(f" Vocabvalidation passed")

    # 🆕 if missing num_tables，calculate from dataset
    if 'num_tables' not in model_config:
        print(f"⚠️  Warning: model_config missing num_tables, computing from dataset...")
        unique_tables = set()
        target_metas = _get_meta_list(target_dataset)
        query_metas = _get_meta_list(query_dataset)
        for meta in target_metas + query_metas:
            table_name = meta.get('table_name', '').replace('.csv', '')
            if table_name:
                unique_tables.add(table_name)
        model_config['num_tables'] = len(unique_tables)
        print(f"   Computed num_tables: {model_config['num_tables']}")

    # 
    print("🔷 Loading Enhanced HyperJoin model...")
    enhanced_config = {
        'vocab_size': model_config.get('vocab_size', 10000),
        'content_dim': model_config.get('content_dim', 1200),
        'num_tables': model_config.get('num_tables', 20),
        'embed_dim': model_config.get('embed_dim', 512),
        'column_pe_dim': 16,
        'num_patches': 100,
        'patch_gnn_layers': 2,
        'num_mixer_layers': 2,
        'mlp_ratio': 4.0,
        'dropout': model_config.get('dropout', 0.1),
        'use_table_pe': True,
        'use_column_pe': True,
        'use_structure_bias': True,
        'use_edge_type': True
    }

    model = EnhancedHyperJoinModel(**enhanced_config).to(device)
    # Load with strict=False to handle parameter mismatches
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=False
    )
    if unexpected_keys:
        print(f'⚠️ Ignored unexpected keys: {len(unexpected_keys)}')
    model.eval()
    print(' Enhanced model loaded successfully')
    print(f"   enhanced features: Table PE  | Column PE  | Patch GNN  | Mixer ")

    # 🆕 Build table_name → table_id mapping (for Table PE)
    print("\n🔧 Building table mapping...")
    unique_tables = set()
    target_metas = _get_meta_list(target_dataset)
    query_metas = _get_meta_list(query_dataset)
    for meta in target_metas + query_metas:
        table_name = meta.get('table_name', '').replace('.csv', '')
        if table_name:
            unique_tables.add(table_name)
    table_name_to_id = {table_name: idx for idx, table_name in enumerate(sorted(unique_tables))}
    print(f" Table mapping built: {len(table_name_to_id)} tables")

    # Build GT adjacency matrix if using MST
    gt_adjacency = None
    if args.use_mst:
        # Check if GT adjacency is needed
        if args.mst_alpha > 0:
            # Need GT adjacency matrix
            if args.joinable_pairs is None:
                raise ValueError("--joinable_pairs is required when using MST with mst_alpha > 0")

            print('\n Building GT adjacency matrix for MST...')
            cache_path = None
            if args.mst_cache_graph:
                cache_dir = f"datasets/Lake/{args.dataset}/cache"
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = f"{cache_dir}/gt_adjacency.npy"

            graph_builder = GraphBuilderSearch(
                target_dataset,
                args.joinable_pairs,
                only_can=True,
                cache_path=cache_path
            )
            gt_adjacency = graph_builder.build_adjacency_matrix(verbose=True)
        else:
            # Label-Free mode: alpha=0, no GT needed
            print('\n⚡ Label-Free mode: mst_alpha=0, skipping GT adjacency matrix')
            gt_adjacency = None

        # Initialize MST reranker (Label-Free only)
        if args.mst_alpha != 0:
            raise ValueError(f"Only Label-Free mode (alpha=0) is supported. Got alpha={args.mst_alpha}")

        if args.use_explicit_maxst:
            print(f'   Using ExplicitMaxSTReranker (λ={args.mst_lambda}, L={args.mst_top_l_neighbors}) [SLOW]')
            from mst_explicit_maxst import ExplicitMaxSTReranker
            mst_reranker = ExplicitMaxSTReranker(
                lambda_=args.mst_lambda,
                top_l_neighbors=args.mst_top_l_neighbors,
                verbose=False
            )
        else:
            print(f'   Using LabelFreeMSTReranker (λ={args.mst_lambda}, L={args.mst_top_l_neighbors})')
            mst_reranker = LabelFreeMSTReranker(
                lambda_=args.mst_lambda,
                normalize_coherence=True,  # normalize，keep stablecoherencescale
                top_l_neighbors=args.mst_top_l_neighbors,
                verbose=False
            )
        print(' MST reranker initialized')

    # Encode all data
    print("\n🚀 Encoding data...")
    target_embeddings = encode_dataset(model, target_dataset, vocab, table_name_to_id, device, vocab_size, desc='Encoding targets')
    query_embeddings = encode_dataset(model, query_dataset, vocab, table_name_to_id, device, vocab_size, desc='Encoding queries')

    # Quality check
    print(f"\n📐 Embedding quality check:")
    print(f"   Target: shape={tuple(target_embeddings.shape)}, "
          f"mean={target_embeddings.mean().item():.4f}, "
          f"std={target_embeddings.std().item():.4f}")
    print(f"   Query:  shape={tuple(query_embeddings.shape)}, "
          f"mean={query_embeddings.mean().item():.4f}, "
          f"std={query_embeddings.std().item():.4f}")

    # Execute search
    print('\n🔍 Executing search...')
    results = []

    if args.use_mst:
        # MST reranking path
        print(f"   Using MST reranking (B={args.mst_candidate_size} → K={args.top_k})")
        for i in tqdm(range(len(query_embeddings)), desc='Searching with MST'):
            q = query_embeddings[i]

            # Step 1: Retrieve Top-B candidates
            sims = torch.cosine_similarity(q.unsqueeze(0), target_embeddings, dim=1)
            B = min(args.mst_candidate_size, sims.size(0))
            topb_vals, topb_idxs = torch.topk(sims, k=B)

            # Step 2: MST rerank to Top-K (Label-Free only)
            selected = mst_reranker.rerank(
                query_emb=q,
                candidate_embs=target_embeddings[topb_idxs],
                candidate_indices=topb_idxs.cpu().numpy(),
                K=args.top_k
            )
            results.append(selected)
    else:
        # Original path: Direct Top-K
        print(f"   Using direct Top-K retrieval (K={args.top_k})")
        for i in tqdm(range(len(query_embeddings)), desc='Searching'):
            sims = torch.cosine_similarity(
                query_embeddings[i].unsqueeze(0),
                target_embeddings,
                dim=1
            )
            topk = min(args.top_k, sims.size(0))
            vals, idxs = torch.topk(sims, k=topk)
            results.append(idxs.tolist())

    # Evaluation
    print('\n🎯 Evaluating results...')

    # Set method name for detailed logs
    if args.use_mst:
        os.environ['METHOD_NAME'] = 'MST Reranking'
        # Use coherence evaluator
        evaluator = CoherenceEvaluator(args.dataset, gt_adjacency)
        eval_results = evaluator.run_complete_evaluation_with_coherence(results)
    else:
        os.environ['METHOD_NAME'] = 'Baseline (Hypergraph)'
        # Use standard evaluator
        evaluator = DetailedEvaluator(args.dataset)
        eval_results = evaluator.run_complete_evaluation(results)

    # Print results
    print("\n" + "="*60)
    print("  Performance Summary")
    print("="*60)

    key_metrics = [1, 5, 10, 15, 20, 25]

    # Precision@K
    print("\n  Precision@K:")
    for k in key_metrics:
        key = f'Precision@{k}'
        if key in eval_results:
            print(f"    P@{k:2d}: {eval_results[key]:.4f} ({eval_results[key]*100:.2f}%)")

    # Recall@K
    print("\n  Recall@K:")
    for k in key_metrics:
        key = f'Recall@{k}'
        if key in eval_results:
            print(f"    R@{k:2d}: {eval_results[key]:.4f} ({eval_results[key]*100:.2f}%)")

    # F1@K
    print("\n  F1@K:")
    for k in key_metrics:
        key = f'F1@{k}'
        if key in eval_results:
            print(f"    F1@{k:2d}: {eval_results[key]:.4f} ({eval_results[key]*100:.2f}%)")

    # Coherence metrics (if using MST)
    if args.use_mst:
        print("\n  Coherence Metrics:")
        for k in [5, 15, 25]:
            coherence_key = f'Coherence@{k}'
            diversity_key = f'TableDiversity@{k}'
            ccr_key = f'CCR@{k}'
            if coherence_key in eval_results:
                print(f"\n    @ K={k}:")
                print(f"      Coherence:       {eval_results[coherence_key]:.4f}")
                print(f"      Table Diversity: {eval_results[diversity_key]:.4f}")
                print(f"      CCR:             {eval_results[ccr_key]:.4f}")

    print("\n" + "="*60)

    # Save results
    experiment_dir = os.environ.get('EXPERIMENT_DIR', 'results/results_mst')
    os.makedirs(experiment_dir, exist_ok=True)

    suffix = '_mst' if args.use_mst else ''
    out_json = os.path.join(
        experiment_dir,
        f'search_results_{args.dataset}{suffix}.json'
    )

    # Include hyperparameters in saved results
    save_data = {
        'metrics': eval_results,
        'config': {
            'dataset': args.dataset,
            'model_path': args.model_path,
            'top_k': args.top_k,
            'use_mst': args.use_mst,
        }
    }

    if args.use_mst:
        save_data['config'].update({
            'mst_candidate_size': args.mst_candidate_size,
            'mst_alpha': args.mst_alpha,
            'mst_beta': args.mst_beta,
            'mst_lambda': args.mst_lambda,
        })

    with open(out_json, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n Results saved to: {out_json}")
    print(f"🎉 Search evaluation complete!\n")


if __name__ == '__main__':
    main()
