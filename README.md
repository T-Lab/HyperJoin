# HyperJoin
<img width="1107" height="622" alt="image" src="https://github.com/user-attachments/assets/bff044c9-112e-4a01-b9d9-1e3109469309" />

# HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery

## Description

This repository provides the codebase for the paper "HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery".

## 1. Train the Model

```bash
python src/train.py \
    --dataset UK_SG_LabelFree \
    --loss_type triplet \
    --lr 4e-4 \
    --dropout 0.05 \
    --embed_dim 512 \
    --patch_gnn_layers 2 \
    --num_mixer_layers 2 \
    --num_layers 1 \
    --use_residual 1 \
    --margin 1.0 \
    --hard_neg_ratio 1.0 \
    --epochs 30 \
    --batch_size 64 \
    --device cuda \
    --output_dir ./results/UK_SG
```

## 2. Search for Joinable Columns

After training, run the search pipeline to discover joinable columns:

```bash
python src/search.py \
    --dataset UK_SG \
    --model_path ./results/UK_SG/best_model.pth \
    --use_mst \
    --mst_alpha 0.0 \
    --mst_lambda 1.0 \
    --mst_candidate_size 50 \
    --mst_top_l_neighbors 20 \
    --top_k 25 \
    --device cuda \
    --seed 42
```

The search module outputs evaluation metrics including Precision@K, Recall@K.

## Dataset

We provide the dataset used in this project here: **[UK_SG / UK_SG_LabelFree](DATASET_LINK)**.

## Key Components

**Hypergraph Construction:**
- Intra-table hyperedges: Connect columns within the same table
- Inter-table hyperedges: Connect joinable columns across tables using LLM-augmented data generation
- Formulates joinable table discovery as link prediction on the constructed hypergraph

**Hierarchical Interaction Network (HIN):**
- Text and content encoders for column representation
- Three-level positional encoding (Table PE + Column PE)
- Patch GNN encoder: Local message passing for intra-hyperedge aggregation
- Hypergraph-aware Mixer: Global message passing for inter-hyperedge interaction

**Label-Free Self-Supervised Learning:**
- Table splitting and column perturbation for automatic training data generation
- Triplet loss with hard negative mining
- No manual annotations required

**Coherence-Aware Reranking:**
- Maximum Spanning Tree (MST) algorithm for pruning noisy connections
- Balances query-candidate relevance and inter-candidate coherence
- Produces internally consistent result sets

## Project Structure

```
HyperJoin/
├── src/
│   ├── train.py           # Model training
│   ├── search.py          # Joinable column search
│   ├── model.py           # Hypergraph neural network model
│   ├── model_base.py      # Base model components
│   ├── reranker.py        # MST reranking
│   ├── data.py            # Data loaders
│   ├── evaluator.py       # Evaluation metrics
│   └── utils.py           # Utility functions
└── datasets_*/            # Dataset directories
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- numpy
- scikit-learn
- networkx

Install dependencies:
```bash
pip install torch transformers numpy scikit-learn networkx
```

