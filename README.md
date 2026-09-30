# HyperJoin
<img width="1107" height="622" alt="image" src="https://github.com/user-attachments/assets/bff044c9-112e-4a01-b9d9-1e3109469309" />

# HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery

## Description

This repository provides the codebase for the paper "HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery".

## Artifact Status

This repository contains the main source code used for the HyperJoin paper: column-level hypergraph construction, the HIN model, label-free self-supervised training, MST reranking, and evaluation. It is provided as a research artifact; dataset preprocessing and full end-to-end reproduction require the benchmark files described below.

## Quick Demo

A self-contained smoke test ships in `test/`. It runs the full chain — data generation, training, MST-reranked search, evaluation — on five tiny sample tables:

```bash
python test/run_demo.py
```

See `test/README.md` for what each file does and how to plug in your own tables or enable LLM perturbation.

## Dataset

We provide the dataset used in this project here: **[UK_SG](https://drive.google.com/drive/folders/1Ttb6nrt05J6VG2FdF6_jEKoU-zWpSyL2?usp=sharing)**.

The scripts expect the downloaded dataset under `datasets/Lake/<dataset>/`, including `target.npy`, `query.npy`, `target_metadata.pkl`, `query_metadata.pkl`, and either `self_supervised_pairs.pkl` or the corresponding joinable-pair CSV/index files. Pass `--data_root` to `train.py`/`search.py` or set `HYPERJOIN_DATA_ROOT` to use a different location.

## 1. Generate Label-Free Training Pairs (Optional)

`src/datagen.py` builds self-supervised joinable pairs by table splitting and column perturbation; `src/llm_augmentation.py` generates the LLM-augmented inter-table descriptions used for hyperedge construction. Skip this step if `self_supervised_pairs.pkl` already exists under `datasets/Lake/<dataset>/`.

```bash
python src/datagen.py --datasets <name> --data_dir <raw_tables_dir> --use_llm 0
```

Rule-based perturbation is used by default. Pass `--use_llm 1` and set `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_ID` to enable LLM perturbation (see `src/llm_augmentation.py`).

## 2. Train the Model

```bash
python src/hypergraph/train.py \
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
    --edge_mask_ratio 0.2 \
    --hard_neg_ratio 1.0 \
    --epochs 30 \
    --batch_size 64 \
    --device cuda \
    --output_dir ./results/UK_SG
```

## 3. Search for Joinable Columns

After training, run the search pipeline to discover joinable columns:

```bash
python src/hypergraph/search.py \
    --dataset UK_SG \
    --model_path ./results/UK_SG/UK_SG_LabelFree/best_model.pth \
    --use_mst \
    --mst_alpha 0.0 \
    --mst_lambda 1.0 \
    --mst_candidate_size 50 \
    --mst_top_l_neighbors 20 \
    --top_k 25 \
    --device cuda \
    --seed 42
```

The search module outputs evaluation metrics including Precision@K, Recall@K, and F1@K, plus per-query logs under `results/`.

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
│   ├── hypergraph/                            # Core model & pipeline
│   │   ├── construction.py                    # Hypergraph construction, batching, losses
│   │   ├── model.py                           # EnhancedHyperJoinModel (HIN)
│   │   ├── layers.py                          # Positional encoding & mixer layers
│   │   ├── intra_edge_gnn.py                  # Intra-hyperedge GNN encoder
│   │   ├── train.py                           # Label-free self-supervised training
│   │   └── search.py                          # Search + MST reranking
│   ├── mst_reranker/                          # Coherence-aware reranking
│   │   ├── mst.py                             # Greedy label-free MST reranker
│   │   ├── explicit_maxst.py                  # Explicit MaxST variant
│   │   ├── graph_builder.py                   # Joinability graph builder
│   │   └── coherence_eval.py                  # Coherence metrics
│   ├── datagen.py                             # Label-free pair generation
│   ├── llm_augmentation.py                    # LLM-augmented inter-table hyperedges
│   ├── perturbation_cache.py                  # Column perturbation cache
│   ├── data.py                                # Dataset classes
│   ├── evaluator.py                           # Precision/Recall/F1 evaluation
│   └── utils.py                               # Utility functions
├── test/                                      # Self-contained demo (see test/README.md)
├── full_version/                              # Online full paper with appendix
└── datasets/Lake/                             # Expected dataset root (not included)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy, pandas, scipy, tqdm
- `openai` (optional, only for `llm_augmentation.py`)
- `fasttext` (optional, only for `datagen.py`; falls back to deterministic embeddings without it)
- `scikit-learn` + `matplotlib` (optional, only for the embedding-visualisation helper)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026hyperjoin,
  author    = {Liu, Shiyuan and Wang, Jianwei and Lin, Xuemin and Qin, Lu and Zhang, Wenjie and Zhang, Ying},
  title     = {HyperJoin: {LLM}-augmented Hypergraph Link Prediction for Joinable Table Discovery},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {19},
  number    = {13},
  year      = {2026},
  url       = {https://github.com/T-Lab/HyperJoin}
}
```

