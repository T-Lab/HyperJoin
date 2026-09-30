# Demo / Smoke Test

A minimal end-to-end run of the HyperJoin pipeline on a tiny bundled dataset.
Run from the repository root:

```bash
python test/run_demo.py
```

It executes the full chain — label-free data generation, model training
(3 epochs on CPU), and search with MST reranking — then prints
Precision/Recall/F1@K. Results land in `results/` and `datasets/Lake/DEMO*/`
(both git-ignored).

## What this folder contains

The files below are exactly the artefacts you need to prepare for your own
dataset:

| File | Role |
|---|---|
| `tables/*.csv` | Raw source tables (header row + >= 10 data rows). Input for label-free training-pair generation. |
| `query.csv` | Evaluation queries — **one CSV row per query column's cell values**. |
| `target.csv` | Search pool — **one CSV row per target column's cell values**. |
| `index.csv` | Ground truth — row *i* lists the target-column indices that are correct answers for query *i*. |
| `query_metadata.json` | Display names (`table_name`, `column_name`) for each query row. |
| `target_metadata.json` | Same, for the target pool. |
| `run_demo.py` | Converts the above into `datasets/Lake/DEMO`, runs `datagen.py`, `train.py`, `search.py`. |

In the demo ground truth, query 0 (person names) is joinable to target columns
0 and 1; queries 1–4 (cities / emails / products / customer ids) each map to
one target column; target columns 6–14 are distractors.

## LLM perturbation

Perturbation defaults to the built-in rule-based perturber so the demo runs
with no API access. To enable LLM perturbation, edit `USE_LLM = True` in
`run_demo.py` and export:

```bash
export LLM_BASE_URL="https://api.studio.nebius.com/v1/"
export LLM_API_KEY="<your-api-key>"
export LLM_MODEL_ID="deepseek-ai/DeepSeek-V3.2"
```
