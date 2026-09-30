"""
End-to-end smoke test for the HyperJoin pipeline on the bundled demo data.

Run from anywhere:

    python test/run_demo.py

What it does
------------
1. Materialises a tiny evaluation dataset under ``datasets/Lake/DEMO`` from the
   human-readable files in ``test/``:

       query.csv / target.csv      one CSV row per column's cell values
       query_metadata.json         table/column display names (-> .pkl)
       target_metadata.json
       index.csv                   ground truth: row i = target indices of query i

2. Runs ``src/datagen.py`` on the raw tables in ``test/tables/`` to build the
   label-free training dataset ``datasets/Lake/DEMO_LabelFree``
   (each table needs a header row and >= 10 data rows).

3. Trains ``EnhancedHyperJoinModel`` for a few epochs on CPU and saves
   ``results/demo/DEMO_LabelFree/best_model.pth``.

4. Runs ``src/hypergraph/search.py`` with MST reranking and prints
   Precision/Recall/F1@K for K in {1, 5, 10, 15, 20, 25}.

LLM perturbation is OFF by default: column-name variants and value noise are
produced by the built-in rule-based perturber. To use the LLM variant instead,
set ``USE_LLM = True`` below and export:

    export LLM_BASE_URL="https://api.studio.nebius.com/v1/"
    export LLM_API_KEY="<your-api-key>"        # <- replace with your own key
    export LLM_MODEL_ID="deepseek-ai/DeepSeek-V3.2"
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_LLM = False   # set True to enable LLM perturbation (requires the LLM_* env vars above)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / 'test'
SRC_DIR = REPO_ROOT / 'src'
DATA_DIR = TEST_DIR / 'data'        # all generated datasets live here (git-ignored)
DEMO_DIR = DATA_DIR / 'DEMO'
RESULT_DIR = REPO_ROOT / 'results' / 'demo'

sys.path.insert(0, str(SRC_DIR))
from datagen import text2vec  # noqa: E402  (low-level CSV -> .npy helper)


def run(cmd, **kwargs):
    """Run a pipeline step as a subprocess from the repo root (paths are relative to it)."""
    print('\n' + '=' * 78)
    print('$ ' + ' '.join(cmd))
    print('=' * 78)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, **kwargs)
    if proc.returncode != 0:
        sys.exit(f'Step failed (exit {proc.returncode}): {cmd[1]}')


def write_metadata_pkl(json_path: Path, pkl_path: Path):
    """Convert the human-readable JSON metadata into the pipeline's .pkl format."""
    meta = json.loads(json_path.read_text())['metadata']
    table_names = sorted({m['table_name'] for m in meta})
    column_names = sorted({m['column_name'] for m in meta})
    payload = {
        'metadata': meta,
        'table_to_id': {name: i for i, name in enumerate(table_names)},
        'column_to_id': {name: i for i, name in enumerate(column_names)},
        'total_tables': len(table_names),
        'total_columns': len(column_names),
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f)


def prepare_eval_dataset():
    """Step 1: build datasets/Lake/DEMO from the test/ files."""
    print('\n' + '=' * 78)
    print('Step 1/4  Preparing the evaluation dataset (test/data/DEMO)')
    print('=' * 78)

    gt_dir = DEMO_DIR / 't=0.2' / 'test'
    gt_dir.mkdir(parents=True, exist_ok=True)

    for name in ('query.csv', 'target.csv', 'index.csv'):
        shutil.copy2(TEST_DIR / name, gt_dir / name if name == 'index.csv' else DEMO_DIR / name)

    write_metadata_pkl(TEST_DIR / 'query_metadata.json', DEMO_DIR / 'query_metadata.pkl')
    write_metadata_pkl(TEST_DIR / 'target_metadata.json', DEMO_DIR / 'target_metadata.pkl')

    # Embed the columns (real FastText vectors if installed, deterministic
    # hash-seeded embeddings otherwise -- identical strings map to identical
    # vectors either way, so the demo works offline).
    text2vec(str(DEMO_DIR / 'query.csv'), str(DEMO_DIR / 'query.npy'), plm='fasttext')
    text2vec(str(DEMO_DIR / 'target.csv'), str(DEMO_DIR / 'target.npy'), plm='fasttext')
    print('  test/data/DEMO is ready')


def generate_training_data():
    """Step 2: label-free pair generation from the raw tables in test/tables/."""
    print('\n' + '=' * 78)
    print('Step 2/4  Generating label-free training pairs (test/data/DEMO_LabelFree)')
    if not USE_LLM:
        print('  LLM perturbation disabled -> rule-based perturbation is used.')
        print('  Set USE_LLM = True in test/run_demo.py and export LLM_API_KEY to enable it.')
    print('=' * 78)
    run([
        sys.executable, 'src/datagen.py',
        '--datasets', 'DEMO',
        '--data_dir', str(TEST_DIR / 'tables'),
        '--output_dir', str(DATA_DIR / 'DEMO'),
        '--use_llm', '1' if USE_LLM else '0',
        '--type', 'mat',
    ])


def train_model():
    """Step 3: short training run on the generated label-free dataset."""
    print('\n' + '=' * 78)
    print('Step 3/4  Training EnhancedHyperJoinModel (demo: 3 epochs, CPU)')
    print('=' * 78)
    run([
        sys.executable, 'src/hypergraph/train.py',
        '--dataset', 'DEMO_LabelFree',
        '--data_root', str(DATA_DIR.relative_to(REPO_ROOT)),
        '--epochs', '3',
        '--batch_size', '16',
        '--num_workers', '0',
        '--device', 'cpu',
        '--output_dir', str(RESULT_DIR.relative_to(REPO_ROOT)),
    ])


def search():
    """Step 4: search the DEMO dataset with MST reranking and evaluate."""
    print('\n' + '=' * 78)
    print('Step 4/4  Searching joinable columns (MST reranking) + evaluation')
    print('=' * 78)
    ckpt = RESULT_DIR / 'DEMO_LabelFree' / 'best_model.pth'
    env = dict(os.environ, EXPERIMENT_DIR=str(RESULT_DIR))
    run([
        sys.executable, 'src/hypergraph/search.py',
        '--dataset', 'DEMO',
        '--data_root', str(DATA_DIR.relative_to(REPO_ROOT)),
        '--model_path', str(ckpt),
        '--use_mst',
        '--mst_alpha', '0.0',
        '--mst_lambda', '1.0',
        '--mst_candidate_size', '10',
        '--mst_top_l_neighbors', '5',
        '--top_k', '15',
        '--device', 'cpu',
        '--seed', '42',
    ], env=env)


def main():
    print('=' * 78)
    print(' HyperJoin demo pipeline')
    print('   raw tables   : test/tables/*.csv   (what readers prepare for real data)')
    print('   eval dataset : test/data/DEMO  (query/target/index built from test/)')
    print('   train dataset: test/data/DEMO_LabelFree (generated)')
    print('=' * 78)

    prepare_eval_dataset()
    generate_training_data()
    train_model()
    search()

    print('\n' + '=' * 78)
    print(' Demo finished.')
    print(f'   checkpoint : {RESULT_DIR / "DEMO_LabelFree" / "best_model.pth"}')
    print(f'   metrics    : {RESULT_DIR}/search_results_DEMO_mst.json')
    print('=' * 78)


if __name__ == '__main__':
    main()
