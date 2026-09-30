"""
Detailed evaluation module
=============

Provides ``DetailedEvaluator``, run after a search script produces top-K results:

  1. Load labelled relations from ``ground_truth_pairs.csv`` / ``joinable_pairs.csv``
  2. Compute Precision@K / Recall@K / F1@K for K in {1, 5, 10, 15, 20, 25}
  3. Print detailed hit/miss lists per (query, top-K)
  4. Return a flattened metrics dict for the caller

Called by ``src/hypergraph/search.py`` as the single evaluation entry point.
"""

import os
import io
import csv
import numpy as np
import pandas as pd
import pickle


class DetailedEvaluator:
    """Detailed evaluator - full search-result analysis."""

    def __init__(self, datasets, data_root=None):
        self.datasets = datasets
        # Dataset root: constructor arg > HYPERJOIN_DATA_ROOT env var > default
        self.data_root = data_root or os.environ.get('HYPERJOIN_DATA_ROOT', 'datasets/Lake')
        self.query_real_data = []
        self.target_real_data = []
        self.query_names = None  # display names from the metadata
        self.target_names = None
        self._load_real_data()
        self._load_metadata_names()

    def _load_real_data(self):
        """Load the real CSV contents for detailed display (proper CSV parsing)."""
        try:
            def read_csv_columns(csv_path: str):
                rows = []
                if not os.path.exists(csv_path):
                    return rows
                # Strip NUL chars; use the csv module so quoted newlines are supported
                with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                    content = f.read().replace('\x00', '')
                sio = io.StringIO(content)
                reader = csv.reader(sio, delimiter=',', quotechar='"', skipinitialspace=True)
                for row in reader:
                    # Normalise: strip whitespace, drop fully empty rows
                    cleaned = [c.strip() for c in row]
                    if any(c != '' for c in cleaned):
                        rows.append(cleaned)
                return rows

            # Load the real query data
            query_csv_path = f'{self.data_root}/{self.datasets}/query.csv'
            print(f" Loading query data: {query_csv_path}")
            if os.path.exists(query_csv_path):
                self.query_real_data = read_csv_columns(query_csv_path)
                print(f" Loaded {len(self.query_real_data)} query columns")
            else:
                print(f"Query data file not found: {query_csv_path}")

            # Load the real target data
            target_csv_path = f'{self.data_root}/{self.datasets}/target.csv'
            print(f" Loading target data: {target_csv_path}")
            if os.path.exists(target_csv_path):
                self.target_real_data = read_csv_columns(target_csv_path)
                print(f" Loaded {len(self.target_real_data)} target columns")
            else:
                print(f"Target data file not found: {target_csv_path}")

            # Consistency check against the NPY column counts
            try:
                target_npy_path = f'{self.data_root}/{self.datasets}/target.npy'
                query_npy_path = f'{self.data_root}/{self.datasets}/query.npy'
                if os.path.exists(target_npy_path):
                    target_npy_len = len(np.load(target_npy_path, allow_pickle=True))
                    if target_npy_len != len(self.target_real_data):
                        print(
                            f"Warning: parsed target.csv columns ({len(self.target_real_data)}) do not match target.npy items ({target_npy_len})."
                        )
                if os.path.exists(query_npy_path):
                    query_npy_len = len(np.load(query_npy_path, allow_pickle=True))
                    if query_npy_len != len(self.query_real_data):
                        print(
                            f"Warning: parsed query.csv columns ({len(self.query_real_data)}) do not match query.npy items ({query_npy_len})."
                        )
            except Exception as e_chk:
                print(f"Consistency check failed: {e_chk}")

        except Exception as e:
            print(f"Failed to load real data: {e}")

    def _load_metadata_names(self):
        """Load table.column display names from the metadata PKL (list or dict)."""
        try:
            def build_names(meta_obj):
                # meta_obj may be a dict ({'metadata': [...]}) or a plain list
                items = []
                if isinstance(meta_obj, dict):
                    items = meta_obj.get('metadata', [])
                elif isinstance(meta_obj, list):
                    items = meta_obj
                names_local = []
                for item in items:
                    # item is expected to be a dict
                    if isinstance(item, dict):
                        tname = str(item.get('table_name', ''))
                        cname = str(item.get('column_name', ''))
                    else:
                        # Unrecognised; fall back to empty
                        tname, cname = '', ''
                    if tname and cname:
                        names_local.append(f"{tname}.{cname}")
                    elif cname:
                        names_local.append(cname)
                    elif tname:
                        names_local.append(tname)
                    else:
                        names_local.append('')
                return names_local

            base = f'{self.data_root}/{self.datasets}'
            # Target metadata
            target_meta_pkl = os.path.join(base, 'target_metadata.pkl')
            if os.path.exists(target_meta_pkl):
                with open(target_meta_pkl, 'rb') as f:
                    meta = pickle.load(f)
                names = build_names(meta)
                self.target_names = names if names else None
                if self.target_names is not None:
                    print(f" Loaded target metadata names: {len(self.target_names)} columns")
                    if self.target_real_data and len(self.target_names) != len(self.target_real_data):
                        print(
                            f"Warning: number of target names ({len(self.target_names)}) does not match target.csv columns ({len(self.target_real_data)})"
                        )
            # Query metadata (may not exist)
            query_meta_pkl = os.path.join(base, 'query_metadata.pkl')
            if os.path.exists(query_meta_pkl):
                with open(query_meta_pkl, 'rb') as f:
                    meta = pickle.load(f)
                names = build_names(meta)
                self.query_names = names if names else None
                print(f" Loaded query metadata names: {len(self.query_names)} columns")
                if self.query_real_data and len(self.query_names) != len(self.query_real_data):
                    print(
                        f"Warning: number of query names ({len(self.query_names)}) does not match query.csv columns ({len(self.query_real_data)})"
                    )
        except Exception as e:
            print(f"Failed to load metadata names: {e}")

    def _get_column_info(self, index, is_query=True):
        """Return column info (prefer "table.column" names from the metadata PKL)."""
        try:
            # Build the column-name info
            data_type = "query" if is_query else "target"

            # Prefer the metadata names
            names = self.query_names if is_query else self.target_names
            if names is not None and 0 <= index < len(names) and names[index]:
                return names[index]

            # Fall back to the metadata text files
            metadata_file = f'{self.data_root}/{self.datasets}/{"query" if is_query else "target"}_metadata.txt'

            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    lines = f.readlines()
                    if index < len(lines):
                        column_name = lines[index].strip()
                        return column_name

            # Otherwise infer from the CAN dataset naming scheme
            if self.datasets.startswith('CAN'):
                # Infer from the CAN naming convention
                return f"CAN_Column_{index}"

            return f"{data_type}_Column_{index}"
        except Exception:
            return f"Column_{index}"

    def _get_sample_data(self, index, is_query=True):
        """Return sample values of a column (non-empty samples from the parsed records)."""
        try:
            data_list = self.query_real_data if is_query else self.target_real_data
            if 0 <= index < len(data_list):
                row = data_list[index]
                # Filter invalid/placeholder strings
                candidates = [v for v in row if v and str(v).strip().lower() not in ('nan', 'none', 'null', 'na')]
                # Return the first 3 samples to keep the output short
                return candidates[:3] if candidates else ['N/A']
            return ['N/A']
        except Exception:
            return ['N/A']

    def _get_pipeline_log_dir(self):
        """Return the pipeline log dir - detailed_logs under the experiment dir."""
        # Experiment dir from the environment, else a default
        experiment_dir = os.environ.get('EXPERIMENT_DIR', 'results/baseline_v2')

        # Create the detailed_logs subdirectory under it
        log_dir = f"{experiment_dir}/detailed_logs"

        # Create the directory
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _get_method_name(self):
        """Infer the method name from env vars or the experiment dir."""
        method = os.environ.get('METHOD_NAME', None)
        if method:
            return method

        # Infer from the experiment dir
        experiment_dir = os.environ.get('EXPERIMENT_DIR', '')
        if 'mst' in experiment_dir.lower():
            return 'MST Reranking'
        elif 'baseline' in experiment_dir.lower():
            return 'Baseline'
        elif 'hypergraph' in experiment_dir.lower():
            return 'Hypergraph'
        else:
            return 'Unknown Method'

    def load_ground_truth(self):
        """Load the ground truth - compatible with Snoopy's index.csv format."""
        print(f" Loading ground truth...")

        index_path = f"{self.data_root}/{self.datasets}/t=0.2/test/index.csv"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Ground-truth file does not exist: {index_path}")

        print(f" Reading file: {index_path}")

        # Read with pandas; the CSV is irregular (rows may have different field counts)
        try:
            df = pd.read_csv(index_path, header=None, sep=',', engine='python')
        except pd.errors.ParserError:
            # Fall back to manual line-by-line reading if pandas cannot parse it
            import csv
            rows = []
            with open(index_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)

            # Find the maximum column count
            max_cols = max(len(row) for row in rows) if rows else 0

            # Pad shorter rows
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

            df = pd.DataFrame(rows)
        print(f" Ground-truth shape: {df.shape}")

        # Build the ground-truth dictionary
        ground_truth = {}

        for gt_row_idx in range(len(df)):
            query_idx = gt_row_idx
            row = df.iloc[gt_row_idx]

            # Collect all non-empty target indices
            target_indices = []
            for col_name in df.columns:
                value = row[col_name]
                if pd.notna(value) and str(value).strip() != '':
                    try:
                        target_indices.append(int(float(value)))
                    except (ValueError, TypeError):
                        # Skip values that cannot be parsed as numbers
                        continue

            if target_indices:
                ground_truth[query_idx] = target_indices

        print(f" Ground truth loaded: {len(ground_truth)} queries have real answers")

        # Per-query GT count statistics
        gt_counts = [len(ground_truth[i]) for i in ground_truth.keys()]
        if gt_counts:
            print(f" Ground-truth stats: avg {np.mean(gt_counts):.1f} targets/query, "
                  f"min {min(gt_counts)}, max {max(gt_counts)}")

        return ground_truth

    def evaluate_precision_recall(self, search_results, ground_truth, topk_list=[1, 5, 10, 15, 20, 25]):
        """
        Compute Precision@K and Recall@K together (standard IR definitions).

        Precision@K:
        - formula: sum(|Top-K preds ∩ full GT| / K) / num_queries
        - meaning: how many of the Top-K predictions are true GT
        - denominator: fixed K (number of retrieved results)

        Recall@K:
        - formula: sum(|Top-K preds ∩ full GT| / |GT|) / num_queries
        - meaning: what fraction of the full GT is recalled
        - denominator: fixed |GT| (total number of relevant items)
        - note: min(K, |GT|) is NOT used, per the standard IR definition

        Definition (as in the paper):
        P@k = |T_g ∩ T_q| / |T_q|
        R@k = |T_g ∩ T_q| / |T_g|

        Args:
            search_results: List[List[int]] - Top-25 predicted indices per query
            ground_truth: Dict[int, List[int]] - all GT indices per query (unordered)
            topk_list: List[int] - K values to evaluate

        Returns:
            dict: {
                'recall': {1: 0.033, 5: 0.16, ...},
                'precision': {1: 0.87, 5: 0.79, ...},
                'f1': {1: 0.06, 5: 0.27, ...}
            }
        """
        print(f"\n Computing Precision@K & Recall@K...")

        results = {'recall': {}, 'precision': {}, 'f1': {}}
        max_k = max(topk_list)

        # Use the pipeline log directory
        log_dir = self._get_pipeline_log_dir()
        print(f" Detailed logs saved to: {log_dir}")

        # GT-count distribution statistics
        gt_counts = []
        for query_idx in range(len(search_results)):
            if query_idx in ground_truth:
                gt_counts.append(len(ground_truth[query_idx]))

        if gt_counts:
            print(f"\n Ground-truth count statistics:")
            print(f"   Queries: {len(gt_counts)}")
            print(f"   GT counts: min={min(gt_counts)}, max={max(gt_counts)}, avg={sum(gt_counts)/len(gt_counts):.2f}")
            print(f"   Median: {sorted(gt_counts)[len(gt_counts)//2]}")

        # Compute the metrics for every K
        for kk in topk_list:
            recall_sum = 0.0
            precision_sum = 0.0
            num_queries = 0

            # Create the detailed log file for this K
            log_file = os.path.join(log_dir, f"precision_recall_k{kk}_details.log")

            # Get the method name
            method_name = self._get_method_name()

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Precision@{kk} & Recall@{kk} detailed computation (standard IR definitions) ===\n")
                f.write(f"Method: {method_name}\n")
                f.write(f"Dataset: {self.datasets}\n")
                f.write(f"Queries: {len(search_results)}\n\n")
                f.write(f"Standard IR definitions (as in the paper):\n")
                f.write(f"  P@k = |T_g ∩ T_q| / |T_q|\n")
                f.write(f"  R@k = |T_g ∩ T_q| / |T_g|\n\n")
                f.write(f"Precision@{kk} formula: sum(|Top-{kk} ∩ full GT| / {kk}) / num_queries\n")
                f.write(f"Recall@{kk} formula: sum(|Top-{kk} ∩ full GT| / |GT|) / num_queries\n\n")
                f.write("="*80 + "\n\n")

                for query_idx in range(len(search_results)):
                    if query_idx in ground_truth:
                        # Top-K predictions
                        pred_topk = search_results[query_idx][:kk]
                        pred_set = set(pred_topk)

                        # GT handling (unordered; use the full GT set)
                        gt_all = ground_truth[query_idx]
                        gt_full = set(gt_all)  # full GT set (used by both Recall and Precision)

                        # Hit count (both metrics intersect with the full GT)
                        hits = len(pred_set & gt_full)  # intersection with the full GT

                        # Recall@K (standard IR: denominator fixed to |GT|)
                        recall_denominator = len(gt_all)  # no min(); always |GT|
                        query_recall = hits / recall_denominator if recall_denominator > 0 else 0.0

                        # Precision@K (denominator fixed to K)
                        query_precision = hits / kk

                        recall_sum += query_recall
                        precision_sum += query_precision
                        num_queries += 1

                        # Write the detailed log
                        f.write(f"Query {query_idx}:\n")

                        # Show query-column info
                        if query_idx < len(self.query_names):
                            query_name = self.query_names[query_idx]
                            f.write(f"  Query column: {query_name}\n")
                            if query_idx < len(self.query_real_data):
                                query_samples = self.query_real_data[query_idx][:3]  # first 3 samples
                                f.write(f"  Query samples: {query_samples}\n")

                        f.write(f"  Total GT: {len(gt_all)}\n")
                        f.write(f"  Top-{kk} predictions: {pred_topk}\n")
                        f.write(f"  Hits vs full GT: {hits}/{len(gt_all)} = {query_recall:.4f} (Recall)\n")
                        f.write(f"  Hits vs full GT: {hits}/{kk} = {query_precision:.4f} (Precision)\n")
                        f.write(f"  Hit indices: {sorted(pred_set & gt_full)}\n")

                        # Show GT details (all ground truth)
                        f.write(f"\n  GT details (all {len(gt_all)}):\n")
                        for gt_rank, gt_idx in enumerate(sorted(gt_all), 1):
                            is_predicted = gt_idx in pred_set
                            pred_mark = "" if is_predicted else ""

                            if gt_idx < len(self.target_names):
                                gt_name = self.target_names[gt_idx]
                                if gt_idx < len(self.target_real_data):
                                    gt_samples = self.target_real_data[gt_idx][:3]
                                    f.write(f"    [{gt_rank:2d}] {pred_mark} [{gt_idx:4d}] {gt_name}: {gt_samples}\n")
                                else:
                                    f.write(f"    [{gt_rank:2d}] {pred_mark} [{gt_idx:4d}] {gt_name}\n")
                            else:
                                f.write(f"    [{gt_rank:2d}] {pred_mark} [{gt_idx:4d}]\n")

                        # Show Top-K prediction details
                        f.write(f"\n  Prediction details (Top-{len(pred_topk)}):\n")
                        for rank, target_idx in enumerate(pred_topk, 1):
                            is_hit = target_idx in gt_full
                            hit_mark = "" if is_hit else ""

                            if target_idx < len(self.target_names):
                                target_name = self.target_names[target_idx]
                                if target_idx < len(self.target_real_data):
                                    target_samples = self.target_real_data[target_idx][:3]
                                    f.write(f"    [{rank:2d}] {hit_mark} [{target_idx:4d}] {target_name}: {target_samples}\n")
                                else:
                                    f.write(f"    [{rank:2d}] {hit_mark} [{target_idx:4d}] {target_name}\n")
                            else:
                                f.write(f"    [{rank:2d}] {hit_mark} [{target_idx:4d}]\n")

                        f.write("\n")

            # Compute averages
            avg_recall = recall_sum / num_queries if num_queries > 0 else 0.0
            avg_precision = precision_sum / num_queries if num_queries > 0 else 0.0

            # F1 score
            if avg_recall + avg_precision > 0:
                f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
            else:
                f1 = 0.0

            results['recall'][kk] = avg_recall
            results['precision'][kk] = avg_precision
            results['f1'][kk] = f1

        # Print the summary table
        print(f"\n{'='*70}")
        print(f"{'K':<5} {'Precision@K':<18} {'Recall@K':<18} {'F1@K':<18}")
        print(f"{'='*70}")
        for kk in topk_list:
            print(f"{kk:<5} {results['precision'][kk]:<18.4f} {results['recall'][kk]:<18.4f} {results['f1'][kk]:<18.4f}")
        print(f"{'='*70}\n")

        # Save the summary file
        summary_file = os.path.join(log_dir, "precision_recall_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Precision@K & Recall@K summary ===\n")
            f.write(f"Dataset: {self.datasets}\n")
            f.write(f"Queries: {num_queries}\n\n")
            f.write(f"{'K':<5} {'Precision@K':<18} {'Recall@K':<18} {'F1@K':<18}\n")
            f.write("="*70 + "\n")
            for kk in topk_list:
                f.write(f"{kk:<5} {results['precision'][kk]:<18.4f} {results['recall'][kk]:<18.4f} {results['f1'][kk]:<18.4f}\n")

        print(f" Summary saved: {summary_file}")

        return results


    def detailed_analysis(self, search_results, ground_truth, num_examples=None):
        """Brief analysis - the details live in the separate log files."""

        print(f"\n=== Search results summary ===")
        print(f"Queries: {len(search_results)}")
        print(f"Target dataset size: {len(self.target_real_data)} columns")

        # Hit statistics
        total_hits = 0
        queries_with_hits = 0

        for i in range(len(search_results)):
            if i in ground_truth:
                pred_indices = search_results[i][:25]
                hits = [idx for idx in pred_indices if idx in ground_truth[i]]
                total_hits += len(hits)
                if len(hits) > 0:
                    queries_with_hits += 1

        print(f"Total hits: {total_hits}")
        print(f"Queries with hits: {queries_with_hits}/{len(search_results)}")
        print(f"Avg hits per query: {total_hits/len(search_results):.2f}")

        print(f"\n Detailed per-query analysis was saved to separate log files")

    def run_complete_evaluation(self, search_results):
        """Run the full evaluation pipeline."""
        print(f"\n Starting the full evaluation pipeline...")

        # Load the ground truth
        ground_truth = self.load_ground_truth()

        # Compute Precision@K and Recall@K
        metrics = self.evaluate_precision_recall(search_results, ground_truth)

        # Detailed analysis
        self.detailed_analysis(search_results, ground_truth)

        # External callers (e.g. the search script) expect a flat dict
        #   {'Precision@k': ..., 'Recall@k': ..., 'F1@k': ...}
        # while evaluate_precision_recall returns a nested dict
        #   {'precision': {k: v, ...}, 'recall': {...}, 'f1': {...}}
        # Flatten it here to keep the external interface intact
        flat_results = {}
        for k in metrics['precision'].keys():
            flat_results[f'Precision@{k}'] = metrics['precision'][k]
            flat_results[f'Recall@{k}'] = metrics['recall'][k]
            flat_results[f'F1@{k}'] = metrics['f1'][k]

        return flat_results


def enhance_search_output(datasets, search_results):
    """Main entry for enhanced search-result analysis."""
    print(f"\n Starting enhanced search-result analysis...")
    print(f"Dataset: {datasets}")
    print(f"Number of search results: {len(search_results)}")

    # Create the detailed evaluator
    evaluator = DetailedEvaluator(datasets)

    # Run the full evaluation
    results = evaluator.run_complete_evaluation(search_results)

    print(f"\n Detailed evaluation complete!")

    return results
