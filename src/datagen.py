"""
HyperJoin Label-Free Data Generator
===============================

Following OmniMatch's self-supervised table-splitting idea, this generates positive
and negative column pairs for ``EnhancedHyperJoinModel`` without manual join labels.

Pipeline:
  1. Read the raw CSV tables of a dataset
  2. Row-split each table (self-split) into two views that form positive pairs
  3. Optionally perturb column values via ``llm_augmentation`` (enabled by ``--use_llm``)
  4. Embed column data with FastText into ``target.npy``, ``query.npy``, ``anchor.npy``, ``auglist.npy``
  5. Write ``metadata.pkl`` (loaded by data.TestDatasetHyperJoin)

This file is the single entry point for data generation; the low-level helpers
(``fix_seed``, ``normalize_rows``, ``text2vec``, ``augment_mat_level``) are
defined at the bottom of this module.
"""

import pandas as pd
import numpy as np
import os
import random
import time
import json
from tqdm import tqdm
import pickle
import csv
import argparse
import shutil
from collections import defaultdict
from typing import List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)

import sys
sys.path.append(os.path.dirname(__file__))

# Import the LLM augmentation module
try:
    from llm_augmentation import LLMAugmenter
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print(" llm_augmentation module not found; falling back to basic perturbation")

# Import the cache module
try:
    from perturbation_cache import create_cache_for_dataset
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print(" perturbation_cache module not found; caching disabled")

class LabelFreeDataGenerator:
    """
    Label-Free self-supervised training data generator.

    Core idea (following OmniMatch):
    1. Automatically generate training pairs from single-table splits
    2. No joinable_pairs.csv required
    3. Simulate realistic fuzzy-join scenarios
    """

    def __init__(self, data_dir: str, output_dir: str, num_query: int = 30, use_llm: bool = False,
                 use_cache: bool = True, display_samples: int = 5):
        self.data_dir = data_dir
        self.num_query = num_query
        self.use_llm = use_llm and LLM_AVAILABLE
        self.display_samples = display_samples

        # Append the _LabelFree suffix to the output directory
        self.output_dir = output_dir + '_LabelFree'

        # Create the required directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/t=0.2/train/mat_level", exist_ok=True)
        os.makedirs(f"{self.output_dir}/t=0.2/test", exist_ok=True)

        print(f" Label-Free output directory: {self.output_dir}")

        # Initialise the cache system
        self.cache = None
        if use_cache and CACHE_AVAILABLE:
            dataset_name = os.path.basename(output_dir)  # extract the dataset name
            self.cache = create_cache_for_dataset(dataset_name, output_dir=self.output_dir)
            print(f" Perturbation cache enabled")
        elif use_cache and not CACHE_AVAILABLE:
            print(f"Cache module unavailable; running without cache")

        # Initialise the LLM augmenter (pass the cache through)
        if self.use_llm:
            self.llm_augmenter = LLMAugmenter(enable_llm=True, fallback_to_basic=True, cache=self.cache)
            if self.llm_augmenter.enable_llm:
                print(f" LLM augmentation enabled")
            else:
                print(f"LLM initialisation failed; using basic perturbation")
                self.use_llm = False
        else:
            self.llm_augmenter = None
            print(f"Using basic perturbation (LLM not enabled)")

    def run_complete_pipeline(self, plm='fasttext', tau=0.2, list_size=5):
        """
        Full Label-Free data-generation pipeline.
        """
        _pipeline_start = time.time()

        print("\n" + "="*80)
        print("HyperJoin Label-Free self-supervised data generation")
        print("="*80)

        # Step 1: Load all tables
        print("\n[Step 1] Loading all tables...")
        _t = time.time()
        all_tables = self.load_all_tables()
        _load_time = time.time() - _t

        # Step 2: Generate self-supervised training pairs (including LLM perturbation)
        print("\n[Step 2] Generating self-supervised training samples...")
        _t = time.time()
        positive_pairs, negative_pairs = self.generate_self_supervised_pairs(all_tables)
        _pair_gen_time = time.time() - _t

        # Show sample pairs
        if self.display_samples > 0:
            self.display_sample_pairs(positive_pairs, negative_pairs, self.display_samples)

        # Step 3: Convert to the HyperJoin format (target.csv)
        print("\n[Step 3] Converting to HyperJoin data format...")
        _t = time.time()
        self.convert_to_hyperjoin_format(positive_pairs, negative_pairs)
        _convert_time = time.time() - _t

        # Step 4: Convert CSV to NPY
        print("\n[Step 4] Converting to NPY vector format...")
        _t = time.time()
        self.convert_csv_to_npy(plm)
        _embed_time = time.time() - _t

        # Step 5: Data augmentation
        print("\n[Step 5] Generating augmented data...")
        _t = time.time()
        self.generate_augmented_data(tau, list_size)
        _aug_time = time.time() - _t

        # Step 6: Prepare test data (reuses the original GT)
        print("\n[Step 6] Preparing test data...")
        self.prepare_test_data()

        # Step 7: Save the cache
        if self.cache:
            print("\n[Step 7] Saving the perturbation cache...")
            self.cache.save()
            csv_path = f"{self.output_dir}/column_perturbations.csv"
            self.cache.export_to_csv(csv_path)
            print(f"   Cache saved and exported to CSV")

        _total_time = time.time() - _pipeline_start

        # LLM statistics
        _llm_calls = 0
        _llm_time = 0.0
        if self.cache:
            stats = self.cache.get_stats()
            _llm_calls = stats.get('llm_generated', 0)
        if hasattr(self, '_llm_total_time'):
            _llm_time = self._llm_total_time

        print("\n" + "="*80)
        print(" Label-Free data generation complete!")
        print(f"   Output directory: {self.output_dir}")
        if self.cache:
            stats = self.cache.get_stats()
            print(f"   Perturbation cache: {stats['total']} perturbations ({stats['llm_generated']} LLM-generated)")
        print("="*80)

        # Save timing JSON for scalability experiments
        _timing = {
            'num_tables': len(all_tables),
            'num_positive_pairs': len(positive_pairs),
            'num_negative_pairs': len(negative_pairs),
            'use_llm': self.use_llm,
            'table_loading_time_s': round(_load_time, 2),
            'pair_generation_time_s': round(_pair_gen_time, 2),
            'format_conversion_time_s': round(_convert_time, 2),
            'embedding_time_s': round(_embed_time, 2),
            'augmentation_time_s': round(_aug_time, 2),
            'total_datagen_time_s': round(_total_time, 2),
            'llm_calls': _llm_calls,
            'llm_total_time_s': round(_llm_time, 2),
        }
        _timing_path = f"{self.output_dir}/datagen_timing.json"
        with open(_timing_path, 'w') as f:
            json.dump(_timing, f, indent=2)
        print(f" Timing saved to: {_timing_path}")

    def load_all_tables(self):
        """Load all CSV tables."""
        # Check for the tables subdirectory
        tables_dir = os.path.join(self.data_dir, 'tables')
        if not os.path.exists(tables_dir):
            tables_dir = self.data_dir

        csv_files = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
        all_tables = {}

        for csv_file in tqdm(csv_files, desc="  Loading tables"):
            try:
                df = pd.read_csv(
                    os.path.join(tables_dir, csv_file),
                    low_memory=False,
                    dtype=str,
                    on_bad_lines='skip',
                    encoding='utf-8'
                )
                # Clean column names
                df.columns = [col.strip().replace('ï»¿', '').replace('"', '') for col in df.columns]

                table_name = csv_file.replace('.csv', '')
                all_tables[table_name] = df
            except Exception as e:
                print(f"    Skipping {csv_file}: {e}")

        print(f"   Loaded {len(all_tables)} tables")
        return all_tables

    def identify_potential_join_keys(self, df: pd.DataFrame, table_name: str) -> list:
        """
        Identify potential join-key columns (simplified; strict filtering removed).

         Label-Free strategy: no ground truth, so only basic filtering:
        1. Non-null ratio > 50% (some nulls allowed)
        2. At least 2 distinct values (constant columns are useless)
        """
        if len(df) == 0 or len(df.columns) == 0:
            return []

        candidates = []

        for col in df.columns:
            # Basic statistics
            total_rows = len(df)
            non_null_count = df[col].notna().sum()
            non_null_ratio = non_null_count / total_rows

            # Relaxed filter: keep columns with non-null ratio > 50%
            if non_null_ratio < 0.5:
                continue

            # Distinct-value statistics
            unique_count = df[col].nunique()

            # Relaxed filter: keep columns with at least 2 distinct values
            if unique_count < 2:
                continue

            unique_ratio = unique_count / total_rows

            # Column-name heuristic (for ranking priority only; no filtering)
            col_lower = col.lower()
            key_keywords = ['id', 'code', 'name', 'type', 'category', 'class', 'status', 'country', 'city', 'region']
            has_keyword = any(kw in col_lower for kw in key_keywords)

            # Compute priority (for ranking only; no filtering)
            priority = 0
            if has_keyword:
                priority += 10
            priority += unique_ratio * 5  # higher distinct ratio -> higher priority
            priority += non_null_ratio * 3  # more non-nulls -> higher priority

            candidates.append({
                'column': col,
                'unique_ratio': unique_ratio,
                'non_null_ratio': non_null_ratio,
                'priority': priority
            })

        # Sort by priority
        candidates.sort(key=lambda x: x['priority'], reverse=True)

        # Return the top-3 candidates
        selected = [c['column'] for c in candidates[:3]]

        if selected:
            print(f"    {table_name}: identified join keys {selected}")

        return selected

    def is_valid_column_data(self, data_list: list, col_name: str = "") -> tuple:
        """
        Check column data quality; filter out data that could produce NaN.

        Returns: (is_valid: bool, reason: str)
        """
        if not data_list or len(data_list) == 0:
            return False, "empty_data"

        # Check 1: all values identical (zero variance)
        unique_values = set(data_list)
        if len(unique_values) == 1:
            return False, "all_same_values"

        # Check 2: contains "nan" or "inf" strings
        data_str = ' '.join(str(x).lower() for x in data_list[:100])  # check the first 100 only
        if 'nan' in data_str or 'inf' in data_str:
            return False, "contains_nan_inf"

        # Check 3: overly long text (>10000 chars can break embedding)
        total_length = sum(len(str(x)) for x in data_list[:50])  # total length of the first 50 cells
        if total_length > 10000:
            return False, "text_too_long"

        return True, "ok"

    def _serialize_sample_rows(self, df: pd.DataFrame, k: int = 5) -> List[str]:
        """
        Serialize sample rows into attribute-value-pair format (following OmniMatch).

        Args:
            df: DataFrame
            k: number of rows to sample

        Returns:
            List of serialized sample rows: ["col1: val1, col2: val2, ...", ...]

        Example:
            Input: DataFrame with columns [ID, Name, Email]
            Output: ["ID: 123, Name: John, Email: john@example.com", ...]
        """
        if len(df) == 0:
            return []

        # Randomly sample k rows (or all if fewer)
        n_sample = min(k, len(df))
        sampled_rows = df.sample(n=n_sample, random_state=42)

        serialized = []
        for _, row in sampled_rows.iterrows():
            # Build the attribute-value-pair string
            pairs = []
            for col in df.columns:
                val = str(row[col])
                # Truncate overly long values
                if len(val) > 50:
                    val = val[:47] + "..."
                pairs.append(f"{col}: {val}")

            serialized.append(", ".join(pairs))

        return serialized

    def split_table_for_training(self, df: pd.DataFrame, key_col: str, table_name: str):
        """
         Plan C: keep the table unsplit and create an augmented version of the key column.

        Strategy:
        1. table1 = the full original table (all columns kept)
        2. table2 = the full original table with the key column augmented (~75% row overlap)
        3. Simulate a realistic fuzzy join: two tables partially row-matched

        Advantages:
        - Training data is closer to the test scenario (full table vs full table)
        - Avoids the phantom-column explosion problem
        - Preserves the full table context
        """
        n_rows = len(df)

        # Skip tables with too few rows
        if n_rows < 10:
            return None, None, None

        # Table1: the full original table (all columns and rows kept)
        table1 = df.copy()

        # Table2: the full original table with row-level augmentation on the key column
        # Randomly keep ~75% of the rows to simulate partial matches in real joins
        overlap_ratio = 0.75  # 75% row overlap
        n_keep = int(n_rows * overlap_ratio)

        # Randomly select the rows to keep
        indices = np.arange(n_rows)
        np.random.shuffle(indices)
        keep_indices = sorted(indices[:n_keep])

        # Table2 keeps a subset of rows
        table2 = df.iloc[keep_indices].copy()

        # Strategy: only perturb the key column NAME of table2 (cell values untouched)
        # Prepare context (following OmniMatch)
        all_columns = df.columns.tolist()
        sample_rows = self._serialize_sample_rows(df, k=5)

        if self.use_llm and self.llm_augmenter:
            # Use the LLM to generate a column-name variant (context-aware)
            _llm_start = time.time()
            perturbed_col_name = self.llm_augmenter.perturb_column_name(
                column_name=key_col,
                table_name=table_name,
                all_columns=all_columns,
                sample_rows=sample_rows
            )
            _llm_elapsed = time.time() - _llm_start
            if not hasattr(self, '_llm_total_time'):
                self._llm_total_time = 0.0
            self._llm_total_time += _llm_elapsed
            print(f"     LLM column-name perturbation (context-aware): {key_col} -> {perturbed_col_name}")
        else:
            # Use the basic column-name perturbation
            perturbed_col_name = self._basic_perturb_column_name(key_col)
            print(f"     Basic column-name perturbation: {key_col} -> {perturbed_col_name}")

        # Avoid column-name collisions: if the perturbed name already exists in table2, add a suffix
        original_perturbed_name = perturbed_col_name
        suffix = 1
        while perturbed_col_name in table2.columns and perturbed_col_name != key_col:
            perturbed_col_name = f"{original_perturbed_name}_{suffix}"
            suffix += 1

        if perturbed_col_name != original_perturbed_name:
            print(f"   Column-name collision: {original_perturbed_name} exists; using {perturbed_col_name}")

        # Rename the key column in table2
        table2.rename(columns={key_col: perturbed_col_name}, inplace=True)

        # Return table1, table2 and the perturbed column name
        return table1, table2, perturbed_col_name

    def perturb_key_column(self, df: pd.DataFrame, key_col: str, perturb_ratio: float = 0.2):
        """
        Perturb the key-column values to simulate real-world noise.

        With LLM augmentation enabled:
        - the LLM generates semantically equivalent variants,
        - e.g. "John Smith" -> "J. Smith", "Jon Smith"

        Otherwise basic perturbation is used:
        1. Case changes
        2. Add/remove spaces
        3. Character substitution
        """
        if self.use_llm and self.llm_augmenter:
            # Use LLM-based perturbation
            return self.llm_augmenter.perturb_key_column(df, key_col, perturb_ratio)

        # Basic perturbation (original logic)
        df_copy = df.copy()

        if len(df_copy) == 0:
            return df_copy

        # Randomly select rows to perturb
        n_perturb = max(1, int(len(df_copy) * perturb_ratio))
        perturb_indices = np.random.choice(len(df_copy), min(n_perturb, len(df_copy)), replace=False)

        for idx in perturb_indices:
            original = str(df_copy[key_col].iloc[idx])

            if not original or original == 'nan':
                continue

            # Pick a perturbation method at random
            method = random.choice(['case', 'space', 'char'])

            if method == 'case' and len(original) > 0:
                # Case change
                if original.isupper():
                    df_copy.at[df_copy.index[idx], key_col] = original.lower()
                elif original.islower():
                    df_copy.at[df_copy.index[idx], key_col] = original.upper()
                else:
                    df_copy.at[df_copy.index[idx], key_col] = original.swapcase()

            elif method == 'space':
                # Space change
                if ' ' in original:
                    df_copy.at[df_copy.index[idx], key_col] = original.replace(' ', '_')
                else:
                    # Insert a space at a random position
                    if len(original) > 2:
                        pos = random.randint(1, len(original)-1)
                        df_copy.at[df_copy.index[idx], key_col] = original[:pos] + ' ' + original[pos:]

            elif method == 'char' and len(original) > 2:
                # Character substitution (simulate a typo)
                pos = random.randint(0, len(original)-1)
                char_list = list(original)
                char_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                df_copy.at[df_copy.index[idx], key_col] = ''.join(char_list)

        return df_copy

    def _basic_perturb_column_name(self, column_name: str) -> str:
        """
        Basic column-name perturbation (fallback when the LLM is unavailable).

        Args:
            column_name: original column name

        Returns:
            perturbed column name
        """
        if not column_name:
            return column_name

        # Pick a perturbation method at random
        method = random.choice(['underscore', 'case', 'dash'])

        if method == 'underscore':
            # CamelCase → snake_case
            # CustomerID → customer_id
            import re
            perturbed = re.sub(r'(?<!^)(?=[A-Z])', '_', column_name).lower()
            if perturbed == column_name.lower():
                # No uppercase letters; insert underscores
                perturbed = column_name.replace(' ', '_')
            return perturbed

        elif method == 'case':
            # Case transformation
            if column_name.isupper():
                return column_name.lower()
            elif column_name.islower():
                return column_name.upper()
            else:
                # camelCase <-> PascalCase
                if column_name[0].islower():
                    return column_name[0].upper() + column_name[1:]
                else:
                    return column_name[0].lower() + column_name[1:]

        elif method == 'dash':
            # underscore <-> dash
            if '_' in column_name:
                return column_name.replace('_', '-')
            else:
                return column_name.replace(' ', '-')

        return column_name

    def _generate_cross_table_negatives(self, all_tables_column_info, target_count, quality_stats):
        """
         Core improvement: generate cross-table negatives (closer to the supervised distribution).

        Args:
            all_tables_column_info: {table_name: [(col_name, col_data, split_name), ...]}
            target_count: number of negatives to generate
            quality_stats: quality-statistics dictionary

        Returns:
            cross_table_negatives: list of cross-table negative samples
        """
        cross_table_negatives = []

        # Flatten the columns of all tables into a single list
        all_columns = []
        for table_name, columns in all_tables_column_info.items():
            for col_name, col_data, split_name in columns:
                all_columns.append({
                    'table_name': table_name,
                    'split_name': split_name,
                    'col_name': col_name,
                    'col_data': col_data
                })

        if len(all_columns) < 2:
            print(f"  Not enough columns to generate cross-table negatives")
            return []

        print(f"   Column pool: {len(all_columns)} columns from {len(all_tables_column_info)} tables")

        # Generate cross-table negatives
        attempts = 0
        max_attempts = target_count * 10  # allow extra attempts

        while len(cross_table_negatives) < target_count and attempts < max_attempts:
            # Pick two columns at random
            col1_info, col2_info = random.sample(all_columns, 2)

            # Key constraint: they must come from different tables
            if col1_info['table_name'] == col2_info['table_name']:
                attempts += 1
                continue

            # Avoid pairing two columns with the same name
            if col1_info['col_name'] == col2_info['col_name']:
                attempts += 1
                continue

            # Create the negative sample
            cross_table_negatives.append({
                'table1_name': col1_info['split_name'],
                'col1_name': col1_info['col_name'],
                'col1_data': col1_info['col_data'],
                'table2_name': col2_info['split_name'],
                'col2_name': col2_info['col_name'],
                'col2_data': col2_info['col_data'],
                'label': 0
            })

            attempts += 1

        if len(cross_table_negatives) < target_count:
            print(f"  Generated only {len(cross_table_negatives)}/{target_count} cross-table negatives")

        return cross_table_negatives

    def generate_self_supervised_pairs(self, all_tables):
        """
        Generate self-supervised training pairs from all tables.

        Returns:
            positive_pairs: pairs sharing the key column
            negative_pairs: pairs of unrelated columns
        """
        all_positive_pairs = []
        all_negative_pairs = []
        intra_table_negatives = []  # intra-table negatives (~30%)

        table_count = 0
        skipped_count = 0

        # Collect column info of all tables for cross-table negatives
        all_tables_column_info = {}  # {table_name: [(col_name, col_data), ...]}

        # Data-quality statistics
        quality_stats = {
            'filtered_positive': 0,
            'filtered_negative': 0,
            'reasons': defaultdict(int)
        }

        # Batch-precompute LLM column-name perturbations (multithreaded)
        if self.use_llm and self.llm_augmenter:
            print(f"\n   Precomputing LLM column-name perturbations (batch + multithreaded)...")
            _precompute_start = time.time()
            # Silently collect all column names (no per-table logging)
            all_key_columns = []
            for df_tmp in all_tables.values():
                if len(df_tmp) == 0 or len(df_tmp.columns) == 0:
                    continue
                for col in df_tmp.columns:
                    total_rows = len(df_tmp)
                    if df_tmp[col].notna().sum() / total_rows < 0.5:
                        continue
                    if df_tmp[col].nunique() < 2:
                        continue
                    all_key_columns.append(col)
            # Deduplicate
            unique_keys = list(dict.fromkeys(all_key_columns))
            print(f"   Collected {len(unique_keys)} unique column names")
            if unique_keys:
                self.llm_augmenter.batch_precompute_column_names(
                    unique_keys, batch_size=20, max_workers=5
                )
            _precompute_time = time.time() - _precompute_start
            print(f"   Precompute finished in {_precompute_time:.1f}s")
            if not hasattr(self, '_llm_total_time'):
                self._llm_total_time = 0.0
            self._llm_total_time += _precompute_time

        for table_name, df in tqdm(all_tables.items(), desc="  Generating pairs"):
            # Identify potential join keys
            key_candidates = self.identify_potential_join_keys(df, table_name)

            if not key_candidates:
                skipped_count += 1
                continue

            # Generate samples for each candidate key
            for key_col in key_candidates:
                # Split the table (the return value includes the renamed column of table2)
                result = self.split_table_for_training(df, key_col, table_name)

                if result is None or result[0] is None:
                    continue

                table1, table2, perturbed_col_name = result

                table_count += 1

                # Positive sample: the shared key column
                col1_data = table1[key_col].fillna('').astype(str).tolist()[:50]
                col2_data_perturbed = table2[perturbed_col_name].fillna('').astype(str).tolist()[:50]

                # Data-quality checks
                valid1, reason1 = self.is_valid_column_data(col1_data, key_col)
                valid2, reason2 = self.is_valid_column_data(col2_data_perturbed, perturbed_col_name)

                if not valid1 or not valid2:
                    quality_stats['filtered_positive'] += 1
                    if not valid1:
                        quality_stats['reasons'][reason1] += 1
                    if not valid2:
                        quality_stats['reasons'][reason2] += 1
                    continue

                # Improvement 1: also add the ORIGINAL-name pairing as a positive (closer to supervised).
                # table2 keeps the original key column (only renamed; data intact).
                # Use the same data but with the original column name.
                all_positive_pairs.append({
                    'table1_name': f"{table_name}_split1",
                    'col1_name': key_col,  # original column name
                    'col1_data': col1_data,
                    'table2_name': f"{table_name}_split2",
                    'col2_name': key_col,  # original column name as well (exact match)
                    'col2_data': col2_data_perturbed,  # same data; the column name stays unperturbed
                    'label': 1
                })

                # Improvement 2: add the perturbed-name pairing as a positive (learns variants)
                all_positive_pairs.append({
                    'table1_name': f"{table_name}_split1",
                    'col1_name': key_col,  # table1 keeps the original name
                    'col1_data': col1_data,
                    'table2_name': f"{table_name}_split2",
                    'col2_name': perturbed_col_name,  # perturbed name (fuzzy match)
                    'col2_data': col2_data_perturbed,
                    'label': 1
                })

                # Collect this table's column info (used for cross-table negatives)
                if table_name not in all_tables_column_info:
                    all_tables_column_info[table_name] = []

                # Collect all columns of table1 and table2
                for col in table1.columns:
                    col_data = table1[col].fillna('').astype(str).tolist()[:50]
                    if self.is_valid_column_data(col_data, col)[0]:  # keep valid columns only
                        all_tables_column_info[table_name].append((col, col_data, f"{table_name}_split1"))

                for col in table2.columns:
                    col_data = table2[col].fillna('').astype(str).tolist()[:50]
                    if self.is_valid_column_data(col_data, col)[0]:
                        all_tables_column_info[table_name].append((col, col_data, f"{table_name}_split2"))

                # Intra-table negatives (hard negatives; keep ~30%)
                neg_samples_from_this_split = []
                for col1 in table1.columns:
                    if col1 == key_col:
                        continue
                    for col2 in table2.columns:
                        if col2 == perturbed_col_name:
                            continue

                        # Exclude pairs of the same column
                        if col1 == col2:
                            continue

                        # Low sampling rate (5%; only ~30% intra-table negatives are needed)
                        if random.random() < 0.05:  # reduced from 20% to 5%
                            neg_col1_data = table1[col1].fillna('').astype(str).tolist()[:50]
                            neg_col2_data = table2[col2].fillna('').astype(str).tolist()[:50]

                            # Data-quality checks
                            valid1, reason1 = self.is_valid_column_data(neg_col1_data, col1)
                            valid2, reason2 = self.is_valid_column_data(neg_col2_data, col2)

                            if not valid1 or not valid2:
                                quality_stats['filtered_negative'] += 1
                                if not valid1:
                                    quality_stats['reasons'][reason1] += 1
                                if not valid2:
                                    quality_stats['reasons'][reason2] += 1
                                continue

                            neg_samples_from_this_split.append({
                                'table1_name': f"{table_name}_split1",
                                'col1_name': col1,
                                'col1_data': neg_col1_data,
                                'table2_name': f"{table_name}_split2",
                                'col2_name': col2,
                                'col2_data': neg_col2_data,
                                'label': 0
                            })

                # Keep at most 3 intra-table negatives per split (down from 5)
                if neg_samples_from_this_split:
                    selected_neg = random.sample(
                        neg_samples_from_this_split,
                        min(3, len(neg_samples_from_this_split))
                    )
                    intra_table_negatives.extend(selected_neg)

        print(f"\n   Processed splits for {table_count} tables")
        print(f"  Skipped {skipped_count} tables (no suitable join key)")
        print(f"   Positive pairs generated: {len(all_positive_pairs)}")
        print(f"   Intra-table negatives generated: {len(intra_table_negatives)}")

        # Core improvement: generate cross-table negatives (~70%)
        print(f"\n   Generating cross-table negatives (approaching the supervised distribution)...")
        cross_table_negatives = self._generate_cross_table_negatives(
            all_tables_column_info,
            target_count=int(len(all_positive_pairs) * 1.5 * 0.7),  # ~70% of negatives are cross-table
            quality_stats=quality_stats
        )
        print(f"   Cross-table negatives generated: {len(cross_table_negatives)}")

        # Merge intra-table and cross-table negatives
        all_negative_pairs = intra_table_negatives + cross_table_negatives
        print(f"   Total negatives: {len(all_negative_pairs)} (intra-table: {len(intra_table_negatives)}, cross-table: {len(cross_table_negatives)})")

        # Print the data-quality filtering statistics
        if quality_stats['filtered_positive'] > 0 or quality_stats['filtered_negative'] > 0:
            print(f"\n   Data-quality filtering stats:")
            print(f"     Filtered positives: {quality_stats['filtered_positive']}")
            print(f"     Filtered negatives: {quality_stats['filtered_negative']}")
            print(f"     Filter reasons:")
            for reason, count in quality_stats['reasons'].items():
                print(f"       - {reason}: {count}")

        # Balancing: keep the 70/30 split while capping the total at 1.5x
        target_total_negatives = int(len(all_positive_pairs) * 1.5)
        actual_total = len(all_negative_pairs)

        if actual_total > target_total_negatives:
            # Downsample while keeping the 70/30 ratio
            ratio_cross = len(cross_table_negatives) / actual_total if actual_total > 0 else 0.7
            ratio_intra = len(intra_table_negatives) / actual_total if actual_total > 0 else 0.3

            # Downsample proportionally
            keep_cross = int(target_total_negatives * ratio_cross)
            keep_intra = int(target_total_negatives * ratio_intra)

            cross_table_negatives = random.sample(cross_table_negatives, min(keep_cross, len(cross_table_negatives)))
            intra_table_negatives = random.sample(intra_table_negatives, min(keep_intra, len(intra_table_negatives)))

            all_negative_pairs = intra_table_negatives + cross_table_negatives
            print(f"  Balanced negatives: {len(all_negative_pairs)} (cross-table: {len(cross_table_negatives)}, intra-table: {len(intra_table_negatives)})")

        # Print the final ratio statistics
        total_neg = len(all_negative_pairs)
        if total_neg > 0:
            cross_ratio = len(cross_table_negatives) / total_neg * 100
            intra_ratio = len(intra_table_negatives) / total_neg * 100
            print(f"   Negative distribution: cross-table {cross_ratio:.1f}% | intra-table {intra_ratio:.1f}%")

        # Save the pairs (used to build the graph at training time)
        pairs_path = f"{self.output_dir}/self_supervised_pairs.pkl"
        with open(pairs_path, 'wb') as f:
            pickle.dump({
                'positive': all_positive_pairs,
                'negative': all_negative_pairs
            }, f)
        print(f"   Saved to: {pairs_path}")

        return all_positive_pairs, all_negative_pairs

    def convert_to_hyperjoin_format(self, positive_pairs, negative_pairs):
        """
        Convert self-supervised data into HyperJoin's target.csv format;
        each unique column becomes one row.
        """
        # Collect all unique columns
        all_columns = []
        column_set = set()
        metadata = []

        # Collect all unique table/column names (for the mapping)
        table_names = set()
        column_names = set()

        for pair in positive_pairs + negative_pairs:
            # Process the first column
            key1 = (pair['table1_name'], pair['col1_name'])
            if key1 not in column_set:
                column_set.add(key1)
                all_columns.append(pair['col1_data'])
                table_names.add(pair['table1_name'])
                column_names.add(pair['col1_name'])
                metadata.append({
                    'index': len(metadata),
                    'table_name': pair['table1_name'],
                    'column_name': pair['col1_name']
                })

            # Process the second column
            key2 = (pair['table2_name'], pair['col2_name'])
            if key2 not in column_set:
                column_set.add(key2)
                all_columns.append(pair['col2_data'])
                table_names.add(pair['table2_name'])
                column_names.add(pair['col2_name'])
                metadata.append({
                    'index': len(metadata),
                    'table_name': pair['table2_name'],
                    'column_name': pair['col2_name']
                })

        print(f"  Total columns: {len(all_columns)}")
        print(f"  Total tables: {len(table_names)}")

        # Write target.csv (drop empty columns to stay consistent with the NPY)
        target_csv_path = f"{self.output_dir}/target.csv"
        valid_columns = []
        valid_metadata = []
        skipped_empty = 0

        with open(target_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for idx, col_data in enumerate(all_columns):
                # Sanitize the data
                sanitized = [str(v).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
                           for v in col_data]

                # Drop fully empty columns (avoids index drift when the NPY conversion skips them)
                if not any(v and v != 'nan' for v in sanitized):
                    skipped_empty += 1
                    continue

                writer.writerow(sanitized)
                valid_columns.append(col_data)
                valid_metadata.append(metadata[idx])

        # Replace metadata with the filtered version
        metadata = valid_metadata
        for i, m in enumerate(metadata):
            m['index'] = i  # renumber

        if skipped_empty > 0:
            print(f"   Filtered out {skipped_empty} empty columns")
        print(f"   Saved target.csv: {len(valid_columns)} columns")

        # Save metadata (adds the table_to_id and column_to_id mappings)
        metadata_dict = {
            'metadata': metadata,
            'table_to_id': {name: i for i, name in enumerate(sorted(table_names))},
            'column_to_id': {name: i for i, name in enumerate(sorted(column_names))},
            'total_tables': len(table_names),
            'total_columns': len(column_names)
        }

        metadata_path = f"{self.output_dir}/target_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)

        print(f"   Saved target_metadata.pkl (with mappings for {len(table_names)} tables)")

    def convert_csv_to_npy(self, plm='fasttext'):
        """Convert CSV to NPY format."""
        target_csv = f"{self.output_dir}/target.csv"
        target_npy = f"{self.output_dir}/target.npy"

        print(f"  Converting target.csv -> target.npy...")
        text2vec(target_csv, target_npy, plm=plm, normal=0, max_cells_embed=5000)

        print(f"   Conversion complete")

    def generate_augmented_data(self, tau=0.2, list_size=5):
        """Generate augmented data."""
        print(f"  Generating augmented data (tau={tau}, list_size={list_size})...")

        augment_mat_level(
            f"{self.output_dir}/target.npy",
            f"{self.output_dir}/t={tau}/train/mat_level/anchor.npy",
            f"{self.output_dir}/t={tau}/train/mat_level/auglist.npy",
            f"{self.output_dir}/t={tau}/train/mat_level/auglist_y.csv",
            f"{self.output_dir}/target_metadata.pkl",
            tau=tau,
            k=list_size
        )

        print(f"   Augmented data generated")

    def display_sample_pairs(self, positive_pairs, negative_pairs, n_samples=15):
        """
        Show random positive/negative samples for a data-quality check.

        Args:
            positive_pairs: list of positive samples
            negative_pairs: list of negative samples
            n_samples: number of samples shown per class
        """
        print("\n" + "="*80)
        print(" Sample pairs (randomly drawn)")
        print("="*80)

        # Randomly draw positive samples
        if len(positive_pairs) > 0:
            sample_positive = random.sample(positive_pairs, min(n_samples, len(positive_pairs)))

            print(f"\n Positive examples ({len(sample_positive)} of {len(positive_pairs)}):")
            print("-" * 80)

            for i, pair in enumerate(sample_positive, 1):
                print(f"\n[Positive {i}]")
                print(f"  Table1: {pair['table1_name']}")
                print(f"  Column1: {pair['col1_name']}")
                print(f"  Values1: {self._format_cell_values(pair['col1_data'][:5])}")
                print(f"  ")
                print(f"  Table2: {pair['table2_name']}")
                print(f"  Column2: {pair['col2_name']}")
                print(f"  Values2: {self._format_cell_values(pair['col2_data'][:5])}")
                print(f"  Label: {pair['label']} (should join)")

        # Randomly draw negative samples
        if len(negative_pairs) > 0:
            sample_negative = random.sample(negative_pairs, min(n_samples, len(negative_pairs)))

            print(f"\n\n Negative examples ({len(sample_negative)} of {len(negative_pairs)}):")
            print("-" * 80)

            for i, pair in enumerate(sample_negative, 1):
                print(f"\n[Negative {i}]")
                print(f"  Table1: {pair['table1_name']}")
                print(f"  Column1: {pair['col1_name']}")
                print(f"  Values1: {self._format_cell_values(pair['col1_data'][:5])}")
                print(f"  ")
                print(f"  Table2: {pair['table2_name']}")
                print(f"  Column2: {pair['col2_name']}")
                print(f"  Values2: {self._format_cell_values(pair['col2_data'][:5])}")
                print(f"  Label: {pair['label']} (should NOT join)")

        print("\n" + "="*80)

    def _format_cell_values(self, values):
        """
        Format a list of cell values for display.

        Args:
            values: list of cell values

        Returns:
            formatted string
        """
        if not values:
            return "[]"

        # Convert to strings and truncate long values
        formatted = []
        for v in values:
            v_str = str(v)
            if len(v_str) > 30:
                v_str = v_str[:27] + "..."
            formatted.append(v_str)

        return "[" + ", ".join(formatted) + "]"

    def prepare_test_data(self):
        """
        Prepare the test data.
        Important: the test set still uses the ORIGINAL supervised query and ground truth
        so that label-free and supervised results are fairly comparable.
        """
        print("  The test set reuses the original supervised data (for a fair comparison)")

        # Original data directory
        original_dir = self.output_dir.replace('_LabelFree', '')

        if not os.path.exists(original_dir):
            print(f"  Warning: original directory does not exist: {original_dir}")
            print(f"     Run supervised-mode generation first to produce the test data")
            return

        # Files to copy
        files_to_copy = [
            'query.csv',
            'query.npy',
            'query_metadata.pkl',
            't=0.2/test/index.csv'
        ]

        for file_path in files_to_copy:
            src = os.path.join(original_dir, file_path)
            dst = os.path.join(self.output_dir, file_path)

            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"     Copied: {file_path}")
            else:
                print(f"    Missing: {file_path}")

def parse_ops(parser):
    parser.add_argument('--datasets', type=str, default='CAN_ALL',
                       help='dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='dataset directory (inferred automatically by default)')
    parser.add_argument('--tau', type=float, default=0.2,
                       help='augmentation noise threshold')
    parser.add_argument('--list_size', type=int, default=5,
                       help='number of augmented samples per anchor')
    parser.add_argument('--type', type=str, default='mat',
                       help='PLM type to use')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='output directory (default: datasets/Lake/<datasets>)')
    parser.add_argument('--display_samples', type=int, default=5,
                        help='sample pairs printed per class for inspection (0 = none)')
    parser.add_argument('--num_query', type=int, default=30,
                       help='number of query columns (kept for compatibility)')
    parser.add_argument('--use_llm', type=int, default=None, choices=[0, 1],
                       help='Enable LLM perturbation (1=on, 0=off); must be set explicitly '
                            'to prevent silently falling back to basic perturbation on new datasets')
    return parser.parse_args()

# =============================================================================
# Low-level data-generation helpers
# =============================================================================

_ft_model = None

def _get_fasttext_model():
    global _ft_model
    if _ft_model is not None:
        return _ft_model
    try:
        import fasttext
        fasttext_paths = [
            'fasttext-english/cc.en.300.bin',
            '../fasttext-english/cc.en.300.bin',
            'fasttext_model/cc.en.300.bin',
        ]
        for p in fasttext_paths:
            if os.path.exists(p):
                _ft_model = fasttext.load_model(p)
                print(f"Using FastText model: {p}")
                return _ft_model
    except ImportError:
        pass
    print("FastText model not found, using deterministic random embeddings")
    return None


def fix_seed(seed=2024):
    """Seed used only for data generation: sets random / numpy / PYTHONHASHSEED and deliberately leaves torch alone.

    Note: ``utils.fix_seed`` additionally calls ``torch.manual_seed`` and ``torch.cuda.manual_seed_all``.
    We do not touch torch here to avoid accidentally overwriting the torch RNG state of the training script
    during data generation, preserving the historic reproduction behaviour. Training/search scripts should keep using ``utils.fix_seed``.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def normalize_rows(matrix):
    norm_matrix = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norm_matrix
    return normalized_matrix


def text2vec(path_load, path_save, plm='fasttext', normal=0, max_cells_embed=0):
    """Convert CSV file to NPY embeddings."""
    csv.field_size_limit(10 * 1024 * 1024)

    print(f"Converting {path_load} -> {path_save}")

    with open(path_load, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        data_list = []
        for row_idx, row in enumerate(csv_reader):
            data_list.append(row)
            if row_idx > 0 and row_idx % 1000 == 0:
                print(f"   Read {row_idx} rows...")

    emb_mats = []
    print(f"Processing {len(data_list)} columns...")

    if plm == 'fasttext':
        ft = _get_fasttext_model()
        use_fasttext = ft is not None

        skipped_empty_cols = 0
        downsampled_cols = 0
        for id, col in tqdm(enumerate(data_list), desc="Converting to embeddings"):
            clean_vals = []
            for val in col:
                clean_val = str(val).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
                if not clean_val or clean_val.lower() in ['nan', 'none', 'null', '']:
                    continue
                clean_vals.append(clean_val)

            if max_cells_embed and len(clean_vals) > max_cells_embed:
                idxs = np.linspace(0, len(clean_vals) - 1, num=max_cells_embed, dtype=int)
                selected_vals = [clean_vals[i] for i in idxs]
                downsampled_cols += 1
            else:
                selected_vals = clean_vals

            col_emb = []
            for clean_val in selected_vals:
                if use_fasttext:
                    x = ft.get_sentence_vector(clean_val).astype(np.float32)
                else:
                    seed = hash(clean_val) % (2**32)
                    np.random.seed(seed)
                    x = np.random.normal(0, 1, 300).astype(np.float32)
                col_emb.append(x)

            if len(col_emb) == 0:
                skipped_empty_cols += 1
                continue

            col_emb = np.array(col_emb, dtype=np.float32)
            emb_mats.append(col_emb)

        if skipped_empty_cols > 0:
            print(f"Skipped {skipped_empty_cols} empty columns")
        if downsampled_cols > 0 and max_cells_embed:
            print(f"Downsampled {downsampled_cols} columns to {max_cells_embed} cells")
    else:
        for _, row in tqdm(enumerate(data_list)):
            clean_vals = []
            for val in row:
                s = str(val).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').strip()
                if not s or s.lower() in ['nan', 'none', 'null', '']:
                    continue
                clean_vals.append(s)
            if max_cells_embed and len(clean_vals) > max_cells_embed:
                idxs = np.linspace(0, len(clean_vals) - 1, num=max_cells_embed, dtype=int)
                clean_vals = [clean_vals[i] for i in idxs]

            col_emb = []
            for s in clean_vals:
                random.seed(hash(s) % (2**32))
                x = np.random.normal(0, 1, 300).astype(np.float32)
                col_emb.append(x)
            if len(col_emb) == 0:
                continue
            emb_mats.append(np.array(col_emb, dtype=np.float32))

    # Save
    try:
        shapes = [arr.shape for arr in emb_mats]
        if len(set(shapes)) == 1:
            np.save(path_save, np.array(emb_mats, dtype=np.float32), allow_pickle=False)
        else:
            np.save(path_save, np.array(emb_mats, dtype=object), allow_pickle=True)
    except Exception:
        np.save(path_save, np.array(emb_mats, dtype=object), allow_pickle=True)
    print(f"Saved {len(emb_mats)} column embeddings to {path_save}")

    return skipped_empty_cols if 'skipped_empty_cols' in locals() else 0


def augment_mat_level(path_load, anchor_path, aug_path, y_path, metadata_path, tau=0.1, k=5):
    """Matrix-level data augmentation."""
    X = np.load(path_load, allow_pickle=True)
    anchor_list = []
    aug_list = []
    y_list = []
    anchor_metadata = []

    with open(metadata_path, 'rb') as f:
        target_metadata_dict = pickle.load(f)

    if isinstance(target_metadata_dict, dict):
        target_metadata = target_metadata_dict['metadata']
        table_to_id = target_metadata_dict.get('table_to_id', {})
        column_to_id = target_metadata_dict.get('column_to_id', {})
    else:
        target_metadata = target_metadata_dict
        table_names = set()
        column_names = set()
        for meta in target_metadata:
            table_names.add(meta['table_name'])
            column_names.add(meta['column_name'])
        table_to_id = {name: i for i, name in enumerate(sorted(table_names))}
        column_to_id = {name: i for i, name in enumerate(sorted(column_names))}

    print(f"Starting matrix-level augmentation for {len(X)} columns")

    for i in tqdm(range(len(X)), desc="Augmenting data"):
        y = []
        mat = X[i]
        if len(mat) == 0:
            continue

        if i < len(target_metadata):
            current_metadata = target_metadata[i]
            anchor_metadata.append({
                'anchor_index': len(anchor_list),
                'original_target_index': i,
                'table_name': current_metadata['table_name'],
                'column_name': current_metadata['column_name'],
                'table_id': table_to_id.get(current_metadata['table_name'], 0),
                'column_id': column_to_id.get(current_metadata['column_name'], 0)
            })

        np.random.shuffle(mat)
        anchor_rate = random.uniform(0.6, 1)
        anchor = mat[:int(mat.shape[0] * anchor_rate)]
        anchor_list.append(anchor)
        leave = mat[int(mat.shape[0] * anchor_rate):]

        if random.random() > 0.3:
            random_sequence = [random.uniform(0.7, 1) for _ in range(k)]
        else:
            random_sequence = [random.uniform(0.4, 0.7) for _ in range(k)]
        random_sequence.sort(reverse=True)

        for j in range(k):
            copy_rate = random_sequence[j]
            copy = anchor[:int(len(anchor) * copy_rate)]
            if len(leave) > 0:
                leave_sample = leave[:int(len(leave) * random.uniform(0.8, 1))]
                new_mat = np.concatenate((copy, leave_sample), axis=0)
            else:
                new_mat = copy
            np.random.shuffle(new_mat)
            y.append(str(round(copy_rate, 4)))
            aug_list.append(new_mat)

        y_list.append(y)

    anchor_array = np.array(anchor_list, dtype=object)
    aug_array = np.array(aug_list, dtype=object)

    np.save(anchor_path, anchor_array, allow_pickle=True)
    np.save(aug_path, aug_array, allow_pickle=True)

    with open(y_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in y_list:
            writer.writerow(row)

    anchor_metadata_dict = {
        'metadata': anchor_metadata,
        'table_to_id': table_to_id,
        'column_to_id': column_to_id,
        'total_anchors': len(anchor_list)
    }

    anchor_metadata_path = anchor_path.replace('.npy', '_metadata.pkl')
    with open(anchor_metadata_path, 'wb') as f:
        pickle.dump(anchor_metadata_dict, f)

    print(f"Generated {len(anchor_list)} anchor samples and {len(aug_list)} augmented samples")

if __name__ == "__main__":
    fix_seed(2024)

    parser = argparse.ArgumentParser(description='HyperJoin Label-Free self-supervised data generation')
    args = parse_ops(parser)

    # Guard: --use_llm must be explicit so protocols stay consistent across datasets
    if args.use_llm is None:
        raise SystemExit(
            "Please specify --use_llm 0 or 1 explicitly.\n"
            "(Historical note: the Valentine dataset once silently used basic perturbation "
            "because this flag was omitted, which diverged from the LLM-augmented LakeBench protocol.)")

    # Infer the data directory automatically
    if args.data_dir is None:
        args.data_dir = f"datasets/datasets_{args.datasets}"

    if args.output_dir is None:
        args.output_dir = f"datasets/Lake/{args.datasets}"
    output_dir = args.output_dir

    print("\n" + "="*80)
    print("HyperJoin Label-Free Data Generator")
    print("="*80)
    print(f"Dataset: {args.datasets}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}_LabelFree")
    print(f"LLM augmentation: {'enabled' if args.use_llm else 'disabled'}")
    print("="*80 + "\n")

    # Create the generator and run
    generator = LabelFreeDataGenerator(
        data_dir=args.data_dir,
        output_dir=output_dir,
        num_query=args.num_query,
        use_llm=bool(args.use_llm),
        display_samples=args.display_samples
    )

    generator.run_complete_pipeline(
        plm=args.type if args.type != 'mat' else 'fasttext',
        tau=args.tau,
        list_size=args.list_size
    )
