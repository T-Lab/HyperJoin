"""LLM-based data augmentation for Label-Free HyperJoin.

Uses OpenAI-compatible API to generate intelligent perturbations for join key columns.

Environment variables:
  LLM_BASE_URL: API base URL (default: ModelScope inference API)
  LLM_API_KEY: API key
  LLM_MODEL_ID: Model identifier (default: DeepSeek-V3.1)

Usage:
  from llm_augmentation import LLMAugmenter

  augmenter = LLMAugmenter(enable_llm=True)
  perturbed_values = augmenter.perturb_key_values(original_values, column_name)
"""

import os
import json
import re
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ====== API Configuration ======
# Nebius API (DeepSeek models)
API_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.studio.nebius.com/v1/")
API_KEY = os.environ.get("LLM_API_KEY")
MODEL_ID = os.environ.get("LLM_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")  # DeepSeek model on Nebius

# ====== Rate Limiting Configuration ======
# Global rate limiter to prevent overwhelming the API
_last_request_time = 0
_request_lock = None  # Will be initialized when needed
MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between requests
# ========================================


class LLMAugmenter:
    """LLM-powered data augmentation for join key perturbation."""

    def __init__(self, enable_llm: bool = True, fallback_to_basic: bool = True, cache=None):
        """
        Args:
            enable_llm: Whether to use LLM for augmentation
            fallback_to_basic: If LLM fails, use basic string operations
            cache: PerturbationCache instance for caching LLM results
        """
        self.enable_llm = enable_llm and OpenAI is not None and bool(API_KEY)
        self.fallback_to_basic = fallback_to_basic
        self.client = None
        self.cache = cache  # Perturbation cache

        if enable_llm and OpenAI is None:
            print("OpenAI package not installed. LLM augmentation disabled.")
            print("   Install with: pip install openai>=1.0.0")
        elif enable_llm and not API_KEY:
            print("LLM_API_KEY not set. LLM augmentation disabled.")

        if self.enable_llm:
            try:
                self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
                print(f" LLM Augmenter initialized with model: {MODEL_ID}")
                if self.cache:
                    print(f" Perturbation cache enabled")
            except Exception as e:
                print(f" Failed to initialize LLM client: {e}")
                print("   Falling back to basic augmentation")
                self.enable_llm = False

    def perturb_key_column(self, df: pd.DataFrame, key_col: str,
                           perturb_ratio: float = 0.2,
                           batch_size: int = 5) -> pd.DataFrame:
        """
        Intelligently perturb the key-column values (LLM-augmented version).

        Args:
            df: DataFrame
            key_col: Key column name
            perturb_ratio: Ratio of rows to perturb (0.0-1.0)
            batch_size: Number of values to send to LLM in one request

        Returns:
            Perturbed DataFrame
        """
        df_copy = df.copy()

        if len(df_copy) == 0:
            return df_copy

        # Randomly select the rows to perturb
        n_perturb = max(1, int(len(df_copy) * perturb_ratio))
        perturb_indices = random.sample(range(len(df_copy)), min(n_perturb, len(df_copy)))

        if not self.enable_llm or self.client is None:
            # Fallback to basic augmentation
            return self._basic_perturbation(df_copy, key_col, perturb_indices)

        # Batch process with LLM
        try:
            # Get original values
            original_values = [str(df_copy[key_col].iloc[idx]) for idx in perturb_indices]
            original_values = [v for v in original_values if v and v != 'nan']

            if not original_values:
                return df_copy

            # Process in batches
            all_perturbed = []
            for i in range(0, len(original_values), batch_size):
                batch = original_values[i:i+batch_size]
                perturbed_batch = self._llm_perturb_batch(batch, key_col)
                all_perturbed.extend(perturbed_batch)

            # Apply perturbations
            valid_idx = 0
            for idx in perturb_indices:
                original = str(df_copy[key_col].iloc[idx])
                if original and original != 'nan' and valid_idx < len(all_perturbed):
                    if all_perturbed[valid_idx]:  # Only apply if LLM returned something
                        df_copy.at[df_copy.index[idx], key_col] = all_perturbed[valid_idx]
                    valid_idx += 1

            return df_copy

        except Exception as e:
            print(f" LLM perturbation failed: {e}")
            if self.fallback_to_basic:
                print("   Falling back to basic perturbation")
                return self._basic_perturbation(df_copy, key_col, perturb_indices)
            return df_copy

    def _llm_perturb_batch(self, values: List[str], column_name: str,
                          max_retries: int = 5, base_delay: float = 2.0) -> List[str]:
        """
        Use LLM to generate intelligent perturbations for a batch of values.
        Includes retry logic with exponential backoff for rate limiting.

        Args:
            values: Original values to perturb
            column_name: Column name for context
            max_retries: Maximum retry attempts (default: 5)
            base_delay: Base delay in seconds for exponential backoff (default: 2.0)

        Returns:
            List of perturbed values (same length as input)
        """
        system_prompt = """You are a data-quality expert. Generate realistic variants (perturbations) of table join keys that mimic real-world noise and formatting differences.

Requirements:
1. Keep the semantics identical but change the surface form (e.g. abbreviations, format changes, synonyms)
2. Mimic common data-quality issues (typos, inconsistent casing, different separators)
3. The variants should still be matchable by a fuzzy join
4. Output strict JSON only; no explanations"""

        user_prompt = f"""Column name: {column_name}
Original values: {json.dumps(values, ensure_ascii=False)}

Generate exactly 1 variant per value. Return JSON:
{{"perturbed": ["variant1", "variant2", ...]}}

Variant examples:
- Person name: John Smith -> J. Smith / Jon Smith / Smith, John
- Date: 2024-01-15 -> 01/15/2024 / 15 Jan 2024
- Place: New York -> NY / new york / NewYork
- ID: ABC-123 -> abc123 / ABC 123
- Company: Microsoft Corp -> Microsoft Corporation / MSFT

Return JSON only, no explanations."""

        last_error = None
        for attempt in range(max_retries):
            try:
                # Global rate limiting: ensure minimum interval between requests
                global _last_request_time
                current_time = time.time()
                time_since_last = current_time - _last_request_time
                if time_since_last < MIN_REQUEST_INTERVAL:
                    sleep_time = MIN_REQUEST_INTERVAL - time_since_last
                    time.sleep(sleep_time)

                # Add additional delay for retry attempts
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)

                _last_request_time = time.time()
                response = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                    stream=False
                )

                content = response.choices[0].message.content or ""

                # Try to extract JSON
                result = self._extract_json(content)
                perturbed = result.get("perturbed", [])

                # Ensure we return same length as input
                if len(perturbed) != len(values):
                    # Pad or truncate
                    while len(perturbed) < len(values):
                        perturbed.append(values[len(perturbed)])
                    perturbed = perturbed[:len(values)]

                # Success - global rate limiter will handle next request delay
                return perturbed

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        retry_delay = base_delay * (2 ** attempt) + random.uniform(2, 5)
                        print(f" Rate limit (attempt {attempt+1}/{max_retries}), waiting {retry_delay:.1f}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f" Rate limit after {max_retries} attempts, using basic perturbation")
                else:
                    # Non-rate-limit error, don't retry
                    print(f" LLM API call failed: {e}")
                    break

        # All retries failed, return original values
        return values

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            return {}

        # Remove markdown code fences
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)

        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to find JSON object
        try:
            match = re.search(r'\{[^{}]*"perturbed"[^{}]*\[[^\]]*\][^{}]*\}', text)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # Last resort: find any {...}
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        return {}

    def _basic_perturbation(self, df: pd.DataFrame, key_col: str,
                           indices: List[int]) -> pd.DataFrame:
        """
        Basic string-based perturbation (fallback method).
        Same as original implementation in datagen.py
        """
        for idx in indices:
            original = str(df[key_col].iloc[idx])

            if not original or original == 'nan':
                continue

            # Random perturbation method
            method = random.choice(['case', 'space', 'char'])

            if method == 'case' and len(original) > 0:
                # Case change
                if original.isupper():
                    df.at[df.index[idx], key_col] = original.lower()
                elif original.islower():
                    df.at[df.index[idx], key_col] = original.upper()
                else:
                    df.at[df.index[idx], key_col] = original.swapcase()

            elif method == 'space':
                # Space change
                if ' ' in original:
                    df.at[df.index[idx], key_col] = original.replace(' ', '_')
                else:
                    if len(original) > 2:
                        pos = random.randint(1, len(original)-1)
                        df.at[df.index[idx], key_col] = original[:pos] + ' ' + original[pos:]

            elif method == 'char' and len(original) > 2:
                # Character substitution
                pos = random.randint(0, len(original)-1)
                char_list = list(original)
                char_list[pos] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                df.at[df.index[idx], key_col] = ''.join(char_list)

        return df

    def perturb_column_name(self, column_name: str, table_name: str = None,
                           all_columns: Optional[List[str]] = None,
                           sample_rows: Optional[List[str]] = None,
                           max_retries: int = 3) -> str:
        """
        Generate a semantically equivalent column-name variant with the LLM
        (context-aware, cached).
        Args:
            column_name: original column name
            table_name: table name (context)
            all_columns: all column names of the table (helps the LLM infer the naming style)
            sample_rows: sample rows in attribute-value-pair format
            max_retries: maximum retry attempts

        Returns:
            Perturbed column name (basic-perturbation result if the LLM fails)
        """
        # Step 1: try the cache
        if self.cache:
            cached_result = self.cache.get(table_name or "", column_name, all_columns, sample_rows)
            if cached_result:
                print(f"      Cache hit: {column_name} -> {cached_result}")
                return cached_result

        # Step 2: cache miss - generate with the LLM
        if not self.enable_llm or self.client is None:
            perturbed = self._basic_perturb_column_name(column_name)
            # Cache the result even when the basic method is used
            if self.cache:
                self.cache.put(table_name or "", column_name, perturbed,
                             all_columns, sample_rows, method="basic")
            return perturbed

        # Step 3: call the LLM (with context)
        perturbed = self._llm_perturb_column_name_with_context(
            column_name, table_name, all_columns, sample_rows, max_retries
        )

        # Step 4: store in the cache
        if self.cache:
            self.cache.put(table_name or "", column_name, perturbed,
                         all_columns, sample_rows, method="llm")
            print(f"      Cached: {column_name} -> {perturbed}")

        return perturbed

    def batch_precompute_column_names(self, column_names: List[str],
                                      batch_size: int = 20,
                                      max_workers: int = 5) -> Dict[str, str]:
        """
        Batch-precompute column-name perturbations (multithreaded + batched prompts).

        Args:
            column_names: all column names to perturb
            batch_size: number of names handled per API call
            max_workers: number of concurrent threads

        Returns:
            Mapping {original_name: perturbed_name}
        """
        # Deduplicate + filter out already-cached names
        unique_names = list(dict.fromkeys(column_names))  # order-preserving dedup
        uncached = []
        results = {}

        for name in unique_names:
            if self.cache:
                cached = self.cache.get("", name, None, None)
                if cached:
                    results[name] = cached
                    continue
            uncached.append(name)

        if not uncached:
            print(f"  All {len(unique_names)} column names already cached; skipping LLM calls")
            return results

        print(f"   Column-name perturbation: {len(unique_names)} unique names, "
              f"{len(results)} cached, {len(uncached)} need LLM generation")

        # Split into batches
        batches = []
        for i in range(0, len(uncached), batch_size):
            batches.append(uncached[i:i+batch_size])

        print(f"   {len(batches)} batches of {batch_size}, "
              f"{max_workers} concurrent threads")

        # Thread-safe lock
        cache_lock = threading.Lock()
        completed = [0]
        total_batches = len(batches)

        def process_batch(batch):
            """Process one batch of column names."""
            batch_results = self._llm_batch_column_names(batch)

            # Write to the cache in a thread-safe way
            with cache_lock:
                for orig, perturbed in batch_results.items():
                    results[orig] = perturbed
                    if self.cache:
                        self.cache.put("", orig, perturbed, None, None, method="llm_batch")
                completed[0] += 1
                if completed[0] % 10 == 0 or completed[0] == total_batches:
                    print(f"   Progress: {completed[0]}/{total_batches} batches done")

            return batch_results

        # Process batches concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    batch = futures[future]
                    print(f"  Batch failed ({batch[:2]}...): {e}")
                    # Fallback: basic perturbation
                    with cache_lock:
                        for name in batch:
                            if name not in results:
                                perturbed = self._basic_perturb_column_name(name)
                                results[name] = perturbed
                                if self.cache:
                                    self.cache.put("", name, perturbed, None, None, method="basic_fallback")

        print(f"   Batch precompute done: {len(results)} column-name perturbations")
        return results

    def _llm_batch_column_names(self, column_names: List[str],
                                max_retries: int = 3) -> Dict[str, str]:
        """
        Perturb multiple column names in a single API call.

        Args:
            column_names: list of column names (up to ~20)
            max_retries: maximum retry attempts

        Returns:
            Dict {original_name: perturbed_name}
        """
        system_prompt = """You are a database expert. For each column name, generate 1 semantically equivalent variant that follows common database naming conventions.
Prefer minimal surface-form changes: separators (_ - space camelCase), abbreviations or expansions, case conventions.
Use synonyms ONLY when the column name is a clearly descriptive English phrase; for code-like or very short names (e.g. id, tid, ASSI, CHI), apply surface-form changes only and never swap in semantically different words.
Output strict JSON."""

        # Build the batched prompt
        names_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(column_names)])
        user_prompt = f"""Generate 1 variant for each of the following column names:
{names_list}

Return JSON:
{{"results": {{{", ".join([f'"{name}": "variant"' for name in column_names[:3]])}...}}}}

Return JSON only."""

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = 2.0 * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)

                response = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                    stream=False
                )

                content = response.choices[0].message.content or ""
                result = self._extract_json(content)

                # Extract the results
                batch_results = {}
                results_dict = result.get("results", result)

                for name in column_names:
                    perturbed = results_dict.get(name, "")
                    if perturbed and perturbed != name:
                        batch_results[name] = perturbed
                    else:
                        # LLM returned nothing useful; use the basic method
                        batch_results[name] = self._basic_perturb_column_name(name)

                return batch_results

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        retry_delay = 2.0 * (2 ** attempt) + random.uniform(2, 5)
                        print(f"     Rate limit, waiting {retry_delay:.1f}s...")
                        time.sleep(retry_delay)
                        continue
                else:
                    print(f"     Batch LLM failed: {e}")
                    break

        # All attempts failed; use the basic method
        return {name: self._basic_perturb_column_name(name) for name in column_names}

    def _llm_perturb_column_name_with_context(self, column_name: str,
                                             table_name: Optional[str] = None,
                                             all_columns: Optional[List[str]] = None,
                                             sample_rows: Optional[List[str]] = None,
                                             max_retries: int = 3) -> str:
        """
        Generate a context-aware column-name variant with the LLM
        (prompt strategy inspired by OmniMatch).
        Args:
            column_name: original column name
            table_name: table name
            all_columns: all column names of the table
            sample_rows: sample rows in attribute-value-pair format
            max_retries: maximum retry attempts

        Returns:
            Perturbed column name
        """
        system_prompt = """You are a database expert. Generate column-name variants that keep the semantics identical but differ in form, mimicking real-world database naming differences.

Requirements:
1. Keep the semantics exactly the same
2. Mimic real naming-convention differences (infer the style from the other columns)
3. Infer the column's real meaning from the sample data
4. Output strict JSON only; no explanations"""

        # Build the context-aware prompt (inspired by OmniMatch)
        context_parts = [f"Original column name: {column_name}"]

        if table_name:
            context_parts.append(f"Table name: {table_name}")

        if all_columns:
            # Paper prompt: all column names of the table
            columns_concat = ", ".join(all_columns)
            context_parts.append(f"Other columns in the table: {columns_concat}")

        if sample_rows:
            # Paper prompt: 3-5 sampled rows
            sample_text = "\n".join(sample_rows[:5])
            context_parts.append(f"Sample data rows:\n{sample_text}")

        user_prompt = "\n".join(context_parts) + """

Generate a semantically equivalent variant that follows common database naming conventions. Prefer minimal surface-form changes:
- Separator changes: CustomerID -> Customer_ID, customer-id
- Abbreviations: CustomerID -> CustID; ID -> Identifier
- Case conventions: CustomerID -> customerId, CUSTOMERID
- Synonyms (only for clearly descriptive English phrases): CustomerID -> ClientID
For code-like or very short names (e.g. id, tid, ASSI, CHI), apply casing/separator changes only; never swap in semantically different words.

Return a single most plausible variant as JSON:
{"perturbed": "variant_column_name"}

No explanations."""

        last_error = None
        for attempt in range(max_retries):
            try:
                # Global rate limiting
                global _last_request_time
                current_time = time.time()
                time_since_last = current_time - _last_request_time
                if time_since_last < MIN_REQUEST_INTERVAL:
                    time.sleep(MIN_REQUEST_INTERVAL - time_since_last)

                if attempt > 0:
                    delay = 2.0 * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)

                _last_request_time = time.time()
                response = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    stream=False
                )

                content = response.choices[0].message.content or ""
                result = self._extract_json(content)
                perturbed = result.get("perturbed", "")

                if perturbed and perturbed != column_name:
                    return perturbed
                else:
                    # LLM returned an empty or identical name; use the basic method
                    return self._basic_perturb_column_name(column_name)

            except Exception as e:
                last_error = e
                error_str = str(e)

                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        retry_delay = 2.0 * (2 ** attempt) + random.uniform(2, 5)
                        print(f"Rate limit (column-name perturbation), waiting {retry_delay:.1f}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Rate limit exceeded for the column name; using basic perturbation")
                else:
                    print(f"LLM column-name perturbation failed: {e}")
                    break

        # All retries failed, use basic perturbation
        return self._basic_perturb_column_name(column_name)

    def _basic_perturb_column_name(self, column_name: str) -> str:
        """
        Basic column-name perturbation (fallback method).

        Args:
            column_name: original column name

        Returns:
            perturbed column name
        """
        if not column_name:
            return column_name

        # Pick a perturbation method at random
        method = random.choice(['underscore', 'case', 'abbreviate'])

        if method == 'underscore':
            # CamelCase → snake_case
            # CustomerID → customer_id
            import re
            perturbed = re.sub(r'(?<!^)(?=[A-Z])', '_', column_name).lower()
            if perturbed == column_name.lower():
                # No uppercase letters; try another method
                perturbed = column_name.replace('_', '-')
            return perturbed

        elif method == 'case':
            # Case transformation
            if column_name.isupper():
                return column_name.lower()
            elif column_name.islower():
                return column_name.upper()
            else:
                # camelCase ↔ PascalCase
                if column_name[0].islower():
                    return column_name[0].upper() + column_name[1:]
                else:
                    return column_name[0].lower() + column_name[1:]

        elif method == 'abbreviate':
            # Simple abbreviation strategy
            # CustomerID → CustID
            if len(column_name) > 6:
                # Keep the first 4 characters + suffix
                if column_name.endswith('ID'):
                    return column_name[:4] + 'ID'
                elif column_name.endswith('Name'):
                    return column_name[:4] + 'Name'
                else:
                    return column_name[:6]
            else:
                # Too short; use underscores
                return column_name.replace(' ', '_')

        return column_name

    def suggest_hard_negatives(self, table1_cols: List[str], table2_cols: List[str],
                              key_col: str, sample_data: Optional[Dict[str, List[str]]] = None,
                              max_suggestions: int = 5) -> List[tuple]:
        """
        Use LLM to suggest semantically similar but non-joinable column pairs (hard negatives).

        Args:
            table1_cols: Columns from table 1
            table2_cols: Columns from table 2
            key_col: The actual join key (to exclude)
            sample_data: Optional sample values for each column
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (col1, col2) tuples representing hard negative pairs
        """
        if not self.enable_llm or self.client is None:
            # Fallback: random pairs
            candidates1 = [c for c in table1_cols if c != key_col]
            candidates2 = [c for c in table2_cols if c != key_col]
            if not candidates1 or not candidates2:
                return []
            pairs = []
            for _ in range(min(max_suggestions, len(candidates1) * len(candidates2))):
                pairs.append((random.choice(candidates1), random.choice(candidates2)))
            return pairs

        try:
            system_prompt = """You are a database expert. Identify column pairs that "look related but should NOT be joined" (hard negatives) for training a table-join model.

Good hard negatives:
1. Semantically similar but different entities (e.g. customer_id vs order_id)
2. Hierarchically related but not directly joinable (e.g. city vs country)
3. Related but not matching (e.g. product_name vs category_name)
4. Same data type but different semantics (e.g. two different date columns)

Output strict JSON."""

            user_prompt = f"""Table 1 columns: {json.dumps(table1_cols, ensure_ascii=False)}
Table 2 columns: {json.dumps(table2_cols, ensure_ascii=False)}
Actual join key: {key_col}

Find {max_suggestions} hard-negative column pairs (look related but should not join).

Return JSON:
{{"hard_negatives": [
  {{"col1": "table1_column", "col2": "table2_column", "reason": "why this is a hard negative"}},
  ...
]}}

Return JSON only, no explanations."""

            response = self.client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                stream=False
            )

            content = response.choices[0].message.content or ""
            result = self._extract_json(content)

            pairs = []
            for item in result.get("hard_negatives", []):
                col1 = item.get("col1")
                col2 = item.get("col2")
                if col1 in table1_cols and col2 in table2_cols and col1 != key_col and col2 != key_col:
                    pairs.append((col1, col2))

            return pairs[:max_suggestions]

        except Exception as e:
            print(f"LLM hard-negative suggestion failed: {e}")
            # Fallback to random
            candidates1 = [c for c in table1_cols if c != key_col]
            candidates2 = [c for c in table2_cols if c != key_col]
            if not candidates1 or not candidates2:
                return []
            pairs = []
            for _ in range(min(max_suggestions, len(candidates1), len(candidates2))):
                pairs.append((random.choice(candidates1), random.choice(candidates2)))
            return pairs


# ====== Utility Functions ======

def test_llm_connection():
    """Test if LLM API is accessible."""
    augmenter = LLMAugmenter(enable_llm=True)
    if not augmenter.enable_llm:
        print(" LLM not available")
        return False

    try:
        test_values = ["John Smith", "2024-01-15", "New York"]
        result = augmenter._llm_perturb_batch(test_values, "test_column")
        print(f" LLM connection successful")
        print(f"   Test input: {test_values}")
        print(f"   Test output: {result}")
        return True
    except Exception as e:
        print(f" LLM connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the LLM augmenter
    print("=" * 50)
    print("Testing LLM Augmenter")
    print("=" * 50)
    test_llm_connection()
