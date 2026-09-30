"""
Perturbation result cache manager
Avoids repeated LLM API calls and speeds up data generation.

Usage:
    from perturbation_cache import create_cache_for_dataset

    # Create the cache
    cache = create_cache_for_dataset("CAN_ALL")

    # Query the cache
    result = cache.get("orders", "CustomerID")

    # Add to the cache
    cache.put("orders", "CustomerID", "customer_id", method="llm")

    # Save the cache
    cache.save()
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path


class PerturbationCache:
    """
    Cache manager for column-name perturbation results.

    Features:
    1. Cache LLM-generated column-name perturbations
    2. Context-aware lookup support
    3. Automatic persistence to a JSON file
    4. CSV export for manual editing
    """

    def __init__(self, cache_file: str, llm_model: str = "deepseek-ai/DeepSeek-V3"):
        """
        Args:
            cache_file: cache file path (JSON)
            llm_model: LLM model identifier (recorded as metadata)
        """
        self.cache_file = Path(cache_file)
        self.llm_model = llm_model
        self.cache_data = {
            "metadata": {
                "dataset": "",
                "created_at": datetime.now().isoformat(),
                "llm_model": llm_model,
                "total_perturbations": 0
            },
            "perturbations": {}
        }

        # Load the existing cache
        self._load_cache()

    def _load_cache(self):
        """Load the cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.cache_data = loaded
                    print(f" Loaded cache: {self.cache_file}")
                    print(f"   Existing perturbations: {self.cache_data['metadata']['total_perturbations']}")
            except Exception as e:
                print(f"Failed to load cache: {e}; a new one will be created")
        else:
            print(f"Cache file does not exist; creating a new one: {self.cache_file}")

    def save(self):
        """Persist the cache to disk."""
        try:
            # Ensure the directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Update the metadata
            self.cache_data['metadata']['total_perturbations'] = len(self.cache_data['perturbations'])
            self.cache_data['metadata']['last_updated'] = datetime.now().isoformat()

            # Write the file
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)

            print(f" Cache saved: {self.cache_file}")
            print(f"   Total perturbations: {self.cache_data['metadata']['total_perturbations']}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _make_cache_key(self,
                       table_name: str,
                       column_name: str,
                       all_columns: Optional[List[str]] = None,
                       sample_rows: Optional[List[str]] = None) -> str:
        """
        Build the cache key (table name + column name + context hash).

        Args:
            table_name: table name
            column_name: column name
            all_columns: all column names of the table (optional)
            sample_rows: sample rows (optional)

        Returns:
            Cache key of the form table_name:column_name:context_hash
        """
        # Base key
        base_key = f"{table_name}:{column_name}"

        # Optional context hash
        if all_columns or sample_rows:
            context_str = ""
            if all_columns:
                context_str += "|".join(sorted(all_columns))
            if sample_rows:
                context_str += "|".join(sample_rows)

            # MD5 hash (first 8 hex digits)
            context_hash = hashlib.md5(context_str.encode('utf-8')).hexdigest()[:8]
            return f"{base_key}:{context_hash}"

        return base_key

    def get(self,
            table_name: str,
            column_name: str,
            all_columns: Optional[List[str]] = None,
            sample_rows: Optional[List[str]] = None) -> Optional[str]:
        """
        Look up a perturbation result in the cache.

        Args:
            table_name: table name
            column_name: column name
            all_columns: all column names of the table (optional)
            sample_rows: sample rows (optional)

        Returns:
            The perturbed column name, or None on a cache miss
        """
        # Try the exact match first (with the context hash)
        full_key = self._make_cache_key(table_name, column_name, all_columns, sample_rows)
        if full_key in self.cache_data['perturbations']:
            return self.cache_data['perturbations'][full_key]['perturbed']

        # Fallback: match on table + column only (ignore context)
        base_key = f"{table_name}:{column_name}"
        for key, value in self.cache_data['perturbations'].items():
            if key.startswith(base_key):
                print(f"   Using approximate cache entry: {key}")
                return value['perturbed']

        return None

    def put(self,
            table_name: str,
            column_name: str,
            perturbed_name: str,
            all_columns: Optional[List[str]] = None,
            sample_rows: Optional[List[str]] = None,
            method: str = "llm") -> str:
        """
        Store a perturbation result in the cache.

        Args:
            table_name: table name
            column_name: original column name
            perturbed_name: perturbed column name
            all_columns: all column names of the table (optional)
            sample_rows: sample rows (optional)
            method: perturbation method ("llm" or "basic")

        Returns:
            the cache key
        """
        cache_key = self._make_cache_key(table_name, column_name, all_columns, sample_rows)

        # Hash the sample_rows (recorded only; full data not stored)
        sample_rows_hash = ""
        if sample_rows:
            sample_rows_hash = hashlib.md5("|".join(sample_rows).encode('utf-8')).hexdigest()[:8]

        self.cache_data['perturbations'][cache_key] = {
            "original": column_name,
            "perturbed": perturbed_name,
            "table_name": table_name,
            "all_columns": all_columns[:5] if all_columns else [],  # keep only the first 5 names (saves space)
            "sample_rows_hash": sample_rows_hash,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }

        return cache_key

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        perturbations = self.cache_data['perturbations']

        llm_count = sum(1 for v in perturbations.values() if v.get('method') == 'llm')
        basic_count = sum(1 for v in perturbations.values() if v.get('method') == 'basic')

        return {
            "total": len(perturbations),
            "llm_generated": llm_count,
            "basic_generated": basic_count,
            "cache_file": str(self.cache_file),
            "cache_size_kb": self.cache_file.stat().st_size / 1024 if self.cache_file.exists() else 0
        }

    def export_to_csv(self, output_file: str):
        """Export the cache to CSV (for manual editing)."""
        import csv

        def clean_text(text):
            """Strip newlines and special characters from text."""
            if not text:
                return ""
            # Replace newlines/tabs with spaces and collapse whitespace
            text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Collapse runs of whitespace into one space
            import re
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['table_name', 'original', 'perturbed', 'all_columns', 'method', 'timestamp'])

            for key, value in self.cache_data['perturbations'].items():
                # Clean every field
                table_name = clean_text(value['table_name'])
                original = clean_text(value['original'])
                perturbed = clean_text(value['perturbed'])
                # Clean each name in the all_columns list
                all_cols = [clean_text(col) for col in value.get('all_columns', [])]
                all_columns_str = ','.join(all_cols)

                writer.writerow([
                    table_name,
                    original,
                    perturbed,
                    all_columns_str,
                    value['method'],
                    value['timestamp']
                ])

        print(f" Cache exported to CSV: {output_file}")


# ====== Helper functions ======

def create_cache_for_dataset(dataset_name: str, output_dir: str = None) -> PerturbationCache:
    """
    Create a cache manager for a dataset.

    Args:
        dataset_name: dataset name (e.g. CAN_ALL)
        output_dir: output directory (inferred when None)

    Returns:
        PerturbationCache instance
    """
    if output_dir is None:
        output_dir = f"datasets/Lake/{dataset_name}_LabelFree"

    cache_file = os.path.join(output_dir, "column_perturbations.json")
    cache = PerturbationCache(cache_file)
    cache.cache_data['metadata']['dataset'] = dataset_name

    return cache


# ====== Test code ======

if __name__ == "__main__":
    # Test the cache
    print("=" * 80)
    print("Testing the perturbation cache manager")
    print("=" * 80)

    # Create a test cache
    cache = PerturbationCache("test_cache.json")

    # Add some test data
    cache.put("orders", "CustomerID", "customer_id",
              all_columns=["CustomerID", "OrderDate", "Amount"],
              sample_rows=["CustomerID: 123, OrderDate: 2024-01-01, Amount: 99.99"],
              method="llm")

    cache.put("products", "ProductName", "prod_name",
              method="basic")

    # Query the cache
    result1 = cache.get("orders", "CustomerID")
    print(f"\nQuery result: {result1}")

    result2 = cache.get("products", "ProductName")
    print(f"Query result: {result2}")

    result3 = cache.get("users", "UserID")
    print(f"Query result (miss): {result3}")

    # Save the cache
    cache.save()

    # Show statistics
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")

    # Export to CSV
    cache.export_to_csv("test_cache.csv")

    print("\n Test complete")
