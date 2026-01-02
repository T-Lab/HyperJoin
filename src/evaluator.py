import os
import pandas as pd


class DetailedEvaluator:
    """Simplified evaluator for search results"""

    def __init__(self, datasets):
        self.datasets = datasets

    def load_ground_truth(self):
        """Load ground truth"""
        index_path = f"datasets/Lake/{self.datasets}/t=0.2/test/index.csv"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Ground truth file not found: {index_path}")

        try:
            df = pd.read_csv(index_path, header=None, sep=',', engine='python')
        except pd.errors.ParserError:
            import csv
            rows = []
            with open(index_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)

            max_cols = max(len(row) for row in rows) if rows else 0

            for row in rows:
                while len(row) < max_cols:
                    row.append('')

            df = pd.DataFrame(rows)

        ground_truth = {}

        for gt_row_idx in range(len(df)):
            query_idx = gt_row_idx
            row = df.iloc[gt_row_idx]

            target_indices = []
            for col_name in df.columns:
                value = row[col_name]
                if pd.notna(value) and str(value).strip() != '':
                    try:
                        target_indices.append(int(float(value)))
                    except (ValueError, TypeError):
                        continue

            if target_indices:
                ground_truth[query_idx] = target_indices

        print(f"Ground truth: {len(ground_truth)} queries")

        return ground_truth

    def evaluate_precision_recall(self, search_results, ground_truth, topk_list=[1, 5, 10, 15, 20, 25]):
        """
        Calculate Precision@K and Recall@K
        """
        results = {'recall': {}, 'precision': {}, 'f1': {}}

        for kk in topk_list:
            recall_sum = 0.0
            precision_sum = 0.0
            num_queries = 0

            for query_idx in range(len(search_results)):
                if query_idx in ground_truth:
                    pred_topk = search_results[query_idx][:kk]
                    pred_set = set(pred_topk)

                    gt_all = ground_truth[query_idx]
                    gt_full = set(gt_all)

                    hits = len(pred_set & gt_full)

                    recall_denominator = len(gt_all)
                    query_recall = hits / recall_denominator if recall_denominator > 0 else 0.0

                    query_precision = hits / kk

                    recall_sum += query_recall
                    precision_sum += query_precision
                    num_queries += 1

            avg_recall = recall_sum / num_queries if num_queries > 0 else 0.0
            avg_precision = precision_sum / num_queries if num_queries > 0 else 0.0

            if avg_recall + avg_precision > 0:
                f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
            else:
                f1 = 0.0

            results['recall'][kk] = avg_recall
            results['precision'][kk] = avg_precision
            results['f1'][kk] = f1

        # Print results table
        print(f"\n{'='*70}")
        print(f"{'K':<5} {'Precision@K':<18} {'Recall@K':<18} {'F1@K':<18}")
        print(f"{'='*70}")
        for kk in topk_list:
            print(f"{kk:<5} {results['precision'][kk]:<18.4f} {results['recall'][kk]:<18.4f} {results['f1'][kk]:<18.4f}")
        print(f"{'='*70}\n")

        return results

    def run_complete_evaluation(self, search_results):
        """Run complete evaluation"""
        ground_truth = self.load_ground_truth()

        metrics = self.evaluate_precision_recall(search_results, ground_truth)

        flat_results = {}
        for k in metrics['precision'].keys():
            flat_results[f'Precision@{k}'] = metrics['precision'][k]
            flat_results[f'Recall@{k}'] = metrics['recall'][k]
            flat_results[f'F1@{k}'] = metrics['f1'][k]

        return flat_results


def enhance_search_output(datasets, search_results):
    """Enhanced search output main function"""
    evaluator = DetailedEvaluator(datasets)

    results = evaluator.run_complete_evaluation(search_results)

    return results
