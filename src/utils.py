"""
HyperJoin utility functions - consolidated helpers
Contains the original utils + training logging + model verification + misc helpers
"""

import os
import json
import time
import math
import random
import hashlib
import logging
import warnings
import numpy as np
from io import StringIO
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

# =============================================================================
# Core utility functions (original utils.py)
# =============================================================================

logger = logging.getLogger(__name__)

def fix_seed(seed=2024):
    """Set the random seed."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_seed(seed=2024):
    """Set the random seed - compatibility alias."""
    fix_seed(seed)

def process_csv_without_nul(file):
    """Handle CSV files that contain NUL characters."""
    contents = file.read().replace('\\x00', '')
    f = StringIO(contents)
    return f

def load_query(path):
    """Load the query data."""
    query = np.load(path, allow_pickle=True)
    return query

def cal_score(query, target, t=0.2, device='cpu'):
    """Compute the similarity score."""
    query_vectors = torch.tensor(query).to(device)
    target_vectors = torch.tensor(target).to(device)
    query_vectors = F.normalize(query_vectors, p=2, dim=1)
    target_vectors = F.normalize(target_vectors, p=2, dim=1)
    distance_squared = torch.sum((query_vectors[:, None] - target_vectors) ** 2, dim=2)
    euclidean_distances = torch.sqrt(distance_squared)
    values = torch.min(euclidean_distances, dim=1)[0]
    binary_values = torch.where(values <= t, torch.tensor(1).to(device), torch.tensor(0).to(device))
    result = torch.sum(binary_values)
    return (result*1.0/query.shape[0]).item()

def cal_score_cos(query, target):
    """Compute the cosine-similarity score."""
    query_vectors = torch.tensor(query)
    target_vectors = torch.tensor(target)
    similarity_matrix = F.cosine_similarity(query_vectors[:, None, :], target_vectors[None, :, :], dim=2)
    values, topk_indices = torch.topk(similarity_matrix, k=1, dim=1)
    binary_values = torch.where(values > 0.9, torch.tensor(1), torch.tensor(0))
    result = torch.sum(binary_values)
    return (result*1.0/query.shape[0]).item()

def cal_NDCG(query, lake, pred, gt, k):
    """Compute the NDCG metric - fixed version with zero-division guard."""
    ndcg = 0
    for i in range(len(query)):
        dcg_pred = DCG(query[i], lake[pred[i].tolist()], k)
        dcg_ideal = DCG(query[i], lake[[int(x) for x in gt[i]]], k)
        if dcg_ideal > 0:
            ndcg += dcg_pred / dcg_ideal
    return ndcg/len(query)

def DCG(query, target_set, k):
    """Compute DCG."""
    dcg = 0
    for i in range(min(k, len(target_set))):
        dcg += cal_score(query, target_set[i])/math.log2(i + 2)
    return dcg

def parse_options(parser):
    """Parse command-line arguments."""
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--plm', type=str, default='fasttext')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=32)
    parser.add_argument('--n_proxy_sets', type=int, default=90)
    parser.add_argument('--n_elements', type=int, default=50)
    parser.add_argument('--d', type=int, default=300, help='dimension of each cell')
    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--type', type=str, default="mat")
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--datasets', type=str, default='WikiTable')
    parser.add_argument('--version', type=str, default='t=0.2_gendata_mat_level')
    parser.add_argument('--list_size', type=int, default=3)
    parser.add_argument('--da', type=str, default='True')
    parser.add_argument('--num_hypergraph_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=6)
    return parser.parse_args()

# =============================================================================
# Training-logging utility
# =============================================================================

class TrainingLogger:
    """Training logger."""

    def __init__(self, log_dir="logs", experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        if experiment_name is None:
            experiment_name = f"hyperjoin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(exist_ok=True)

        self.setup_logging()

        self.stats = {
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': 0,
            'batch_losses': [],
            'epoch_losses': [],
            'eval_scores': [],
            'best_score': 0.0,
            'model_saves': [],
            'config': {}
        }

        self.logger.info(f"Training logger initialized: {self.experiment_name}")

    def setup_logging(self):
        """Set up the logging system."""
        log_file = self.exp_dir / "training.log"

        self.logger = logging.getLogger(f"TrainingLogger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log_config(self, config_dict):
        """Record the training configuration."""
        self.stats['config'] = config_dict

        self.logger.info("=" * 60)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 60)

        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")

        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def start_training(self, total_epochs):
        """Mark the start of training."""
        self.stats['start_time'] = time.time()
        self.stats['total_epochs'] = total_epochs

        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total epochs: {total_epochs}")

    def log_epoch_start(self, epoch):
        """Record the start of an epoch."""
        self.stats['current_epoch'] = epoch
        self.logger.info(f"EPOCH {epoch + 1}/{self.stats['total_epochs']}")

    def log_batch(self, epoch, batch_id, total_batches, loss, additional_metrics=None):
        """Record batch information."""
        batch_info = {
            'epoch': epoch,
            'batch': batch_id,
            'loss': float(loss),
            'timestamp': time.time()
        }

        if additional_metrics:
            batch_info.update(additional_metrics)

        self.stats['batch_losses'].append(batch_info)

        if batch_id % max(1, total_batches // 10) == 0 or batch_id == total_batches - 1:
            progress = (batch_id + 1) / total_batches * 100
            self.logger.info(f"Batch {batch_id + 1}/{total_batches} ({progress:.1f}%) - Loss: {loss:.6f}")

    def log_epoch_end(self, epoch, avg_loss, epoch_time):
        """Record the end of an epoch."""
        epoch_info = {
            'epoch': epoch,
            'avg_loss': float(avg_loss),
            'time': float(epoch_time),
            'timestamp': time.time()
        }

        self.stats['epoch_losses'].append(epoch_info)

        self.logger.info(f"Epoch {epoch + 1} completed - Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        self.save_progress()

    def log_evaluation(self, epoch, score, metric_name="Score"):
        """Record an evaluation result."""
        eval_info = {
            'epoch': epoch,
            'score': float(score),
            'metric_name': metric_name,
            'timestamp': time.time()
        }

        self.stats['eval_scores'].append(eval_info)

        if score > self.stats['best_score']:
            self.stats['best_score'] = float(score)
            self.logger.info(f" NEW BEST {metric_name}: {score:.6f} at epoch {epoch + 1}")
        else:
            self.logger.info(f"Evaluation - {metric_name}: {score:.6f} (Best: {self.stats['best_score']:.6f})")

    def log_model_save(self, epoch, score, model_path, additional_info=None):
        """Record a model save."""
        save_info = {
            'epoch': epoch,
            'score': float(score),
            'model_path': str(model_path),
            'timestamp': time.time(),
            'file_exists': os.path.exists(model_path)
        }

        if additional_info:
            save_info.update(additional_info)

        self.stats['model_saves'].append(save_info)

        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / 1e6
            self.logger.info(f" Model saved: {model_path} ({file_size:.2f} MB)")
        else:
            self.logger.error(f" Model save failed: {model_path}")

    def save_progress(self):
        """Persist the training progress."""
        progress_file = self.exp_dir / "progress.json"

        save_data = self.stats.copy()
        save_data['last_update'] = time.time()

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    def create_summary(self):
        """Create the training summary."""
        if not self.stats['start_time']:
            return

        total_time = time.time() - self.stats['start_time']

        total_batches = len(self.stats['batch_losses'])
        avg_epoch_time = np.mean([e['time'] for e in self.stats['epoch_losses']]) if self.stats['epoch_losses'] else 0
        final_loss = self.stats['epoch_losses'][-1]['avg_loss'] if self.stats['epoch_losses'] else 0

        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.stats['current_epoch'] + 1,
            'total_time_hours': total_time / 3600,
            'total_batches': total_batches,
            'best_score': self.stats['best_score'],
            'final_loss': final_loss,
            'avg_epoch_time': avg_epoch_time,
            'models_saved': len(self.stats['model_saves']),
            'completed_at': datetime.now().isoformat(),
            'config': self.stats['config']
        }

        summary_file = self.exp_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info("=" * 60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Total Epochs: {summary['total_epochs']}")
        self.logger.info(f"Total Time: {summary['total_time_hours']:.2f} hours")
        self.logger.info(f"Best Score: {summary['best_score']:.6f}")
        self.logger.info(f"Models Saved: {summary['models_saved']}")

        return summary

# =============================================================================
# Model-save verification utility
# =============================================================================

class ModelSaveVerifier:
    """Model-save verifier."""

    def __init__(self, base_dir="check", backup_dir="model_backups"):
        self.base_dir = Path(base_dir)
        self.backup_dir = Path(backup_dir)

        self.base_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        self.model_registry = {}
        self.load_registry()

    def load_registry(self):
        """Load the model registry."""
        registry_file = self.base_dir / "model_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load model registry: {e}")
                self.model_registry = {}

    def save_registry(self):
        """Save the model registry."""
        registry_file = self.base_dir / "model_registry.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_registry, f, indent=2, ensure_ascii=False)

    def calculate_file_hash(self, file_path: Path) -> str:
        """Compute the file hash."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def verify_model_integrity(self, model_path: Path, model_class=None) -> Dict[str, Any]:
        """Verify model integrity."""
        verification_result = {
            'path': str(model_path),
            'exists': False,
            'loadable': False,
            'file_size': 0,
            'file_hash': None,
            'torch_version': torch.__version__,
            'verification_time': datetime.now().isoformat(),
            'error_messages': []
        }

        try:
            if not model_path.exists():
                verification_result['error_messages'].append(f"Model file does not exist: {model_path}")
                return verification_result

            verification_result['exists'] = True
            verification_result['file_size'] = model_path.stat().st_size
            verification_result['file_hash'] = self.calculate_file_hash(model_path)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    state_dict = torch.load(model_path, map_location='cpu')

                verification_result['loadable'] = True
                verification_result['state_dict_keys'] = list(state_dict.keys())
                verification_result['parameter_count'] = sum(p.numel() for p in state_dict.values())

                if model_class is not None:
                    try:
                        model = model_class()
                        model.load_state_dict(state_dict)
                        verification_result['model_compatible'] = True
                    except Exception as e:
                        verification_result['model_compatible'] = False
                        verification_result['error_messages'].append(f"Model compatibility error: {e}")

            except Exception as e:
                verification_result['error_messages'].append(f"Failed to load model: {e}")

        except Exception as e:
            verification_result['error_messages'].append(f"Verification error: {e}")

        return verification_result

    def safe_save_model(self, model, folder_name: str, version: str,
                       score: float = 0.0, epoch: int = 0,
                       additional_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Save a model safely (with backup and verification)."""
        save_result = {
            'success': False,
            'model_path': None,
            'backup_path': None,
            'verification': None,
            'timestamp': datetime.now().isoformat(),
            'error_messages': []
        }

        try:
            save_dir = self.base_dir / folder_name
            save_dir.mkdir(exist_ok=True)

            model_filename = f"{version}_trained.pth"
            model_path = save_dir / model_filename

            if model_path.exists():
                backup_filename = f"{version}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                backup_path = self.backup_dir / backup_filename

                try:
                    import shutil
                    shutil.copy2(model_path, backup_path)
                    save_result['backup_path'] = str(backup_path)
                    print(f" Previous model backed up to: {backup_path}")
                except Exception as e:
                    save_result['error_messages'].append(f"Backup failed: {e}")

            torch.save(model.state_dict(), model_path)
            save_result['model_path'] = str(model_path)

            verification = self.verify_model_integrity(model_path)
            save_result['verification'] = verification

            if not verification['loadable']:
                save_result['error_messages'].append("Saved model is not loadable")
                return save_result

            metadata = {
                'version': version,
                'score': score,
                'epoch': epoch,
                'timestamp': save_result['timestamp'],
                'file_size': verification['file_size'],
                'file_hash': verification['file_hash'],
                'parameter_count': verification.get('parameter_count', 0),
                'torch_version': torch.__version__,
                'additional_info': additional_info or {}
            }

            metadata_path = save_dir / f"{version}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            model_key = f"{folder_name}/{version}"
            self.model_registry[model_key] = metadata
            self.save_registry()

            save_result['success'] = True

            print(f" Model saved successfully:")
            print(f"   Path: {model_path}")
            print(f"   Size: {verification['file_size'] / 1e6:.2f} MB")
            print(f"   Parameters: {verification.get('parameter_count', 0):,}")
            print(f"   Score: {score:.6f}")

        except Exception as e:
            save_result['error_messages'].append(f"Save failed: {e}")
            print(f" Model save failed: {e}")

        return save_result

# =============================================================================
# Convenience functions
# =============================================================================

def save_model(model, folder_name, version, score=0.0, epoch=0, use_verifier=True):
    """Enhanced model-save function."""
    if use_verifier:
        try:
            verifier = ModelSaveVerifier()
            save_result = verifier.safe_save_model(
                model=model,
                folder_name=folder_name,
                version=version,
                score=score,
                epoch=epoch
            )
            return save_result['model_path'] if save_result['success'] else None
        except Exception as e:
            print(f"Enhanced save failed: {e}, using basic method...")

    # Fall back to the basic save method
    os.makedirs(folder_name, exist_ok=True)
    model_path = os.path.join(folder_name, f"{version}_trained.pth")
    torch.save(model.state_dict(), model_path)
    print("Model checkpoint saved (basic method)!")
    return model_path

def create_experiment_logger(experiment_name=None, log_dir="logs"):
    """Create an experiment logger."""
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)

def validate_data_files(output_dir, tau, args=None):
    """Validate the generated data files (ported from DataGen_hyperjoin.py)."""
    validation_report = []
    is_valid = True

    print("=== Data-generation correctness validation ===")

    files_to_validate = {
        'target_csv': f"{output_dir}/target.csv",
        'query_csv': f"{output_dir}/query.csv",
        'index_csv': f"{output_dir}/t={tau}/test/index.csv",
        'target_npy': f"{output_dir}/target.npy",
        'query_npy': f"{output_dir}/query.npy",
        'anchor_npy': f"{output_dir}/t={tau}/train/mat_level/anchor.npy",
        'auglist_npy': f"{output_dir}/t={tau}/train/mat_level/auglist.npy",
        'auglist_y_csv': f"{output_dir}/t={tau}/train/mat_level/auglist_y.csv"
    }

    # Basic file-existence check
    missing_files = []
    for file_type, file_path in files_to_validate.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            is_valid = False

    if missing_files:
        validation_report.append(f" Missing files: {missing_files}")
        print(f" Missing files found: {len(missing_files)}")
        return False, validation_report
    else:
        print(" All required files exist")

    return is_valid, validation_report

# =============================================================================
# Shared text/column encoding helpers (previously duplicated in train / search / analyze)
#
# These three functions used to exist separately in the training and search
# scripts; they are collected here to remove the duplication. The body matches
# the search version (which adds an UNK/PAD-id fallback for out-of-range
# vocabularies; with normal vocabularies its output is identical to the
# other implementation).
# =============================================================================

def get_gpu_memory_mb():
    """Return the peak GPU memory of the current process in MB (0 without a GPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def get_cpu_memory_mb():
    """Return the RSS memory of the current process in MB (0 when psutil is missing)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0


def tokenize_pad(text: str, vocab: dict, max_len: int = 10, vocab_size: int = None) -> torch.Tensor:
    """Split ``text`` on underscores, map tokens to ids and right-pad with PAD; returns a long tensor.

    Args:
        text: input string (table_name / column_name); a ``.csv`` suffix is stripped.
        vocab: {token: id} vocabulary; must contain ``<PAD>`` and ``<UNK>``.
        max_len: final tensor length; PAD-fills or truncates as needed.
        vocab_size: when given, clamps all returned ids to ``[0, vocab_size-1]``
            to guard against Embedding index errors (mismatched vocab/model seen historically).
    """
    tokens = text.replace('.csv', '').split('_')
    ids = []

    # Safely resolve the special-token ids; fall back to 1/0 when out of range
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
    """Aggregate the row-level FastText vectors of a column into a fixed-dim [mean, std, max, min] feature.

    Returns a 1200-dim zero vector (300*4) for empty inputs.

    Args:
        col_data: (num_rows, feat_dim) Tensor or a numpy array convertible to one.
    """
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


def build_vocabulary(metadata):
    """Build a shared vocabulary from the table_name / column_name fields of the metadata list.

    ``<PAD>`` / ``<UNK>`` are fixed at ids 0/1; other tokens increment by first appearance.
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab_idx = 2

    for meta in metadata:
        table_name = meta.get('table_name', '').replace('.csv', '')
        for token in table_name.split('_'):
            if token and token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1

        column_name = meta.get('column_name', '')
        for token in column_name.split('_'):
            if token and token not in vocab:
                vocab[token] = vocab_idx
                vocab_idx += 1

    return vocab


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core utilities
    'fix_seed', 'set_seed', 'process_csv_without_nul', 'load_query',
    'cal_score', 'cal_score_cos', 'cal_NDCG', 'DCG', 'parse_options',

    # Shared text/column encoding helpers
    'get_gpu_memory_mb', 'get_cpu_memory_mb',
    'tokenize_pad', 'aggregate_content', 'build_vocabulary',

    # Logging and verification
    'TrainingLogger', 'ModelSaveVerifier', 'save_model',
    'create_experiment_logger', 'validate_data_files'
]