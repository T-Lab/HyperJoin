"""
Dataset wrappers for HyperJoin.

Wraps the .npy + .pkl files produced by DataGen / datagen as torch
Datasets so the training and search scripts can index per-column FastText
embeddings together with their metadata.
"""

import pickle
import os

import numpy as np
import torch
import torch.utils.data as Data


class TestDatasetHyperJoin(Data.Dataset):
    """
    Test dataset for HyperJoin, with metadata support.

    Each sample corresponds to one column; ``__getitem__`` returns the row-level
    vector matrix of that column. Metadata is available via ``metadata`` and
    ``get_table_column_info``.
    """
    def __init__(self,
                 test_path_mat,  # path to the test-set matrix
                 metadata_path,  # path to the metadata file
                 da=None,
                 plm='fasttext'):

        self.plm = plm
        self.da = da
        mats = np.load(test_path_mat, allow_pickle=True)
        self.data_mats = mats  # self.data_mats[k] is a matrix of the kth col
        self.data_size = len(self.data_mats)  # of cols in total

        # Load the metadata (two historical formats are supported):
        #   new format (datagen): dict with keys 'metadata' / 'table_to_id' / 'column_to_id'
        #   old format (early data-generation tooling): a plain list [{'table_name': ..., 'column_name': ...}, ...]
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata_loaded = pickle.load(f)
                # Normalise: keep the per-column list in ``self.metadata`` so training code
                # can index it directly, and expose the id maps as separate attributes.
                if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
                    self.metadata = metadata_loaded['metadata']  # new format
                    self.table_to_id = metadata_loaded.get('table_to_id', {})
                    self.column_to_id = metadata_loaded.get('column_to_id', {})
                else:
                    self.metadata = metadata_loaded              # old format
                    self.table_to_id = {}
                    self.column_to_id = {}
        else:
            self.metadata = None
            self.table_to_id = {}
            self.column_to_id = {}

    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset."""
        x = self.data_mats[index]
        return x

    def get_metadata(self, index):
        """Return the metadata of the given index."""
        if self.metadata and index < len(self.metadata):
            return self.metadata[index]
        return None

    def get_table_column_info(self, index):
        """Return the real table name / column name and their IDs."""
        if self.metadata and index < len(self.metadata):
            item_meta = self.metadata[index]
            table_name = item_meta['table_name']
            column_name = item_meta['column_name']

            # Map to IDs (when the id maps are available)
            table_id = self.table_to_id.get(table_name, 0)
            column_id = self.column_to_id.get(column_name, index)

            return table_id, column_id, table_name, column_name
        return 0, index, "unknown_table", "unknown_column"  # graceful fallback

    @staticmethod
    def pad(batch):
        """Merge the embed_mats of different cols into a big mat
        Args:
            batch (list of arrays): a list of arrays, each array is respect to a col
        Returns:
            LongTensor: a big mat
            LongTensor: index, which indicates that each col contains ? cells
        """
        mat = batch
        index_mat = []
        concat_array = np.concatenate(mat, axis=0)
        for arr in mat:
            index_mat.append(arr.shape[0])
        return torch.tensor(concat_array, dtype=torch.float32), torch.tensor(index_mat)


class MyDatasetHyperJoin(Data.Dataset):
    """
    Training dataset for HyperJoin, with metadata support.
    """
    def __init__(self,
                 anchor_path,
                 auglist_path,
                 list_size,
                 metadata_path=None,
                 training='true',
                 plm='fasttext'):

        query = np.load(anchor_path, allow_pickle=True)
        self.anchor_mats = query
        self.data_size = len(self.anchor_mats)

        item = np.load(auglist_path, allow_pickle=True)
        item_list = []

        for i in range(0, len(item), list_size):
            sub_list = item[i:i + list_size].tolist()
            item_list.append(sub_list)
        self.item_mats = item_list
        indicies = [k * list_size for k in range(len(query))]
        pos = item[indicies]
        self.pos_mats = pos
        self.plm = plm
        self.training = training
        self.list_size = list_size

        # Load the metadata (two historical formats are supported):
        #   new format (datagen): dict with keys 'metadata' / 'table_to_id' / 'column_to_id'
        #   old format (early data-generation tooling): a plain list [{'table_name': ..., 'column_name': ...}, ...]
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata_loaded = pickle.load(f)
                if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
                    self.metadata = metadata_loaded['metadata']  # new format
                    self.table_to_id = metadata_loaded.get('table_to_id', {})
                    self.column_to_id = metadata_loaded.get('column_to_id', {})
                else:
                    self.metadata = metadata_loaded              # old format
                    self.table_to_id = {}
                    self.column_to_id = {}
        else:
            self.metadata = None
            self.table_to_id = {}
            self.column_to_id = {}

    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset."""
        query = self.anchor_mats[index]
        item_list = self.item_mats[index]
        pos = self.pos_mats[index]
        index_list = []
        for item in item_list:
            index_list.append(item.shape[0])
        return query, pos, item_list, index_list

    def get_metadata(self, index):
        """Return the metadata of the given index."""
        if self.metadata and index < len(self.metadata):
            return self.metadata[index]
        return None

    def get_table_column_info(self, index):
        """Return the real table name / column name and their IDs."""
        if self.metadata and index < len(self.metadata):
            item_meta = self.metadata[index]
            table_name = item_meta['table_name']
            column_name = item_meta['column_name']

            # Map to IDs (when the id maps are available)
            table_id = self.table_to_id.get(table_name, 0)
            column_id = self.column_to_id.get(column_name, index)

            return table_id, column_id, table_name, column_name
        return 0, index, "unknown_table", "unknown_column"  # graceful fallback

    @staticmethod
    def pad(batch):
        query, pos, item_list, index_list = zip(*batch)

        # Fix: handle batched data correctly -
        # each query/pos is a single column, not a group of columns
        concat_query = np.concatenate(query, axis=0)  # cells of all query columns
        concat_pos = np.concatenate(pos, axis=0)      # cells of all pos columns

        # item_list: each sample's item_list is a list of columns
        all_items = []
        merged_list = []

        for sample_items in item_list:
            for item in sample_items:
                all_items.append(item)
                merged_list.append(item.shape[0])  # cell count of each item column

        concat_item = np.concatenate(all_items, axis=0)

        # query/pos indices: each sample contributes one column
        query_index = [arr.shape[0] for arr in query]  # cell count of each query column
        pos_index = [arr.shape[0] for arr in pos]      # cell count of each pos column

        return (torch.tensor(concat_query, dtype=torch.float32),
                torch.tensor(concat_pos, dtype=torch.float32),
                torch.tensor(concat_item, dtype=torch.float32),
                torch.tensor(query_index),
                torch.tensor(pos_index),
                torch.tensor(merged_list))


class RankDatasetHyperJoin(Data.Dataset):
    """
    Ranking dataset for HyperJoin.
    """
    def __init__(self,
                 anchor_path,
                 auglist_path,
                 list_size,
                 metadata_path=None,
                 training='true',
                 plm='fasttext'):

        query = np.load(anchor_path, allow_pickle=True)
        self.anchor_mats = query

        item = np.load(auglist_path, allow_pickle=True)
        item_list = []

        for i in range(0, len(item), list_size):
            sub_list = item[i:i + list_size].tolist()
            item_list.append(sub_list)
        self.item_mats = item_list
        self.plm = plm
        self.training = training
        self.data_size = len(self.anchor_mats)
        self.list_size = list_size

        # Load the metadata (two historical formats are supported):
        #   new format (datagen): dict with keys 'metadata' / 'table_to_id' / 'column_to_id'
        #   old format (early data-generation tooling): a plain list [{'table_name': ..., 'column_name': ...}, ...]
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata_loaded = pickle.load(f)
                if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
                    self.metadata = metadata_loaded['metadata']  # new format
                else:
                    self.metadata = metadata_loaded              # old format
        else:
            self.metadata = None

    def __len__(self):
        """Return the size of the dataset."""
        return self.data_size

    def __getitem__(self, index):
        """Return a item of the dataset."""
        query = self.anchor_mats[index]
        item_list = self.item_mats[index]
        index_list = []
        for item in item_list:
            index_list.append(item.shape[0])
        return query, item_list, index_list

    def get_metadata(self, index):
        """Return the metadata of the given index."""
        if self.metadata and index < len(self.metadata):
            return self.metadata[index]
        return None

    @staticmethod
    def pad(batch):
        query, item_list, index_list = zip(*batch)

        # Fix: handle batched data correctly
        concat_query = np.concatenate(query, axis=0)  # cells of all query columns

        # item_list: each sample's item_list is a list of columns
        all_items = []
        merged_list = []

        for sample_items in item_list:
            for item in sample_items:
                all_items.append(item)
                merged_list.append(item.shape[0])  # cell count of each item column

        concat_item = np.concatenate(all_items, axis=0)

        # query indices: each sample contributes one column
        query_index = [arr.shape[0] for arr in query]  # cell count of each query column

        return (torch.tensor(concat_query, dtype=torch.float32, requires_grad=True),
                torch.tensor(concat_item, dtype=torch.float32, requires_grad=True),
                torch.tensor(query_index),
                torch.tensor(merged_list))


# Keep the original dataset-class aliases for backward compatibility
TestDataset = TestDatasetHyperJoin
MyDataset = MyDatasetHyperJoin
RankDataset = RankDatasetHyperJoin