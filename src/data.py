import csv
import itertools
import random
import pickle
import os

import numpy as np
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
import fasttext
from faker import Faker


class TestDatasetHyperJoin(Data.Dataset):
 """
 HyperJoin version of test dataset with metadata support
 """
 def __init__(self,
 test_path_mat, # Path to test set matrix
 metadata_path, # Metadata path
 da=None,
 plm='fasttext'):

 self.plm = plm
 self.da = da
 mats = np.load(test_path_mat, allow_pickle=True)
 self.data_mats = mats # self.data_mats[k] is a matrix of the kth col
 self.data_size = len(self.data_mats) # of cols in total
 
 # Load metadata (compatible with two formats)
 if metadata_path and os.path.exists(metadata_path):
 with open(metadata_path, 'rb') as f:
 metadata_loaded = pickle.load(f)
 # Compatibility handling: detect if dict or list
 if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
 # New format: dict contains 'metadata' key
 self.metadata = metadata_loaded['metadata']
 else:
 # Old format: directly a list
 self.metadata = metadata_loaded
 else:
 self.metadata = None

 def __len__(self):
 """Return the size of the dataset."""
 return self.data_size

 def __getitem__(self, index):
 """Return a item of the dataset."""
 x = self.data_mats[index]
 return x

 def get_metadata(self, index):
 """Get metadata for specified index"""
 if self.metadata and index < len(self.metadata['metadata']):
 return self.metadata['metadata'][index]
 return None

 def get_table_column_info(self, index):
 """Get real table name and column name ID information"""
 if self.metadata and index < len(self.metadata['metadata']):
 item_meta = self.metadata['metadata'][index]
 table_name = item_meta['table_name']
 column_name = item_meta['column_name']

 # Convert to ID
 table_id = self.metadata['table_to_id'][table_name]
 column_id = self.metadata['column_to_id'][column_name]

 return table_id, column_id, table_name, column_name
 return 0, index, "unknown_table", "unknown_column" # Fallback handling

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
 HyperJoin version of training dataset with metadata support
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
 
 # Load metadata (compatible with two formats)
 if metadata_path and os.path.exists(metadata_path):
 with open(metadata_path, 'rb') as f:
 metadata_loaded = pickle.load(f)
 # Compatibility handling: detect if dict or list
 if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
 # New format: dict contains 'metadata' key
 self.metadata = metadata_loaded['metadata']
 else:
 # Old format: directly a list
 self.metadata = metadata_loaded
 else:
 self.metadata = None

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
 """Get metadata for specified index"""
 if self.metadata and index < len(self.metadata['metadata']):
 return self.metadata['metadata'][index]
 return None

 def get_table_column_info(self, index):
 """Get real table name and column name ID information"""
 if self.metadata and index < len(self.metadata['metadata']):
 item_meta = self.metadata['metadata'][index]
 table_name = item_meta['table_name']
 column_name = item_meta['column_name']

 # Convert to ID
 table_id = self.metadata['table_to_id'][table_name]
 column_id = self.metadata['column_to_id'][column_name]

 return table_id, column_id, table_name, column_name
 return 0, index, "unknown_table", "unknown_column" # Fallback handling

 @staticmethod
 def pad(batch):
 query, pos, item_list, index_list = zip(*batch)
 
 # Fix: Correctly handle batch data
 # Each query/pos is a single column, not a combination of multiple columns
 concat_query = np.concatenate(query, axis=0) # Concatenate all query column cells
 concat_pos = np.concatenate(pos, axis=0) # Concatenate all pos column cells

 # Process item_list: each sample's item_list is a list of multiple columns
 all_items = []
 merged_list = []

 for sample_items in item_list:
 for item in sample_items:
 all_items.append(item)
 merged_list.append(item.shape[0]) # Number of cells in each item column

 concat_item = np.concatenate(all_items, axis=0)

 # Query and pos indices: each sample contributes one column
 query_index = [arr.shape[0] for arr in query] # Number of cells in each query column
 pos_index = [arr.shape[0] for arr in pos] # Number of cells in each pos column
 
 return (torch.tensor(concat_query, dtype=torch.float32), 
 torch.tensor(concat_pos, dtype=torch.float32), 
 torch.tensor(concat_item, dtype=torch.float32),
 torch.tensor(query_index), 
 torch.tensor(pos_index), 
 torch.tensor(merged_list))


class RankDatasetHyperJoin(Data.Dataset):
 """
 HyperJoin version of ranking dataset
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
 
 # Load metadata (compatible with two formats)
 if metadata_path and os.path.exists(metadata_path):
 with open(metadata_path, 'rb') as f:
 metadata_loaded = pickle.load(f)
 # Compatibility handling: detect if dict or list
 if isinstance(metadata_loaded, dict) and 'metadata' in metadata_loaded:
 # New format: dict contains 'metadata' key
 self.metadata = metadata_loaded['metadata']
 else:
 # Old format: directly a list
 self.metadata = metadata_loaded
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
 """Get metadata for specified index"""
 if self.metadata and index < len(self.metadata):
 return self.metadata[index]
 return None

 @staticmethod
 def pad(batch):
 query, item_list, index_list = zip(*batch)

 # Fix: Correctly handle batch data
 concat_query = np.concatenate(query, axis=0) # Concatenate all query column cells

 # Process item_list: each sample's item_list is a list of multiple columns
 all_items = []
 merged_list = []

 for sample_items in item_list:
 for item in sample_items:
 all_items.append(item)
 merged_list.append(item.shape[0]) # Number of cells in each item column

 concat_item = np.concatenate(all_items, axis=0)

 # Query indices: each sample contributes one column
 query_index = [arr.shape[0] for arr in query] # Number of cells in each query column
 
 return (torch.tensor(concat_query, dtype=torch.float32, requires_grad=True), 
 torch.tensor(concat_item, dtype=torch.float32, requires_grad=True), 
 torch.tensor(query_index), 
 torch.tensor(merged_list))

# Keep original dataset classes for backward compatibility
TestDataset = TestDatasetHyperJoin
MyDataset = MyDatasetHyperJoin
RankDataset = RankDatasetHyperJoin