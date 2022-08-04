"""
Mar. 10, 2022.

Version of python is implemented as:
Python: v3.9.6

This code is the code for loading recommendation data used in our Uncertainty-Aware Multiple Imputation (UA-MI) framework: Amazon-Book (A-Book), CiteULike (CUL)-a, Epinions, MovieLens (ML)-20M.
Original source code from "Variational autoencoders for collaborative filtering" is partially used.

Versions of the libraries are implemented as:
Numpy: v1.20.3
Pandas: v1.1.5
"""

import numpy as np
import os
import pandas as pd
from scipy import sparse

def set_item_lists(data):

    data_path = os.path.join('./data', data)
    item_id_lists_path = data_path + '/item_id_lists'
    train_item_id = list()

    with open(item_id_lists_path + '/train_item_id_{}.txt'.format(data), 'r') as f:
        for line in f:
            train_item_id.append(line.strip())

    n_items = len(train_item_id)

    return n_items, train_item_id

def load_train_data(data):
    
    n_items, _ = set_item_lists(data)

    data_path = os.path.join('./data', data)
    csv_file = os.path.join(data_path, 'final_df/final_train_df_{}.csv'.format(data))

    df = pd.read_csv(csv_file)
    n_users = df['user_id'].max() + 1
    rows, cols = df['user_id'], df['item_id']
    train_data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)),
                             dtype = 'float32',
                             shape = (n_users, n_items))

    return train_data

def load_valid_fi_fo_data(data):

    n_items, _ = set_item_lists(data)

    data_path = os.path.join('./data', data)
    csv_file_fi = os.path.join(data_path, 'final_df/final_valid_df_{}_fi.csv'.format(data))
    csv_file_fo = os.path.join(data_path, 'final_df/final_valid_df_{}_fo.csv'.format(data))

    df_fi = pd.read_csv(csv_file_fi)
    df_fo = pd.read_csv(csv_file_fo)

    start_idx = min(df_fi['user_id'].min(), df_fo['user_id'].min())
    end_idx = max(df_fi['user_id'].max(), df_fo['user_id'].max())

    rows_fi, cols_fi = df_fi['user_id'] - start_idx, df_fi['item_id']
    rows_fo, cols_fo = df_fo['user_id'] - start_idx, df_fo['item_id']

    valid_data_fi = sparse.csr_matrix((np.ones_like(rows_fi), 
                                (rows_fi, cols_fi)), 
                                dtype = 'float32', 
                                shape = (end_idx - start_idx + 1, n_items))
    valid_data_fo = sparse.csr_matrix((np.ones_like(rows_fo),
                                (rows_fo, cols_fo)), 
                                dtype = 'float32', 
                                shape = (end_idx - start_idx + 1, n_items))

    return valid_data_fi, valid_data_fo

def load_test_fi_fo_data(data):

    n_items, _ = set_item_lists(data)

    data_path = os.path.join('./data', data)
    csv_file_fi = os.path.join(data_path, 'final_df/final_test_df_{}_fi.csv'.format(data))
    csv_file_fo = os.path.join(data_path, 'final_df/final_test_df_{}_fo.csv'.format(data))

    df_fi = pd.read_csv(csv_file_fi)
    df_fo = pd.read_csv(csv_file_fo)

    start_idx = min(df_fi['user_id'].min(), df_fo['user_id'].min())
    end_idx = max(df_fi['user_id'].max(), df_fo['user_id'].max())

    rows_fi, cols_fi = df_fi['user_id'] - start_idx, df_fi['item_id']
    rows_fo, cols_fo = df_fo['user_id'] - start_idx, df_fo['item_id']

    test_data_fi = sparse.csr_matrix((np.ones_like(rows_fi), 
                                (rows_fi, cols_fi)), 
                                dtype = 'float32', 
                                shape = (end_idx - start_idx + 1, n_items))
    test_data_fo = sparse.csr_matrix((np.ones_like(rows_fo),
                                (rows_fo, cols_fo)), 
                                dtype = 'float32', 
                                shape = (end_idx - start_idx + 1, n_items))

    return test_data_fi, test_data_fo