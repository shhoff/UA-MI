"""
Mar. 10, 2022.

Version of python is implemented as:
Python: v3.9.6

This code is the code for Uncertainty-Aware Multiple Imputation (UA-MI) framework based on ensemble-based method.
It consists of 3 parts: Training Multinomial Variational Autoencoder (MultVAE) considering ensemble size, performing UA-MI with ensemble, re-training MultVAE with filled matrix.

Versions of the libraries are implemented as:
Bottleneck: v1.3.2
Cupy: v9.4.0
Numpy: v1.20.3
Pandas: v1.1.5
Tensorflow-GPU for RTX-3090(CUDA v11.4.100, cuDNN v8.2.2): v2.5.0
"""

import bottleneck as bn
import cupy as cp
from evaluation import ndcg_binary_at_N_batch, recall_at_N_batch
from load_data import set_item_lists, load_train_data, load_valid_fi_fo_data, load_test_fi_fo_data
import math
from ua_mi_base import UA_MI_base
import numpy as np
import os
import pandas as pd
from scipy import sparse
import shutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# This function is for training base model, MultVAE.

def training_ens(data, n_ensembles, batch_size_train = 500, batch_size_valid = 2000, batch_size_test = 2000, total_anneal_steps = 200000, anneal_max = 0.2, n_epochs = 200):

    # Loading data & setting hyperparameters for training MultVAE.

    n_items, train_item_id = set_item_lists(data)

    train_data = load_train_data(data)
    valid_data_fi, valid_data_fo = load_valid_fi_fo_data(data)
    test_data_fi, test_data_fo = load_test_fi_fo_data(data)

    N_train = train_data.shape[0]
    N_valid = valid_data_fi.shape[0]
    N_test = test_data_fi.shape[0]

    idxlist_train = list(range(N_train))
    idxlist_valid = list(range(N_valid))
    idxlist_test = list(range(N_test))

    batches_per_epoch = int(np.ceil(float(N_train) / batch_size_train))

    dec_dim = [200, 600, n_items]

    tf.reset_default_graph()

    ua_mi_base = UA_MI_base(dec_dim, lam = 0.0, random_seed = 98765)
    saver, pred_rate, neg_ELBO, train_opt, merged_var = ua_mi_base.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype = tf.float64, shape = None)

    recall_var = tf.Variable(0.0)
    recall_dist_var = tf.placeholder(dtype = tf.float64, shape = None)

    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid_ndcg = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    recall_summary = tf.summary.scalar('recall_at_k_validation', recall_var)
    recall_dist_summary = tf.summary.histogram('recall_at_k_hist_validation', recall_dist_var)
    merged_valid_recall = tf.summary.merge([recall_summary, recall_dist_summary])

    # Setting the path for logs & checkpoints.

    data_path = os.path.join('./data', data)

    log_path = data_path + '/logs'

    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    summary_writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())

    chkpt_path = data_path + '/checkpoints'

    if not os.path.isdir(chkpt_path):
        os.makedirs(chkpt_path)

    # Training MultVAE.

    for batch in range(n_ensembles):
        ndcgs_valid, recalls_valid = [], []

        with tf.Session() as train_sess:
            init = tf.global_variables_initializer()
            train_sess.run(init)

            best_ndcg = -np.inf
            update_count = 0.0

            for epoch in range(n_epochs):
                new_idxlist_train = np.copy(idxlist_train)
                np.random.shuffle(new_idxlist_train)

                for bnum, st_idx in enumerate(range(0, N_train, batch_size_train)):
                    end_idx = min(st_idx + batch_size_train, N_train)

                    X = train_data[new_idxlist_train[st_idx:end_idx]]

                    if sparse.isspmatrix(X):
                        X = X.toarray()

                    X = X.astype('float32')

                    if total_anneal_steps > 0:
                        anneal = min(anneal_max, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = anneal_max

                    feed_dict = {ua_mi_base.input_interaction: X,
                                ua_mi_base.keeping_prob_dropout: 0.5,
                                ua_mi_base.anneal_param: anneal,
                                ua_mi_base.check_training: 1}

                    train_sess.run(train_opt, feed_dict = feed_dict)

                    if bnum % 100 == 0:
                        summary_train = train_sess.run(merged_var, feed_dict = feed_dict)
                        summary_writer.add_summary(summary_train, 
                                                  global_step = epoch * batches_per_epoch + bnum)

                    update_count += 1

                # Compute validation (select between NDCG & Recall, default = NDCG)

                ndcg_dist = []
                recall_dist = []

                for bnum, st_idx in enumerate(range(0, N_valid, batch_size_valid)):
                    end_idx = min(st_idx + batch_size_valid, N_valid)

                    X = valid_data_fi[idxlist_valid[st_idx:end_idx]]

                    if sparse.isspmatrix(X):
                        X = X.toarray()

                    X = X.astype('float32')
                    pred_val = train_sess.run(pred_rate, feed_dict = {ua_mi_base.input_interaction: X})

                    pred_val[X.nonzero()] = -np.inf

                    ndcg_dist.append(ndcg_binary_at_N_batch(pred_val, valid_data_fo[idxlist_valid[st_idx:end_idx]]))
                    recall_dist.append(recall_at_N_batch(pred_val, valid_data_fo[idxlist_valid[st_idx:end_idx]]))

                ndcg_dist = np.concatenate(ndcg_dist)
                ndcg_ = ndcg_dist.mean()
                ndcgs_valid.append(ndcg_)
                merged_valid_ndcg_val = train_sess.run(merged_valid_ndcg, feed_dict = {ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
                summary_writer.add_summary(merged_valid_ndcg_val, epoch)

                recall_dist = np.concatenate(recall_dist)
                recall_ = recall_dist.mean()
                recalls_valid.append(recall_)
                merged_valid_recall_val = train_sess.run(merged_valid_recall, feed_dict = {recall_var: recall_, recall_dist_var: recall_dist})
                summary_writer.add_summary(merged_valid_recall_val, epoch)

                # Update the best model(select between NDCG & Recall, default = NDCG)

                if ndcg_ > best_ndcg:
                    saver.save(train_sess, '{}/trained_multvae_{}'.format(chkpt_path, batch + 1))
                    best_ndcg = ndcg_

        return batch_size_train, batch_size_test, chkpt_path, dec_dim, idxlist_train, idxlist_test, N_train, N_test, n_items, train_data, test_data_fi, test_data_fo


def softmax(x):

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


# This function is for performing UA-MI based on ensemble-based method.

def UA_MI_ens(data, fill_rate, n_ensembles):

    batch_size_train, batch_size_test, chkpt_path, dec_dim, idxlist_train, idxlist_test, N_train, N_test, n_items, train_data, test_data_fi, test_data_fo = training_ens(data)

    tf.reset_default_graph()

    ua_mi_base = UA_MI_base(dec_dim, lam = 0.0, random_seed = 98765)
    saver, pred_rate, _, _, _ = ua_mi_base.build_graph()

    with tf.Session() as ua_mi_ens_sess:
        filled_X = np.array([])

        for bnum, st_idx in enumerate(range(0, N_train, batch_size_train)):
            end_idx = min(st_idx + batch_size_train, N_train)

            X = train_data[idxlist_train[st_idx:end_idx]]
            
            if sparse.isspmatrix(X):
                X = X.toarray()

            X = X.astype('float32')

            total_fill_counts = np.count_nonzero(X == 0)
            fill_counts = math.floor(total_fill_counts * fill_rate)

            pred_X = np.array([])

            for ens_num in range(n_ensembles):
                saver.restore(ua_mi_ens_sess, chkpt_path + '/trained_multvae_{}'.format(ens_num + 1))
                pred_X = np.append(pred_X, softmax(ua_mi_ens_sess.run(pred_rate, feed_dict = {ua_mi_base.input_interaction: X})))

            pred_X = pred_X.reshape(n_ensembles, X.shape[0], X.shape[1])

            pred_mean = cp.mean(pred_X, axis = 0, dtype = 'float32')
            pred_var = cp.var(pred_X, axis = 0, dtype = 'float32')

            pred_var_idx = np.where(X == 0, True, False)
            pred_var_list = pred_var[pred_var_idx]
            pred_var_list.sort()

            thr = pred_var_list[fill_counts]

            pred_var_thr_idx = pred_var < thr

            X[pred_var_thr_idx & pred_var_idx] = pred_mean[pred_var_thr_idx & pred_var_idx]

            filled_X = np.append(filled_X, X)

        X_shape = train_data

        if sparse.isspmatrix(X_shape):
            X_shape = X_shape.toarray()

        X_shape = X_shape.astype('float32')

        filled_X = filled_X.reshape(X_shape.shape[0], X_shape.shape[1])

    return filled_X


# This function is for re-training base model, MultVAE with filled matrix data.

def re_training_ens(data, fill_rate, n_ensembles, batch_size_train = 500, batch_size_valid = 2000, batch_size_test = 2000, total_anneal_steps = 200000, anneal_max = 0.2, n_epochs = 200):

    filled_X = UA_MI_ens(data, fill_rate, n_ensembles)

    # Loading data & setting hyperparameters for training MultVAE.

    n_items, train_item_id = set_item_lists(data)

    train_data = load_train_data(data)
    valid_data_fi, valid_data_fo = load_valid_fi_fo_data(data)
    test_data_fi, test_data_fo = load_test_fi_fo_data(data)

    N_train = train_data.shape[0]
    N_valid = valid_data_fi.shape[0]
    N_test = test_data_fi.shape[0]

    idxlist_train = list(range(N_train))
    idxlist_valid = list(range(N_valid))
    idxlist_test = list(range(N_test))

    batches_per_epoch = int(np.ceil(float(N_train) / batch_size_train))

    dec_dim = [200, 600, n_items]

    tf.reset_default_graph()

    ua_mi_base = UA_MI_base(dec_dim, lam = 0.0, random_seed = 98765)
    saver, pred_rate, neg_ELBO, train_opt, merged_var = ua_mi_base.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype = tf.float64, shape = None)

    recall_var = tf.Variable(0.0)
    recall_dist_var = tf.placeholder(dtype = tf.float64, shape = None)

    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid_ndcg = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    recall_summary = tf.summary.scalar('recall_at_k_validation', recall_var)
    recall_dist_summary = tf.summary.histogram('recall_at_k_hist_validation', recall_dist_var)
    merged_valid_recall = tf.summary.merge([recall_summary, recall_dist_summary])

    # Setting the path for logs & checkpoints.

    data_path = os.path.join('./data', data)

    log_path = data_path + '/logs'

    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    summary_writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())

    chkpt_path = data_path + '/checkpoints'

    if not os.path.isdir(chkpt_path):
        os.makedirs(chkpt_path)

    # Re-training MultVAE.

    ndcgs_valid, recalls_valid = [], []

    with tf.Session() as re_train_sess:
        init = tf.global_variables_initializer()
        re_train_sess.run(init)

        best_ndcg = -np.inf
        update_count = 0.0

        for epoch in range(n_epochs):
            new_idxlist_train = np.copy(idxlist_train)
            np.random.shuffle(new_idxlist_train)

            for bnum, st_idx in enumerate(range(0, N_train, batch_size_train)):
                end_idx = min(st_idx + batch_size_train, N_train)

                X = filled_X[new_idxlist_train[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()

                X = X.astype('float32')

                if total_anneal_steps > 0:
                    anneal = min(anneal_max, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_max

                feed_dict = {ua_mi_base.input_interaction: X,
                            ua_mi_base.keeping_prob_dropout: 0.5,
                            ua_mi_base.anneal_param: anneal,
                            ua_mi_base.check_training: 1}

                re_train_sess.run(train_opt, feed_dict = feed_dict)

                if bnum % 100 == 0:
                    summary_train = re_train_sess.run(merged_var, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_train, 
                                              global_step = epoch * batches_per_epoch + bnum)

                update_count += 1

            # Compute validation (select between NDCG & Recall, default = NDCG)

            ndcg_dist = []
            recall_dist = []

            for bnum, st_idx in enumerate(range(0, N_valid, batch_size_valid)):
                end_idx = min(st_idx + batch_size_valid, N_valid)

                X = valid_data_fi[idxlist_valid[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()

                X = X.astype('float32')
                pred_val = re_train_sess.run(pred_rate, feed_dict = {ua_mi_base.input_interaction: X})

                pred_val[X.nonzero()] = -np.inf

                ndcg_dist.append(ndcg_binary_at_N_batch(pred_val, valid_data_fo[idxlist_valid[st_idx:end_idx]]))
                recall_dist.append(recall_at_N_batch(pred_val, valid_data_fo[idxlist_valid[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_valid.append(ndcg_)
            merged_valid_ndcg_val = re_train_sess.run(merged_valid_ndcg, feed_dict = {ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_ndcg_val, epoch)

            recall_dist = np.concatenate(recall_dist)
            recall_ = recall_dist.mean()
            recalls_valid.append(recall_)
            merged_valid_recall_val = re_train_sess.run(merged_valid_recall, feed_dict = {recall_var: recall_, recall_dist_var: recall_dist})
            summary_writer.add_summary(merged_valid_recall_val, epoch)

            # Update the best model(select between NDCG & Recall, default = NDCG)

            if ndcg_ > best_ndcg:
                saver.save(re_train_sess, '{}/re_trained_ens_fill_{}_k_{}'.format(chkpt_path, fill_rate, n_ensembles))
                best_ndcg = ndcg_

    return batch_size_train, batch_size_test, chkpt_path, dec_dim, idxlist_train, idxlist_test, N_train, N_test, n_items, train_data, test_data_fi, test_data_fo