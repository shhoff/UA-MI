"""
Mar. 10, 2022.

Version of python is implemented as:
Python: v3.9.6

This code is the code for evaluating each recommendation session of Uncertainty-Aware Multiple Imputation (UA-MI) framework.
Original source code from "Variational autoencoders for collaborative filtering" is partially used.
In this code, we will only provide the evaluation functions of UA-MI based on ensemble-based method.

Versions of the libraries are implemented as:
Bottleneck: v1.3.2
Numpy: v1.20.3
Tensorflow-GPU for RTX-3090(CUDA v11.4.100, cuDNN v8.2.2): v2.5.0
"""

import bottleneck as bn
from ua_mi_base import UA_MI_base
import numpy as np
from scipy import sparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from ua_mi_ens import training_ens, re_training_ens

def ndcg_binary_at_N_batch(X_pred, heldout_batch, N = 100):

    batch_users = X_pred.shape[0]
    idx_top_N_part = bn.argpartition(-X_pred, N, axis = 1)
    top_N_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_top_N_part[:, :N]]
    idx_part = np.argsort(-top_N_part, axis = 1)

    idx_top_N = idx_top_N_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, N + 2))
    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_top_N].toarray() * tp).sum(axis = 1)
    idcg = np.array([(tp[:min(n, N)]).sum() for n in heldout_batch.getnnz(axis = 1)])

    return dcg / idcg

def recall_at_N_batch(X_pred, heldout_batch, N = 100):

    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, N, axis = 1)

    X_pred_binary = np.zeros_like(X_pred, dtype = bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :N]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis = 1)).astype(np.float32)
    recall = tmp / np.minimum(N, X_true_binary.sum(axis = 1))
    
    return recall


# Compute test metrics (NDCG@N & Recall@N)

def training_evaluation(data, ens_num):

    batch_size_train, batch_size_test, chkpt_path, dec_dim, idxlist_train, idxlist_test, N_train, N_test, n_items, train_data, test_data_fi, test_data_fo = training_ens(data)

    tf.reset_default_graph()
    ua_mi_base = UA_MI_base(dec_dim, lam = 0.0)
    saver, pred_rate, _, _, _ = ua_mi_base.build_graph()

    r1_list, r5_list, r10_list, r50_list = [], [], [], []
    n1_list, n5_list, n10_list, n50_list = [], [], [], []

    with tf.Session() as eval_sess:
        saver.restore(eval_sess, chkpt_path + '/trained_multvae_{}'.format(ens_num))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)

            X = test_data_fi[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()

            X = X.astype('float32')
            pred_val = eval_sess.run(pred_rate, feed_dict = {ua_mi_base.input_interaction: X})

            pred_val[X.nonzero()] = -np.inf
            
            r1_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 1))
            r5_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 5))
            r10_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 10))
            r50_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 50))

            n1_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 1))
            n5_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 5))
            n10_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 10))
            n50_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 50))
            
    r1_list = np.concatenate(r1_list)
    r5_list = np.concatenate(r5_list)
    r10_list = np.concatenate(r10_list)
    r50_list = np.concatenate(r50_list)

    n1_list = np.concatenate(n1_list)
    n5_list = np.concatenate(n5_list)
    n10_list = np.concatenate(n10_list)
    n50_list = np.concatenate(n50_list)

    print('Test Recall@1 = %.5f (%.5f)' % (np.mean(r1_list), np.std(r1_list) / np.sqrt(len(r1_list))))
    print('Test Recall@5 = %.5f (%.5f)' % (np.mean(r5_list), np.std(r5_list) / np.sqrt(len(r5_list))))
    print('Test Recall@10 = %.5f (%.5f)' % (np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
    print('Test Recall@50 = %.5f (%.5f)' % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    print('Test NDCG@1 = %.5f (%.5f)' % (np.mean(n1_list), np.std(n1_list) / np.sqrt(len(n1_list))))
    print('Test NDCG@5 = %.5f (%.5f)' % (np.mean(n5_list), np.std(n5_list) / np.sqrt(len(n5_list))))
    print('Test NDCG@10 = %.5f (%.5f)' % (np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
    print('Test NDCG@50 = %.5f (%.5f)' % (np.mean(n50_list), np.std(n50_list) / np.sqrt(len(n50_list))))


def re_training_evaluation(data, fill_rate, n_ensembles):

    batch_size_train, batch_size_test, chkpt_path, dec_dim, idxlist_train, idxlist_test, N_train, N_test, n_items, train_data, test_data_fi, test_data_fo = re_training_ens(data, fill_rate, n_ensembles)

    tf.reset_default_graph()
    ua_mi_base = UA_MI_base(dec_dim, lam = 0.0)
    saver, pred_rate, _, _, _ = ua_mi_base.build_graph()

    r1_list, r5_list, r10_list, r50_list = [], [], [], []
    n1_list, n5_list, n10_list, n50_list = [], [], [], []

    with tf.Session() as eval_sess:
        saver.restore(eval_sess, chkpt_path + '/re_trained_ens_fill_{}_k_{}'.format(fill_rate, n_ensembles))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)

            X = test_data_fi[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()

            X = X.astype('float32')
            pred_val = eval_sess.run(pred_rate, feed_dict = {ua_mi_base.input_interaction: X})

            pred_val[X.nonzero()] = -np.inf
            
            r1_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 1))
            r5_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 5))
            r10_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 10))
            r50_list.append(recall_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 50))

            n1_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 1))
            n5_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 5))
            n10_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 10))
            n50_list.append(ndcg_binary_at_N_batch(pred_val, test_data_fo[idxlist_test[st_idx:end_idx]], N = 50))
            
    r1_list = np.concatenate(r1_list)
    r5_list = np.concatenate(r5_list)
    r10_list = np.concatenate(r10_list)
    r50_list = np.concatenate(r50_list)

    n1_list = np.concatenate(n1_list)
    n5_list = np.concatenate(n5_list)
    n10_list = np.concatenate(n10_list)
    n50_list = np.concatenate(n50_list)

    print('Test Recall@1 = %.5f (%.5f)' % (np.mean(r1_list), np.std(r1_list) / np.sqrt(len(r1_list))))
    print('Test Recall@5 = %.5f (%.5f)' % (np.mean(r5_list), np.std(r5_list) / np.sqrt(len(r5_list))))
    print('Test Recall@10 = %.5f (%.5f)' % (np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
    print('Test Recall@50 = %.5f (%.5f)' % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    print('Test NDCG@1 = %.5f (%.5f)' % (np.mean(n1_list), np.std(n1_list) / np.sqrt(len(n1_list))))
    print('Test NDCG@5 = %.5f (%.5f)' % (np.mean(n5_list), np.std(n5_list) / np.sqrt(len(n5_list))))
    print('Test NDCG@10 = %.5f (%.5f)' % (np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
    print('Test NDCG@50 = %.5f (%.5f)' % (np.mean(n50_list), np.std(n50_list) / np.sqrt(len(n50_list))))