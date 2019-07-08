#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/10/18

@author: Maurizio Ferrari Dacrema
"""


import os
import shutil
import sys

import numpy as np
from scipy import sparse


import seaborn as sn
sn.set()

import pandas as pd

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

import bottleneck as bn

from VAE_CF_github.MultiVae_Dae import MultiDAE, MultiVAE
# Data splitting procedure
# Select 10K users as heldout users, 10K users as validation users, and the rest of the users for training
# Use all the items from the training users as item set
# For each of both validation and test user, subsample 80% as fold-in data and the rest for prediction
# Only keep items that are clicked on by at least 5 users



def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count



def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount





def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])





def split_train_validation_test_VAE_CF(URM_dataframe, pro_dir):


    raw_data, user_activity, item_popularity = filter_triplets(URM_dataframe)


    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


    unique_uid = user_activity.index

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 10000

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]


    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))


    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]


    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)





    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)



    return train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te





def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te





def load_data_VAE_CF(pro_dir):


    unique_sid = list()

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)


    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)


    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                               os.path.join(pro_dir, 'validation_te.csv'), n_items)


    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'), n_items)




    return train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, n_items



import  scipy.sparse as sps


def offset_sparse_matrix_row(URM, offset_row):

    URM_coo = sps.coo_matrix(URM.copy())

    URM_coo.row += offset_row

    return sps.csr_matrix((URM_coo.data, (URM_coo.row, URM_coo.col)))





dataset = "movielens20m"
#dataset = "movielens1m"


if dataset == "movielens20m":

    DATA_DIR = '../data/Movielens20M/ml-20m/'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)

    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > 3.5]


elif dataset == "movielens1m":

    DATA_DIR = '../data/Movielens1M/ml-1m/'

    column_names = ["userId","movieId","rating","timestamp"]
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), header = None, names = column_names, sep="::")

    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > 3.5]





output_directory = "../result_experiments/VAE_CF_{}/".format(dataset)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)




try:

    train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, n_items = load_data_VAE_CF(output_directory + "split/")

except:

    split_train_validation_test_VAE_CF(raw_data, output_directory + "split/")
    train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, n_items = load_data_VAE_CF(output_directory + "split/")





from Base.Recommender_utils import reshapeSparse


URM_train_all = sps.vstack([train_data, vad_data_tr, test_data_tr])

URM_train_all_shape = URM_train_all.shape

## OFFSET all row indices
n_train_users = train_data.shape[0]

URM_validation = offset_sparse_matrix_row(vad_data_te, n_train_users)

n_train_and_validation_users = URM_validation.shape[0]

URM_validation = reshapeSparse(URM_validation, URM_train_all_shape)
URM_test = offset_sparse_matrix_row(test_data_te, n_train_and_validation_users)












##############################################################################################################################################################
##### Set up training hyperparameters


N = train_data.shape[0]
idxlist = list(range(N))

# training batch size
batch_size = 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))

N_vad = vad_data_tr.shape[0]
idxlist_vad = list(range(N_vad))

# validation batch size (since the entire validation set might not fit into GPU memory)
batch_size_vad = 2000

# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2




##### Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall





##############################################################################################################################################################



















############################ Train a Multi-VAE^{PR}

p_dims = [200, 600, n_items]



tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0, random_seed=98765)

saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])




arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))


# log_dir = '/volmount/log/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
#     total_anneal_steps/1000, anneal_cap, arch_str)

log_dir = output_directory + 'log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())



# chkpt_dir = '/volmount/chkpt/ml-20m/VAE_anneal{}K_cap{:1.1E}/{}'.format(
#     total_anneal_steps/1000, anneal_cap, arch_str)

chkpt_dir = output_directory + 'chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)

if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir)

print("chkpt directory: %s" % chkpt_dir)


n_epochs = 0




ndcgs_vad = []

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    best_ndcg = -np.inf

    update_count = 0.0

    for epoch in range(n_epochs):

        np.random.shuffle(idxlist)
        # train for one epoch
        for bnum, st_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(st_idx + batch_size, N)
            X = train_data[idxlist[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap

            feed_dict = {vae.input_ph: X,
                         vae.keep_prob_ph: 0.5,
                         vae.anneal_ph: anneal,
                         vae.is_training_ph: 1}
            sess.run(train_op_var, feed_dict=feed_dict)

            if bnum % 100 == 0:
                summary_train = sess.run(merged_var, feed_dict=feed_dict)
                summary_writer.add_summary(summary_train,
                                           global_step=epoch * batches_per_epoch + bnum)

            update_count += 1

        # compute validation NDCG

        ndcg_dist = []
        for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
            end_idx = min(st_idx + batch_size_vad, N_vad)
            X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X} )
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

        ndcg_dist = np.concatenate(ndcg_dist)
        ndcg_ = ndcg_dist.mean()
        ndcgs_vad.append(ndcg_)
        merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
        summary_writer.add_summary(merged_valid_val, epoch)

        print("Validation epoch {} - NDCG {}".format(epoch, ndcg_))

        # update the best model (if necessary)
        if ndcg_ > best_ndcg:
            saver.save(sess, '{}/model'.format(chkpt_dir))
            best_ndcg = ndcg_



# plt.figure(figsize=(12, 3))
# plt.plot(ndcgs_vad)
# plt.ylabel("Validation NDCG@100")
# plt.xlabel("Epochs")
# pass
#


########################## Load the test data and compute test metrics


N_test = test_data_tr.shape[0]
idxlist_test = list(range(N_test))

batch_size_test = 2000



tf.reset_default_graph()
vae = MultiVAE(p_dims, lam=0.0)
saver, logits_var, _, _, _ = vae.build_graph()

# Load the best performing model on the validation set


chkpt_dir = output_directory + 'chkpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps/1000, anneal_cap, arch_str)
print("chkpt directory: %s" % chkpt_dir)


n100_list, r20_list, r50_list = [], [], []

with tf.Session() as sess:
    saver.restore(sess, '{}/model'.format(chkpt_dir))

    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))


n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)


print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))






########################## MY wrapper



import numpy as np
from Base.BaseRecommender import BaseRecommender

import scipy.sparse as sps


class VAE_CF_RecommenderWrapper(BaseRecommender):

    RECOMMENDER_NAME = "VAE_CF_RecommenderWrapper"

    def __init__(self, URM_train, chkpt_dir, test_data_tr):
        super(VAE_CF_RecommenderWrapper, self).__init__()

        # convert to csc matrix for faster column-wise sum
        self.chkpt_dir = chkpt_dir
        self.URM_train = URM_train

        self._compute_item_score = self._compute_score_VAE
        self.test_data_tr = sps.csr_matrix(test_data_tr)


    def _remove_seen_on_scores(self, user_id, scores):

        seen = self.test_data_tr.indices[self.test_data_tr.indptr[user_id]:self.test_data_tr.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores


    def _compute_score_VAE(self, user_id):


        X = test_data_tr[user_id]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')


        pred_val = self.sess.run(logits_var, feed_dict={vae.input_ph: X})

        pred_val[X.nonzero()] = -np.inf


        return pred_val



    def fit(self):

        self.sess = tf.Session()
        saver.restore(self.sess, '{}/model'.format(self.chkpt_dir))






recommender = VAE_CF_RecommenderWrapper(train_data, chkpt_dir, test_data_tr)
recommender.fit()





from Base.Evaluation.metrics import recall, ndcg


n100_list, r20_list, r50_list = [], [], []

for user_index in range(len(idxlist_test)):
    #end_idx = min(st_idx + batch_size_test, N_test)
    # X = test_data_tr[idxlist_test[st_idx:end_idx]]
    #
    # if sparse.isspmatrix(X):
    #     X = X.toarray()
    # X = X.astype('float32')
    #
    # pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
    # # exclude examples from training and validation (if any)
    # pred_val[X.nonzero()] = -np.inf

    pred_val = recommender._compute_score_VAE(idxlist_test[user_index])

    pos_items_sparse = test_data_te[idxlist_test[user_index]]
    pos_items_array = test_data_te[idxlist_test[user_index]].indices

    n100_list.append(NDCG_binary_at_k_batch(pred_val, pos_items_sparse, k=100))
    r20_list.append(Recall_at_k_batch(pred_val, pos_items_sparse, k=20))
    r50_list.append(Recall_at_k_batch(pred_val, pos_items_sparse, k=50))

    k = 100
    #
    # batch_users = pred_val.shape[0]
    # idx_topk_part = bn.argpartition(-pred_val, k, axis=1)
    # topk_part = pred_val[np.arange(batch_users)[:, np.newaxis],
    #                    idx_topk_part[:, :k]]
    # idx_part = np.argsort(-topk_part, axis=1)
    # # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # # topk predicted score
    # idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    recommended_items = np.argsort(-pred_val, axis=1).ravel()[:k]

    is_relevant = np.in1d(recommended_items, pos_items_array, assume_unique=True)

    # his_recall = Recall_at_k_batch(pred_val, pos_items_sparse, k=20)[0]
    # my_recall = recall(is_relevant, pos_items_array)

    his_ndcg = NDCG_binary_at_k_batch(pred_val, pos_items_sparse, k=100)[0]
    my_ndcg = ndcg(recommended_items, pos_items_array)



    if not np.allclose(my_ndcg, his_ndcg, atol=0.0001):
        pass


n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)


print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))




























from Base.Evaluation.Evaluator import EvaluatorHoldout

evaluator = EvaluatorHoldout(test_data_te, cutoff_list=[20, 50, 100])

results_dict, results_run_string = evaluator.evaluateRecommender(recommender)


print(results_run_string)