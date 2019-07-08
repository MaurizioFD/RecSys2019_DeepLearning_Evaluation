#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/10/18

@author: Maurizio Ferrari Dacrema
"""



import os
import sys

import numpy as np
from scipy import sparse


import pandas as pd



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





def split_train_validation_test_VAE_CF_original(URM_dataframe, pro_dir, n_heldout_users):


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



import shutil


def split_train_validation_test_VAE_CF(URM_dataframe, n_heldout_users):

    split_dir = "./result_experiments/__Temp_MultiVAE_Splitter/"

    split_train_validation_test_VAE_CF_original(URM_dataframe, split_dir, n_heldout_users)
    train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, n_items = load_data_VAE_CF(split_dir)

    # Remove temp files
    shutil.rmtree(split_dir, ignore_errors=True)


    from Base.Recommender_utils import reshapeSparse

    URM_train_only = train_data.copy()
    URM_train_all = sps.vstack([train_data, vad_data_tr, test_data_tr])

    URM_train_all_shape = URM_train_all.shape

    ## OFFSET all row indices
    n_train_users = train_data.shape[0]

    URM_validation = offset_sparse_matrix_row(vad_data_te, n_train_users)

    n_train_and_validation_users = URM_validation.shape[0]

    URM_validation = reshapeSparse(URM_validation, URM_train_all_shape)

    URM_test = offset_sparse_matrix_row(test_data_te, n_train_and_validation_users)
    URM_test = reshapeSparse(URM_test, URM_train_all_shape)


    URM_train_only = sps.csr_matrix(URM_train_only)
    URM_train_all = sps.csr_matrix(URM_train_all)
    URM_validation = sps.csr_matrix(URM_validation)
    URM_test = sps.csr_matrix(URM_test)


    return URM_train_only, URM_train_all, URM_validation, URM_test