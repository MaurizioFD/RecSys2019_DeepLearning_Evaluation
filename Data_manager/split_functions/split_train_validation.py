#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix



def split_train_validation_percentage_user_wise(URM_train, train_percentage = 0.1, verbose=True):

    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    # ensure to use csr matrix or we get big problem
    URM_train = URM_train.tocsr()


    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)

    user_no_item_train = 0
    user_no_item_validation = 0

    for user_id in range(URM_train.shape[0]):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = round(user_profile_length*train_percentage)

        if n_train_items == len(user_profile_items) and n_train_items > 1:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            user_no_item_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            user_no_item_validation += 1


        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if user_no_item_train != 0:
        print("Warning split: {} users with 0 train items ({} total users)".format(user_no_item_train, URM_train.shape[0]))
    if user_no_item_validation != 0:
        print("Warning split: {} users with 0 validation items ({} total users)".format(user_no_item_validation, URM_train.shape[0]))

    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation



def split_train_validation_leave_one_out_user_wise(URM_train, verbose=True, at_least_n_train_items=0):

    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)

    count_train = 0
    count_validation = 0
    for user_id in range(URM_train.shape[0]):

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = user_profile_length

        if n_train_items > at_least_n_train_items:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            count_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            count_validation += 1

        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if count_train>0:
        print("{} users with 0 train items".format(count_train))
    if count_validation>0:
        print("{} users with 0 validation items".format(count_validation))


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation


def split_data_train_validation_test_negative_user_wise(URM_all, negative_items_per_positive = 50):
    """
    This function creates a Train, Test, Validation split with negative items sampled
    The split is perfomed user-wise, 20% is test, 80% is train. Train is further divided in 90% final train and 10% validation
    :param URM_all:
    :param negative_items_per_positive:
    :return:
    """

    URM_all = sps.csr_matrix(URM_all)

    n_rows, n_cols = URM_all.shape


    URM_train_all, URM_test = split_train_validation_percentage_user_wise(URM_all, train_percentage = 0.8)

    URM_train, URM_validation = split_train_validation_percentage_user_wise(URM_train_all, train_percentage = 0.9)


    URM_negative_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)

    all_items = np.arange(0, n_cols, dtype=np.int)

    for user_index in range(URM_train_all.shape[0]):

        if user_index % 10000 == 0:
            print("split_data_train_validation_test_negative: user {} of {}".format(user_index, URM_all.shape[0]))

        start_pos = URM_all.indptr[user_index]
        end_pos = URM_all.indptr[user_index+1]

        user_profile = URM_all.indices[start_pos:end_pos]

        unobserved_index = np.in1d(all_items, user_profile, assume_unique=True, invert=True)

        unobserved_items = all_items[unobserved_index]
        np.random.shuffle(unobserved_items)

        n_test_items = URM_test.indptr[user_index+1] - URM_test.indptr[user_index]

        num_negative_items = n_test_items * negative_items_per_positive

        if num_negative_items > len(unobserved_items):
            print("split_data_train_validation_test_negative: WARNING number of negative to sample for user {} is greater than available negative items {}".format(num_negative_items, len(unobserved_items)))
            num_negative_items = min(num_negative_items, len(unobserved_items))

        URM_negative_builder.add_single_row(user_index, unobserved_items[:num_negative_items], 1.0)



    URM_negative = URM_negative_builder.get_SparseMatrix()




    return URM_train, URM_validation, URM_test, URM_negative






def split_train_validation_test_negative_leave_one_out_user_wise(URM_all, negative_items_per_positive = 50, verbose=True, at_least_n_train_items_test=0, at_least_n_train_items_validation=0):
    """
    This function creates a Train, Test, Validation split with negative items sampled
    The split is perfomed user-wise, hold 1 out for validation and test
    :param URM_all:
    :param negative_items_per_positive:
    :return:
    """

    URM_all = sps.csr_matrix(URM_all)

    n_rows, n_cols = URM_all.shape


    print('Creation test...')
    URM_train_all, URM_test = split_train_validation_leave_one_out_user_wise(URM_all, at_least_n_train_items=at_least_n_train_items_test,verbose=verbose)

    print('Creation validation...')
    URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train_all, at_least_n_train_items=at_least_n_train_items_validation, verbose=verbose)


    URM_negative_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)

    all_items = np.arange(0, n_cols, dtype=np.int)

    for user_index in range(URM_train_all.shape[0]):

        if user_index % 10000 == 0:
            print("split_data_train_validation_test_negative: user {} of {}".format(user_index, URM_all.shape[0]))

        start_pos = URM_all.indptr[user_index]
        end_pos = URM_all.indptr[user_index+1]

        user_profile = URM_all.indices[start_pos:end_pos]

        unobserved_index = np.in1d(all_items, user_profile, assume_unique=True, invert=True)

        unobserved_items = all_items[unobserved_index]
        np.random.shuffle(unobserved_items)

        n_test_items = URM_test.indptr[user_index+1] - URM_test.indptr[user_index]

        num_negative_items = n_test_items * negative_items_per_positive

        if num_negative_items > len(unobserved_items):
            print("split_data_train_validation_test_negative: WARNING number of negative to sample for user {} is greater than available negative items {}".format(num_negative_items, len(unobserved_items)))
            num_negative_items = min(num_negative_items, len(unobserved_items))

        URM_negative_builder.add_single_row(user_index, unobserved_items[:num_negative_items], 1.0)



    URM_negative = URM_negative_builder.get_SparseMatrix()




    return URM_train, URM_validation, URM_test, URM_negative






def split_train_validation_percentage_random_holdout(URM_train, train_percentage = 0.8):

    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)


    URM_train = sps.coo_matrix(URM_train)

    train_mask = np.random.rand(URM_train.nnz) <= train_percentage
    validation_mask = np.logical_not(train_mask)


    URM_train_builder.add_data_lists(URM_train.row[train_mask], URM_train.col[train_mask], URM_train.data[train_mask])
    URM_validation_builder.add_data_lists(URM_train.row[validation_mask], URM_train.col[validation_mask], URM_train.data[validation_mask])


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()


    return URM_train, URM_validation



def split_train_validation_cold_start_user_wise(URM_train, full_train_percentage = 0.0, cold_items=1 ,verbose=True):

    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

    # ensure to use csr matrix or we get big problem
    URM_train = URM_train.tocsr()

    num_users, num_items = URM_train.shape

    URM_train_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)

    user_no_item_train = 0
    user_no_item_validation = 0

    # if we split two time train-test and train-validation we could get users with no items in the second split,
    # in order to get good test with enough non empty users, get the random users within the users with at least <cold_items>
    nnz_per_row = URM_train.getnnz(axis=1)

    users_enough_items = np.where(nnz_per_row > cold_items)[0]
    users_no_enough_items = np.where(nnz_per_row <= cold_items)[0]

    np.random.shuffle(users_enough_items)

    n_train_users = round(len(users_enough_items)*full_train_percentage)

    print("Users enough items: {}".format(len(users_enough_items)))
    print("Users no enough items: {}".format(len(users_no_enough_items)))

    # create full train part without coldstart
    for user_id in np.concatenate((users_enough_items[0:n_train_users], users_no_enough_items), axis=0):
        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id + 1]
        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)
        URM_train_builder.add_data_lists([user_id] * user_profile_length, user_profile_items, user_profile_ratings)


    # create test + train for the cold start users
    for user_id in users_enough_items[n_train_users:]:

        start_pos = URM_train.indptr[user_id]
        end_pos = URM_train.indptr[user_id+1]


        user_profile_items = URM_train.indices[start_pos:end_pos]
        user_profile_ratings = URM_train.data[start_pos:end_pos]
        user_profile_length = len(user_profile_items)

        n_train_items = min(cold_items, user_profile_length)

        if n_train_items == len(user_profile_items) and n_train_items > 1:
            n_train_items -= 1

        indices_for_sampling = np.arange(0, user_profile_length, dtype=np.int)
        np.random.shuffle(indices_for_sampling)

        train_items = user_profile_items[indices_for_sampling[0:n_train_items]]
        train_ratings = user_profile_ratings[indices_for_sampling[0:n_train_items]]

        validation_items = user_profile_items[indices_for_sampling[n_train_items:]]
        validation_ratings = user_profile_ratings[indices_for_sampling[n_train_items:]]

        if len(train_items) == 0:
            if verbose: print("User {} has 0 train items".format(user_id))
            user_no_item_train += 1

        if len(validation_items) == 0:
            if verbose: print("User {} has 0 validation items".format(user_id))
            user_no_item_validation += 1

        URM_train_builder.add_data_lists([user_id]*len(train_items), train_items, train_ratings)
        URM_validation_builder.add_data_lists([user_id]*len(validation_items), validation_items, validation_ratings)

    if user_no_item_train != 0:
        print("Warning split: {} users with 0 train items ({} total users)".format(user_no_item_train, URM_train.shape[0]))
    if user_no_item_validation != 0:
        print("Warning split: {} users with 0 validation items ({} total users)".format(user_no_item_validation, URM_train.shape[0]))

    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()

    return URM_train, URM_validation