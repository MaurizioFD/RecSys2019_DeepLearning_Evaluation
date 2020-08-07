#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2019

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix




def split_train_leave_k_out_user_wise(URM, k_out = 1, use_validation_set = True, leave_random_out = True):
    """
    The function splits an URM in two matrices selecting the k_out interactions one user at a time
    :param URM:
    :param k_out:
    :param use_validation_set:
    :param leave_random_out:
    :return:
    """

    assert k_out > 0, "k_out must be a value greater than 0, provided was '{}'".format(k_out)

    URM = sps.csr_matrix(URM)
    n_users, n_items = URM.shape


    URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                        auto_create_col_mapper=False, n_cols = n_items)

    if use_validation_set:
         URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows = n_users,
                                                          auto_create_col_mapper=False, n_cols = n_items)



    for user_id in range(n_users):

        start_user_position = URM.indptr[user_id]
        end_user_position = URM.indptr[user_id+1]

        user_profile = URM.indices[start_user_position:end_user_position]


        if leave_random_out:
            indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

            np.random.shuffle(indices_to_suffle)

            user_interaction_items = user_profile[indices_to_suffle]
            user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

        else:

            # The first will be sampled so the last interaction must be the first one
            interaction_position = URM.data[start_user_position:end_user_position]

            sort_interaction_index = np.argsort(-interaction_position)

            user_interaction_items = user_profile[sort_interaction_index]
            user_interaction_data = URM.data[start_user_position:end_user_position][sort_interaction_index]



        #Test interactions
        user_interaction_items_test = user_interaction_items[0:k_out]
        user_interaction_data_test = user_interaction_data[0:k_out]

        URM_test_builder.add_data_lists([user_id]*len(user_interaction_items_test), user_interaction_items_test, user_interaction_data_test)


        #validation interactions
        if use_validation_set:
            user_interaction_items_validation = user_interaction_items[k_out:k_out*2]
            user_interaction_data_validation = user_interaction_data[k_out:k_out*2]

            URM_validation_builder.add_data_lists([user_id]*k_out, user_interaction_items_validation, user_interaction_data_validation)


        #Train interactions
        user_interaction_items_train = user_interaction_items[k_out*2:]
        user_interaction_data_train = user_interaction_data[k_out*2:]

        URM_train_builder.add_data_lists([user_id]*len(user_interaction_items_train), user_interaction_items_train, user_interaction_data_train)



    URM_train = URM_train_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()


    URM_train = sps.csr_matrix(URM_train)
    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train, user_no_item_train/n_users*100, n_users))



    if use_validation_set:
        URM_validation = URM_validation_builder.get_SparseMatrix()

        URM_validation = sps.csr_matrix(URM_validation)
        user_no_item_validation = np.sum(np.ediff1d(URM_validation.indptr) == 0)

        if user_no_item_validation != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Validation items".format(user_no_item_validation, user_no_item_validation/n_users*100, n_users))


        return URM_train, URM_validation, URM_test


    return URM_train, URM_test


