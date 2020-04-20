#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/10/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix



def split_data_on_timestamp(URM_all, URM_timestamp, negative_items_per_positive=100):

    URM_all = sps.csr_matrix(URM_all)
    URM_timestamp = sps.csr_matrix(URM_timestamp)

    n_rows, n_cols = URM_all.shape


    URM_train_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)
    URM_test_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)
    URM_validation_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)
    URM_negative_builder = IncrementalSparseMatrix(n_rows=n_rows, n_cols=n_cols)

    all_items = np.arange(0, n_cols, dtype=np.int)


    for user_index in range(URM_all.shape[0]):

        if user_index % 10000 == 0:
            print("split_data_on_sequence: user {} of {}".format(user_index, URM_all.shape[0]))

        start_pos = URM_all.indptr[user_index]
        end_pos = URM_all.indptr[user_index+1]

        user_profile = URM_all.indices[start_pos:end_pos]
        user_data = URM_all.data[start_pos:end_pos]
        user_sequence = URM_timestamp.data[start_pos:end_pos]


        unobserved_index = np.in1d(all_items, user_profile, assume_unique=True, invert=True)

        unobserved_items = all_items[unobserved_index]
        np.random.shuffle(unobserved_items)

        URM_negative_builder.add_single_row(user_index, unobserved_items[:negative_items_per_positive], 1.0)


        if len(user_profile) >= 3:



            # Test contain the first one, validation the second
            min_pos = np.argmax(user_sequence)

            venue_index = user_profile[min_pos]
            venue_data = user_data[min_pos]

            URM_test_builder.add_data_lists([user_index], [venue_index], [venue_data])

            user_profile = np.delete(user_profile, min_pos)
            user_data = np.delete(user_data, min_pos)
            user_sequence = np.delete(user_sequence, min_pos)


            min_pos = np.argmax(user_sequence)

            venue_index = user_profile[min_pos]
            venue_data = user_data[min_pos]

            URM_validation_builder.add_data_lists([user_index], [venue_index], [venue_data])

            user_profile = np.delete(user_profile, min_pos)
            user_data = np.delete(user_data, min_pos)
            #user_sequence = np.delete(user_sequence, min_pos)


            URM_train_builder.add_data_lists([user_index]*len(user_profile), user_profile, user_data)


    URM_train = URM_train_builder.get_SparseMatrix()
    URM_validation = URM_validation_builder.get_SparseMatrix()
    URM_test = URM_test_builder.get_SparseMatrix()
    URM_negative = URM_negative_builder.get_SparseMatrix()



    return URM_train, URM_validation, URM_test, URM_negative


