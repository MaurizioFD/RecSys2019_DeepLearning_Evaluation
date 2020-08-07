#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import os
import numpy as np
import scipy.sparse as sps
import Data_manager.Utility as ut

from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise


class Movielens1MReader():

    URM_DICT = {}
    ICM_DICT = {}

    CONFERENCE = "IJCAI"
    MODEL = "CoupledCF"
    DATASET_NAME = "Movielens1M"

    def __init__(self, pre_splitted_path, type='original'):
        assert type in ["original", "ours"]

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:
            print("Dataset_{}: Attempting to load pre-splitted data".format(self.DATASET_NAME))

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_{}: Pre-splitted data not found, building new one".format(self.DATASET_NAME))

            from Conferences.IJCAI.CoupledCF_original import LoadMovieDataCnn as DatareaderOriginal
            path = "Conferences/IJCAI/CoupledCF_original/ml-1m/"

            n_users, gender, age, occupation = DatareaderOriginal.load_user_attributes(path=path, split=True)
            n_items, items_genres_mat = DatareaderOriginal.load_itemGenres_as_matrix(path=path)
            ratings = DatareaderOriginal.load_rating_train_as_matrix(path=path)
            testRatings = DatareaderOriginal.load_rating_file_as_list(path=path)
            testNegatives = DatareaderOriginal.load_negative_file(path=path)

            URM_all = ratings.tocsr()

            UCM_gender = gender.tocsr()
            UCM_age = age.tocsr()
            UCM_occupation = occupation.tocsr()
            UCM_all = sps.hstack((UCM_gender, UCM_age, UCM_occupation)).tocsr()

            ICM_all = sps.csr_matrix(items_genres_mat)

            testRatings = np.array(testRatings).T
            URM_test_builder = IncrementalSparseMatrix(n_rows=n_users+1, n_cols=n_items+1)
            URM_test_builder.add_data_lists(testRatings[0], testRatings[1], np.ones(len(testRatings[0])))

            URM_test = URM_test_builder.get_SparseMatrix()


            URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=n_users+1, n_cols=n_items+1)

            # care here, the test negative start from index 0 but it refer to user index 1 (user index start from 1)
            n_negative_samples = 99
            for index in range(len(testNegatives)):
                user_test_items = testNegatives[index]
                if len(user_test_items) != n_negative_samples:
                    print("user id: {} has {} negative items instead {}".format(index+1, len(user_test_items), n_negative_samples))
                URM_test_negatives_builder.add_single_row(index + 1, user_test_items, data=1.0)

            URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()
            URM_test_negative.data = np.ones_like(URM_test_negative.data)

            if type=='original':
                URM_test = URM_test
                URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_all.copy(), verbose=False)

            else: # redo the split
                URM_full = URM_all + URM_test
                URM_temp, URM_test = split_train_validation_leave_one_out_user_wise(URM_full.copy(), verbose=False)
                URM_train , URM_validation = split_train_validation_leave_one_out_user_wise(URM_temp.copy(), verbose=False)


            self.ICM_DICT = {
                "UCM_gender": UCM_gender,
                "UCM_occupation": UCM_occupation,
                "UCM_age": UCM_age,
                "UCM_all": UCM_all,
                "ICM_all": ICM_all,
            }

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
                "URM_test_negative": URM_test_negative,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


        print("{}: Dataset loaded".format(self.DATASET_NAME))

        ut.print_stat_datareader(self)
