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


class TafengReader:

    URM_DICT = {}
    ICM_DICT = {}

    CONFERENCE = "IJCAI"
    MODEL = "CoupledCF"
    DATASET_NAME = "Tafeng"

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

            from Conferences.IJCAI.CoupledCF_original import LoadTafengDataCnn as DatareaderOriginal
            path = "Conferences/IJCAI/CoupledCF_original/tafeng/"

            n_users, user_attributes_mat = DatareaderOriginal.load_user_attributes(path=path)
            n_items, items_genres_mat = DatareaderOriginal.load_itemGenres_as_matrix(path=path)
            ratings = DatareaderOriginal.load_rating_train_as_matrix(path=path)
            testRatings = DatareaderOriginal.load_rating_file_as_list(path=path)
            testNegatives = DatareaderOriginal.load_negative_file(path=path)

            URM_all = ratings.tocsr()

            UCM_all = sps.csc_matrix(user_attributes_mat)
            UCM_age = UCM_all[:, 0:11].tocsr()
            UCM_region = UCM_all[:, 11:19].tocsr()
            UCM_all = UCM_all.tocsr()

            # col: 0->category, 2->asset(0-1), 1->price(0-1)
            ICM_original = sps.csc_matrix(items_genres_mat)

            # category could be used as matrix, not single row
            ICM_sub_class = ICM_original[:, 0:1].tocsr()
            max = ICM_sub_class.shape[0]
            rows, cols, data = [], [], []
            for idx in range(max):
                # we have only index 0 as col
                data_vect = ICM_sub_class.data[ICM_sub_class.indptr[idx]:ICM_sub_class.indptr[idx+1]]
                if len(data_vect) == 0:
                    # handle category value 0 that in a csr matrix is not present
                    cols.append(int(0))
                else:
                    cols.append(int(data_vect[0]))
                rows.append(idx)
                data.append(1.0)

            ICM_sub_class = sps.csr_matrix((data, (rows, cols)))
            ICM_asset = ICM_original[:, 1:2].tocsr()
            ICM_price = ICM_original[:, 2:3].tocsr()

            ICM_original = ICM_original.tocsc()
            ICM_all = sps.hstack((ICM_sub_class, ICM_asset,ICM_price))


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

            if type == 'original':
                URM_test = URM_test
                URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_all.copy(), verbose=False)
            else:  # redo the split
                URM_full = URM_all + URM_test
                URM_temp, URM_test = split_train_validation_leave_one_out_user_wise(URM_full.copy(), verbose=False)
                URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_temp.copy(), verbose=False)




            self.ICM_DICT = {
                "UCM_age": UCM_age,
                "UCM_region": UCM_region,
                "UCM_all": UCM_all,
                "ICM_all": ICM_all,
                "ICM_original": ICM_original,
                "ICM_sub_class": ICM_sub_class,
                "ICM_asset": ICM_asset,
                "ICM_price": ICM_price,
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
