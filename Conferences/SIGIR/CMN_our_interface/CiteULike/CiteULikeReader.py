#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/07/18

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sps
import os

from Conferences.SIGIR.CMN_github.util.data import Dataset

from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

class CiteULikeReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(CiteULikeReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("CiteULikeReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("CiteULikeReader: Pre-splitted data not found, building new one")

            print("CiteULikeReader: loading URM")

            filename = "Conferences/SIGIR/CMN_github/data/citeulike-a.npz"

            URM_train_original, URM_test, URM_test_negative = self.build_sparse_matrix(filename)

            URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train_original.copy())


            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,

            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


        print("CiteULikeReader: Dataset loaded")










    def build_sparse_matrix(self, file_name):

        dataset = Dataset(file_name)

        dataset.test_data.items()

        print("Building sparse matrix from train data")

        row_ind_train = []
        col_ind_train = []
        data_train = []

        for user_id, item_id in dataset.train_data:
           row_ind_train.append(user_id)
           col_ind_train.append(item_id)
           data_train.append(1.0)

        URM_train = sps.csr_matrix((data_train, (row_ind_train, col_ind_train)))
        URM_train_shape = URM_train.shape


        print("Building sparse matrix from test data")

        row_ind_test_pos = []
        col_ind_test_pos = []
        data_test_pos = []

        for user_id, (pos_item, _) in dataset.test_data.items():
           row_ind_test_pos.append(user_id)
           col_ind_test_pos.append(pos_item)
           data_test_pos.append(1.0)

        URM_test = sps.csr_matrix((data_test_pos, (row_ind_test_pos, col_ind_test_pos)))
        URM_test_shape = URM_test.shape



        row_ind_test_neg = []
        col_ind_test_neg = []
        data_test_neg = []

        for user_id, (_, neg_item) in dataset.test_data.items():
           row_ind_test_neg.extend([user_id]*len(neg_item))
           col_ind_test_neg.extend(neg_item)
           data_test_neg.extend([1.0]*len(neg_item))

        URM_test_negative = sps.csr_matrix((data_test_neg, (row_ind_test_neg, col_ind_test_neg)))
        URM_test_negative_shape = URM_test_negative.shape

        if URM_train_shape != URM_test_shape or URM_train_shape != URM_test_negative_shape:

            print("Reshaping...")

            new_shape = (max(URM_train_shape[0], URM_test_shape[0], URM_test_negative_shape[0]),
                         max(URM_train_shape[1], URM_test_shape[1], URM_test_negative_shape[1]))

            URM_train = sps.csr_matrix((data_train, (row_ind_train, col_ind_train)), shape=new_shape)
            URM_test = sps.csr_matrix((data_test_pos, (row_ind_test_pos, col_ind_test_pos)), shape=new_shape)
            URM_test_negative = sps.csr_matrix((data_test_neg, (row_ind_test_neg, col_ind_test_neg)), shape=new_shape)


        print("Building sparse matrix is complete!")

        return URM_train, URM_test, URM_test_negative


