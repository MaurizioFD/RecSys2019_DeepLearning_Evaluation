#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import os, pickle
import scipy.sparse as sps

from Conferences.WWW.NeuMF_github.Dataset import Dataset as Dataset_github

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from Data_manager.load_and_save_data import save_data_dict, load_data_dict

class PinterestICCVReader(object):


    def __init__(self):

        super(PinterestICCVReader, self).__init__()

        pre_splitted_path = "Data_manager_split_datasets/PinterestICCV/WWW/NeuMF_our_interface/"

        pre_splitted_filename = "splitted_data"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Dataset_Pinterest: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_Pinterest: Pre-splitted data not found, building new one")

            # Ensure file is loaded as matrix
            Dataset_github.load_rating_file_as_list = Dataset_github.load_rating_file_as_matrix

            dataset = Dataset_github("Conferences/WWW/NeuMF_github/Data/pinterest-20")

            self.URM_train_original, self.URM_test = dataset.trainMatrix, dataset.testRatings

            self.URM_train_original = self.URM_train_original.tocsr()
            self.URM_test = self.URM_test.tocsr()


            from Base.Recommender_utils import reshapeSparse


            shape = (max(self.URM_train_original.shape[0], self.URM_test.shape[0]),
                     max(self.URM_train_original.shape[1], self.URM_test.shape[1]))


            self.URM_train_original = reshapeSparse(self.URM_train_original, shape)
            self.URM_test = reshapeSparse(self.URM_test, shape)


            URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=shape[0], n_cols=shape[1])

            for user_index in range(len(dataset.testNegatives)):

                user_test_items = dataset.testNegatives[user_index]

                URM_test_negatives_builder.add_single_row(user_index, user_test_items, data=1.0)


            self.URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()



            self.URM_train, self.URM_validation = split_train_validation_leave_one_out_user_wise(self.URM_train_original.copy())


            data_dict = {
                "URM_train_original": self.URM_train_original,
                "URM_train": self.URM_train,
                "URM_test": self.URM_test,
                "URM_test_negative": self.URM_test_negative,
                "URM_validation": self.URM_validation,

            }

            save_data_dict(data_dict, pre_splitted_path, pre_splitted_filename)


        print("Dataset_Pinterest: Dataset loaded")

        print("N_items {}, n_users {}".format(self.URM_train.shape[1], self.URM_train.shape[0]))



