#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import os, tarfile, shutil
import numpy as np
import scipy.sparse as sps
import Data_manager.Utility as ut

from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from Data_manager.split_functions.split_data_on_timestamp import split_data_on_timestamp
from Data_manager.Utility import filter_urm

from Conferences.IJCAI.ConvNCF_github.Dataset import Dataset as Dataset_github



class YelpReader:
    DATASET_NAME = "Yelp"

    URM_DICT = {}
    ICM_DICT = {}


    def __init__(self, pre_splitted_path, original=True):

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

            compressed_file_folder = "Conferences/IJCAI/ConvNCF_github/Data/"
            decompressed_file_folder = "Data_manager_split_datasets/Yelp/"

            # compressed_file = tarfile.open(compressed_file_folder + "yelp.test.negative.gz", "r:gz")
            # compressed_file.extract("yelp.test.negative", path=decompressed_file_folder + "decompressed/")
            # compressed_file.close()
            #
            # compressed_file = tarfile.open(compressed_file_folder + "yelp.test.rating.gz", "r:gz")
            # compressed_file.extract("yelp.test.rating", path=decompressed_file_folder + "decompressed/")
            # compressed_file.close()
            #
            # compressed_file = tarfile.open(compressed_file_folder + "yelp.train.rating.gz", "r:gz")
            # compressed_file.extract("yelp.train.rating", path=decompressed_file_folder + "decompressed/")
            # compressed_file.close()


            # if original:

            Dataset_github.load_rating_file_as_list = Dataset_github.load_training_file_as_matrix

            try:
                dataset = Dataset_github(compressed_file_folder + "yelp")

            except FileNotFoundError as exc:

                print("Dataset_{}: Uncompressed files not found, please manually decompress the *.gz files in this folder: '{}'".format(self.DATASET_NAME, compressed_file_folder))

                raise exc


            URM_train_original, URM_test = dataset.trainMatrix, dataset.testRatings

            n_users = max(URM_train_original.shape[0], URM_test.shape[0])
            n_items = max(URM_train_original.shape[1], URM_test.shape[1])

            URM_train_original = sps.csr_matrix(URM_train_original, shape=(n_users, n_items))
            URM_test = sps.csr_matrix(URM_test, shape=(n_users, n_items))

            URM_train_original.data = np.ones_like(URM_train_original.data)
            URM_test.data = np.ones_like(URM_test.data)

            URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=n_users, n_cols=n_items)

            n_negative_samples = 999
            for user_index in range(len(dataset.testNegatives)):
                user_test_items = dataset.testNegatives[user_index]
                if len(user_test_items) != n_negative_samples:
                    print("user id: {} has {} negative items instead {}".format(user_index, len(user_test_items), n_negative_samples))
                URM_test_negatives_builder.add_single_row(user_index, user_test_items, data=1.0)

            URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()
            URM_test_negative.data = np.ones_like(URM_test_negative.data)

            URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train_original.copy(),verbose=False)

            #
            # else:
            #     data_reader = YelpReader_DataManager()
            #     loaded_dataset = data_reader.load_data()
            #
            #     URM_all = loaded_dataset.get_URM_all()
            #
            #     URM_timestamp = URM_all.copy()
            #
            #     URM_all.data = np.ones_like(URM_all.data)
            #
            #     URM_train, URM_validation, URM_test, URM_negative = split_data_on_timestamp(URM_all, URM_timestamp, negative_items_per_positive=999)
            #     URM_train = URM_train + URM_validation
            #     URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train, verbose=False)

            shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
                "URM_test_negative": URM_test_negative,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

        print("{}: Dataset loaded".format(self.DATASET_NAME))

        ut.print_stat_datareader(self)
