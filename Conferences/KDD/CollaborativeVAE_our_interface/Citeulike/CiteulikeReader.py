#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""


from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_random_holdout
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import scipy.io
import scipy.sparse as sps
import h5py, os
import numpy as np

from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

from Base.Recommender_utils import reshapeSparse

class CiteulikeReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path, dataset_variant = "a", train_interactions = 1):

        super(CiteulikeReader, self).__init__()

        assert dataset_variant in ["a", "t"], "CiteulikeReader: dataset_variant must be either 'a' or 't'"
        assert train_interactions in [1, 10, "all"], "CiteulikeReader: train_interactions must be: 1, 10 or 'all'"

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = "Conferences/KDD/CollaborativeVAE_github/data/citeulike-{}/".format(dataset_variant)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("CiteulikeReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("CiteulikeReader: Pre-splitted data not found, building new one")

            print("CiteulikeReader: loading URM")

            if train_interactions=="all":
                train_interactions_file_suffix = 10
            else:
                train_interactions_file_suffix = train_interactions


            URM_train_builder = self._load_data_file(original_data_path + "cf-train-{}-users.dat".format(train_interactions_file_suffix))
            URM_test_builder = self._load_data_file(original_data_path + "cf-test-{}-users.dat".format(train_interactions_file_suffix))

            URM_test = URM_test_builder.get_SparseMatrix()
            URM_train = URM_train_builder.get_SparseMatrix()


            if dataset_variant == "a":
                ICM_tokens_TFIDF = scipy.io.loadmat(original_data_path + "mult_nor.mat")['X']
            else:
                # Variant "t" uses a different file format and is transposed
                ICM_tokens_TFIDF = h5py.File(original_data_path + "mult_nor.mat").get('X')
                ICM_tokens_TFIDF = sps.csr_matrix(ICM_tokens_TFIDF).T

            ICM_tokens_TFIDF = sps.csr_matrix(ICM_tokens_TFIDF)

            ICM_tokens_bool = ICM_tokens_TFIDF.copy()
            ICM_tokens_bool.data = np.ones_like(ICM_tokens_bool.data)


            n_rows = max(URM_test.shape[0], URM_train.shape[0])
            n_cols = max(URM_test.shape[1], URM_train.shape[1], ICM_tokens_TFIDF.shape[0])

            newShape = (n_rows, n_cols)

            URM_test = reshapeSparse(URM_test, newShape)
            URM_train = reshapeSparse(URM_train, newShape)


            if train_interactions == "all":

                URM_train += URM_test

                URM_train, URM_test = split_train_validation_percentage_random_holdout(URM_train, train_percentage = 0.8)
                URM_train, URM_validation = split_train_validation_percentage_random_holdout(URM_train.copy(), train_percentage = 0.8)

            elif train_interactions == 10:
                # If train interactions == 10 the train will NOT contain the validation data
                URM_train, URM_validation = split_train_validation_percentage_random_holdout(URM_train.copy(), train_percentage = 0.8)

            else:
                # If train interactions == 10 the train WILL contain the validation data
                _, URM_validation = split_train_validation_percentage_random_holdout(URM_train.copy(), train_percentage = 0.8)


            self.ICM_DICT = {
                "ICM_tokens_TFIDF": ICM_tokens_TFIDF,
                "ICM_tokens_bool": ICM_tokens_bool,
            }

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

            print("CiteulikeReader: loading complete")







    def _load_data_file(self, filePath, separator = " "):

        URM_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, auto_create_col_mapper=False)

        fileHandle = open(filePath, "r")
        user_index = 0


        for line in fileHandle:

            if (user_index % 1000000 == 0):
                print("Processed {} cells".format(user_index))

            if (len(line)) > 1:

                line = line.replace("\n", "")
                line = line.split(separator)

                if len(line)>0:

                    if line[0]!="0":

                        line = [int(line[i]) for i in range(len(line))]

                        URM_builder.add_single_row(user_index, line[1:], data=1.0)

            user_index += 1


        fileHandle.close()

        return  URM_builder