#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""


from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_random_holdout
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
import os, pickle

import scipy.io
import scipy.sparse as sps
import h5py

from Data_manager.load_and_save_data import save_data_dict, load_data_dict

from Base.Recommender_utils import reshapeSparse

class CiteulikeReader(object):


    def __init__(self, dataset_variant = "a", train_interactions = 1):

        super(CiteulikeReader, self).__init__()

        assert dataset_variant in ["a", "t"], "CiteulikeReader: dataset_variant must be either 'a' or 't'"
        assert train_interactions in [1, 10, "all"], "CiteulikeReader: train_interactions must be: 1, 10 or 'all'"



        pre_splitted_path = "Data_manager_split_datasets/CiteULike/KDD/CollaborativeVAE_our_interface/"


        pre_splitted_filename = "splitted_data_citeulike-{}-{}-items".format(dataset_variant, train_interactions)


        original_data_path = "Conferences/KDD/CollaborativeVAE_github/data/citeulike-{}/".format(dataset_variant)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("CiteulikeReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict(pre_splitted_path, pre_splitted_filename).items():
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

            self.URM_test = URM_test_builder.get_SparseMatrix()
            self.URM_train = URM_train_builder.get_SparseMatrix()


            if dataset_variant == "a":
                self.ICM_title_abstract = scipy.io.loadmat(original_data_path + "mult_nor.mat")['X']
            else:
                # Variant "t" uses a different file format and is transposed
                self.ICM_title_abstract = h5py.File(original_data_path + "mult_nor.mat").get('X')
                self.ICM_title_abstract = sps.csr_matrix(self.ICM_title_abstract).T

            self.ICM_title_abstract = sps.csr_matrix(self.ICM_title_abstract)


            n_rows = max(self.URM_test.shape[0], self.URM_train.shape[0])
            n_cols = max(self.URM_test.shape[1], self.URM_train.shape[1], self.ICM_title_abstract.shape[0])

            newShape = (n_rows, n_cols)

            self.URM_test = reshapeSparse(self.URM_test, newShape)
            self.URM_train = reshapeSparse(self.URM_train, newShape)


            if train_interactions == "all":

                self.URM_train += self.URM_test

                self.URM_train, self.URM_test = split_train_validation_percentage_random_holdout(self.URM_train, train_percentage = 0.8)
                self.URM_train, self.URM_validation = split_train_validation_percentage_random_holdout(self.URM_train, train_percentage = 0.8)

            else:

                self.URM_train, self.URM_validation = split_train_validation_percentage_random_holdout(self.URM_train, train_percentage = 0.8)



            data_dict = {
                "URM_train": self.URM_train,
                "URM_test": self.URM_test,
                "URM_validation": self.URM_validation,
                "ICM_title_abstract": self.ICM_title_abstract

            }

            save_data_dict(data_dict, pre_splitted_path, pre_splitted_filename)

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