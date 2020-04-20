#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import os
import scipy.sparse as sps
import numpy as np

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_random_holdout
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

class PinterestICCVReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(PinterestICCVReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("PinterestICCVReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("PinterestICCVReader: Pre-splitted data not found, building new one")

            print("PinterestICCVReader: loading URM")


            # data_reader = PinterestICCVReader()
            # loaded_dataset = data_reader.load_data()
            #
            # URM_all = loaded_dataset.get_URM_all()
            #
            # URM_train, URM_validation, URM_test, URM_negative = split_train_validation_test_negative_leave_one_out_user_wise(URM_all, negative_items_per_positive=100)

            dataset = Dataset_NeuralCollaborativeFiltering("Conferences/WWW/NeuMF_github/Data/pinterest-20")

            URM_train_original, URM_test, URM_test_negative = dataset.URM_train, dataset.URM_test, dataset.URM_test_negative

            URM_train, URM_validation = split_train_validation_percentage_random_holdout(URM_train_original.copy(), train_percentage=0.8)



            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,
            }


            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


            print("PinterestICCVReader: loading complete")








class Dataset_NeuralCollaborativeFiltering(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        testRatings = self.load_rating_file_as_matrix(path + ".test.rating")
        testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(testRatings) == len(testNegatives)

        self.num_users, self.num_items = trainMatrix.shape

        from Base.Recommender_utils import reshapeSparse

        self.URM_train = trainMatrix.tocsr()
        self.URM_test = testRatings.tocsr()

        shape = (max(self.URM_train.shape[0], self.URM_test.shape[0]),
                 max(self.URM_train.shape[1], self.URM_test.shape[1]))


        self.URM_train = reshapeSparse(self.URM_train, shape)
        self.URM_test = reshapeSparse(self.URM_test, shape)


        URM_test_negatives_builder = IncrementalSparseMatrix(n_rows=shape[0], n_cols=shape[1])

        for user_index in range(len(testNegatives)):

            user_test_items = testNegatives[user_index]

            URM_test_negatives_builder.add_single_row(user_index, user_test_items, data=1.0)


        self.URM_test_negative = URM_test_negatives_builder.get_SparseMatrix()








    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sps.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat
