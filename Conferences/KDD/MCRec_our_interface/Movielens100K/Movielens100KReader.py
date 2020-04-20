#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""
import os, zipfile, shutil

import numpy as np
import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

from Data_manager.Movielens.Movielens100KReader import Movielens100KReader as Movielens100KReader_DataManager

class Movielens100KReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(Movielens100KReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        original_data_path = "Conferences/KDD/MCRec_github/data/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Movielens100KReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Movielens100KReader: Pre-splitted data not found, building new one")

            print("Movielens100KReader: loading URM")


            from Conferences.KDD.MCRec_github.code.Dataset import Dataset

            dataset = 'ml-100k'

            dataset = Dataset(original_data_path + dataset)
            URM_train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

            # Dataset adds 1 to user and item id, removing it to restore 0 indexing
            URM_train = sps.coo_matrix(URM_train)
            URM_train.row -= 1
            URM_train.col -= 1

            URM_train = sps.csr_matrix((np.ones_like(URM_train.data), (URM_train.row, URM_train.col)))


            num_users, num_items = URM_train.shape



            # Build sparse matrices from lists
            URM_test_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)
            URM_test_negative_builder = IncrementalSparseMatrix(n_rows=num_users, n_cols=num_items)


            for user_index in range(len(testRatings)):

                user_id = testRatings[user_index][0]
                current_user_test_items = testRatings[user_index][1:]
                current_user_test_negative_items = testNegatives[user_index]

                current_user_test_items = np.array(current_user_test_items) -1
                current_user_test_negative_items = np.array(current_user_test_negative_items) -1

                URM_test_builder.add_single_row(user_id -1, current_user_test_items, 1.0)
                URM_test_negative_builder.add_single_row(user_id -1, current_user_test_negative_items, 1.0)



            # the test data has repeated data, apparently
            URM_test = URM_test_builder.get_SparseMatrix()

            URM_test_negative = URM_test_negative_builder.get_SparseMatrix()


            # Split validation from train as 10%
            from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_user_wise

            URM_train, URM_validation = split_train_validation_percentage_user_wise(URM_train, train_percentage=0.9)


            # Load features

            data_reader = Movielens100KReader_DataManager()
            loaded_dataset = data_reader.load_data()

            zipFile_path = data_reader.DATASET_SPLIT_ROOT_FOLDER + data_reader.DATASET_SUBFOLDER
            dataFile = zipfile.ZipFile(zipFile_path + "ml-100k.zip")

            ICM_path = dataFile.extract("ml-100k/u.item", path=zipFile_path + "decompressed/")

            ICM_genre = self._loadICM(ICM_path)
            ICM_genre = ICM_genre.get_SparseMatrix()

            shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

            self.ICM_DICT = {
                "ICM_genre": ICM_genre
            }

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

            print("Movielens100KReader: loading complete")




    def _loadICM (self, filePath, header = False, separator="|"):

        ICM_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        fileHandle = open(filePath, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                genre_list = [int(genre_bit) for genre_bit in line[5:]]
                item_id = int(line[0])-1

                ICM_builder.add_data_lists([item_id]*len(genre_list), genre_list, [1.0]*len(genre_list))


        fileHandle.close()

        return  ICM_builder
