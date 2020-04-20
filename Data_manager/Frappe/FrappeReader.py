#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Simone Boglio
"""



import zipfile, shutil, os
import numpy as np
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

from Base.Recommender_utils import reshapeSparse


class FrappeReader(DataReader):

    DATASET_URL = "https://github.com/hexiangnan/neural_factorization_machine/archive/master.zip"
    DATASET_SUBFOLDER = "Frappe/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self):
        super(FrappeReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "neural_factorization_machine-master.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")



        inner_path_in_zip = "neural_factorization_machine-master/data/frappe/"


        URM_train_path = dataFile.extract(inner_path_in_zip + "frappe.train.libfm", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(inner_path_in_zip + "frappe.test.libfm", path=zipFile_path + "decompressed/")
        URM_validation_path = dataFile.extract(inner_path_in_zip + "frappe.validation.libfm", path=zipFile_path + "decompressed/")


        tmp_URM_train, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_train_path,
                                                                                             item_original_ID_to_index = None,
                                                                                             user_original_ID_to_index = None)

        tmp_URM_test, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_test_path,
                                                                                             item_original_ID_to_index = item_original_ID_to_index,
                                                                                             user_original_ID_to_index = user_original_ID_to_index)

        tmp_URM_validation, item_original_ID_to_index, user_original_ID_to_index = self._loadURM(URM_validation_path,
                                                                                             item_original_ID_to_index = item_original_ID_to_index,
                                                                                             user_original_ID_to_index = user_original_ID_to_index)

        shape = (len(user_original_ID_to_index), len(item_original_ID_to_index))

        tmp_URM_train = reshapeSparse(tmp_URM_train, shape)
        tmp_URM_test = reshapeSparse(tmp_URM_test, shape)
        tmp_URM_validation = reshapeSparse(tmp_URM_validation, shape)


        URM_occurrence = tmp_URM_train + tmp_URM_test + tmp_URM_validation

        URM_all = URM_occurrence.copy()
        URM_all.data = np.ones_like(URM_all.data)

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_occurrence": URM_occurrence}

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = None,
                                 ICM_feature_mapper_dictionary = None,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= user_original_ID_to_index,
                                 item_original_ID_to_index= item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )


        self._print("cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset





    def _loadURM(self, file_name, header = False,
                 separator = " ",
                 item_original_ID_to_index = None,
                 user_original_ID_to_index = None):



        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index, on_new_col = "add",
                                                        preinitialized_row_mapper = user_original_ID_to_index, on_new_row = "add")


        fileHandle = open(file_name, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:

            if (numCells % 100000 == 0 and numCells!=0):
                print("Processed {} cells".format(numCells))

            line = line.split(separator)
            if (len(line)) > 1:
                if line[0]=='-1':
                    numCells += 1
                    continue
                elif line[0]=='1':
                    item = int(line[2].split(':')[0])
                    user = int(line[1].split(':')[0])
                    value = 1.0
                else:
                    print('ERROR READING DATASET')
                    break



            numCells += 1

            URM_builder.add_data_lists([user], [item], [value])



        fileHandle.close()


        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


