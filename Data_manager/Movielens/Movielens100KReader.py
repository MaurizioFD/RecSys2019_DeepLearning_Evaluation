#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class Movielens100KReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DATASET_SUBFOLDER = "Movielens100K/"
    AVAILABLE_ICM = []

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-100k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-100k.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-100k.zip")


        URM_path = dataFile.extract("ml-100k/u.data", path=zipFile_path + "decompressed/")

        URM_all, URM_timestamp, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=False, timestamp=True)


        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

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

