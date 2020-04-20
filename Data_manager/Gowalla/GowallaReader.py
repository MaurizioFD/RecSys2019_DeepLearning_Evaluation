#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/04/2019

@author: Simone Boglio
"""



import gzip, os
import numpy as np
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import load_CSV_into_SparseBuilder
from Data_manager.DataReader_utils import download_from_URL


class GowallaReader(DataReader):

    DATASET_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    DATASET_SUBFOLDER = "Gowalla/"


    ZIP_NAME = "loc-gowalla_totalCheckins.txt.gz"
    FILE_RATINGS_PATH =   "loc-gowalla_totalCheckins.txt"


    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            compressed_file = gzip.open(folder_path + self.ZIP_NAME, )

        except FileNotFoundError:

            self._print("Unable to find data zip file. Downloading...")
            download_from_URL(self.DATASET_URL, folder_path, self.ZIP_NAME)

            compressed_file = gzip.open(folder_path + self.ZIP_NAME)


        URM_path = folder_path + self.FILE_RATINGS_PATH

        decompressed_file = open(URM_path, "w")

        self._save_GZ_in_text_file(compressed_file, decompressed_file)

        decompressed_file.close()

        self._print("loading URM")
        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path,
                                                                                                    header = False,
                                                                                                    separator="\t",
                                                                                                    remove_duplicates=True,
                                                                                                    custom_user_item_rating_columns = [0, 4, 2])

        # URM_all contains the coordinates in textual format
        URM_all.data = np.ones_like(URM_all.data)

        loaded_URM_dict = {"URM_all": URM_all}

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

        os.remove(URM_path)

        self._print("loading complete")

        return loaded_dataset





    def _save_GZ_in_text_file(self, compressed_file, decompressed_file):

        print("GowallaReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("GowallaReader: decompressing file... done!")

