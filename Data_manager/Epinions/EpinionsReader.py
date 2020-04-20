#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import bz2, os
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder


class EpinionsReader(DataReader):

    DATASET_URL = "http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
    DATASET_SUBFOLDER = "Epinions/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        compressed_file_path = folder_path + "ratings_data.txt.bz2"
        decompressed_file_path = folder_path + "ratings_data.txt"


        try:

            open(decompressed_file_path, "r")

        except FileNotFoundError:

            self._print("Unable to find decompressed data file. Decompressing...")

            try:

                compressed_file = bz2.open(compressed_file_path, "rb")

            except Exception:

                self._print("Unable to find or open compressed data file. Downloading...")

                download_from_URL(self.DATASET_URL, folder_path, "ratings_data.txt.bz2")

                compressed_file = bz2.open(compressed_file_path, "rb")


            decompressed_file = open(decompressed_file_path, "w")

            self._save_BZ2_in_text_file(compressed_file, decompressed_file)

            decompressed_file.close()


        self._print("loading URM")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(decompressed_file_path, separator=" ", header = False, timestamp = False)

        loaded_URM_dict = {"URM_all": URM_all}

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = None,
                                 ICM_feature_mapper_dictionary = None,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= self.user_original_ID_to_index,
                                 item_original_ID_to_index= self.item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )


        self._print("cleaning temporary files")

        os.remove(decompressed_file_path)

        self._print("loading complete")

        return loaded_dataset




    def _save_BZ2_in_text_file(self, compressed_file, decompressed_file):

        print("EpinionsReader: decompressing file...")

        for line in compressed_file:
            decoded_line = line.decode("utf-8")
            if len(decoded_line.split(" ")) == 3:
                decompressed_file.write(decoded_line)

        decompressed_file.flush()

        print("EpinionsReader: decompressing file... done!")

