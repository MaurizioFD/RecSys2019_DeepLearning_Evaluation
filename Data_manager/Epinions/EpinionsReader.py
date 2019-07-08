#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import bz2

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder


class EpinionsReader(DataReader):

    DATASET_URL = "http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
    DATASET_SUBFOLDER = "Epinions/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("EpinionsReader: Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        compressed_file_path = folder_path + "ratings_data.txt.bz2"
        decompressed_file_path = folder_path + "ratings_data.txt"


        try:

            open(decompressed_file_path, "r")

        except FileNotFoundError:

            print("EpinionsReader: Unable to find decompressed data file. Decompressing...")

            try:

                compressed_file = bz2.open(compressed_file_path, "rb")

            except Exception:

                print("EpinionsReader: Unable to find or open compressed data file. Downloading...")

                downloadFromURL(self.DATASET_URL, folder_path, "ratings_data.txt.bz2")

                compressed_file = bz2.open(compressed_file_path, "rb")


            decompressed_file = open(decompressed_file_path, "w")

            self._save_BZ2_in_text_file(compressed_file, decompressed_file)

            decompressed_file.close()


        print("EpinionsReader: loading URM")

        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(decompressed_file_path, separator=" ", header = True)



        print("EpinionsReader: cleaning temporary files")

        import os

        os.remove(decompressed_file_path)

        print("EpinionsReader: loading complete")




    def _save_BZ2_in_text_file(self, compressed_file, decompressed_file):

        print("EpinionsReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("EpinionsReader: decompressing file... done!")

