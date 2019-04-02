#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous authors
"""


import gzip


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


class BrightkiteReader(DataReader):

    DATASET_URL = "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz"
    DATASET_SUBFOLDER = "Brightkite/"
    AVAILABLE_ICM = []



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("BrightkiteReader: Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            compressed_file = gzip.open(folder_path + "loc-brightkite_edges.txt.gz", 'rb')

        except (FileNotFoundError):

            print("BrightkiteReader: Unable to find or extract data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, folder_path, "loc-brightkite_edges.txt.gz")

            compressed_file = gzip.open(folder_path + "loc-brightkite_edges.txt.gz", 'rb')

        URM_path = folder_path + "loc-brightkite_edges.txt"

        decompressed_file = open(URM_path, "w")

        self._save_GZ_in_text_file(compressed_file, decompressed_file)

        decompressed_file.close()


        print("BrightkiteReader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = self._loadURM(URM_path, separator="\t", header=False)


        print("BrightkiteReader: cleaning temporary files")

        import os

        os.remove(URM_path)

        print("BrightkiteReader: loading complete")




    def _loadURM(self, filePath, header=False, separator="::"):

        URM_all_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        fileHandle = open(filePath, "r")
        numCells = 0

        current_user = None
        checkins_user_sequence_counter = 1
        checkins_user_dict = {}
        checkins_user_multiplicity_dict = {}

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                user_id = line[0]
                venue_id = line[1]

                # Dataset has duplicate checkins, keep only the most recent one
                if current_user is None:
                    current_user = user_id

                if current_user != user_id:

                    # New user, flush data structure
                    venue_list_previous_user = list(checkins_user_dict.keys())

                    for venue_id_previous_user in venue_list_previous_user:
                        URM_all_builder.add_data_lists([current_user], [venue_id_previous_user],
                                                       [checkins_user_multiplicity_dict[venue_id_previous_user]])


                    current_user = user_id
                    checkins_user_sequence_counter = 1
                    checkins_user_dict = {}
                    checkins_user_multiplicity_dict = {}

                # checkins_user_profile_len_counter += 1

                if venue_id not in checkins_user_dict:
                    checkins_user_dict[venue_id] = checkins_user_sequence_counter
                    checkins_user_multiplicity_dict[venue_id] = 1
                    checkins_user_sequence_counter += 1

                else:
                    checkins_user_multiplicity_dict[venue_id] += 1

        # Save last user
        venue_list_previous_user = list(checkins_user_dict.keys())

        for venue_id_previous_user in venue_list_previous_user:
            URM_all_builder.add_data_lists([current_user], [venue_id_previous_user],
                                           [checkins_user_multiplicity_dict[venue_id_previous_user]])


        fileHandle.close()


        return URM_all_builder.get_SparseMatrix(), URM_all_builder.get_column_token_to_id_mapper(), URM_all_builder.get_row_token_to_id_mapper()






    def _save_GZ_in_text_file(self, compressed_file, decompressed_file):

        print("BrightkiteReader: decompressing file...")

        for line in compressed_file:
            decompressed_file.write(line.decode("utf-8"))

        decompressed_file.flush()

        print("BrightkiteReader: decompressing file... done!")

