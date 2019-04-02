#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous authors
"""


import zipfile


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder


class LastFMHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    DATASET_SUBFOLDER = "LastFMHetrec2011/"
    AVAILABLE_ICM = ["ICM_tags"]



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("LastFMHetrec2011Reader: Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("LastFMHetrec2011Reader: Unable to find or extract data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, folder_path, "hetrec2011-lastfm-2k.zip")

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")



        URM_path = dataFile.extract("user_artists.dat", path=folder_path + "decompressed")
        tags_path = dataFile.extract("user_taggedartists-timestamps.dat", path=folder_path + "decompressed")


        print("LastFMHetrec2011Reader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=True)

        print("LastFMHetrec2011Reader: loading tags")
        self.ICM_tags, self.tokenToFeatureMapper_ICM_tags, _ = self._loadICM_tags(tags_path, header=True, separator='\t', if_new_item = "ignore")

        print("LastFMHetrec2011Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("LastFMHetrec2011Reader: loading complete")




    def _loadURM (self, filePath, header = True, separator="::", if_new_user = "add", if_new_item = "ignore"):



        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user)


        fileHandle = open(filePath, "r", encoding='latin1')
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

                user_id = line[0]
                item_id = line[1]

                # If 0 rating is implicit
                # To avoid removin it accidentaly, set ti to -1
                rating = float(line[2])

                if rating == 0:
                    rating = -1

                URM_builder.add_data_lists([user_id], [item_id], [rating])

        fileHandle.close()

        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()




    def _loadICM_tags(self, tags_path, header=True, separator=',', if_new_item = "ignore"):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)



        fileHandle = open(tags_path, "r", encoding="latin1")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 100000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # If a movie has no genre, ignore it
                item_id = line[1]
                tag = line[2]

                # Rows movie ID
                # Cols features
                ICM_builder.add_data_lists([item_id], [tag], [1.0])


        fileHandle.close()



        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()


