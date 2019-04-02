#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous authors
"""


import zipfile


from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder


class DeliciousHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip"
    DATASET_SUBFOLDER = "DeliciousHetrec2011/"
    AVAILABLE_ICM = []



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("DeliciousHetrec2011Reader: Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("DeliciousHetrec2011Reader: Unable to find or extract data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, folder_path, "hetrec2011-delicious-2k.zip")

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-delicious-2k.zip")



        URM_path = dataFile.extract("user_taggedbookmarks-timestamps.dat", path=folder_path + "decompressed")

        print("DeliciousHetrec2011Reader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t", header=True)


        print("DeliciousHetrec2011Reader: cleaning temporary files")

        import shutil

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        print("DeliciousHetrec2011Reader: loading complete")




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

                URM_builder.add_data_lists([user_id], [item_id], [1.0])

        fileHandle.close()

        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()


