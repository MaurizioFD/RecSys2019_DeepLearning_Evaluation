#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""



import zipfile, shutil

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL

from Data_manager.Movielens20M.Movielens20MReader import _loadICM_genres, _loadURM_preinitialized_item_id





def _loadUCM(UCM_path, header=True, separator=','):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(UCM_path, "r", encoding="latin1")
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

            token_list = []
            token_list.append("gender_" + str(line[1]))
            token_list.append("age_group_" + str(line[2]))
            token_list.append("occupation_" + str(line[3]))
            token_list.append("zip_code_" + str(line[4]))

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(user_id, token_list, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()





















class Movielens1MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens1M/"
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("Movielens1MReader: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens1MReader: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        ICM_genre_path = dataFile.extract("ml-1m/movies.dat", path=zipFile_path + "decompressed/")
        UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")


        self.tokenToFeatureMapper_ICM_genres = {}
        self.tokenToFeatureMapper_UCM_all = {}

        print("Movielens1MReader: loading genres")
        self.ICM_genres, self.tokenToFeatureMapper_ICM_genres, self.item_original_ID_to_index = _loadICM_genres(ICM_genre_path, header=True, separator='::', genresSeparator="|")

        print("Movielens1MReader: loading UCM")
        self.UCM_all, self.tokenToFeatureMapper_UCM_all, self.user_original_ID_to_index = _loadUCM(UCM_path, header=True, separator='::')

        print("Movielens1MReader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM_preinitialized_item_id(URM_path, separator="::",
                                                                                          header = True, if_new_user = "ignore", if_new_item = "ignore",
                                                                                          item_original_ID_to_index = self.item_original_ID_to_index,
                                                                                          user_original_ID_to_index = self.user_original_ID_to_index)



        print("Movielens1MReader: cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens1MReader: loading complete")

