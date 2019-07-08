#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL




def _loadURM_preinitialized_item_id (filePath, header = False, separator="::",
                                     if_new_user = "add", if_new_item = "ignore",
                                     item_original_ID_to_index = None,
                                     user_original_ID_to_index = None):


    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index,
                                                    on_new_col = if_new_item,
                                                    preinitialized_row_mapper = user_original_ID_to_index,
                                                    on_new_row = if_new_user)


    fileHandle = open(filePath, "r")
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


        try:
            value = float(line[2])

            if value != 0.0:

                URM_builder.add_data_lists([user_id], [item_id], [value])

        except:
            pass

    fileHandle.close()


    return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()



def _loadICM_genres(genres_path, header=True, separator=',', genresSeparator="|"):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(genres_path, "r", encoding="latin1")
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

            movie_id = line[0]

            title = line[1]
            # In case the title contains commas, it is enclosed in "..."
            # genre list will always be the last element
            genreList = line[-1]

            genreList = genreList.split(genresSeparator)

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(movie_id, genreList, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






def _loadICM_tags(tags_path, header=True, separator=',', if_new_item = "ignore",
                  item_original_ID_to_index = None, preinitialized_col_mapper = None):

    # Tags
    from Data_manager.TagPreprocessing import tagFilterAndStemming


    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = preinitialized_col_mapper, on_new_col = "add",
                                                    preinitialized_row_mapper = item_original_ID_to_index, on_new_row = if_new_item)



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
            movie_id = line[1]

            tagList = line[2]

            # Remove non alphabetical character and split on spaces
            tagList = tagFilterAndStemming(tagList)

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(movie_id, tagList, data = 1.0)


    fileHandle.close()



    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()








class Movielens20MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    DATASET_SUBFOLDER = "Movielens20M/"
    AVAILABLE_ICM = ["ICM_all", "ICM_genres", "ICM_tags"]

    IS_IMPLICIT = True




    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original

        print("Movielens20MReader: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("Movielens20MReader: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "ml-20m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-20m.zip")


        genres_path = dataFile.extract("ml-20m/movies.csv", path=zipFile_path + "decompressed/")
        tags_path = dataFile.extract("ml-20m/tags.csv", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-20m/ratings.csv", path=zipFile_path + "decompressed/")


        self.tokenToFeatureMapper_ICM_genres = {}
        self.tokenToFeatureMapper_ICM_tags = {}

        print("Movielens20MReader: loading genres")
        self.ICM_genres, self.tokenToFeatureMapper_ICM_genres, self.item_original_ID_to_index = _loadICM_genres(genres_path, header=True, separator=',', genresSeparator="|")

        print("Movielens20MReader: loading tags")
        self.ICM_tags, self.tokenToFeatureMapper_ICM_tags, _ = _loadICM_tags(tags_path, header=True, separator=',', if_new_item = "ignore",
                                                                             item_original_ID_to_index = self.item_original_ID_to_index)

        print("Movielens20MReader: loading URM")
        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM_preinitialized_item_id(URM_path, separator=",",
                                                                                          header = True, if_new_user = "add", if_new_item = "ignore",
                                                                                          item_original_ID_to_index = self.item_original_ID_to_index)

        self.ICM_all, self.tokenToFeatureMapper_ICM_all = self._merge_ICM(self.ICM_genres, self.ICM_tags,
                                                                          self.tokenToFeatureMapper_ICM_genres,
                                                                          self.tokenToFeatureMapper_ICM_tags)


        print("Movielens20MReader: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("Movielens20MReader: saving URM and ICM")








