#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import zipfile, shutil
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.Movielens._utils_movielens_parser import _loadICM_tags, _loadURM_preinitialized_item_id, _loadICM_genres, _loadUCM



class Movielens1MReader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    DATASET_SUBFOLDER = "Movielens1M/"
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):
        # Load data from original
        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to fild data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "ml-1m.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "ml-1m.zip")


        ICM_genre_path = dataFile.extract("ml-1m/movies.dat", path=zipFile_path + "decompressed/")
        UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        URM_path = dataFile.extract("ml-1m/ratings.dat", path=zipFile_path + "decompressed/")


        self._print("loading genres")
        ICM_genres, tokenToFeatureMapper_ICM_genres, item_original_ID_to_index = _loadICM_genres(ICM_genre_path, header=True, separator='::', genresSeparator="|")

        self._print("loading UCM")
        UCM_all, tokenToFeatureMapper_UCM_all, user_original_ID_to_index = _loadUCM(UCM_path, header=True, separator='::')

        self._print("loading URM")
        URM_all, item_original_ID_to_index, user_original_ID_to_index, URM_timestamp = _loadURM_preinitialized_item_id(URM_path, separator="::",
                                                                                          header = False, if_new_user = "ignore", if_new_item = "ignore",
                                                                                          item_original_ID_to_index = item_original_ID_to_index,
                                                                                          user_original_ID_to_index = user_original_ID_to_index)
        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

        loaded_ICM_dict = {"ICM_genres": ICM_genres}
        loaded_ICM_mapper_dict = {"ICM_genres": tokenToFeatureMapper_ICM_genres}

        loaded_UCM_dict = {"UCM_all": UCM_all}
        loaded_UCM_mapper_dict = {"UCM_all": tokenToFeatureMapper_UCM_all}



        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = loaded_ICM_mapper_dict,
                                 UCM_dictionary = loaded_UCM_dict,
                                 UCM_feature_mapper_dictionary = loaded_UCM_mapper_dict,
                                 user_original_ID_to_index= user_original_ID_to_index,
                                 item_original_ID_to_index= item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )


        self._print("cleaning temporary files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("loading complete")

        return loaded_dataset

