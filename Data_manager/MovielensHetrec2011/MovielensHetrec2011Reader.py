#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Simone Boglio
"""



import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder




class MovielensHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
    DATASET_SUBFOLDER = "MovielensHetrec2011/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("MovielensHetrec2011: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("MovielensHetrec2011: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "hetrec2011-movielens-2k-v2.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "hetrec2011-movielens-2k-v2.zip")


        URM_path = dataFile.extract("user_ratedmovies.dat", path=zipFile_path + "decompressed/")


        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator="\t")


        print("MovielensHetrec2011: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("MovielensHetrec2011: loading complete")

