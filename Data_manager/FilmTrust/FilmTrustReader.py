#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Anonymous authors
"""



import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder




class FilmTrustReader(DataReader):

    DATASET_URL = "https://www.librec.net/datasets/filmtrust.zip"
    DATASET_SUBFOLDER = "FilmTrust/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = False


    def __init__(self):
        super(FilmTrustReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("FilmTrust: Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("FilmTrust: Unable to fild data zip file. Downloading...")

            downloadFromURL(self.DATASET_URL, zipFile_path, "filmtrust.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")


        URM_path = dataFile.extract("ratings.txt", path=zipFile_path + "decompressed/")


        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=" ")


        print("FilmTrust: cleaning temporary files")

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("FilmTrust: loading complete")

