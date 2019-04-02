#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Anonymous authors
"""



import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import downloadFromURL, load_CSV_into_SparseBuilder




class GowallaReader(DataReader):

    DATASET_NAME = "Gowalla"

    DATASET_URL = "http://dawenl.github.io/data/gowalla_pro.zip"
    ZIP_NAME = "gowalla_pro.zip"
    FILE_RATINGS_PATH =   "gowalla_pro/gwl_checkins.tsv"
    DATASET_SUBFOLDER = DATASET_NAME+"/"
    SEPARATOR = '\t'

    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True


    def __init__(self):
        super(GowallaReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("{}: Loading original data".format(self.DATASET_NAME))

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + self.ZIP_NAME)

        except (FileNotFoundError, zipfile.BadZipFile):

            print("{}: Unable to fild data zip file. Downloading...".format(self.DATASET_NAME))

            downloadFromURL(self.DATASET_URL, zipFile_path, self.ZIP_NAME)

            dataFile = zipfile.ZipFile(zipFile_path + self.ZIP_NAME)


        URM_path = dataFile.extract(self.FILE_RATINGS_PATH, path=zipFile_path + "decompressed/")


        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=self.SEPARATOR)


        print("{}: cleaning temporary files".format(self.DATASET_NAME))

        import shutil

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        print("{}: loading complete".format(self.DATASET_NAME))

