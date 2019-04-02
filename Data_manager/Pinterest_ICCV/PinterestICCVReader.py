#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous authors
"""


import bson, os, zipfile

from Data_manager.DataReader import DataReader

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


class PinterestICCVReader(DataReader):

    DATASET_URL = "https://sites.google.com/site/xueatalphabeta/academic-projects"
    DATASET_SUBFOLDER = "Pinterest_ICCV/"

    def __init__(self):
        super(PinterestICCVReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original


        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(compressed_file_folder + "pinterest_iccv.zip")
            dataFile.extractall(path=decompressed_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("PinterestICCVReader: Unable to fild data zip file.")
            print("PinterestICCVReader: Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            print("PinterestICCVReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")



        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = self._load_pins(decompressed_file_folder + "decompressed/")


        print("PinterestICCVReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_file_folder + "decompressed", ignore_errors=True)

        print("PinterestICCVReader: loading complete")





    def _load_pins(self, folder_path):

        file = open(folder_path + 'pinterest_iccv/subset_iccv_pin_im.bson', "rb")
        #data = bson.decode_all(file.read())
        #data = bson.loads(file.read())

        # Load mapping pin_id to image_id

        pin_id_to_image_id = {}

        for line in file:

            try:

                data_row = bson.loads(line)

            except:
                pass

            pin_id = data_row["pin_id"]

            image_id = data_row["im_name"]

            pin_id_to_image_id[pin_id] = image_id






        file = open(folder_path + 'pinterest_iccv/subset_iccv_board_pins.bson', "rb")
        #data = bson.decode_all(file.read())

        URM_pins = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        for line in file:

            data_row = bson.loads(line)

            user_id = data_row["board_id"]

            pins_list = data_row["pins"]

            image_id_list = [pin_id_to_image_id[pin_id] for pin_id in pins_list]

            URM_pins.add_single_row(user_id, image_id_list, data=1.0)



        return URM_pins.get_SparseMatrix(), URM_pins.get_column_token_to_id_mapper(), URM_pins.get_row_token_to_id_mapper()







