#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import os
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Data_manager.Epinions.EpinionsReader import EpinionsReader as EpinionsReader_DataManager
import numpy as np

from Data_manager.split_functions.split_train_validation import split_train_validation_test_negative_leave_one_out_user_wise

class EpinionsReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(EpinionsReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("EpinionsReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("EpinionsReader: Pre-splitted data not found, building new one")

            print("EpinionsReader: loading URM")


            data_reader = EpinionsReader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_all()
            URM_all.data = np.ones_like(URM_all.data)

            URM_train, URM_validation, URM_test, URM_test_negative = split_train_validation_test_negative_leave_one_out_user_wise(URM_all, negative_items_per_positive=100)

            # Compatibility with the other two datasets
            URM_train_original = URM_train + URM_validation

            self.URM_DICT = {
                "URM_train_original": URM_train_original,
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_test_negative": URM_test_negative,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


            print("EpinionsReader: loading complete")



