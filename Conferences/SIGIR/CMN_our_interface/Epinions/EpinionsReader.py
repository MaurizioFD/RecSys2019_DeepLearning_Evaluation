#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import os, pickle
from Data_manager.load_and_save_data import save_data_dict, load_data_dict
from Data_manager.Epinions.EpinionsReader import EpinionsReader as EpinionsReader_DataManager
import numpy as np

from Data_manager.split_functions.split_train_validation import split_train_validation_test_negative_leave_one_out_user_wise

class EpinionsReader(object):

    def __init__(self):
        super(EpinionsReader, self).__init__()


        pre_splitted_path = "Data_manager_split_datasets/Epinions/SIGIR/CMN_our_interface/"

        pre_splitted_filename = "splitted_data"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("EpinionsReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("EpinionsReader: Pre-splitted data not found, building new one")

            print("EpinionsReader: loading URM")


            data_reader = EpinionsReader_DataManager()
            data_reader.load_data()

            URM_all = data_reader.get_URM_all()
            URM_all.data = np.ones_like(URM_all.data)

            self.URM_train, self.URM_validation, self.URM_test, self.URM_test_negative = split_train_validation_test_negative_leave_one_out_user_wise(URM_all, negative_items_per_positive=100)

            # Compatibility with the other two datasets
            self.URM_train_original = self.URM_train + self.URM_validation

            data_dict = {
                "URM_train_original": self.URM_train_original,
                "URM_train": self.URM_train,
                "URM_test": self.URM_test,
                "URM_test_negative": self.URM_test_negative,
                "URM_validation": self.URM_validation,
            }

            save_data_dict(data_dict, pre_splitted_path, pre_splitted_filename)


            print("EpinionsReader: loading complete")



