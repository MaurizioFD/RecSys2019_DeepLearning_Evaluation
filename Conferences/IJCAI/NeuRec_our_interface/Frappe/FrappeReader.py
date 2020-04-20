#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/03/19

@author: Simone Boglio
"""


import os
import numpy as np
import Data_manager.Utility as ut


from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_random_holdout
from Data_manager.Frappe.FrappeReader import FrappeReader as FrappeReader_DataManager
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

class FrappeReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(FrappeReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Dataset_Frappe: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_Frappe: Pre-splitted data not found, building new one")
            data_reader = FrappeReader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_all()

            URM_all.data = np.ones_like(URM_all.data)


            URM_train, URM_test = split_train_validation_percentage_random_holdout(URM_all, train_percentage=0.8)

            URM_train, URM_validation = split_train_validation_percentage_random_holdout(URM_train, train_percentage=0.9)


            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)



        print("FrappeReader: Dataset loaded")

        ut.print_stat_datareader(self)


