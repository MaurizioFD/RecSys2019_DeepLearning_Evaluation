#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import os
import Data_manager.Utility as ut

from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

from Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader as MovielensHetrec2011Reader_DataManager
from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_user_wise

class MovielensHetrec2011Reader:


    URM_DICT = {}
    ICM_DICT = {}


    def __init__(self, pre_splitted_path):

        test_percentage = 0.2
        validation_percentage = 0.2

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:
            print("Dataset_MovielensHetrec2011: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_MovielensHetrec2011: Pre-splitted data not found, building new one")

            data_reader = MovielensHetrec2011Reader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_all()

            # keep only ratings 5
            URM_all.data = URM_all.data==5
            URM_all.eliminate_zeros()

            # create train - test - validation
            URM_train_original, URM_test = split_train_validation_percentage_user_wise(URM_all, train_percentage=1-test_percentage, verbose=False)

            URM_train, URM_validation = split_train_validation_percentage_user_wise(URM_train_original, train_percentage=1-validation_percentage, verbose=False)


            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,

            }


            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


        print("Dataset_MovielensHetrec2011: Dataset loaded")

        ut.print_stat_datareader(self)

