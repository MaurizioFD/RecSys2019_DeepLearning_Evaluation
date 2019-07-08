#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import shutil
import zipfile
import os, pickle
import pandas as pd
import numpy as np

import Data_manager.Utility as ut

from Data_manager.DataReader_utils import downloadFromURL
from Data_manager.load_and_save_data import save_data_dict, load_data_dict

from Data_manager.MovielensHetrec2011.MovielensHetrec2011Reader import MovielensHetrec2011Reader as MovielensHetrec2011Reader_DataManager
from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_user_wise

class MovielensHetrec2011Reader:


    def __init__(self):

        test_percentage = 0.2
        validation_percentage = 0.2


        pre_splitted_path = "Data_manager_split_datasets/MovielensHetrec2011/RecSys/SpectralCF_our_interface/"

        pre_splitted_filename = "splitted_data"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:
            print("Dataset_MovielensHetrec2011: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_MovielensHetrec2011: Pre-splitted data not found, building new one")

            data_reader = MovielensHetrec2011Reader_DataManager()
            data_reader.load_data()

            URM_all = data_reader.get_URM_all()

            # keep only ratings 5
            URM_all.data = URM_all.data==5
            URM_all.eliminate_zeros()

            # create train - test - validation
            URM_train_original, self.URM_test = split_train_validation_percentage_user_wise(URM_all, train_percentage=1-test_percentage, verbose=False)

            self.URM_train, self.URM_validation = split_train_validation_percentage_user_wise(URM_train_original, train_percentage=1-validation_percentage, verbose=False)


            data_dict = {
                "URM_train": self.URM_train,
                "URM_test": self.URM_test,
                "URM_validation": self.URM_validation,

            }


            save_data_dict(data_dict, pre_splitted_path, pre_splitted_filename)


        print("Dataset_MovielensHetrec2011: Dataset loaded")

        ut.print_stat_datareader(self)

