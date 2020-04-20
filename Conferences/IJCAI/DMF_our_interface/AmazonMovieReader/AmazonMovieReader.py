#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Simone Boglio
"""

import os

from Data_manager.AmazonReviewData.AmazonMoviesTVReader import AmazonMoviesTVReader as AmazonMoviesTVReader_DataManager
from Data_manager.split_functions.split_data_on_timestamp import split_data_on_timestamp
from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Data_manager.Utility import filter_urm, print_stat_datareader



class AmazonMovieReader:

    DATASET_NAME = "AmazonMovie"

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:
            print("Dataset_{}: Attempting to load pre-splitted data".format(self.DATASET_NAME))

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:
            print("Dataset_{}: Pre-splitted data not found, building new one".format(self.DATASET_NAME))


        # create from full dataset with leave out one time wise from ORIGINAL full dateset
            data_reader = AmazonMoviesTVReader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_from_name("URM_all")
            URM_timestamp = loaded_dataset.get_URM_from_name("URM_timestamp")

            # use this function 2 time because the order could change slightly the number of final interactions
            URM_all = filter_urm(URM_all, user_min_number_ratings=1, item_min_number_ratings=5)
            URM_all = filter_urm(URM_all, user_min_number_ratings=20, item_min_number_ratings=1)
            URM_timestamp = filter_urm(URM_timestamp, user_min_number_ratings=1, item_min_number_ratings=5)
            URM_timestamp = filter_urm(URM_timestamp, user_min_number_ratings=20, item_min_number_ratings=1)

            URM_timestamp = URM_timestamp

            URM_train, URM_validation, URM_test, URM_test_negative = split_data_on_timestamp(URM_all, URM_timestamp, negative_items_per_positive=99)

            # We want the validation to be sampled at random, not as the last interaction
            URM_train = URM_train + URM_validation
            URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train, verbose=False)


            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
                "URM_test_negative": URM_test_negative,
                "URM_timestamp": URM_timestamp,
            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)

        print("{}: Dataset loaded".format(self.DATASET_NAME))

        print_stat_datareader(self)
