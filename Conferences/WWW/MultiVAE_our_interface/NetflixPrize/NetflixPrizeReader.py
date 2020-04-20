#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Maurizio Ferrari Dacrema
"""


import os
import scipy.sparse as sps
import pandas as pd
from Base.Recommender_utils import reshapeSparse

from Conferences.WWW.MultiVAE_our_interface.VAE_CF_data_splitter import split_train_validation_test_VAE_CF
from Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader as NetflixPrizeReader_DataManager
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

class NetflixPrizeReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):
        super(NetflixPrizeReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"


        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("NetflixPrizeReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("NetflixPrizeReader: Pre-splitted data not found, building new one")

            data_reader = NetflixPrizeReader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_all()

            # binarize the data (only keep ratings >= 4)
            URM_all.data = URM_all.data >= 4.0
            URM_all.eliminate_zeros()


            URM_all = sps.coo_matrix(URM_all)

            dict_for_dataframe = {"userId": URM_all.row,
                                  "movieId": URM_all.col,
                                  "rating": URM_all.data
                                }

            URM_all_dataframe = pd.DataFrame(data = dict_for_dataframe)


            URM_train, URM_train_all, URM_validation, URM_test = split_train_validation_test_VAE_CF(URM_all_dataframe, n_heldout_users = 40000)


            n_rows = max(URM_train.shape[0], URM_train_all.shape[0], URM_validation.shape[0], URM_test.shape[0])
            n_cols = max(URM_train.shape[1], URM_train_all.shape[1], URM_validation.shape[1], URM_test.shape[1])

            newShape = (n_rows, n_cols)

            URM_test = reshapeSparse(URM_test, newShape)
            URM_train = reshapeSparse(URM_train, newShape)
            URM_train_all = reshapeSparse(URM_train_all, newShape)
            URM_test = reshapeSparse(URM_test, newShape)



            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_train_all": URM_train_all,
                "URM_test": URM_test,
                "URM_validation": URM_validation,

            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)




            print("NetflixPrizeReader: Dataset loaded")

