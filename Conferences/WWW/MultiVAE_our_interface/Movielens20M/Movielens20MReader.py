#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps

from Data_manager.Movielens.Movielens20MReader import Movielens20MReader as Movielens20MReader_DataManager
from Data_manager.split_functions.split_train_validation import split_train_validation_test_negative_leave_one_out_user_wise
from Base.Recommender_utils import reshapeSparse


import os
import pandas as pd

from Conferences.WWW.MultiVAE_our_interface.VAE_CF_data_splitter import split_train_validation_test_VAE_CF
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip


class Movielens20MReader(object):

    URM_DICT = {}
    ICM_DICT = {}


    def __init__(self, pre_splitted_path, split_type = "cold_user"):
        super(Movielens20MReader, self).__init__()

        assert split_type in ["cold_user", "warm_user"]

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("Movielens20MReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Movielens20MReader: Pre-splitted data not found, building new one")

            data_reader = Movielens20MReader_DataManager()
            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.get_URM_all()

            # binarize the data (only keep ratings >= 4)
            URM_all.data = URM_all.data >= 4.0
            URM_all.eliminate_zeros()


            if split_type == "cold_user":

                URM_all = sps.coo_matrix(URM_all)

                dict_for_dataframe = {"userId": URM_all.row,
                                      "movieId": URM_all.col,
                                      "rating": URM_all.data
                                      }

                URM_all_dataframe = pd.DataFrame(data=dict_for_dataframe)


                URM_train, URM_train_all, URM_validation, URM_test = split_train_validation_test_VAE_CF(URM_all_dataframe,
                                                                                                                             n_heldout_users = 10000)


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



            elif split_type == "warm_user":


                URM_all = sps.csr_matrix(URM_all)
                users_to_keep = np.ediff1d(URM_all.indptr) >= 4
                URM_all = URM_all[users_to_keep,:]

                URM_all = sps.csc_matrix(URM_all)
                items_to_keep = np.ediff1d(URM_all.indptr) >= 1
                URM_all = URM_all[:,items_to_keep]


                URM_all = sps.csr_matrix(URM_all)

                URM_train, URM_validation, URM_test, _ = split_train_validation_test_negative_leave_one_out_user_wise(URM_all)


                self.URM_DICT = {
                    "URM_train": URM_train,
                    "URM_test": URM_test,
                    "URM_validation": URM_validation

                }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)





            print("Movielens20MReader: Dataset loaded")



