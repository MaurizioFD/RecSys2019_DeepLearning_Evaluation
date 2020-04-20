#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

import os
import scipy.sparse as sps
import numpy as np
import Data_manager.Utility as ut


from Conferences.RecSys.SpectralCF_github.load_data import Data
from Conferences.RecSys.SpectralCF_github.params import *
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip

from Data_manager.Movielens.Movielens1MReader import Movielens1MReader as Movielens1MReader_DataManager

from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_user_wise
from Data_manager.split_functions.split_train_validation import split_train_validation_cold_start_user_wise


class Movielens1MReader:

    URM_DICT = {}
    ICM_DICT = {}


    def __init__(self, pre_splitted_path, type = "original", cold_start=False , cold_items=None):

        assert type in ["original", "ours"]

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # their mode in cold start
        mode = 1

        # path for pre existed movielens1M split
        movielens_splitted_path = "Conferences/RecSys/SpectralCF_github/data/ml-1m/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:
            print("Dataset_Movielens1M: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict_zip(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("Dataset_Movielens1M: Pre-splitted data not found, building new one")

            if type == "original":
                assert(cold_start is False)

                # use the SpectralCF class to read data
                data_generator = Data(train_file=movielens_splitted_path + 'train_users.dat',
                                      test_file=movielens_splitted_path + 'test_users.dat', batch_size=BATCH_SIZE)

                # convert train into csr
                full_train_matrix = sps.csr_matrix(data_generator.R)
                URM_train_original = full_train_matrix

                # convert test into csr
                test_set = data_generator.test_set
                uids, items = [], []
                for uid in test_set.keys():
                    uids += np.full(len(test_set[uid]), uid).tolist()
                    items += test_set[uid]
                test_matrix = sps.csr_matrix((np.ones(len(items)), (uids, items)), shape=(full_train_matrix.shape))

                if not cold_start:
                    URM_test = test_matrix

                    # create validation
                    URM_train, URM_validation = split_train_validation_percentage_user_wise(URM_train_original, train_percentage=0.9, verbose=False)

                else:
                    print('nothing')




            elif type == "ours":

                data_reader = Movielens1MReader_DataManager()
                loaded_dataset = data_reader.load_data()

                URM_all = loaded_dataset.get_URM_all()

                URM_all.data = URM_all.data==5
                URM_all.eliminate_zeros()

                if not cold_start:
                    URM_train, URM_test = split_train_validation_percentage_user_wise(URM_all, train_percentage=0.8, verbose=False)

                    URM_train, URM_validation = split_train_validation_percentage_user_wise(URM_train, train_percentage=0.9, verbose=False)

                else:

                    if mode==1: # their mode, cold start for full dataset
                        URM_train, URM_test = split_train_validation_cold_start_user_wise(URM_all, full_train_percentage=0.0, cold_items=cold_items, verbose=False)

                        URM_test, URM_validation = split_train_validation_percentage_user_wise(URM_test, train_percentage=0.9, verbose=False)


                    if mode==2: # cold start only for some users
                        URM_train, URM_test = split_train_validation_cold_start_user_wise(URM_all, full_train_percentage=0.8, cold_items=cold_items, verbose=False)

                        URM_train, URM_validation = split_train_validation_cold_start_user_wise(URM_train, full_train_percentage=0.9, cold_items=cold_items, verbose=False)




            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,

            }

            save_data_dict_zip(self.URM_DICT, self.ICM_DICT, pre_splitted_path, pre_splitted_filename)


        print("Dataset_Movielens1M: Dataset loaded")

        ut.print_stat_datareader(self)


