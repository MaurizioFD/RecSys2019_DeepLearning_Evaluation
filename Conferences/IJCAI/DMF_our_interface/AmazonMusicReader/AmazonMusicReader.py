#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Simone Boglio
"""

import os
import pandas as pd

from Data_manager.AmazonReviewData.AmazonMusicReader import AmazonMusicReader as AmazonMusicReader_DataManager
from Data_manager.split_functions.split_data_on_timestamp import split_data_on_timestamp
from Data_manager.split_functions.split_train_validation import split_train_validation_leave_one_out_user_wise
from Data_manager.load_and_save_data import save_data_dict_zip, load_data_dict_zip
from Data_manager.Utility import filter_urm, print_stat_datareader
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


class AmazonMusicReader:

    DATASET_NAME = "AmazonMusic"

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path, original=True):

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

            if original:

                URM_path = 'Conferences/IJCAI/DMF_original/data_www/Amazon_ratings_Digital_Music_pruned.txt'
                #
                # dataFile = open(URM_path, "r")
                #
                # # textData = dataFile.readlines()
                # dataFile.close()
                #
                # u_map = {}
                # discarded = 0
                # for line in tqdm(textData):
                #     line = line.split(' ')
                #     u, i, rating, new_time = int(line[0]), int(line[1]), float(line[2]), int(line[3])
                #
                #     # convert u id and i id in integer starting from 0 and initialize u_map
                #     if u not in u_map:
                #         u_map[u] = {}
                #
                #     if i not in u_map[u]:
                #         u_map[u][i] = [rating, new_time]
                #     else:  # rating already exist, keep the most recent timestamp
                #         discarded += 1
                #         current_time = u_map[u][i][1]
                #         if new_time > current_time:
                #             u_map[u][i] = [rating, new_time]
                #
                # print('Merged {} interactions, kept the most recent timestamps'.format(discarded))
                #
                # UTM_builder = IncrementalSparseMatrix()
                # URM_builder = IncrementalSparseMatrix()
                #
                # for u in u_map:
                #     items, ratings, timestamps = [], [], []
                #     for i in u_map[u]:
                #         items.append(i)
                #         timestamps.append(u_map[u][i][1])
                #         ratings.append(u_map[u][i][0])
                #     UTM_builder.add_data_lists(row_list_to_add=np.full(len(items), int(u)), col_list_to_add=items, data_list_to_add=timestamps)
                #     URM_builder.add_data_lists(row_list_to_add=np.full(len(items), int(u)), col_list_to_add=items, data_list_to_add=ratings)
                #

                URM_rating_builder = IncrementalSparseMatrix( auto_create_col_mapper = True, auto_create_row_mapper = True)
                URM_timestamp_builder = IncrementalSparseMatrix( auto_create_col_mapper = True, auto_create_row_mapper = True)

                # URM_duplicate_assert_builder = IncrementalSparseMatrix( auto_create_col_mapper = True, auto_create_row_mapper = True)


                df_original = pd.read_csv(filepath_or_buffer=URM_path, sep=" ", header= None,
                                dtype={0:int, 1:int, 2:float, 3:int})

                df_original.columns = ['userId', 'itemId', 'rating', 'timestamp']

                userId_list = df_original['userId'].values
                itemId_list = df_original['itemId'].values
                rating_list = df_original['rating'].values
                timestamp_list = df_original['timestamp'].values


                URM_rating_builder.add_data_lists(userId_list, itemId_list, rating_list)
                URM_timestamp_builder.add_data_lists(userId_list, itemId_list, timestamp_list)

                # URM_duplicate_assert_builder.add_data_lists(userId_list, itemId_list, np.ones_like(rating_list))
                # URM_duplicate_assert = URM_duplicate_assert_builder.get_SparseMatrix()
                #
                # assert np.all(URM_duplicate_assert.data == 1.0), "Duplicates detected"


                # Check if duplicates exist
                num_unique_user_item_ids = df_original.drop_duplicates(['userId', 'itemId'], keep='first', inplace=False).shape[0]
                assert num_unique_user_item_ids == len(userId_list), "Duplicate (user, item) values found"


                URM_timestamp = URM_timestamp_builder.get_SparseMatrix()
                URM_all = URM_rating_builder.get_SparseMatrix()

                URM_train, URM_validation, URM_test, URM_test_negative = split_data_on_timestamp(URM_all, URM_timestamp, negative_items_per_positive=99)

                # We want the validation to be sampled at random, not as the last interaction
                URM_train = URM_train + URM_validation
                URM_train, URM_validation = split_train_validation_leave_one_out_user_wise(URM_train, verbose=False)


            else:
                # create from full dataset with leave out one time wise from ORIGINAL full dateset
                data_reader = AmazonMusicReader_DataManager()
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
