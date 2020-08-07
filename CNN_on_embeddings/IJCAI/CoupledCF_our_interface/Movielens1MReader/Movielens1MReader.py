#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Simone Boglio
"""

from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Conferences.IJCAI.CoupledCF_our_interface.Movielens1MReader.Movielens1MReader import Movielens1MReader as Movielens1MReader_original


class Movielens1MReader_Wrapper(DataReader):

    DATASET_SUBFOLDER = "Movielens1MReader_Wrapper/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ['ICM_all']
    AVAILABLE_UCM = ["UCM_all"]


    def __init__(self, pre_splitted_path, type='original'):
        super(Movielens1MReader_Wrapper, self).__init__()
        self._originalReader = Movielens1MReader_original(pre_splitted_path, type=type)


    def _load_from_original_file(self):

        URM_all = self._originalReader.URM_DICT['URM_train'] + \
                  self._originalReader.URM_DICT['URM_validation'] + \
                  self._originalReader.URM_DICT['URM_test']

        n_users, n_items = URM_all.shape

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_test_negative": self._originalReader.URM_DICT['URM_test_negative']}

        loaded_ICM_dict = {"ICM_all": self._originalReader.ICM_DICT["ICM_all"]}
        loaded_ICM_mapper_dict = {"ICM_all": { i:i for i in range(self._originalReader.ICM_DICT["ICM_all"].shape[1])}}

        loaded_UCM_dict = {"UCM_all": self._originalReader.ICM_DICT["UCM_all"]}
        loaded_UCM_mapper_dict = {"UCM_all": { i:i for i in range(self._originalReader.ICM_DICT["UCM_all"].shape[1])}}

        user_original_ID_to_index = { i:i for i in range(n_users) }
        item_original_ID_to_index = { i:i for i in range(n_items) }

        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = loaded_ICM_mapper_dict,
                                 UCM_dictionary = loaded_UCM_dict,
                                 UCM_feature_mapper_dictionary = loaded_UCM_mapper_dict,
                                 user_original_ID_to_index= user_original_ID_to_index,
                                 item_original_ID_to_index= item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )

        return loaded_dataset


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
