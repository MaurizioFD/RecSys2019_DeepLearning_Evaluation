#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/19

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps

from Base.DataIO import DataIO
from Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, remove_features
from Data_manager.data_consistency_check import assert_URM_ICM_mapper_consistency
from Data_manager.DataReader_utils import compute_density

def _clone_dictionary(original_dict):
    clone_dict = {key:value.copy() for key,value in original_dict.items()}
    return clone_dict

def gini_index(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array, dtype=np.float)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient



class Dataset(object):

    DATASET_NAME = None

    # Available URM split
    AVAILABLE_URM = {}

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = {}
    AVAILABLE_ICM_feature_mapper = {}
    _HAS_ICM = False

    # Available UCM for the given dataset, there might be no UCM, one or many
    AVAILABLE_UCM = {}
    AVAILABLE_UCM_feature_mapper = {}
    _HAS_UCM = False

    item_original_ID_to_index = {}
    user_original_ID_to_index = {}

    additional_data_mapper = {}
    _HAS_additional_mapper = False

    _IS_IMPLICIT = False

    # Mappers specific for a given dataset, they might be related to more complex data structures or FEATURE_TOKENs
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self, dataset_name = None,
                 URM_dictionary = None,
                 ICM_dictionary = None,
                 ICM_feature_mapper_dictionary = None,
                 UCM_dictionary = None,
                 UCM_feature_mapper_dictionary = None,
                 user_original_ID_to_index = None,
                 item_original_ID_to_index = None,
                 is_implicit = False,
                 additional_data_mapper = None,
                 ):
        """
        :param URM_dictionary:                      Dictionary of "URM_name":URM_object
        :param ICM_dictionary:                      Dictionary of "ICM_name":ICM_object
        :param ICM_feature_mapper_dictionary:       Dictionary of "ICM_name":feature_original_id_to_index
        :param UCM_dictionary:                      Dictionary of "UCM_name":UCM_object
        :param UCM_feature_mapper_dictionary:       Dictionary of "UCM_name":feature_original_id_to_index
        :param user_original_ID_to_index:           Dictionary of "user_original_id":user_index
        :param item_original_ID_to_index:           Dictionary of "item_original_id":user_index
        """
        super(Dataset, self).__init__()

        self.DATASET_NAME = dataset_name
        self.AVAILABLE_URM = URM_dictionary

        if ICM_dictionary is not None:
            self.AVAILABLE_ICM = ICM_dictionary
            self.AVAILABLE_ICM_feature_mapper = ICM_feature_mapper_dictionary
            self._HAS_ICM = True

        if UCM_dictionary is not None:
            self.AVAILABLE_UCM = UCM_dictionary
            self.AVAILABLE_UCM_feature_mapper = UCM_feature_mapper_dictionary
            self._HAS_UCM = True

        if additional_data_mapper is not None:
            self.additional_data_mapper = additional_data_mapper
            self._HAS_additional_mapper = True

        self.item_original_ID_to_index = item_original_ID_to_index
        self.user_original_ID_to_index = user_original_ID_to_index

        self._IS_IMPLICIT = is_implicit




    def _assert_is_initialized(self):
         assert self.AVAILABLE_URM is not None, "DataReader {}: Unable to load data split. The split has not been generated yet, call the load_data function to do so.".format(self._get_dataset_name())

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_ICM_from_name(self, ICM_name):
        self._assert_is_initialized()
        return self.AVAILABLE_ICM[ICM_name].copy()

    def get_URM_from_name(self, URM_name):
        self._assert_is_initialized()
        return self.AVAILABLE_URM[URM_name].copy()

    def get_ICM_feature_to_index_mapper_from_name(self, ICM_name):
        self._assert_is_initialized()
        return self.AVAILABLE_ICM_feature_mapper[ICM_name].copy()

    def get_loaded_URM_names(self):
        return list(self.AVAILABLE_URM.keys())

    def get_item_original_ID_to_index_mapper(self):
        return self.item_original_ID_to_index.copy()

    def get_user_original_ID_to_index_mapper(self):
        return self.user_original_ID_to_index.copy()

    def get_loaded_URM_dict(self):
        return _clone_dictionary(self.AVAILABLE_URM)

    def get_loaded_ICM_dict(self):
        return _clone_dictionary(self.AVAILABLE_ICM)

    def get_loaded_ICM_feature_mapper_dict(self):
        return self.AVAILABLE_ICM_feature_mapper.copy()

    def get_loaded_UCM_dict(self):
        return _clone_dictionary(self.AVAILABLE_UCM)

    def get_loaded_UCM_feature_mapper_dict(self):
        return self.AVAILABLE_UCM_feature_mapper.copy()

    def get_URM_all(self):
        return self.get_URM_from_name("URM_all")

    def get_global_mapper_dict(self):
        return {"user_original_ID_to_index": self.user_original_ID_to_index,
                "item_original_ID_to_index": self.item_original_ID_to_index}

    def is_implicit(self):
        return self._IS_IMPLICIT


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def save_data(self, save_folder_path):

        dataIO = DataIO(folder_path = save_folder_path)

        global_attributes_dict = {
            "item_original_ID_to_index": self.item_original_ID_to_index,
            "user_original_ID_to_index": self.user_original_ID_to_index,
            "DATASET_NAME": self.DATASET_NAME,
            "_IS_IMPLICIT": self._IS_IMPLICIT,
            "_HAS_ICM": self._HAS_ICM,
            "_HAS_UCM": self._HAS_UCM,
            "_HAS_additional_mapper": self._HAS_additional_mapper
        }

        dataIO.save_data(data_dict_to_save = global_attributes_dict,
                         file_name = "dataset_global_attributes")

        dataIO.save_data(data_dict_to_save = self.AVAILABLE_URM,
                         file_name = "dataset_URM")

        if self._HAS_ICM:
            dataIO.save_data(data_dict_to_save = self.AVAILABLE_ICM,
                             file_name = "dataset_ICM")

            dataIO.save_data(data_dict_to_save = self.AVAILABLE_ICM_feature_mapper,
                             file_name = "dataset_ICM_mappers")

        if self._HAS_UCM:
            dataIO.save_data(data_dict_to_save = self.AVAILABLE_UCM,
                             file_name = "dataset_UCM")

            dataIO.save_data(data_dict_to_save = self.AVAILABLE_UCM_feature_mapper,
                             file_name = "dataset_UCM_mappers")

        if self._HAS_additional_mapper:
            dataIO.save_data(data_dict_to_save = self.additional_data_mapper,
                             file_name = "dataset_additional_mappers")



    def load_data(self, save_folder_path):

        dataIO = DataIO(folder_path = save_folder_path)

        global_attributes_dict = dataIO.load_data(file_name = "dataset_global_attributes")

        for attrib_name, attrib_object in global_attributes_dict.items():
            self.__setattr__(attrib_name, attrib_object)

        self.AVAILABLE_URM = dataIO.load_data(file_name = "dataset_URM")

        if self._HAS_ICM > 0:
            self.AVAILABLE_ICM = dataIO.load_data(file_name = "dataset_ICM")
            self.AVAILABLE_ICM_feature_mapper = dataIO.load_data(file_name = "dataset_ICM_mappers")

        if self._HAS_UCM > 0:
            self.AVAILABLE_UCM = dataIO.load_data(file_name = "dataset_UCM")
            self.AVAILABLE_UCM_feature_mapper = dataIO.load_data(file_name = "dataset_UCM_mappers")

        if self._HAS_additional_mapper:
            self.dataset_additional_mappers = dataIO.load_data(file_name = "dataset_additional_mappers")


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATASET STATISTICS                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def print_statistics(self):

        self._assert_is_initialized()

        URM_all = self.get_URM_all()

        n_users, n_items = URM_all.shape

        n_interactions = URM_all.nnz


        URM_all = sps.csr_matrix(URM_all)
        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions/n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(URM_all)
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions/n_items
        min_interactions_per_item = item_profile_length.min()


        print("DataReader: current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in URM_all: {}\n"
              "\tValue range in URM_all: {:.2f}-{:.2f}\n"
              "\tInteraction density: {:.2E}\n"
              "\tInteractions per user:\n"
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n"     
              "\tInteractions per item:\n"    
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n"
              "\tGini Index: {:.2f}\n".format(
            self.__class__,
            n_items,
            n_users,
            n_interactions,
            np.min(URM_all.data), np.max(URM_all.data),
            compute_density(URM_all),
            min_interactions_per_user,
            avg_interactions_per_user,
            max_interactions_per_user,
            min_interactions_per_item,
            avg_interactions_per_item,
            max_interactions_per_item,
            gini_index(user_profile_length),
        ))



        if self._HAS_ICM:

            for ICM_name, ICM_object in self.AVAILABLE_ICM.items():

                n_items, n_features = ICM_object.shape

                min_value = np.min(ICM_object.data)
                max_value = np.max(ICM_object.data)

                format_string = "2E" if np.max([np.abs(min_value), np.abs(max_value)])>100 else "2f"

                statistics_string = "\tICM name: {}, Value range: {:.{format_string}} / {:.{format_string}}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
                    ICM_name,
                    min_value, max_value,
                    n_features,
                    ICM_object.nnz,
                    compute_density(ICM_object),
                    format_string = format_string
                )

                print(statistics_string)

            print("\n")

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                CLONE                                                ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def copy(self):

        loaded_URM_dict = _clone_dictionary(self.AVAILABLE_URM)
        user_original_ID_to_index = self.user_original_ID_to_index.copy()
        item_original_ID_to_index = self.item_original_ID_to_index.copy()

        if self.AVAILABLE_ICM is not None:
            loaded_ICM_dict = _clone_dictionary(self.AVAILABLE_ICM)
            loaded_ICM_mapper_dict = self.AVAILABLE_ICM_feature_mapper.copy()
        else:
            loaded_ICM_dict = None
            loaded_ICM_mapper_dict = None


        if self.AVAILABLE_UCM is not None:
            loaded_UCM_dict = _clone_dictionary(self.AVAILABLE_UCM)
            loaded_UCM_mapper_dict = self.AVAILABLE_UCM_feature_mapper.copy()
        else:
            loaded_UCM_dict = None
            loaded_UCM_mapper_dict = None


        if self.additional_data_mapper is not None:
            additional_data_mapper = self.additional_data_mapper.copy()
        else:
            additional_data_mapper = None

        loaded_dataset = Dataset(dataset_name = self.get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = loaded_ICM_mapper_dict,
                                 UCM_dictionary = loaded_UCM_dict,
                                 UCM_feature_mapper_dictionary = loaded_UCM_mapper_dict,
                                 user_original_ID_to_index= user_original_ID_to_index,
                                 item_original_ID_to_index= item_original_ID_to_index,
                                 is_implicit = self.is_implicit(),
                                 additional_data_mapper = additional_data_mapper,
                                 )

        loaded_dataset.verify_data_consistency()

        return loaded_dataset



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def verify_data_consistency(self):

        self._assert_is_initialized()

        print_preamble = "{} consistency check: ".format(self.DATASET_NAME)

        URM_all = self.get_URM_all()
        n_interactions = URM_all.nnz

        assert n_interactions != 0, print_preamble + "Number of interactions in URM is 0"

        if self.is_implicit():
            assert np.all(URM_all.data == 1.0), print_preamble + "The DataReader is stated to be implicit but the main URM is not"

        assert_URM_ICM_mapper_consistency(URM_DICT = self.AVAILABLE_URM,
                                          user_original_ID_to_index = self.user_original_ID_to_index,
                                          item_original_ID_to_index = self.item_original_ID_to_index,
                                          ICM_DICT = self.AVAILABLE_ICM,
                                          ICM_MAPPER_DICT = self.AVAILABLE_ICM_feature_mapper,
                                          UCM_DICT = self.AVAILABLE_UCM,
                                          UCM_MAPPER_DICT = self.AVAILABLE_UCM_feature_mapper,
                                          DATA_SPLITTER_NAME = self.DATASET_NAME)




    def _remove_items_and_users(self, items_to_remove = None, users_to_remove = None):

        if len(items_to_remove) == 0: items_to_remove = None
        if len(users_to_remove) == 0: users_to_remove = None

        n_items = len(self.item_original_ID_to_index)
        n_users = len(self.user_original_ID_to_index)



        if items_to_remove is not None:

            items_to_keep_mask = np.ones(n_items, dtype=np.bool)
            items_to_keep_mask[items_to_remove] = False

            self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, items_to_remove)

            for URM_name, URM_obj in self.AVAILABLE_URM.items():
                self.AVAILABLE_URM[URM_name] = URM_obj[:, items_to_keep_mask]


            if self._HAS_ICM:

                items_to_keep_mask = np.ones(n_items, dtype=np.bool)
                items_to_keep_mask[items_to_remove] = False

                for ICM_name, ICM_object in self.AVAILABLE_ICM.items():

                    print("Dataset: Removing items from {}".format(ICM_name))

                    ICM_object = ICM_object[items_to_keep_mask,:]
                    ICM_mapper_object = self.AVAILABLE_ICM_feature_mapper[ICM_name]

                    ICM_object, _, ICM_mapper_object = remove_features(ICM_object,
                                                                       min_occurrence= 1,
                                                                       max_percentage_occurrence= 1.00,
                                                                       reconcile_mapper = ICM_mapper_object)

                    self.AVAILABLE_ICM[ICM_name] = ICM_object
                    self.AVAILABLE_ICM_feature_mapper[ICM_name] = ICM_mapper_object





        if users_to_remove is not None:

            users_to_keep_mask = np.ones(n_users, dtype=np.bool)
            users_to_keep_mask[users_to_remove] = False

            self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, users_to_remove)

            for URM_name, URM_obj in self.AVAILABLE_URM.items():
                self.AVAILABLE_URM[URM_name] = URM_obj[users_to_keep_mask,:]


            if self._HAS_UCM:

                users_to_keep_mask = np.ones(n_users, dtype=np.bool)
                users_to_keep_mask[users_to_remove] = False

                for UCM_name, UCM_object in self.AVAILABLE_UCM.items():

                    print("Dataset: Removing users from {}".format(UCM_name))

                    UCM_object = UCM_object[users_to_keep_mask,:]
                    UCM_mapper_object = self.AVAILABLE_UCM_feature_mapper[UCM_name]

                    UCM_object, _, UCM_mapper_object = remove_features(UCM_object,
                                                                       min_occurrence= 1,
                                                                       max_percentage_occurrence= 1.00,
                                                                       reconcile_mapper = UCM_mapper_object)

                    self.AVAILABLE_UCM[UCM_name] = UCM_object
                    self.AVAILABLE_UCM_feature_mapper[UCM_name] = UCM_mapper_object
