#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2018

@author: Maurizio Ferrari Dacrema
"""


import scipy.sparse as sps
import numpy as np
import pickle, os




#################################################################################################################
#############################
#############################               DATA READER
#############################
#################################################################################################################



class DataReader(object):
    """
    Abstract class for the DataReaders, each shoud be implemented for a specific dataset
    DataReader has the following functions:
     - It loads the data of the original dataset and saves it into sparse matrices
     - It exposes the following functions
        - load_data(save_folder_path = None)        loads the data and saves into the specified folder, if None uses default, if False des not save
        - get_URM_all()                             returns a copy of the whole URM
        - get_ICM_from_name(ICM_name)               returns a copy of the specified ICM
        - get_loaded_ICM_names()                    returns a copy of the loaded ICM names, which can be used in get_ICM_from_name
        - get_loaded_ICM_dict()                     returns a copy of the loaded ICM in a dictionary [ICM_name]->ICM_sparse
        - DATASET_SUBFOLDER_DEFAULT                 path of the data folder
        - item_original_ID_to_index
        - user_original_ID_to_index

    """
    DATASET_SPLIT_ROOT_FOLDER = "Data_manager_split_datasets/"
    DATASET_OFFLINE_ROOT_FOLDER = "Data_manager_offline_datasets/"


    # This subfolder contains the preprocessed data, already loaded from the original data file
    DATASET_SUBFOLDER_ORIGINAL = "original/"

    # Available URM split
    AVAILABLE_URM = ["URM_all"]

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = []

    # Mappers existing for all datasets, associating USER_ID and ITEM_ID to the new designation
    GLOBAL_MAPPER = ["item_original_ID_to_index", "user_original_ID_to_index"]

    # Mappers specific for a given dataset, they might be related to more complex data structures or FEATURE_TOKENs
    DATASET_SPECIFIC_MAPPER = []

    # This flag specifies if the given dataset contains implicit preferences or explicit ratings
    IS_IMPLICIT = False


    def __init__(self, reload_from_original_data = False, ICM_to_load_list = None):

        super(DataReader, self).__init__()

        self.reload_from_original_data = reload_from_original_data
        if self.reload_from_original_data:
            print("DataReader: reload_from_original_data is True, previously loaded data will be ignored")

        self.item_original_ID_to_index = {}
        self.user_original_ID_to_index = {}

        if ICM_to_load_list is None:
            self.ICM_to_load_list = self.AVAILABLE_ICM.copy()
        else:
            assert all([ICM_to_load in self.AVAILABLE_ICM for ICM_to_load in ICM_to_load_list]), \
                "DataReader: ICM_to_load_list contains ICM names which are not available for the current DataReader"
            self.ICM_to_load_list = ICM_to_load_list.copy()


    def is_implicit(self):
        return self.IS_IMPLICIT


    def _get_dataset_name(self):
        return self._get_dataset_name_root()[:-1]

    def get_ICM_from_name(self, ICM_name):
        return getattr(self, ICM_name).copy()

    def get_URM_from_name(self, URM_name):
        return getattr(self, URM_name).copy()


    def get_ICM_feature_to_index_mapper_from_name(self, ICM_name):
        return getattr(self, "tokenToFeatureMapper_" + ICM_name).copy()

    def get_loaded_ICM_names(self):
        return self.ICM_to_load_list.copy()

    def get_all_available_ICM_names(self):
        return self.AVAILABLE_ICM.copy()

    def get_loaded_URM_names(self):
        return self.AVAILABLE_URM.copy()

    def get_loaded_ICM_dict(self):

        ICM_dict = {}

        for ICM_name in self.get_loaded_ICM_names():

            ICM_dict[ICM_name] = self.get_ICM_from_name(ICM_name)


        return ICM_dict


    def get_URM_all(self):
        return self.URM_all.copy()


    def print_statistics(self):

        n_users, n_items = self.URM_all.shape

        n_interactions = self.URM_all.nnz


        URM_all = sps.csr_matrix(self.URM_all)
        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions/n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(self.URM_all)
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions/n_items
        min_interactions_per_item = item_profile_length.min()


        print("DataReader: current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in URM_all: {}\n"
              "\tInteraction density: {:.2E}\n"
              "\tInteractions per user:\n"
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n"     
              "\tInteractions per item:\n"    
              "\t\t Min: {:.2E}\n"
              "\t\t Avg: {:.2E}\n"    
              "\t\t Max: {:.2E}\n".format(
            self.__class__,
            n_items,
            n_users,
            n_interactions,
            n_interactions/(n_items*n_users),
            min_interactions_per_user,
            avg_interactions_per_user,
            max_interactions_per_user,
            min_interactions_per_item,
            avg_interactions_per_item,
            max_interactions_per_item
        ))



    def _get_dataset_name_root(self):
        """
        Returns the root of the folder tree which contains all of the dataset data/splits and files

        :return: Dataset_name/
        """
        raise NotImplementedError("DataReader: The following method was not implemented for the required dataset. Impossible to load the data")



    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """
        return self.DATASET_SUBFOLDER_ORIGINAL


    def load_data(self, save_folder_path = None):
        """

        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        # Use default e.g., "dataset_name/original/"
        if save_folder_path is None:
            save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root() + self._get_dataset_name_data_subfolder()


        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.reload_from_original_data:

            try:

                self._load_from_saved_sparse_matrix(save_folder_path)

                print("DataReader: verifying data consistency...")
                self._verify_data_consistency()
                print("DataReader: verifying data consistency... Passed!")

                self.print_statistics()
                return

            except:

                print("DataReader: Preloaded data not found, reading from original files...")
                pass


        self._load_from_original_file()

        print("DataReader: verifying data consistency...")
        self._verify_data_consistency()
        print("DataReader: verifying data consistency... Passed!")

        if save_folder_path not in [False]:

            # If directory does not exist, create
            if not os.path.exists(save_folder_path):
                print("DataReader: Creating folder '{}'".format(save_folder_path))
                os.makedirs(save_folder_path)

            else:
                print("DataReader: Found already existing folder '{}'".format(save_folder_path))

            for URM_name in self.get_loaded_URM_names():
                print("DataReader: Saving {}...".format(URM_name))
                sps.save_npz(save_folder_path + "{}.npz".format(URM_name), self.get_URM_from_name(URM_name))

            for ICM_name in self.get_loaded_ICM_names():
                print("DataReader: Saving {}...".format(ICM_name))
                sps.save_npz(save_folder_path + "{}.npz".format(ICM_name), self.get_ICM_from_name(ICM_name))


        self._save_mappers(save_folder_path)

        print("DataReader: Saving complete!")

        self.print_statistics()



    def _verify_data_consistency(self):

        URM_all = self.get_URM_all()
        n_users, n_items_URM = URM_all.shape
        n_interactions = URM_all.nnz

        assert n_users != 0, "DataReader consistency check: Number of users in URM is 0"
        assert n_items_URM != 0, "DataReader consistency check: Number of items in URM is 0"
        assert n_interactions != 0, "DataReader consistency check: Number of interactions in URM is 0"

        # Check if item index-id and user index-id are consistent
        assert n_users >= len(self.user_original_ID_to_index), "DataReader consistency check: user it-to-index mapper contains more keys than users"
        assert n_items_URM >= len(self.item_original_ID_to_index), "DataReader consistency check: item it-to-index mapper contains more keys than items"

        assert n_users >= max(self.user_original_ID_to_index.values()), "DataReader consistency check: user it-to-index mapper contains indices greater than number of users"
        assert n_items_URM >= max(self.item_original_ID_to_index.values()), "DataReader consistency check: item it-to-index mapper contains indices greater than number of item"


        # Check if every non-empty user and item has a mapper value
        URM_all = sps.csc_matrix(URM_all)
        nonzero_items_mask = np.ediff1d(URM_all.indptr)>0
        nonzero_items = np.arange(0, n_items_URM, dtype=np.int)[nonzero_items_mask]
        assert np.isin(nonzero_items, np.array(list(self.item_original_ID_to_index.values()))).all(), "DataReader consistency check: there exist items with interactions that do not have a mapper entry"


        URM_all = sps.csr_matrix(URM_all)
        nonzero_users_mask = np.ediff1d(URM_all.indptr)>0
        nonzero_users = np.arange(0, n_users, dtype=np.int)[nonzero_users_mask]
        assert np.isin(nonzero_users, np.array(list(self.user_original_ID_to_index.values()))).all(), "DataReader consistency check: there exist users with interactions that do not have a mapper entry"


        for ICM_name in self.get_loaded_ICM_names():

            ICM_object = self.get_ICM_from_name(ICM_name)
            feature_original_id_to_index = self.get_ICM_feature_to_index_mapper_from_name(ICM_name)

            n_items_ICM, n_features = ICM_object.shape
            n_feature_occurrences = ICM_object.nnz

            assert n_items_ICM == n_items_URM, "DataReader consistency check: Number of users in ICM {} is {} while in URM is {}".format(ICM_name, n_items_ICM, n_items_URM)
            assert n_features != 0, "DataReader consistency check: Number of features in ICM {} is 0".format(ICM_name)
            assert n_feature_occurrences != 0, "DataReader consistency check: Number of interactions in ICM {} is 0".format(ICM_name)


            assert n_features >= len(feature_original_id_to_index), "DataReader consistency check: feature id-to-index mapper contains more keys than features in ICM {}".format(ICM_name)
            assert n_features >= max(feature_original_id_to_index.values()), "DataReader consistency check: feature id-to-index mapper contains indices greater than number of features in ICM {}".format(ICM_name)

            # Check if every non-empty item and feature has a mapper value
            ICM_object = sps.csr_matrix(ICM_object)
            nonzero_items_mask = np.ediff1d(ICM_object.indptr)>0
            nonzero_items = np.arange(0, n_items_URM, dtype=np.int)[nonzero_items_mask]
            assert np.isin(nonzero_items, np.array(list(self.item_original_ID_to_index.values()))).all(), "DataReader consistency check: there exist items with features that do not have a mapper entry in ICM {}".format(ICM_name)


            ICM_object = sps.csc_matrix(ICM_object)
            nonzero_features_mask = np.ediff1d(ICM_object.indptr)>0
            nonzero_features = np.arange(0, n_features, dtype=np.int)[nonzero_features_mask]
            assert np.isin(nonzero_features, np.array(list(feature_original_id_to_index.values()))).all(), "DataReader consistency check: there exist users with interactions that do not have a mapper entry in ICM {}".format(ICM_name)




    def _load_from_original_file(self):

        raise NotImplementedError("DataReader: The following method was not implemented for the required dataset. Impossible to load the data")






    def _load_from_saved_sparse_matrix(self, save_folder_path):

        file_names_to_load = self.get_loaded_ICM_names()

        file_names_to_load.extend(self.get_loaded_URM_names())

        for file_name in file_names_to_load:

            print("DataReader: Loading {}...".format(save_folder_path + file_name))
            self.__setattr__(file_name, sps.load_npz("{}.npz".format(save_folder_path + file_name)))

        self._load_mappers(save_folder_path)

        print("DataReader: Loading complete!")





    def _save_mappers(self, save_folder_path):
        """
        Saves the mappers for the given dataset. Mappers associate the original ID of user, item, feature, to the
        index in the sparse matrix
        :param dataset_specific_mappers_list:
        :return:
        """

        mappers_list = list(self.GLOBAL_MAPPER)
        mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for ICM_name in self.get_loaded_ICM_names():
            mappers_list.append("tokenToFeatureMapper_{}".format(ICM_name))


        for mapper_name in mappers_list:
            mapper_data = self.__getattribute__(mapper_name)
            pickle.dump(mapper_data, open(save_folder_path + mapper_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)




    def _load_mappers(self, save_folder_path):
        """
        Loads all saved mappers for the given dataset. Mappers are the union of GLOBAL mappers and dataset specific ones
        :return:
        """

        mappers_list = list(self.GLOBAL_MAPPER)
        mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for ICM_name in self.get_loaded_ICM_names():
            mappers_list.append("tokenToFeatureMapper_{}".format(ICM_name))

        for mapper_name in mappers_list:
            self.__setattr__(mapper_name, pickle.load(open(save_folder_path + mapper_name, "rb")))




    def _merge_ICM(self, ICM1, ICM2, mapper_ICM1, mapper_ICM2):

        ICM_all = sps.hstack([ICM1, ICM2], format='csr')

        mapper_ICM_all = mapper_ICM1.copy()

        for key in mapper_ICM2.keys():
            mapper_ICM_all[key] = mapper_ICM2[key] + len(mapper_ICM1)

        return  ICM_all, mapper_ICM_all


