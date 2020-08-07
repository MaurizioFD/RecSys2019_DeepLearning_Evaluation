#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Maurizio Ferrari Dacrema
"""

import traceback, os
from Data_manager.DataReader import DataReader

class DataSplitter(object):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    """
     - It exposes the following functions
        - load_data(save_folder_path = None, force_new_split = False)   loads the data or creates a new split
    
    """

    __DATASET_SPLIT_SUBFOLDER = "Data_manager_split_datasets/"
    DATASET_SPLIT_ROOT_FOLDER = None


    ICM_SPLIT_SUFFIX = [""]

    DATA_SPLITTER_NAME = "DataSplitter"


    def __init__(self, dataReader_object:DataReader, forbid_new_split = False, force_new_split = False):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        """
        super(DataSplitter, self).__init__()

        self.DATASET_SPLIT_ROOT_FOLDER = os.path.join(os.path.dirname(__file__), '..', self.__DATASET_SPLIT_SUBFOLDER)

        self.dataReader_object = dataReader_object
        self.forbid_new_split = forbid_new_split
        self.force_new_split = force_new_split


    def get_dataReader_object(self):
        return self.dataReader_object

    # Allow to use ICM functions on the DataSplitter
    def _get_dataset_name(self):
        return self.get_dataReader_object()._get_dataset_name()

    def get_ICM_from_name(self, ICM_name):
        return self.SPLIT_ICM_DICT[ICM_name].copy()

    def get_loaded_ICM_names(self):
        return self.get_dataReader_object().get_loaded_ICM_names()

    def get_all_available_ICM_names(self):
        return self.get_dataReader_object().get_loaded_ICM_names().copy()

    def get_UCM_from_name(self, UCM_name):
        return self.SPLIT_UCM_DICT[UCM_name].copy()

    def get_loaded_UCM_names(self):
        return self.get_dataReader_object().get_loaded_UCM_names()

    def get_all_available_UCM_names(self):
        return self.get_dataReader_object().get_loaded_ICM_names().copy()

    def get_loaded_ICM_dict(self):
        # return self.get_dataset_object().get_loaded_ICM_dict()

        ICM_dict = {}

        for ICM_name in self.get_loaded_ICM_names():

            ICM_dict[ICM_name] = self.get_ICM_from_name(ICM_name)

        return ICM_dict

    def get_loaded_UCM_dict(self):
        # return self.get_dataset_object().get_loaded_UCM_dict()

        UCM_dict = {}

        for UCM_name in self.get_loaded_UCM_names():

            UCM_dict[UCM_name] = self.get_UCM_from_name(UCM_name)

        return UCM_dict

    def _print(self, message):
        print("{}: {}".format(self.DATA_SPLITTER_NAME, message))


    def _get_default_save_path(self):
        """
        Returns the default path in which to save the splitted data
        # Use default "dataset_name/split_name/original" or "dataset_name/split_name/k-cores"
        :return:
        """

        save_folder_path = self.DATASET_SPLIT_ROOT_FOLDER + \
                           self.get_dataReader_object()._get_dataset_name_root() + \
                           self._get_split_subfolder_name() + \
                           self.get_dataReader_object()._get_dataset_name_data_subfolder()

        return save_folder_path



    def load_data(self, save_folder_path = None):
        """

        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        :return:
        """

        # Use default "dataset_name/split_name/original" or "dataset_name/split_name/k-cores"
        if save_folder_path is None:
            save_folder_path = self._get_default_save_path()


        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not False and not self.force_new_split:

            try:

                self._load_previously_built_split_and_attributes(save_folder_path)

                self._print("Verifying data consistency...")
                self._verify_data_consistency()
                self._print("Verifying data consistency... Passed!")

            except FileNotFoundError:

                # Split not found, either stop or create a new one
                if self.forbid_new_split:
                    raise ValueError("{}: Preloaded data not found, but creating a new split is forbidden. Terminating".format(self.DATA_SPLITTER_NAME))

                else:
                    self._print("Preloaded data not found, reading from original files...")

                    # If directory does not exist, create
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)

                    self._split_data_from_original_dataset(save_folder_path)
                    self._load_previously_built_split_and_attributes(save_folder_path)

                    self._print("Verifying data consistency...")
                    self._verify_data_consistency()
                    self._print("Verifying data consistency... Passed!")

                    self._print("Preloaded data not found, reading from original files... Done")


            except Exception:

                self._print("Reading split from {} caused the following exception...".format(save_folder_path))
                traceback.print_exc()
                raise Exception("{}: Exception while reading split".format(self.DATA_SPLITTER_NAME))

        else:

            self._print("Reading from original files...")

            self._split_data_from_original_dataset(save_folder_path)

            self._print("Reading from original files...Done")




        self.get_statistics_URM()
        self.get_statistics_ICM()

        self._print("Done.")




    def _load_from_DataReader_ICM_and_mappers(self, loaded_dataset):

        self.SPLIT_ICM_DICT = loaded_dataset.get_loaded_ICM_dict()
        self.SPLIT_ICM_MAPPER_DICT = loaded_dataset.get_loaded_ICM_feature_mapper_dict()

        self.SPLIT_UCM_DICT = loaded_dataset.get_loaded_UCM_dict()
        self.SPLIT_UCM_MAPPER_DICT = loaded_dataset.get_loaded_UCM_feature_mapper_dict()

        self.SPLIT_GLOBAL_MAPPER_DICT = loaded_dataset.get_global_mapper_dict()





    def _get_split_subfolder_name(self):
        """
        :return: Dataset_name/split_name/
        """
        raise NotImplementedError("{}: _get_split_subfolder_name was not implemented for the required dataset. Impossible to load the data".format(self.DATA_SPLITTER_NAME))




    def _split_data_from_original_dataset(self, save_folder_path):
        raise NotImplementedError("{}: _split_data_from_original_dataset was not implemented for the required dataset. Impossible to load the data".format(self.DATA_SPLITTER_NAME))


    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """
        raise NotImplementedError("{}: _load_previously_built_split_and_attributes was not implemented for the required dataset. Impossible to load the data".format(self.DATA_SPLITTER_NAME))


    def get_statistics_URM(self):

        raise NotImplementedError("{}: get_statistics_URM was not implemented for the required dataset. Impossible to load the data".format(self.DATA_SPLITTER_NAME))



    def get_statistics_ICM(self):

        raise NotImplementedError("{}: get_statistics_ICM was not implemented for the required dataset. Impossible to load the data".format(self.DATA_SPLITTER_NAME))


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                DATA CONSISTENCY                                     ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _verify_data_consistency(self):

        self._print("WARNING WARNING WARNING _verify_data_consistency not implemented for the current DataSplitter, unable to validate current split.")
