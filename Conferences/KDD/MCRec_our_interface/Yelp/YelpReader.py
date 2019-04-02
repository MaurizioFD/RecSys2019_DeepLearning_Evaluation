#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Anonymous authors
"""

import os

import scipy.sparse as sps

from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Data_manager.load_and_save_data import load_data_dict, save_data_dict
from Data_manager.split_functions.split_train_validation import split_data_train_validation_test_negative_user_wise


class YelpReader(object):


    def __init__(self):

        super(YelpReader, self).__init__()

        pre_splitted_path = "Data_manager_split_datasets/Yelp/KDD/MCRec_our_interface/"

        pre_splitted_filename = "splitted_data"

        original_data_path = "Conferences/KDD/MCRec_github/Dataset-In-Papers_master/Yelp/"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        try:

            print("YelpReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in load_data_dict(pre_splitted_path, pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("YelpReader: Pre-splitted data not found, building new one")

            print("YelpReader: loading URM")

            URM_all_builder = self._loadURM(original_data_path + "user_business.dat", separator="\t")

            URM_all = URM_all_builder.get_SparseMatrix()


            self.URM_train, self.URM_validation, self.URM_test, self.URM_test_negative = split_data_train_validation_test_negative_user_wise(URM_all, negative_items_per_positive = 50)


            item_id_to_index_mappper = URM_all_builder.get_column_token_to_id_mapper()

            ICM_category = self._loadICM (original_data_path + "business_category.dat", item_id_to_index_mappper,
                                     header = False, separator="\t")

            ICM_category = ICM_category.get_SparseMatrix()


            ICM_city = self._loadICM (original_data_path + "business_city.dat", item_id_to_index_mappper,
                                     header = False, separator="\t")

            ICM_city = ICM_city.get_SparseMatrix()



            ICM_all = sps.hstack([ICM_category, ICM_city])


            self.ICM_dict = {"ICM_category": ICM_category,
                            "ICM_city": ICM_city,
                             "ICM_all": ICM_all}


            data_dict = {
                "URM_train": self.URM_train,
                "URM_test": self.URM_test,
                "URM_validation": self.URM_validation,
                "URM_test_negative": self.URM_test_negative,
                "ICM_dict": self.ICM_dict,
            }

            save_data_dict(data_dict, pre_splitted_path, pre_splitted_filename)

            print("YelpReader: loading complete")





    def _loadURM (self, filePath, header = False, separator="::"):

        URM_all_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                user_id = line[0]
                item_id = line[1]

                URM_all_builder.add_data_lists([user_id], [item_id], [1.0])


        fileHandle.close()

        return  URM_all_builder




    def _loadICM (self, filePath, item_id_to_index_mappper, header = False, separator="::"):

        ICM_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=False)

        fileHandle = open(filePath, "r")
        numCells = 0

        if header:
            fileHandle.readline()

        for line in fileHandle:
            numCells += 1
            if (numCells % 1000000 == 0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                item_id = line[0]
                feature_id = line[1]

                item_index = item_id_to_index_mappper[item_id]

                ICM_builder.add_data_lists([item_index], [feature_id], [1.0])


        fileHandle.close()

        return  ICM_builder
