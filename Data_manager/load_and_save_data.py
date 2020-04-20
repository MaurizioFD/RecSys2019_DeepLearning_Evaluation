#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/02/19

@author: Maurizio Ferrari Dacrema
"""

from Base.DataIO import DataIO



def save_data_dict_zip(URM_DICT, ICM_DICT, splitted_data_path, file_name_prefix):

    dataIO = DataIO(folder_path = splitted_data_path)

    URM_DICT["__ICM_available"] = len(ICM_DICT)>0

    dataIO.save_data(data_dict_to_save = URM_DICT, file_name=file_name_prefix + "URM_dict")

    del URM_DICT["__ICM_available"]

    if len(ICM_DICT)>0:
        dataIO.save_data(data_dict_to_save = ICM_DICT, file_name=file_name_prefix + "ICM_dict")






def load_data_dict_zip(splitted_data_path, file_name_prefix):

    URM_DICT = {}
    ICM_DICT = {}


    dataIO = DataIO(folder_path = splitted_data_path)

    URM_DICT = dataIO.load_data(file_name=file_name_prefix + "URM_dict")

    if URM_DICT["__ICM_available"]:
        ICM_DICT = dataIO.load_data(file_name=file_name_prefix + "ICM_dict")


    del URM_DICT["__ICM_available"]


    loaded_data_dict = {
        "URM_DICT": URM_DICT,
        "ICM_DICT": ICM_DICT,
    }


    return loaded_data_dict
