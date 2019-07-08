#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/02/19

@author: Maurizio Ferrari Dacrema
"""

import pickle
import scipy.sparse as sps
from pandas import DataFrame
import pandas as pd



def save_data_dict(data_dict, splitted_data_path, file_name_prefix):

    pickle_attribute_name_to_path_dict = {}
    pickle_attribute_name_to_type_dict = {}

    if "ICM_dict" in data_dict:

        ICM_dict = data_dict["ICM_dict"]
        print("ICM_dict: " + str(list(ICM_dict.keys())))

        for ICM_name, ICM_object in ICM_dict.items():
            data_dict["ICM_dict" + "_" + ICM_name] = ICM_object

        del data_dict["ICM_dict"]


    for attrib_name in data_dict:

        attrib_object = data_dict[attrib_name]

        attrib_file_name = file_name_prefix + "_" + attrib_name
        pickle_attribute_name_to_path_dict[attrib_name] = attrib_file_name

        npz_file_name = splitted_data_path + attrib_file_name

        if isinstance(attrib_object, DataFrame):
            attrib_object.to_csv(npz_file_name, index=False)
            pickle_attribute_name_to_type_dict[attrib_name] = DataFrame

        elif isinstance(attrib_object, sps.spmatrix):
            sps.save_npz(npz_file_name, attrib_object)
            pickle_attribute_name_to_type_dict[attrib_name] = sps.spmatrix

        else:
            raise Exception("Attribute type not recognized for: '{}' of class: '{}'".format(attrib_name, attrib_object.__class__))


        print("Saved: " + npz_file_name)


    pickle.dump(pickle_attribute_name_to_path_dict.copy(),
                open(splitted_data_path + file_name_prefix + "_file_list", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump(pickle_attribute_name_to_type_dict.copy(),
                open(splitted_data_path + file_name_prefix + "_file_type", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)







def load_data_dict(splitted_data_path, file_name_prefix):

    file_list = pickle.load(open(splitted_data_path + file_name_prefix + "_file_list", "rb"))

    pickle_attribute_name_to_path_dict = {}

    for attrib_name in file_list.keys():
         pickle_attribute_name_to_path_dict[attrib_name] = file_list[attrib_name]


    pickle_attribute_name_to_type_dict = {}

    try:
        type_list = pickle.load(open(splitted_data_path + file_name_prefix + "_file_type", "rb"))

        for attrib_name in type_list.keys():
             pickle_attribute_name_to_type_dict[attrib_name] = type_list[attrib_name]

    except FileNotFoundError:

         for attrib_name in file_list.keys():
             pickle_attribute_name_to_type_dict[attrib_name] = sps.spmatrix


    loaded_data_dict = {}
    ICM_dict = {}

    for object_name, object_path in pickle_attribute_name_to_path_dict.items():

        object_type = pickle_attribute_name_to_type_dict[object_name]

        if object_type is DataFrame:
            object_data = pd.read_csv(splitted_data_path + object_path, index_col=False)

        elif object_type is sps.spmatrix:
            object_data = sps.load_npz(splitted_data_path + object_path + ".npz")

        else:
            raise Exception("Attribute type not recognized for: '{}' of class: '{}'".format(object_name, object_type))


        if "ICM_dict" in object_name:
            object_name = object_name.replace("ICM_dict_", "")
            ICM_dict[object_name] = object_data

        else:
            loaded_data_dict[object_name] = object_data


        print("Loaded: " + object_name)


    if len(ICM_dict)>0:
        loaded_data_dict["ICM_dict"] = ICM_dict

    return loaded_data_dict
