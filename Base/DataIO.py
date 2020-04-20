#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Maurizio Ferrari Dacrema
"""

import os, json, zipfile, shutil, platform

import scipy.sparse as sps
from pandas import DataFrame
import pandas as pd
import numpy as np


def json_not_serializable_handler(o):
    """
    Json cannot serialize automatically some data types, for example numpy integers (int32).
    This may be a limitation of numpy-json interfaces for Python 3.6 and may not occur in Python 3.7
    :param o:
    :return:
    """

    if isinstance(o, np.integer):
        return int(o)

    raise TypeError("json_not_serializable_handler: object '{}' is not serializable.".format(type(o)))



class DataIO(object):
    """ DataIO"""

    _DEFAULT_TEMP_FOLDER = ".temp_DataIO_"

    # _MAX_PATH_LENGTH_LINUX = 4096
    _MAX_PATH_LENGTH_WINDOWS = 255

    def __init__(self, folder_path):
        super(DataIO, self).__init__()

        self._is_windows = platform.system() == "Windows"

        self.folder_path = folder_path
        self._key_string_alert_done = False

        # if self._is_windows:
        #     self.folder_path = "\\\\?\\" + self.folder_path


    def _print(self, message):
        print("{}: {}".format("DataIO", message))


    def _get_temp_folder(self, file_name):
        """
        Creates a temporary folder to be used during the data saving
        :return:
        """

        # Ignore the .zip extension
        file_name = file_name[:-4]

        current_temp_folder = "{}{}_{}/".format(self.folder_path, self._DEFAULT_TEMP_FOLDER, file_name)

        if os.path.exists(current_temp_folder):
            self._print("Folder {} already exists, could be the result of a previous failed save attempt or multiple saver are active in parallel. " \
            "Folder will be removed.".format(current_temp_folder))

            shutil.rmtree(current_temp_folder, ignore_errors=True)

        os.makedirs(current_temp_folder)

        return current_temp_folder


    def _check_dict_key_type(self, dict_to_save):
        """
        Check whether the keys of the dictionary are string. If not, transforms them into strings
        :param dict_to_save:
        :return:
        """

        all_keys_are_str = all(isinstance(key, str) for key in dict_to_save.keys())

        if all_keys_are_str:
            return dict_to_save

        if not self._key_string_alert_done:
            self._print("Json dumps supports only 'str' as dictionary keys. Transforming keys to string, note that this will alter the mapper content.")
            self._key_string_alert_done = True

        dict_to_save_key_str = {str(key):val for (key,val) in dict_to_save.items()}

        assert all(dict_to_save_key_str[str(key)] == val for (key,val) in dict_to_save.items()), \
            "DataIO: Transforming dictionary keys into strings altered its content. Duplicate keys may have been produced."

        return dict_to_save_key_str


    def save_data(self, file_name, data_dict_to_save):

        # If directory does not exist, create with .temp_model_folder
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        if file_name[-4:] != ".zip":
            file_name += ".zip"


        current_temp_folder = self._get_temp_folder(file_name)

        attribute_to_file_name = {}
        attribute_to_json_file = {}

        for attrib_name, attrib_data in data_dict_to_save.items():

            current_file_path = current_temp_folder + attrib_name

            if isinstance(attrib_data, DataFrame):
                attrib_data.to_csv(current_file_path, index=False)
                attribute_to_file_name[attrib_name] = attrib_name + ".csv"

            elif isinstance(attrib_data, sps.spmatrix):
                sps.save_npz(current_file_path, attrib_data)
                attribute_to_file_name[attrib_name] = attrib_name + ".npz"

            elif isinstance(attrib_data, np.ndarray):
                # allow_pickle is FALSE to prevent using pickle and ensure portability
                np.save(current_file_path, attrib_data, allow_pickle=False)
                attribute_to_file_name[attrib_name] = attrib_name + ".npy"

            else:
                attribute_to_json_file[attrib_name] = attrib_data
                attribute_to_file_name[attrib_name] = attrib_name + ".json"


        # Save list objects
        attribute_to_json_file[".DataIO_attribute_to_file_name"] = attribute_to_file_name.copy()

        for attrib_name, attrib_data in attribute_to_json_file.items():

            current_file_path = current_temp_folder + attrib_name
            attribute_to_file_name[attrib_name] = attrib_name + ".json"

            # if self._is_windows and len(current_file_path + ".json") >= self._MAX_PATH_LENGTH_WINDOWS:
            #     current_file_path = "\\\\?\\" + current_file_path

            absolute_path = current_file_path + ".json" if current_file_path.startswith(os.getcwd()) else os.getcwd() + current_file_path + ".json"

            assert not self._is_windows or (self._is_windows and len(absolute_path) <= self._MAX_PATH_LENGTH_WINDOWS), \
                "DataIO: Path of file exceeds {} characters, which is the maximum allowed under standard paths for Windows.".format(self._MAX_PATH_LENGTH_WINDOWS)


            with open(current_file_path + ".json", 'w') as outfile:

                if isinstance(attrib_data, dict):
                    attrib_data = self._check_dict_key_type(attrib_data)

                json.dump(attrib_data, outfile, default=json_not_serializable_handler)



        with zipfile.ZipFile(self.folder_path + file_name, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:

            for file_name in attribute_to_file_name.values():
                myzip.write(current_temp_folder + file_name, arcname = file_name)



        shutil.rmtree(current_temp_folder, ignore_errors=True)


    def load_data(self, file_name):

        if file_name[-4:] != ".zip":
            file_name += ".zip"

        dataFile = zipfile.ZipFile(self.folder_path + file_name)

        dataFile.testzip()

        current_temp_folder = self._get_temp_folder(file_name)

        try:

            try:
                attribute_to_file_name_path = dataFile.extract(".DataIO_attribute_to_file_name.json", path = current_temp_folder)
            except KeyError:
                attribute_to_file_name_path = dataFile.extract("__DataIO_attribute_to_file_name.json", path = current_temp_folder)


            with open(attribute_to_file_name_path, "r") as json_file:
                attribute_to_file_name = json.load(json_file)

            data_dict_loaded = {}

            for attrib_name, file_name in attribute_to_file_name.items():

                attrib_file_path = dataFile.extract(file_name, path = current_temp_folder)
                attrib_data_type = file_name.split(".")[-1]

                if attrib_data_type == "csv":
                    attrib_data = pd.read_csv(attrib_file_path, index_col=False)

                elif attrib_data_type == "npz":
                    attrib_data = sps.load_npz(attrib_file_path)

                elif attrib_data_type == "npy":
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    attrib_data = np.load(attrib_file_path, allow_pickle=False)

                elif attrib_data_type == "json":
                    with open(attrib_file_path, "r") as json_file:
                        attrib_data = json.load(json_file)

                else:
                    raise Exception("Attribute type not recognized for: '{}' of class: '{}'".format(attrib_file_path, attrib_data_type))

                data_dict_loaded[attrib_name] = attrib_data


        except Exception as exec:

            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec

        shutil.rmtree(current_temp_folder, ignore_errors=True)


        return data_dict_loaded
