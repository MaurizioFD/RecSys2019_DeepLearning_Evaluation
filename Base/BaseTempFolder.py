#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2019

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import get_unique_temp_folder
import os, shutil


class BaseTempFolder(object):

    def __init__(self):
        super(BaseTempFolder, self).__init__()

        self.DEFAULT_TEMP_FILE_FOLDER = './result_experiments/__Temp_{}/'.format(self.RECOMMENDER_NAME)


    def _get_unique_temp_folder(self, input_temp_file_folder = None):

        if input_temp_file_folder is None:
            print("{}: Using default Temp folder '{}'".format(self.RECOMMENDER_NAME, self.DEFAULT_TEMP_FILE_FOLDER))
            self._use_default_temp_folder = True
            output_temp_file_folder = get_unique_temp_folder(self.DEFAULT_TEMP_FILE_FOLDER)
        else:
            print("{}: Using Temp folder '{}'".format(self.RECOMMENDER_NAME, input_temp_file_folder))
            self._use_default_temp_folder = False
            output_temp_file_folder = get_unique_temp_folder(input_temp_file_folder)

        if not os.path.isdir(output_temp_file_folder):
            os.makedirs(output_temp_file_folder)


        return output_temp_file_folder



    def _clean_temp_folder(self, temp_file_folder):
        """
        Clean temporary folder only if the default one
        :return:
        """

        if  self._use_default_temp_folder:
            print("{}: Cleaning temporary files from '{}'".format(self.RECOMMENDER_NAME, temp_file_folder))
            shutil.rmtree(temp_file_folder, ignore_errors=True)

        else:
            print("{}: Maintaining temporary files due to a custom temp folder being selected".format(self.RECOMMENDER_NAME))
