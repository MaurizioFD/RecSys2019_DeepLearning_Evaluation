#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Maurizio Ferrari Dacrema
"""


import zipfile, os


from Data_manager.DataReader import DataReader




class NetflixPrizeReader(DataReader):

    DATASET_URL = "https://www.kaggle.com/netflix-inc/netflix-prize-data"
    DATASET_SUBFOLDER = "NetflixPrize/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self.zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        self.decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            self.dataFile = zipfile.ZipFile(self.zip_file_folder + "netflix-prize-data.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("NetflixPrizeReader: Unable to find data zip file.")
            print("NetflixPrizeReader: Automatic download not available, please ensure the ZIP data file is in folder {}.".format(self.zip_file_folder))
            print("NetflixPrizeReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(self.zip_file_folder):
                os.makedirs(self.zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")



        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = self._loadURM()


        print("NetflixPrizeReader: loading complete")





    def _loadURM(self):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

        numCells = 0
        URM_builder = IncrementalSparseMatrix(auto_create_col_mapper=True, auto_create_row_mapper=True)


        for current_split in [1, 2, 3, 4]:

            current_split_path = self.dataFile.extract("combined_data_{}.txt".format(current_split), path=self.decompressed_zip_file_folder + "decompressed/")

            fileHandle = open(current_split_path, "r")

            print("NetflixPrizeReader: loading split {}".format(current_split))

            currentMovie_id = None

            for line in fileHandle:


                if numCells % 1000000 == 0 and numCells!=0:
                    print("Processed {} cells".format(numCells))

                if (len(line)) > 1:

                    line_split = line.split(",")

                    # If line has 3 components, it is a 'user_id,rating,date' row
                    if len(line_split) == 3 and currentMovie_id!= None:

                        user_id = line_split[0]

                        URM_builder.add_data_lists([user_id], [currentMovie_id], [float(line_split[1])])

                        numCells += 1

                    # If line has 1 component, it MIGHT be a 'item_id:' row
                    elif len(line_split) == 1:
                        line_split = line.split(":")

                        # Confirm it is a 'item_id:' row
                        if len(line_split) == 2:
                            currentMovie_id = line_split[0]

                        else:
                            print("Unexpected row: '{}'".format(line))

                    else:
                        print("Unexpected row: '{}'".format(line))


            fileHandle.close()


            print("NetflixPrizeReader: cleaning temporary files")

            import shutil

            shutil.rmtree(self.decompressed_zip_file_folder + "decompressed/", ignore_errors=True)


        return  URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()

