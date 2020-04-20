#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/19

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd



def _loadURM_preinitialized_item_id (filePath, header = False, separator="::",
                                     if_new_user = "add", if_new_item = "ignore",
                                     item_original_ID_to_index = None,
                                     user_original_ID_to_index = None):


    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index,
                                                    on_new_col = if_new_item,
                                                    preinitialized_row_mapper = user_original_ID_to_index,
                                                    on_new_row = if_new_user)

    URM_timestamp_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index,
                                                on_new_col = if_new_item,
                                                preinitialized_row_mapper = user_original_ID_to_index,
                                                on_new_row = if_new_user)

    if header:
        df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator, header= 0 if header else None,
                        usecols=['userId', 'movieId', 'rating', 'timestamp'],
                        dtype={'userId':str, 'movieId':str, 'rating':float, 'timestamp':float})
    else:
        df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator, header= 0 if header else None,
                        dtype={0:str, 1:str, 2:float, 3:float})

        df_original.columns = ['userId', 'movieId', 'rating', 'timestamp']

    # Remove data with rating non valid
    df_original.drop(df_original[df_original.rating == 0.0].index, inplace=True)

    user_id_list = df_original['userId'].values
    item_id_list = df_original['movieId'].values
    rating_list = df_original['rating'].values
    timestamp_list = df_original['timestamp'].values

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    URM_timestamp_builder.add_data_lists(user_id_list, item_id_list, timestamp_list)



    return  URM_all_builder.get_SparseMatrix(), \
            URM_all_builder.get_column_token_to_id_mapper(), \
            URM_all_builder.get_row_token_to_id_mapper(),\
            URM_timestamp_builder.get_SparseMatrix()





def _loadICM_genres(genres_path, header=True, separator=',', genresSeparator="|"):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(genres_path, "r", encoding="latin1")
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

            movie_id = line[0]

            title = line[1]
            # In case the title contains commas, it is enclosed in "..."
            # genre list will always be the last element
            genreList = line[-1]

            genreList = genreList.split(genresSeparator)

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(movie_id, genreList, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






def _loadICM_tags(tags_path, header=True, separator=',', if_new_item = "ignore",
                  item_original_ID_to_index = None, preinitialized_col_mapper = None):

    # Tags
    from Data_manager.TagPreprocessing import tagFilterAndStemming


    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = preinitialized_col_mapper, on_new_col = "add",
                                                    preinitialized_row_mapper = item_original_ID_to_index, on_new_row = if_new_item)



    fileHandle = open(tags_path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 100000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            # If a movie has no genre, ignore it
            movie_id = line[1]

            tagList = line[2]

            # Remove non alphabetical character and split on spaces
            tagList = tagFilterAndStemming(tagList)

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(movie_id, tagList, data = 1.0)


    fileHandle.close()



    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()





def _loadUCM(UCM_path, header=True, separator=','):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(UCM_path, "r", encoding="latin1")
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

            token_list = []
            token_list.append("gender_" + str(line[1]))
            token_list.append("age_group_" + str(line[2]))
            token_list.append("occupation_" + str(line[3]))
            token_list.append("zip_code_" + str(line[4]))

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(user_id, token_list, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






