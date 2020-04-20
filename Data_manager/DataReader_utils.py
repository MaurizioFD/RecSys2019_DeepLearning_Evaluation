#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/18

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys, os
from Base.Recommender_utils import check_matrix
import  scipy.sparse as sps




def split_big_CSR_in_columns(sparse_matrix_to_split, num_split = 2):
    """
    The function returns a list of split for the given matrix
    :param sparse_matrix_to_split:
    :param num_split:
    :return:
    """

    assert sparse_matrix_to_split.shape[1]>0, "split_big_CSR_in_columns: sparse_matrix_to_split has no columns"
    assert num_split>=1 and num_split <= sparse_matrix_to_split.shape[1], "split_big_CSR_in_columns: num_split parameter not valid, value must be between 1 and {}, provided was {}".format(sparse_matrix_to_split.shape[1], num_split)


    if num_split == 1:
        return [sparse_matrix_to_split]



    n_column_split = int(sparse_matrix_to_split.shape[1]/num_split)

    sparse_matrix_split_list = []

    for num_current_split in range(num_split):

        start_col = n_column_split*num_current_split

        if num_current_split +1 == num_split:
            end_col = sparse_matrix_to_split.shape[1]
        else:
            end_col = n_column_split*(num_current_split + 1)

        print("split_big_CSR_in_columns: Split {}, columns: {}-{}".format(num_current_split, start_col, end_col))

        sparse_matrix_split_list.append(sparse_matrix_to_split[:,start_col:end_col])

    return sparse_matrix_split_list









def remove_empty_rows_and_cols(URM, ICM = None):

    URM = check_matrix(URM, "csr")
    rows = URM.indptr
    numRatings = np.ediff1d(rows)
    user_mask = numRatings >= 1

    URM = URM[user_mask,:]

    cols = URM.tocsc().indptr
    numRatings = np.ediff1d(cols)
    item_mask = numRatings >= 1

    URM = URM[:,item_mask]

    removedUsers = np.arange(0, len(user_mask))[np.logical_not(user_mask)]
    removedItems = np.arange(0, len(item_mask))[np.logical_not(item_mask)]

    if ICM is not None:

        ICM = ICM[item_mask,:]

        return URM.tocsr(), ICM.tocsr(), removedUsers, removedItems


    return URM.tocsr(), removedUsers, removedItems





from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
import pandas as pd


def load_CSV_into_SparseBuilder (filePath, header = False, separator="::", timestamp = False, remove_duplicates = False,
                                 custom_user_item_rating_columns = None):

    URM_all_builder = IncrementalSparseMatrix(auto_create_col_mapper = True, auto_create_row_mapper = True)
    URM_timestamp_builder = IncrementalSparseMatrix(auto_create_col_mapper = True, auto_create_row_mapper = True)

    if timestamp:
        dtype={0:str, 1:str, 2:float, 3:float}
        columns = ['userId', 'itemId', 'interaction', 'timestamp']

    else:
        dtype={0:str, 1:str, 2:float}
        columns = ['userId', 'itemId', 'interaction']

    df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator, header= 0 if header else None,
                    dtype=dtype, usecols=custom_user_item_rating_columns)

    # If the original file has more columns, keep them but ignore them
    df_original.columns = columns


    user_id_list = df_original['userId'].values
    item_id_list = df_original['itemId'].values
    interaction_list = df_original['interaction'].values

    # Check if duplicates exist
    num_unique_user_item_ids = df_original.drop_duplicates(['userId', 'itemId'], keep='first', inplace=False).shape[0]
    contains_duplicates_flag = num_unique_user_item_ids != len(user_id_list)

    if contains_duplicates_flag:
        if remove_duplicates:
            # # Remove duplicates.

            # This way of removing the duplicates keeping the last tiemstamp without removing other columns
            # would be the simplest, but it is so slow to the point of being unusable on any dataset but ML100k
            # idxs = df_original.groupby(by=['userId', 'itemId'], as_index=False)["timestamp"].idxmax()
            # df_original = df_original.loc[idxs]

            # Alternative faster way:
            # 1 - Sort in ascending order so that the last (bigger) timestamp is in the last position. Set Nan to be in the first position, to remove them if possible
            # 2 - Then remove duplicates for user-item keeping the last row, which will be the last timestamp.

            if timestamp:
                sort_by = ["userId", "itemId", "timestamp"]
            else:
                sort_by = ["userId", "itemId", 'interaction']

            df_original.sort_values(by=sort_by, ascending=True, inplace=True, kind="quicksort", na_position="first")
            df_original.drop_duplicates(["userId", "itemId"], keep='last', inplace=True)

            user_id_list = df_original['userId'].values
            item_id_list = df_original['itemId'].values
            interaction_list = df_original['interaction'].values

            assert num_unique_user_item_ids == len(user_id_list), "load_CSV_into_SparseBuilder: duplicate (user, item) values found"

        else:
            assert num_unique_user_item_ids == len(user_id_list), "load_CSV_into_SparseBuilder: duplicate (user, item) values found"




    URM_all_builder.add_data_lists(user_id_list, item_id_list, interaction_list)

    if timestamp:
        timestamp_list = df_original['timestamp'].values
        URM_timestamp_builder.add_data_lists(user_id_list, item_id_list, timestamp_list)

        return  URM_all_builder.get_SparseMatrix(), URM_timestamp_builder.get_SparseMatrix(), \
                URM_all_builder.get_column_token_to_id_mapper(), URM_all_builder.get_row_token_to_id_mapper()



    return  URM_all_builder.get_SparseMatrix(), \
            URM_all_builder.get_column_token_to_id_mapper(), URM_all_builder.get_row_token_to_id_mapper()






def merge_ICM(ICM1, ICM2, mapper_ICM1, mapper_ICM2):

    ICM_all = sps.hstack([ICM1, ICM2], format='csr')

    mapper_ICM_all = mapper_ICM1.copy()

    for key in mapper_ICM2.keys():
        mapper_ICM_all[key] = mapper_ICM2[key] + len(mapper_ICM1)

    return  ICM_all, mapper_ICM_all



def compute_density(URM):

    n_users, n_items = URM.shape
    n_interactions = URM.nnz

    # This avoids the fixed bit representation of numpy preventing
    # an overflow when computing the product
    n_items = float(n_items)
    n_users = float(n_users)

    if n_interactions == 0:
        return 0.0

    return n_interactions/(n_items*n_users)




def remove_features(ICM, min_occurrence = 5, max_percentage_occurrence = 0.30, reconcile_mapper = None):
    """
    The function eliminates the values associated to feature occurring in less than the minimal percentage of items
    or more then the max. Shape of ICM is reduced deleting features.
    :param ICM:
    :param minPercOccurrence:
    :param max_percentage_occurrence:
    :param reconcile_mapper: DICT mapper [token] -> index
    :return: ICM
    :return: deletedFeatures
    :return: DICT mapper [token] -> index
    """

    ICM = check_matrix(ICM, 'csc')

    n_items = ICM.shape[0]

    cols = ICM.indptr
    numOccurrences = np.ediff1d(cols)

    feature_mask = np.logical_and(numOccurrences >= min_occurrence, numOccurrences <= n_items * max_percentage_occurrence)

    ICM = ICM[:,feature_mask]

    deletedFeatures = np.arange(0, len(feature_mask))[np.logical_not(feature_mask)]

    print("RemoveFeatures: removed {} features with less then {} occurrences, removed {} features with more than {} occurrencies".format(
        sum(numOccurrences < min_occurrence), min_occurrence,
        sum(numOccurrences > n_items * max_percentage_occurrence), int(n_items * max_percentage_occurrence)
    ))

    if reconcile_mapper is not None:
        reconcile_mapper = reconcile_mapper_with_removed_tokens(reconcile_mapper, deletedFeatures)

        return ICM, deletedFeatures, reconcile_mapper


    return ICM, deletedFeatures



def reconcile_mapper_with_removed_tokens(key_to_value_dict, values_to_remove):
    """

    :param mapper_dict: must be a mapper of [token] -> index
    :param indices_to_remove:
    :return:
    """

    # When an index has to be removed:
    # - Delete the corresponding key
    # - Decrement all greater indices

    # Get all values of the mapper into an array to speed-up the decrementing process
    # We need a 1-to-1 association between the mapper key and the array position

    # Assumptions: in dictionary mapper_dict there is a 1-to-1 association to an index
    assert len(set(key_to_value_dict.values())) == len(key_to_value_dict), "mapper_dict values do not have a 1-to-1 correspondance with the key"

    # The value is an index, so we can use it to be both the value and the index of an array.
    # We do not assume values to be contiguous, the missing ones will be -np.inf
    mapper_values_array = np.ones(max(key_to_value_dict.values())+1, dtype=np.int) * -np.inf

    value_to_key = invert_dictionary(key_to_value_dict)


    # Set all old indices
    for key, old_index in key_to_value_dict.items():
        mapper_values_array[old_index] = old_index


    # Set to -np.inf all indices to be removed
    # Remove keys in original dictionary
    for value_to_remove in values_to_remove:

        mapper_values_array[value_to_remove] = -np.inf

        assert value_to_remove in value_to_key, "Value to be removed from dictionary is not in dictionary"

        key_to_remove = value_to_key[value_to_remove]

        del key_to_value_dict[key_to_remove]


    # To update the indices, start from 0 and allocate the index n to the n-th finite value in mapper_values_array
    # Use cumulative sum, each cell is equals to the number of finite (e.g. valid) cells before
    # Ensure the first index is 0 and not 1
    mapper_values_array_finite = np.isfinite(mapper_values_array)

    mapper_values_array_new_indices = np.cumsum(mapper_values_array_finite)
    mapper_values_array_new_indices -= 1

    # Replace old value with new
    for key, old_index in key_to_value_dict.items():

        new_index = mapper_values_array_new_indices[old_index]
        key_to_value_dict[key] = new_index


    return key_to_value_dict




def download_from_URL(URL, folder_path, file_name):

    import urllib
    from urllib.request import urlretrieve

    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:

        urlretrieve (URL, folder_path + file_name, reporthook=urllretrieve_reporthook)

    except urllib.request.URLError as urlerror:

        print("Unable to complete automatic download, network error")
        raise urlerror




    sys.stdout.write("\n")
    sys.stdout.flush()








def urllretrieve_reporthook(count, block_size, total_size):

    global start_time_urllretrieve

    if count == 0:
        start_time_urllretrieve = time.time()
        return

    if total_size < 0 or not np.isfinite(total_size):
        total_size = np.nan

    duration = time.time() - start_time_urllretrieve + 1

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(float(count*block_size*100/total_size),100)

    sys.stdout.write("\rDataReader: Downloaded {:.2f}%, {:.2f} MB, {:.0f} KB/s, {:.0f} seconds passed".format(
                    percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()






def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]

        assert index not in index_to_id, "Dictionary is not invertible as it contains duplicate values."
        index_to_id[index] = id

    return index_to_id





def add_boolean_matrix_iterator(original_data_dict):

    output_data_dict = {}

    for matrix_name, matrix_object in original_data_dict.items():
        output_data_dict[matrix_name] = matrix_object

        if np.max(matrix_object.data) != 1.0 or np.min(matrix_object.data) != 1.0:
            matrix_object_implicit = matrix_object.copy()
            matrix_object_implicit.astype(np.bool, copy=True)
            matrix_object_implicit.data = np.ones_like(matrix_object.data)

            output_data_dict[matrix_name + "_bool"] = matrix_object_implicit

    return output_data_dict