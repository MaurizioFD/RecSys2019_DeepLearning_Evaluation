#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/04/2019

@author: Maurizio Ferrari Dacrema
"""



import numpy as np
import scipy.sparse as sps

def assert_implicit_data(URM_list):
    """
    Checks whether the URM in the list only contain implicit data in the form 1 or 0
    :param URM_list:
    :return:
    """

    for URM in URM_list:

        assert np.all(URM.data == np.ones_like(URM.data)), "assert_implicit_data: URM is not implicit as it contains data other than 1.0"


    print("Assertion assert_implicit_data: Passed")


def assert_disjoint_matrices(URM_list):
    """
    Checks whether the URM in the list have an empty intersection, therefore there is no data point contained in more than one
    URM at a time
    :param URM_list:
    :return:
    """

    URM_implicit_global = None

    cumulative_nnz = 0

    for URM in URM_list:

        cumulative_nnz += URM.nnz
        URM_implicit = URM.copy()
        URM_implicit.data = np.ones_like(URM_implicit.data)

        if URM_implicit_global is None:
            URM_implicit_global = URM_implicit

        else:
            URM_implicit_global += URM_implicit


    assert cumulative_nnz == URM_implicit_global.nnz, \
        "assert_disjoint_matrices: URM in list are not disjoint, {} data points are in more than one URM".format(cumulative_nnz-URM_implicit_global.nnz)


    return True





def assert_URM_ICM_mapper_consistency(URM_DICT, user_original_ID_to_index, item_original_ID_to_index,
                                      ICM_DICT, ICM_MAPPER_DICT,
                                      UCM_DICT, UCM_MAPPER_DICT,
                                      DATA_SPLITTER_NAME):

    print_preamble = "{} consistency check: ".format(DATA_SPLITTER_NAME)

    URM_shape = None

    for URM_name, URM_object in URM_DICT.items():

        if URM_shape is None:
            URM_shape = URM_object.shape

            URM_all = URM_object.copy()
            URM_all.data = np.ones_like(URM_all.data)

            n_users_URM, n_items_URM = URM_shape

            assert n_users_URM != 0, print_preamble + "Number of users in URM is 0"
            assert n_items_URM != 0, print_preamble + "Number of items in URM is 0"

        else:
            URM_implicit = URM_object.copy()
            URM_implicit.data = np.ones_like(URM_implicit.data)

            URM_all += URM_implicit

        assert URM_shape == URM_object.shape, print_preamble + "URM shape is inconsistent"


    assert n_users_URM != 0, print_preamble + "Number of users in URM is 0"
    assert n_items_URM != 0, print_preamble + "Number of items in URM is 0"

    # Check if item index-id and user index-id are consistent
    assert len(set(user_original_ID_to_index.values())) == len(user_original_ID_to_index), "user it-to-index mapper values do not have a 1-to-1 correspondance with the key"
    assert len(set(item_original_ID_to_index.values())) == len(item_original_ID_to_index), "item it-to-index mapper values do not have a 1-to-1 correspondance with the key"

    assert n_users_URM == len(user_original_ID_to_index), print_preamble + "user ID-to-index mapper contains a number of keys different then the number of users"
    assert n_items_URM == len(item_original_ID_to_index), print_preamble + "item ID-to-index mapper contains a number of keys different then the number of items"

    assert n_users_URM >= max(user_original_ID_to_index.values()), print_preamble + "user ID-to-index mapper contains indices greater than number of users"
    assert n_items_URM >= max(item_original_ID_to_index.values()), print_preamble + "item ID-to-index mapper contains indices greater than number of item"


    # Check if every non-empty user and item has a mapper value
    URM_all = sps.csc_matrix(URM_all)
    nonzero_items_mask = np.ediff1d(URM_all.indptr)>0
    nonzero_items = np.arange(0, n_items_URM, dtype=np.int)[nonzero_items_mask]
    assert np.isin(nonzero_items, np.array(list(item_original_ID_to_index.values()))).all(), print_preamble + "there exist items with interactions that do not have a mapper entry"


    URM_all = sps.csr_matrix(URM_all)
    nonzero_users_mask = np.ediff1d(URM_all.indptr)>0
    nonzero_users = np.arange(0, n_users_URM, dtype=np.int)[nonzero_users_mask]
    assert np.isin(nonzero_users, np.array(list(user_original_ID_to_index.values()))).all(), print_preamble + "there exist users with interactions that do not have a mapper entry"

    if ICM_MAPPER_DICT is not None:

        assert len(ICM_DICT) == len(ICM_MAPPER_DICT), print_preamble + "The available ICM and the available ICM mappers do not have the same length. ICMs are {}, mappers are {}".format(len(ICM_DICT), len(ICM_MAPPER_DICT))

        assert all(ICM_name in ICM_MAPPER_DICT for ICM_name in ICM_DICT.keys()), print_preamble + "Not all ICM sparse matrix have a corresponding ICM mapper"
        assert all(ICM_name in ICM_DICT for ICM_name in ICM_MAPPER_DICT.keys()), print_preamble + "Not all ICM mappers have a corresponding ICM sparse matrix"


        for ICM_name, ICM_object in ICM_DICT.items():

            assert ICM_name in ICM_MAPPER_DICT, print_preamble + "No mapper is available for ICM '{}'".format(ICM_name)

            feature_original_id_to_index = ICM_MAPPER_DICT[ICM_name]

            n_items_ICM, n_features = ICM_object.shape
            n_feature_occurrences = ICM_object.nnz

            assert n_items_ICM == n_items_URM, print_preamble + "Number of items in ICM {} is {} while in URM is {}".format(ICM_name, n_items_ICM, n_items_URM)
            assert n_features != 0, print_preamble + "Number of features in ICM {} is 0".format(ICM_name)
            assert n_feature_occurrences != 0, print_preamble + "Number of interactions in ICM {} is 0".format(ICM_name)


            assert n_features >= len(feature_original_id_to_index), print_preamble + "feature ID-to-index mapper contains more keys than features in ICM {}".format(ICM_name)
            assert n_features >= max(feature_original_id_to_index.values()), print_preamble + "feature ID-to-index mapper contains indices greater than number of features in ICM {}".format(ICM_name)

            # Check if every non-empty item and feature has a mapper value
            ICM_object = sps.csr_matrix(ICM_object)
            nonzero_items_mask = np.ediff1d(ICM_object.indptr)>0
            nonzero_items = np.arange(0, n_items_URM, dtype=np.int)[nonzero_items_mask]
            assert np.isin(nonzero_items, np.array(list(item_original_ID_to_index.values()))).all(), print_preamble + "there exist items with features that do not have a mapper entry in ICM {}".format(ICM_name)


            ICM_object = sps.csc_matrix(ICM_object)
            nonzero_features_mask = np.ediff1d(ICM_object.indptr)>0
            nonzero_features = np.arange(0, n_features, dtype=np.int)[nonzero_features_mask]
            assert np.isin(nonzero_features, np.array(list(feature_original_id_to_index.values()))).all(), print_preamble + "there exist users with interactions that do not have a mapper entry in ICM {}".format(ICM_name)






    if UCM_MAPPER_DICT is not None:

        assert len(UCM_DICT) == len(UCM_MAPPER_DICT), print_preamble + "The available UCM and the available UCM mappers do not have the same length. UCMs are {}, mappers are {}".format(len(UCM_DICT), len(UCM_MAPPER_DICT))

        assert all(UCM_name in UCM_MAPPER_DICT for UCM_name in UCM_DICT.keys()), print_preamble + "Not all UCM sparse matrix have a corresponding UCM mapper"
        assert all(UCM_name in UCM_DICT for UCM_name in UCM_MAPPER_DICT.keys()), print_preamble + "Not all UCM mappers have a corresponding UCM sparse matrix"


        for UCM_name, UCM_object in UCM_DICT.items():

            assert UCM_name in UCM_MAPPER_DICT, print_preamble + "No mapper is available for UCM '{}'".format(UCM_name)

            feature_original_id_to_index = UCM_MAPPER_DICT[UCM_name]

            n_users_UCM, n_features = UCM_object.shape
            n_feature_occurrences = UCM_object.nnz

            assert n_users_UCM == n_users_URM, print_preamble + "Number of users in UCM {} is {} while in URM is {}".format(UCM_name, n_users_UCM, n_users_URM)
            assert n_features != 0, print_preamble + "Number of features in UCM {} is 0".format(UCM_name)
            assert n_feature_occurrences != 0, print_preamble + "Number of interactions in UCM {} is 0".format(UCM_name)


            assert n_features >= len(feature_original_id_to_index), print_preamble + "feature ID-to-index mapper contains more keys than features in UCM {}".format(UCM_name)
            assert n_features >= max(feature_original_id_to_index.values()), print_preamble + "feature ID-to-index mapper contains indices greater than number of features in UCM {}".format(UCM_name)

            # Check if every non-empty user and feature has a mapper value
            UCM_object = sps.csr_matrix(UCM_object)
            nonzero_users_mask = np.ediff1d(UCM_object.indptr)>0
            nonzero_users = np.arange(0, n_users_URM, dtype=np.int)[nonzero_users_mask]
            assert np.isin(nonzero_users, np.array(list(user_original_ID_to_index.values()))).all(), print_preamble + "there exist users with features that do not have a mapper entry in UCM {}".format(UCM_name)


            UCM_object = sps.csc_matrix(UCM_object)
            nonzero_features_mask = np.ediff1d(UCM_object.indptr)>0
            nonzero_features = np.arange(0, n_features, dtype=np.int)[nonzero_features_mask]
            assert np.isin(nonzero_features, np.array(list(feature_original_id_to_index.values()))).all(), print_preamble + "there exist users with interactions that do not have a mapper entry in UCM {}".format(UCM_name)



