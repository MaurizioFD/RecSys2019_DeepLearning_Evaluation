#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/12/2018

@author: Maurizio Ferrari Dacrema
"""


import numpy as np

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


    print("Assertion assert_disjoint_matrices: Passed")